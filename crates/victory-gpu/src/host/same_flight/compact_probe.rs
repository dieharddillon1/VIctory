//! Software-only proof for mapping a bounded compact gather at a flight fence.

use super::compact::GATHER;
use super::tests::{fill, prelude};
use crate::host::{Prelude, compile, warp_adapter};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};

const HEADER: usize = 16;
const PIECE: usize = 64 << 10;
const SOURCE: usize = PIECE * 4;
const TAG: u32 = 0x877b2ac1;
const CANARY: u8 = 0xcd;

const ROWS: [[u32; 4]; 4] = [
    [0, PIECE as u32, 0, 2], // Duplicate partial declarations, first flag only.
    [PIECE as u32, PIECE as u32, 2, 1], // Unchanged.
    [2 * PIECE as u32 + 16, 32, 3, 1], // Short, aligned envelope.
    [3 * PIECE as u32 + 16, 32, 4, 0], // No compare, must be read.
];
const FLAGS: [u32; 8] = [0, 1, 0, 0, 1, 0, 0, 0];
const REFS: [u32; 4] = [1, 7, 2, 4];
const REQUIRED: usize = (HEADER + PIECE) + 2 * (HEADER + 32);

fn buffer(
    device: &ID3D12Device,
    heap: D3D12_HEAP_PROPERTIES,
    state: D3D12_RESOURCE_STATES,
    flags: D3D12_RESOURCE_FLAGS,
    size: usize,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: known test properties for a new buffer.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut resource,
        )
    }
    .unwrap();
    resource.unwrap()
}

fn upload(device: &ID3D12Device, data: &[u8]) -> ID3D12Resource {
    let buffer = buffer(
        device,
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_FLAG_NONE,
        data.len(),
    );
    fill(&buffer, data);
    buffer
}

fn encode(words: impl IntoIterator<Item = u32>) -> Vec<u8> {
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn word(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn decode(bytes: &[u8], capacity: usize) -> Option<[Option<&[u8]>; ROWS.len()]> {
    if word(bytes, 4)? != 0 {
        return None;
    }
    let count = word(bytes, 0)? as usize;
    if count > capacity {
        return None;
    }
    let mut seen = [None; ROWS.len()];
    let mut data = bytes.get(HEADER..HEADER + count)?;
    while !data.is_empty() {
        let id = word(data, 0)? as usize;
        let length = word(data, 4)? as usize;
        let row = ROWS.get(id)?;
        if word(data, 12)? != TAG
            || length != row[1] as usize
            || word(data, 8)? != row[0]
            || seen[id].is_some()
        {
            return None;
        }
        let len = HEADER.checked_add(length)?;
        seen[id] = Some(data.get(HEADER..len)?);
        data = data.get(len..)?;
    }
    if seen.iter().enumerate().any(|(i, data)| {
        let row = ROWS[i];
        let changed =
            row[3] == 0 || (0..row[3]).any(|f| FLAGS[REFS[(row[2] + f) as usize] as usize] != 0);
        changed != data.is_some()
    }) {
        return None;
    }
    Some(seen)
}

struct Capture {
    buffer: ID3D12Resource,
    list: Prelude,
    data: *mut u8,
    total: usize,
    capacity: usize,
}

impl Capture {
    fn new(device: &ID3D12Device, capacity: usize) -> Self {
        let total = HEADER + capacity + 64;
        let buffer = buffer(
            device,
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            total,
        );
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the new buffer is idle and has `total` mapped bytes.
        // Keep the map for the flight as the live shared-buffer path does.
        let data = unsafe {
            buffer
                .Map(0, Some(&D3D12_RANGE::default()), Some(&mut mapped))
                .unwrap();
            let data = mapped.cast::<u8>();
            assert!(!data.is_null());
            std::ptr::write_bytes(data, CANARY, total);
            std::ptr::write_bytes(data, 0, 12);
            data
        };
        Self {
            buffer,
            list: prelude(device),
            data,
            total,
            capacity,
        }
    }

    fn record(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        source: &ID3D12Resource,
        flags: &ID3D12Resource,
        work: &ID3D12Resource,
    ) -> ID3D12CommandList {
        self.record_with_words(root, pso, source, flags, work, FLAGS.len() as u32)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_with_words(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        source: &ID3D12Resource,
        flags: &ID3D12Resource,
        work: &ID3D12Resource,
        flag_words: u32,
    ) -> ID3D12CommandList {
        let (allocator, list) = &self.list;
        let constants = [
            SOURCE as u32,
            flag_words,
            ROWS.len() as u32,
            self.capacity as u32,
            REFS.len() as u32,
            0,
        ];
        // SAFETY: isolated idle list and valid buffers; every resource
        // stays alive through the software queue fence.
        unsafe {
            allocator.Reset().unwrap();
            list.Reset(allocator, Some(pso)).unwrap();
            list.SetComputeRootSignature(root);
            list.SetComputeRoot32BitConstants(
                0,
                constants.len() as u32,
                constants.as_ptr().cast(),
                0,
            );
            list.SetComputeRootShaderResourceView(1, source.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, flags.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(3, work.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(4, self.buffer.GetGPUVirtualAddress());
            list.Dispatch(ROWS.len() as u32, 1, 1);
            list.Close().unwrap();
            list.cast().unwrap()
        }
    }

    fn check(&self, expected: Option<&[u8]>) {
        // SAFETY: the software queue fence has completed and the
        // read-only test view is contained in the persistent buffer map.
        let data = unsafe { std::slice::from_raw_parts(self.data, self.total) };
        assert_eq!(word(data, 0), Some(REQUIRED as u32));
        assert_eq!(&data[12..16], &[CANARY; 4]);
        assert_eq!(&data[HEADER + self.capacity..], &[CANARY; 64]);
        let parse = decode(data, self.capacity);
        let Some(source) = expected else {
            assert_eq!(word(data, 4), Some(1));
            assert!(parse.is_none());
            return;
        };
        let found = parse.expect("bounded snapshot with all flagged records");
        for i in [0, 2, 3] {
            let [at, bytes, _, _] = ROWS[i];
            assert_eq!(
                found[i].unwrap(),
                &source[at as usize..(at + bytes) as usize]
            );
        }
        assert!(found[1].is_none());
        // Both disjoint declarations for this piece must be delivered even
        // though the second flag was zero. The intervening bytes stay canary.
        let mut title = [CANARY; 16];
        let payload = found[0].unwrap();
        for span in [3..5, 9..15] {
            title[span.clone()].copy_from_slice(&payload[span]);
        }
        assert_eq!(&title[3..5], &source[3..5]);
        assert_eq!(&title[9..15], &source[9..15]);
        assert_eq!(&title[5..9], &[CANARY; 4]);
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // SAFETY: the map belongs to this buffer, and the test waits for
        // all submissions before destroying the capture.
        unsafe { self.buffer.Unmap(0, None) };
    }
}

#[test]
fn warp_compacts_only_flagged_pieces_into_bounded_flight_uav() {
    let adapter = warp_adapter().expect("software DXGI adapter");
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: the existing helper refuses any adapter without the software flag.
    unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.unwrap();
    let device = device.unwrap();
    let shader = compile(GATHER, s!("main"), s!("cs_5_1")).expect("compile bounded gather");
    // SAFETY: shader bytes contain the root signature and stay live
    // until the software device has created the root and PSO.
    let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &shader).unwrap();
        let pso = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: shader.as_ptr().cast(),
                    BytecodeLength: shader.len(),
                },
                ..Default::default()
            })
            .unwrap();
        (root, pso)
    };
    let source = buffer(
        &device,
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_NONE,
        SOURCE,
    );
    let first = (0..SOURCE).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    let second = (0..SOURCE)
        .map(|i| ((i + 87) % 251) as u8)
        .collect::<Vec<_>>();
    let first_upload = upload(&device, &first);
    let second_upload = upload(&device, &second);
    let flags = upload(&device, &encode(FLAGS));
    let work = upload(&device, &encode(ROWS.into_iter().flatten().chain(REFS)));
    let a = Capture::new(&device, REQUIRED);
    let b = Capture::new(&device, REQUIRED);
    let overflow = Capture::new(&device, HEADER + 32);
    let before = prelude(&device);
    let after = prelude(&device);
    let write = |pair: &Prelude, data: &ID3D12Resource| -> ID3D12CommandList {
        // SAFETY: idle list; the byte counts exactly cover both live buffers.
        unsafe {
            pair.0.Reset().unwrap();
            pair.1.Reset(&pair.0, None).unwrap();
            pair.1.CopyBufferRegion(&source, 0, data, 0, SOURCE as u64);
            pair.1.Close().unwrap();
            pair.1.cast().unwrap()
        }
    };
    let produce_a = write(&before, &first_upload);
    let produce_b = write(&after, &second_upload);
    let gather_a = a.record(&root, &pso, &source, &flags, &work);
    let gather_b = b.record(&root, &pso, &source, &flags, &work);
    let gather_overflow = overflow.record(&root, &pso, &source, &flags, &work);
    // SAFETY: the queue can only submit to WARP, all objects are kept
    // until its fence completes and every list executes once.
    unsafe {
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
            .unwrap();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.ExecuteCommandLists(&[Some(produce_a)]);
        queue.ExecuteCommandLists(&[Some(gather_a)]);
        queue.Signal(&fence, 1).unwrap();
        queue.ExecuteCommandLists(&[Some(produce_b)]);
        queue.ExecuteCommandLists(&[Some(gather_b)]);
        queue.Signal(&fence, 2).unwrap();
        queue.ExecuteCommandLists(&[Some(gather_overflow)]);
        queue.Signal(&fence, 3).unwrap();
        let event = CreateEventW(None, false, false, None).unwrap();
        fence.SetEventOnCompletion(3, event).unwrap();
        assert_eq!(WaitForSingleObject(event, 10_000).0, 0);
        windows::Win32::Foundation::CloseHandle(event).unwrap();
        device.GetDeviceRemovedReason().unwrap();
    }
    a.check(Some(&first));
    b.check(Some(&second));
    overflow.check(None);
}

fn compile_compute(
    device: &ID3D12Device,
    source: &str,
) -> (ID3D12RootSignature, ID3D12PipelineState) {
    let bytes = compile(source, s!("main"), s!("cs_5_1")).expect("compile compute");
    // SAFETY: the shader contains its root signature, and the bytes and
    // borrowed root live through pipeline creation.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytes).unwrap();
        let pipeline = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })
            .unwrap();
        (root, pipeline)
    }
}

fn compute_device() -> ID3D12Device {
    let adapter = warp_adapter().expect("software DXGI adapter");
    let mut device = None;
    // SAFETY: the helper refuses adapters without the software flag.
    unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.unwrap();
    device.unwrap()
}

#[test]
fn warp_handoff_from_real_compare_to_compaction_and_flight_fence() {
    let device = compute_device();
    let (compare_root, compare) = compile_compute(&device, crate::host::CARD_COMPARE);
    let (root, gather) = compile_compute(&device, GATHER);
    let default_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let shared_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let source = buffer(
        &device,
        default_heap,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_NONE,
        SOURCE,
    );
    let hashes = buffer(
        &device,
        default_heap,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        64,
    );
    // Same COMMON, CPU-mapped custom WRITE_BACK/L0 buffer as CardWork::differs.
    let flags = buffer(
        &device,
        shared_heap,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        64,
    );
    fill(&flags, &[0u8; 64]);
    let before = vec![22u8; SOURCE];
    let scene = |seed: usize| {
        (0..SOURCE)
            .map(|byte| {
                if byte / PIECE == 1 {
                    22
                } else {
                    ((byte + seed) % 251) as u8
                }
            })
            .collect::<Vec<_>>()
    };
    let first = scene(29);
    let second = scene(73);
    let uploads = [
        upload(&device, &before),
        upload(&device, &first),
        upload(&device, &second),
    ];
    let producer = [prelude(&device), prelude(&device), prelude(&device)];
    let record_compare = |flight: usize| -> ID3D12CommandList {
        let (allocator, list) = &producer[flight];
        // SAFETY: owned idle lists, valid copy ranges, and resources retained
        // until the last fence. All compare indices refer to assigned words.
        unsafe {
            allocator.Reset().unwrap();
            list.Reset(allocator, Some(&compare)).unwrap();
            list.CopyBufferRegion(&source, 0, &uploads[flight], 0, SOURCE as u64);
            list.ResourceBarrier(&[crate::host::transition(
                &source,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            list.SetComputeRootSignature(&compare_root);
            list.SetComputeRootShaderResourceView(1, source.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, flags.GetGPUVirtualAddress());
            if flight == 0 {
                let constants = [0, 0, SOURCE as u32, 4, 0, 0];
                list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                list.Dispatch(4, 1, 1);
            } else {
                let base = ((flight - 1) * FLAGS.len()) as u32;
                for (piece, flag) in [(0, 1), (1, 2), (2, 4), (0, 7)] {
                    let constants = [piece, 0, SOURCE as u32, 1, (base + flag) * 4, 2];
                    list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                    list.Dispatch(1, 1, 1);
                    // The second declaration of piece 0 compares against
                    // the first one's new hash and writes a distinct flag.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                }
            }
            list.Close().unwrap();
            list.cast().unwrap()
        }
    };
    let mut gathers = Vec::new();
    let mut captures = Vec::new();
    let mut work_uploads = Vec::new();
    for i in 0..2 {
        let offset = i as u32 * FLAGS.len() as u32;
        let work = upload(
            &device,
            &encode(ROWS.into_iter().flatten().chain(REFS.map(|r| r + offset))),
        );
        let capture = Capture::new(&device, REQUIRED);
        gathers.push(capture.record_with_words(
            &root,
            &gather,
            &source,
            &flags,
            &work,
            (2 * FLAGS.len()) as u32,
        ));
        work_uploads.push(work);
        captures.push(capture);
    }
    let overflow = Capture::new(&device, HEADER + 32);
    let overflow_list = overflow.record_with_words(
        &root,
        &gather,
        &source,
        &flags,
        &work_uploads[1],
        (2 * FLAGS.len()) as u32,
    );
    // SAFETY: an isolated WARP queue and closed, owned command lists. Each
    // compare/gather pair has a real flight fence; no CPU flag read or second
    // relay enqueue intervenes. Separate Execute calls let buffers decay.
    unsafe {
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
            .unwrap();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.ExecuteCommandLists(&[Some(record_compare(0))]);
        for (i, gather) in gathers.into_iter().enumerate() {
            queue.ExecuteCommandLists(&[Some(record_compare(i + 1))]);
            queue.ExecuteCommandLists(&[Some(gather)]);
            queue.Signal(&fence, i as u64 + 1).unwrap();
        }
        queue.ExecuteCommandLists(&[Some(overflow_list)]);
        queue.Signal(&fence, 3).unwrap();
        let event = CreateEventW(None, false, false, None).unwrap();
        fence.SetEventOnCompletion(3, event).unwrap();
        assert_eq!(WaitForSingleObject(event, 10_000).0, 0);
        windows::Win32::Foundation::CloseHandle(event).unwrap();
        device.GetDeviceRemovedReason().unwrap();
        let mut mapped = std::ptr::null_mut();
        flags
            .Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 64 }),
                Some(&mut mapped),
            )
            .unwrap();
        let observed = std::slice::from_raw_parts(mapped.cast::<u32>(), 16);
        for block in observed.chunks_exact(FLAGS.len()) {
            for flag in REFS {
                assert_eq!(block[flag as usize], FLAGS[flag as usize], "flag {flag}");
            }
        }
        flags.Unmap(0, Some(&D3D12_RANGE::default()));
    }
    captures[0].check(Some(&first));
    captures[1].check(Some(&second));
    overflow.check(None);
}
