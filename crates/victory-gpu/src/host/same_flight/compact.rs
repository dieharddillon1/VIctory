pub(super) const GATHER: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), SRV(t2), UAV(u0)"
cbuffer Control : register(b0) {
    uint source_bytes;
    uint flag_words;
    uint work_count;
    uint capacity;
    uint refs_count;
    uint first_work;
};
ByteAddressBuffer source : register(t0);
ByteAddressBuffer differs : register(t1);
ByteAddressBuffer work : register(t2);
RWByteAddressBuffer output : register(u0);
groupshared uint record;

void overflow() {
    uint ignored;
    output.InterlockedOr(4u, 1u, ignored);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint job = first_work + group.x;
    if (job >= work_count) return;
    const uint4 row = work.Load4(job * 16u);
    // Rows contain aligned source offset and length, flag index and count.
    // Flags are validated by the host before providing the work upload.
    if (lane == 0u) {
        record = 0u;
        bool valid = ((row.x | row.y) & 15u) == 0u && row.y != 0u &&
            row.y <= 65552u && row.x <= source_bytes && row.y <= source_bytes - row.x;
        valid = valid && row.z <= refs_count && row.w <= refs_count - row.z;
        if (!valid) {
            overflow();
        } else {
            bool changed = row.w == 0u;
            for (uint i = 0u; i < row.w; ++i) {
                const uint flag = work.Load(work_count * 16u + (row.z + i) * 4u);
                if (flag < flag_words) {
                    changed = changed || differs.Load(flag * 4u) != 0u;
                } else {
                    valid = false;
                }
            }
            if (!valid) {
                overflow();
            } else if (changed) {
                const uint bytes = 16u + row.y;
                uint at;
                output.InterlockedAdd(0u, bytes, at);
                if (at <= capacity && bytes <= capacity - at) {
                    record = 16u + at;
                    output.Store4(record, uint4(job, row.y, row.x, 0x877b2ac1u));
                    uint ignored;
                    output.InterlockedAdd(8u, row.y, ignored);
                } else {
                    overflow();
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (record == 0u) return;
    for (uint at = lane * 16u; at < row.y; at += 64u * 16u) {
        output.Store4(record + 16u + at, source.Load4(row.x + at));
    }
}
"#;

use super::{COPY_BACK, CardPlan, CardWork, Host, Order, Snapshot};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

const HEADER: usize = 16;
const TAG: u32 = 0x877b2ac1;
const CAPACITY: usize = 16 << 20;
const MAX_SLOTS: usize = 6;
const MAX_ROWS: usize = 2048;
const MAX_PIECES: usize = 8192;
const MAX_GROUPS: usize = 32;
const MAX_ENVELOPE: usize = 65_552;

#[derive(Clone, Copy)]
pub(super) enum Reject {
    Bounds,
    Pool,
    Host,
    Alias,
    Order,
    Overflow,
    Integrity,
    Landing,
}

static REJECTED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static SUBMITTED: AtomicU64 = AtomicU64::new(0);
static ACCEPTED: AtomicU64 = AtomicU64::new(0);
static GPU_PAYLOAD: AtomicU64 = AtomicU64::new(0);
static TRANSFERRED: AtomicU64 = AtomicU64::new(0);
static MERGED: AtomicU64 = AtomicU64::new(0);

pub(super) fn rejected(reason: Reject) {
    REJECTED[reason as usize].fetch_add(1, Relaxed);
}

pub(super) fn submitted() {
    SUBMITTED.fetch_add(1, Relaxed);
}

pub(super) fn report(frames: f64) {
    let [
        bounds,
        pool,
        host,
        alias,
        order,
        overflow,
        integrity,
        landing,
    ] = REJECTED.each_ref().map(|counter| counter.swap(0, Relaxed));
    let [submitted, accepted, transferred, merged] =
        [&SUBMITTED, &ACCEPTED, &TRANSFERRED, &MERGED].map(|counter| counter.swap(0, Relaxed));
    tracing::info!(
        frames,
        submitted,
        accepted,
        bounds,
        pool,
        host,
        alias,
        order,
        overflow,
        integrity,
        landing,
        gpu_payload_mib_per_frame = GPU_PAYLOAD.swap(0, Relaxed) as f64 / frames / 1_048_576.0,
        copied_mib_per_frame = transferred as f64 / frames / 1_048_576.0,
        merged_mib_per_frame = merged as f64 / frames / 1_048_576.0,
        "compact flight readback interval"
    );
}

struct Group {
    range: usize,
    owner: usize,
    width: u32,
    first: u32,
    count: u32,
}

struct Row {
    lo: usize,
    hi: usize,
    flags: Vec<u32>,
    always: bool,
}

struct Piece {
    row: usize,
    range: usize,
    lo: usize,
    hi: usize,
}

#[derive(Default)]
struct Layout {
    groups: Vec<Group>,
    rows: Vec<Row>,
    pieces: Vec<Piece>,
}

struct Declaration {
    range: usize,
    owner: usize,
    width: u32,
    index: u32,
    lo: usize,
    hi: usize,
    flag: Option<u32>,
}

impl Layout {
    fn build(declarations: impl Iterator<Item = Declaration>, flag_words: u32) -> Option<Self> {
        let mut result = Self::default();
        let mut rows = HashMap::new();
        for d in declarations {
            if d.lo >= d.hi {
                continue;
            }
            if result.pieces.len() == MAX_PIECES
                || d.hi > d.width as usize
                || d.flag.is_some_and(|index| index >= flag_words)
            {
                return None;
            }
            let lo = d.lo & !15;
            let hi = d.hi.checked_add(15)? & !15;
            if hi > d.width as usize || hi - lo > MAX_ENVELOPE {
                return None;
            }
            let row_id = match rows.entry((d.owner, d.index)) {
                std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let next = result.rows.len();
                    if next == MAX_ROWS {
                        return None;
                    }
                    if !result
                        .groups
                        .last()
                        .is_some_and(|last| last.owner == d.owner)
                    {
                        if result.groups.len() == MAX_GROUPS {
                            return None;
                        }
                        result.groups.push(Group {
                            range: d.range,
                            owner: d.owner,
                            width: d.width,
                            first: next as u32,
                            count: 0,
                        });
                    }
                    result.groups.last_mut()?.count += 1;
                    result.rows.push(Row {
                        lo,
                        hi,
                        flags: Vec::new(),
                        always: false,
                    });
                    *e.insert(next)
                }
            };
            let row = &mut result.rows[row_id];
            row.lo = row.lo.min(lo);
            row.hi = row.hi.max(hi);
            if row.hi - row.lo > MAX_ENVELOPE {
                return None;
            }
            if let Some(index) = d.flag {
                if !row.always {
                    row.flags.push(index);
                }
            } else {
                row.always = true;
                row.flags.clear();
            }
            result.pieces.push(Piece {
                row: row_id,
                range: d.range,
                lo: d.lo,
                hi: d.hi,
            });
        }
        (!result.rows.is_empty()).then_some(result)
    }

    fn upload(&self) -> (Vec<u32>, u32) {
        let mut rows = Vec::with_capacity(self.rows.len() * 4 + self.pieces.len());
        let mut refs = Vec::new();
        for row in &self.rows {
            let first = refs.len() as u32;
            refs.extend_from_slice(&row.flags);
            rows.extend_from_slice(&[
                row.lo as u32,
                (row.hi - row.lo) as u32,
                first,
                row.flags.len() as u32,
            ]);
        }
        let total = refs.len() as u32;
        rows.extend(refs);
        (rows, total)
    }

    fn validate<'a>(
        &self,
        output: &'a [u8],
        flags: &[u32],
    ) -> Result<Vec<Option<&'a [u8]>>, Reject> {
        if word(output, 4).ok_or(Reject::Integrity)? != 0 {
            return Err(Reject::Overflow);
        }
        let size = word(output, 0).ok_or(Reject::Integrity)? as usize;
        let expected = word(output, 8).ok_or(Reject::Integrity)? as usize;
        if size > CAPACITY {
            return Err(Reject::Overflow);
        }
        let mut rows = output.get(HEADER..HEADER + size).ok_or(Reject::Integrity)?;
        let mut present = vec![None; self.rows.len()];
        let mut payload = 0usize;
        while !rows.is_empty() {
            let id = word(rows, 0).ok_or(Reject::Integrity)? as usize;
            let len = word(rows, 4).ok_or(Reject::Integrity)? as usize;
            let lo = word(rows, 8).ok_or(Reject::Integrity)? as usize;
            let spec = self.rows.get(id).ok_or(Reject::Integrity)?;
            if spec.lo != lo
                || spec.hi - spec.lo != len
                || present[id].is_some()
                || word(rows, 12) != Some(TAG)
            {
                return Err(Reject::Integrity);
            }
            let (body, rest) = rows
                .get(HEADER..)
                .and_then(|p| p.split_at_checked(len))
                .ok_or(Reject::Integrity)?;
            present[id] = Some(body);
            rows = rest;
            payload += len;
        }
        if payload != expected
            || present.iter().zip(&self.rows).any(|(found, row)| {
                let needed = row.always
                    || row
                        .flags
                        .iter()
                        .any(|&index| flags.get(index as usize).is_none_or(|&flag| flag != 0));
                needed != found.is_some()
            })
        {
            return Err(Reject::Integrity);
        }
        Ok(present)
    }
}

fn word(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

struct Kernel {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}

impl Kernel {
    fn new(device: &ID3D12Device) -> Option<Self> {
        let bytes = super::super::compile(GATHER, s!("main"), s!("cs_5_1")).ok()?;
        // SAFETY: the shader contains its root signature. The binary and
        // the borrowed pointer live through pipeline creation.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytes).ok()?;
            let pipeline = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            Some(Self { root, pipeline })
        }
    }
}

#[derive(Default)]
pub(super) struct Pool {
    buffers: Vec<(ID3D12Resource, usize)>,
    active: usize,
    kernel: std::sync::OnceLock<Option<Kernel>>,
}

impl Pool {
    pub(super) fn retire(&mut self, buffer: ID3D12Resource, capture: Capture) {
        // SAFETY: card_done is after this flight's fence. The header is
        // available even on overflow and when the CPU used legacy fallback.
        let header = unsafe { std::slice::from_raw_parts(capture.data as *const u8, HEADER) };
        let payload = u64::from(word(header, 8).unwrap_or(0)).min(CAPACITY as u64);
        GPU_PAYLOAD.fetch_add(payload, Relaxed);
        self.buffers.push((buffer, capture.data));
        self.active -= 1;
    }
}

pub(super) struct Capture {
    data: usize,
    _input: ID3D12Resource,
    layout: Layout,
}

pub(super) fn prepare(
    host: &Host,
    work: &mut CardWork,
    plan: &CardPlan,
) -> Result<(Snapshot, super::super::Prelude), Reject> {
    if work.snapshots.compact.active == MAX_SLOTS {
        return Err(Reject::Pool);
    }
    let flag_words = u32::try_from(work.differs.2 / 4).map_err(|_| Reject::Bounds)?;
    let mut declarations = Vec::new();
    for (range, r) in plan.ranges.iter().enumerate() {
        let (_, _, width) = r.shadow.card_sides().ok_or(Reject::Bounds)?;
        let width = u32::try_from(width).map_err(|_| Reject::Bounds)?;
        let owner = Arc::as_ptr(&r.shadow) as usize;
        for i in 0..r.span.pieces {
            if declarations.len() == MAX_PIECES {
                return Err(Reject::Bounds);
            }
            let (lo, hi) = r.shadow.card_piece(&r.span, i);
            let index = r.span.first.checked_add(i).ok_or(Reject::Bounds)?;
            let flag = if r.compared {
                Some(r.at.checked_add(i).ok_or(Reject::Bounds)?)
            } else {
                None
            };
            declarations.push(Declaration {
                range,
                owner,
                width,
                index,
                lo,
                hi,
                flag,
            });
        }
    }
    let layout = Layout::build(declarations.into_iter(), flag_words).ok_or(Reject::Bounds)?;
    // InterlockedAdd still advances on overflow, so prove the pessimistic
    // aggregate cursor cannot ever wrap into the already written records.
    if layout.rows.len() * (HEADER + MAX_ENVELOPE) > (u32::MAX as usize) - HEADER {
        return Err(Reject::Bounds);
    }
    let (input, refs) = layout.upload();
    let input_bytes = input.len() * 4;
    let pool = &mut work.snapshots.compact;
    let kernel = pool
        .kernel
        .get_or_init(|| Kernel::new(&host.device))
        .as_ref()
        .ok_or(Reject::Host)?;
    let ((upload, upload_ptr), copy) = (
        host.upload_buffer(input_bytes as u64).ok_or(Reject::Host)?,
        work.check
            .pop()
            .or_else(|| host.make_prelude())
            .ok_or(Reject::Host)?,
    );
    let pick = pool.buffers.pop().or_else(|| {
        host.shared_buffer((HEADER + CAPACITY) as u64)
            .map(|(buffer, ptr)| (buffer, ptr as usize))
    });
    let Some((buffer, mapping)) = pick.filter(|&(_, ptr)| ptr != 0 && !upload_ptr.is_null()) else {
        return Err(Reject::Host);
    };
    let (allocator, list) = &copy;
    // SAFETY: this slot's last flight has fully retired. Only this CPU
    // writes its header until Execute. Upload data and UAV are held until
    // the flight fence. Distinct source dispatches share the output UAV.
    let recorded = unsafe {
        (|| -> windows::core::Result<ID3D12CommandList> {
            std::ptr::copy_nonoverlapping(input.as_ptr().cast::<u8>(), upload_ptr, input_bytes);
            std::ptr::write_bytes(mapping as *mut u8, 0, HEADER);
            allocator.Reset()?;
            list.Reset(allocator, Some(&kernel.pipeline))?;
            list.SetComputeRootSignature(&kernel.root);
            list.SetComputeRootShaderResourceView(2, work.differs.0.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(3, upload.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(4, buffer.GetGPUVirtualAddress());
            for (i, group) in layout.groups.iter().enumerate() {
                if i != 0 {
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(std::mem::transmute_copy(
                                    &buffer,
                                )),
                            }),
                        },
                    }]);
                }
                let source = &plan.ranges[group.range].shadow.buffer;
                let constants = [
                    group.width,
                    flag_words,
                    layout.rows.len() as u32,
                    CAPACITY as u32,
                    refs,
                    group.first,
                ];
                list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                list.SetComputeRootShaderResourceView(1, source.GetGPUVirtualAddress());
                list.Dispatch(group.count, 1, 1);
            }
            list.Close()?;
            list.cast()
        })()
    };
    let recorded = match recorded {
        Ok(list) => list,
        Err(_) => {
            pool.buffers.push((buffer, mapping));
            return Err(Reject::Host);
        }
    };
    pool.active += 1;
    let compact = Capture {
        data: mapping,
        _input: upload,
        layout,
    };
    Ok((
        Snapshot {
            buffer,
            list: recorded,
            pieces: Vec::new(),
            bytes: 0,
            order: None,
            compact: Some(compact),
        },
        copy,
    ))
}

pub(super) struct Validated<'a> {
    records: Vec<Option<&'a [u8]>>,
    transferred: u64,
}

pub(super) fn validate<'a>(capture: &'a Capture, flags: &[u32]) -> Result<Validated<'a>, Reject> {
    // SAFETY: this flight's fence has completed and the slot is not yet
    // retired. The backing allocation is HEADER+CAPACITY in every slot.
    let output =
        unsafe { std::slice::from_raw_parts(capture.data as *const u8, HEADER + CAPACITY) };
    let records = capture.layout.validate(output, flags)?;
    Ok(Validated {
        records,
        transferred: word(output, 0).unwrap_or(0) as u64,
    })
}

pub(super) fn merge(capture: &Capture, plan: &CardPlan, order: Order, valid: Validated<'_>) {
    let mut title = Vec::new();
    let mut written = 0;
    // Do not apply output append order or the widened GPU-copy envelope.
    // Replay only the original declared slices, after complete validation.
    for p in &capture.layout.pieces {
        if let Some(record) = valid.records[p.row] {
            let offset = p.lo - capture.layout.rows[p.row].lo;
            let gpu = &record[offset..offset + p.hi - p.lo];
            written += plan.ranges[p.range]
                .shadow
                .card_merge_relay(p.lo, gpu, order.sent, &mut title);
        }
    }
    COPY_BACK[1].fetch_add(written as u64, Relaxed);
    ACCEPTED.fetch_add(1, Relaxed);
    TRANSFERRED.fetch_add(valid.transferred, Relaxed);
    MERGED.fetch_add(written as u64, Relaxed);
    crate::shadow::relay_fetched(order.sent);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_payload(id: usize, row: &Row, bytes: u8) -> Vec<u8> {
        let len = row.hi - row.lo;
        let mut out: Vec<_> = [id as u32, len as u32, row.lo as u32, TAG]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        out.extend(std::iter::repeat_n(bytes, len));
        out
    }

    fn buffer(rows: &[Vec<u8>]) -> Vec<u8> {
        let stored = rows.iter().map(Vec::len).sum::<usize>() as u32;
        let payload = stored - (HEADER * rows.len()) as u32;
        [stored, 0, payload, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(rows.iter().flatten().copied())
            .collect()
    }

    #[test]
    fn validates_reordered_compaction_that_keeps_exact_duplicate_partial_slices() {
        let declaration = |owner, range, index, lo, hi, flag| Declaration {
            owner,
            range,
            index,
            lo,
            hi,
            flag,
            width: 262_144,
        };
        let layout = Layout::build(
            [
                declaration(1, 0, 7, 3, 5, Some(1)),
                declaration(1, 1, 7, 18, 31, Some(2)),
                declaration(1, 2, 8, 65_540, 65_545, Some(3)),
                declaration(2, 3, 7, 23, 24, None),
            ]
            .into_iter(),
            4,
        )
        .unwrap();
        let (upload, refs) = layout.upload();
        assert_eq!(refs, 3);
        assert_eq!(&upload[..4], &[0, 32, 0, 2]);
        assert_eq!(layout.groups.len(), 2);
        assert_eq!(layout.pieces[0].row, layout.pieces[1].row);
        assert_eq!((layout.pieces[1].lo, layout.pieces[1].hi), (18, 31));
        let flags = [0, 1, 0, 0];
        let a = row_payload(0, &layout.rows[0], 0xa0);
        let b = row_payload(2, &layout.rows[2], 0xb0);
        let packed = buffer(&[b.clone(), a.clone()]);
        let validated = layout.validate(&packed, &flags).ok().unwrap();
        assert_eq!(validated[0].unwrap(), &[0xa0; 32]);
        assert!(validated[1].is_none());
        assert_eq!(validated[2].unwrap(), &[0xb0; 16]);
        assert!(matches!(
            layout.validate(&buffer(&[a.clone(), a]), &flags),
            Err(Reject::Integrity)
        ));
        assert!(matches!(
            layout.validate(&buffer(&[b]), &flags),
            Err(Reject::Integrity)
        ));
        assert!(matches!(
            layout.validate(&packed[..packed.len() - 1], &flags),
            Err(Reject::Integrity)
        ));
        let mut overflow = packed;
        overflow[4] = 1;
        assert!(matches!(
            layout.validate(&overflow, &flags),
            Err(Reject::Overflow)
        ));
    }

    #[test]
    fn warp_production_gather_merges_two_sources_without_overwriting_a_later_send() {
        use crate::{host, shadow::Shadow};
        use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
        if std::env::var_os("VICTORY_D3D12_WARP").is_none() {
            eprintln!("enable VICTORY_D3D12_WARP=1 to run the integrated relay GPU test");
            return;
        }
        let host = host::host().expect("WARP host");
        let aligned = |store: &mut [u8]| {
            let at = (store.as_ptr() as usize).wrapping_neg() & 65_535;
            &mut store[at..at + 65_536] as *mut [u8]
        };
        let mut a_store = vec![0x22; 2 * 65_536];
        let mut b_store = vec![0x33; 2 * 65_536];
        let a_title = aligned(&mut a_store);
        let b_title = aligned(&mut b_store);
        // SAFETY: both slices refer to independent stable private arrays.
        let (a, a_cpu) = Shadow::compact_test_copy(host, unsafe { &*a_title });
        let (b, _) = Shadow::compact_test_copy(host, unsafe { &*b_title });
        let mut work = host.make_card_work().unwrap();
        // This fixture supplies flags directly. The companion WARP test
        // proves the real compare writes the same canonical flag buffer.
        unsafe {
            *(work.differs.1 as *mut u32) = 1;
            *(work.differs.1 as *mut u32).add(1) = 0;
        }
        let range =
            |shadow: &Arc<Shadow>, start: usize, lo: usize, hi: usize, at: u32, compared: bool| {
                let declared = (start + lo, start + hi);
                super::super::super::CardRange {
                    shadow: shadow.clone(),
                    declared,
                    span: shadow.card_span(declared.0, declared.1).unwrap(),
                    at,
                    compared,
                }
            };
        let a_at = a_title.cast::<u8>() as usize;
        let b_at = b_title.cast::<u8>() as usize;
        let plan = CardPlan {
            ranges: vec![
                range(&a, a_at, 3, 9, 0, true),
                range(&a, a_at, 17, 23, 1, true),
                range(&b, b_at, 8, 16, 0, false),
            ],
            check: None,
            snapshot: None,
            _synced: Vec::new(),
            lists: Vec::new(),
        };
        let (mut snapshot, _copy) = prepare(host, &mut work, &plan)
            .ok()
            .expect("production compaction plan");
        let width = 65_536;
        let uploads = [
            host.upload_buffer(width).unwrap(),
            host.upload_buffer(width).unwrap(),
        ];
        let producer = host.new_list().unwrap();
        // SAFETY: private mapped buffers, a fresh list and exact allocations;
        // the uploads and both source buffers outlive the flight.
        unsafe {
            for (i, (shadow, byte)) in [(&a, 0xa1), (&b, 0xb2)].into_iter().enumerate() {
                std::ptr::write_bytes(uploads[i].1, byte, width as usize);
                producer
                    .list
                    .CopyBufferRegion(&shadow.buffer, 0, &uploads[i].0, 0, width);
            }
        }
        producer.close();
        let produced = [Some(producer.list.cast::<ID3D12CommandList>().unwrap())];
        // SAFETY: these are private objects on the software adapter.
        unsafe {
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            super::super::execute(&host.queue, &fence, 1, &produced, &mut snapshot).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            fence.SetEventOnCompletion(1, event).unwrap();
            assert_eq!(WaitForSingleObject(event, 10_000).0, 0);
            windows::Win32::Foundation::CloseHandle(event).unwrap();
            host.device.GetDeviceRemovedReason().unwrap();
        }
        let order = snapshot.order.unwrap();
        // A newer send has put [5,7) on the CPU side since this snapshot.
        // Its older GPU payload must leave those exact bytes alone.
        unsafe {
            std::ptr::write_bytes((a_cpu + 5) as *mut u8, 0x77, 2);
            std::ptr::write_bytes((a_at + 5) as *mut u8, 0x77, 2);
        }
        a.compact_test_note_send(5, 7, order.sent + 1);
        let capture = snapshot.compact.as_ref().unwrap();
        let valid = validate(capture, &[1, 0]).ok().unwrap();
        merge(capture, &plan, order, valid);
        // SAFETY: the test allocations outlive both fully waited GPU lists.
        let (a_bytes, b_bytes) = unsafe { (&*a_title, &*b_title) };
        assert_eq!(
            &a_bytes[..24],
            &[
                0x22, 0x22, 0x22, 0xa1, 0xa1, 0x77, 0x77, 0xa1, 0xa1, 0x22, 0x22, 0x22, 0x22, 0x22,
                0x22, 0x22, 0x22, 0xa1, 0xa1, 0xa1, 0xa1, 0xa1, 0xa1, 0x22,
            ]
        );
        assert_eq!(&b_bytes[..8], &[0x33; 8]);
        assert_eq!(&b_bytes[8..16], &[0xb2; 8]);
        assert_eq!(&b_bytes[16..24], &[0x33; 8]);
        work.snapshots.retire(snapshot);
        assert_eq!(work.snapshots.compact.active, 0);
    }
}
