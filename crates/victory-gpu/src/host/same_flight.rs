mod compact;
mod compact_alias;
pub(super) use compact_alias::AliasGuard;
#[cfg(test)]
mod compact_probe;

use super::{
    CARD_WORK, COPY_BACK, CardPlan, CardWork, Host, QUEUE_ORDER, copy_fetch_runs, flagged_pieces,
};

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

struct Limits {
    bytes: u64,
    pieces: usize,
    slots: usize,
}

const SMALL: Limits = Limits { bytes: 1 << 20, pieces: 16, slots: 24 };
const WIDE: Limits = Limits { bytes: 16 << 20, pieces: 256, slots: 8 };
// Bound the CPU send history even if every card comparison is clean.
const HISTORY_SENDS: u64 = 128;

static LEGACY_EPOCH: AtomicU64 = AtomicU64::new(0);
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_SAME_FLIGHT_READBACK").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        Ok("3") if crate::inplace::level() == 0 => 3,
        Ok("3") => {
            tracing::warn!("compact same-flight readback needs declared writes; using legacy relay with in-place title memory");
            0
        }
        _ => 0,
    })
}

pub(super) fn enabled() -> bool {
    mode() != 0
}

pub(super) fn compact_active() -> bool {
    mode() == 3
}

fn can_defer_empty_retirement(sent: u64, retired: u64) -> bool {
    sent.saturating_sub(retired) < HISTORY_SENDS
}

// An empty fetch observed no GPU bytes. In wide mode keep its CPU send
// history available to earlier snapshots. A real fetch always invalidates
// them, as does hitting the bounded retention limit.
pub(super) fn retire_empty(sent: u64) {
    if mode() >= 2 && can_defer_empty_retirement(sent, crate::shadow::relay_retired()) {
        return;
    }
    retire_legacy_history();
    crate::shadow::relay_fetched(sent);
}

/// The caller holds the queue order lock. A legacy fetch can observe
/// later flights, and both a legacy fetch and an empty retirement can
/// discard send history needed to merge an older captured flight.
pub(super) fn retire_legacy_history() {
    if enabled() {
        LEGACY_EPOCH.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy)]
struct Order {
    sent: u64,
    legacy: u64,
}

impl Order {
    fn still_valid(self, legacy: u64) -> bool {
        self.legacy == legacy
    }
}

struct Piece {
    range: usize,
    index: u32,
    lo: usize,
    hi: usize,
    offset: u64,
}

pub(super) struct Snapshot {
    buffer: ID3D12Resource,
    list: ID3D12CommandList,
    pieces: Vec<Piece>,
    bytes: u64,
    order: Option<Order>,
    compact: Option<compact::Capture>,
}

#[derive(Default)]
pub(super) struct Pool {
    buffers: Vec<ID3D12Resource>,
    active: usize,
    compact: compact::Pool,
}

impl Pool {
    pub(super) fn retire(&mut self, snapshot: Snapshot) {
        if let Some(capture) = snapshot.compact {
            self.compact.retire(snapshot.buffer, capture);
        } else {
            self.buffers.push(snapshot.buffer);
            self.active -= 1;
        }
    }
}

/// Captures a bounded plan before its own fence. Wide mode also captures
/// ranges without comparisons, which always require a CPU merge.
pub(super) fn prepare(host: &Host, work: &mut CardWork, plan: &mut CardPlan, alias: Option<&AliasGuard>) {
    if compact_active() {
        if plan.ranges.is_empty() {
            return;
        }
        if alias.is_none_or(AliasGuard::blocked) {
            compact::rejected(compact::Reject::Alias);
            return;
        }
        match compact::prepare(host, work, plan) {
            Ok((snapshot, list)) => {
                plan.snapshot = Some(snapshot);
                plan.lists.push((true, list));
            }
            Err(reason) => compact::rejected(reason),
        }
        return;
    }
    let wide = mode() == 2;
    let limits = if wide { &WIDE } else { &SMALL };
    if work.snapshots.active >= limits.slots
        || plan.ranges.is_empty()
        || (!wide && plan.ranges.iter().any(|r| !r.compared))
    {
        return;
    }

    let mut pieces = Vec::new();
    let mut bytes = 0;
    for (range, r) in plan.ranges.iter().enumerate() {
        for index in 0..r.span.pieces {
            let (lo, hi) = r.shadow.card_piece(&r.span, index);
            if lo >= hi {
                continue;
            }
            let next = bytes + (hi - lo) as u64;
            if pieces.len() == limits.pieces || next > limits.bytes {
                return;
            }
            pieces.push(Piece {
                range,
                index,
                lo,
                hi,
                offset: bytes,
            });
            bytes = next;
        }
    }
    if pieces.is_empty() {
        return;
    }

    let Some(buffer) = work
        .snapshots
        .buffers
        .pop()
        .or_else(|| host.read_back_buffer(limits.bytes))
    else {
        return;
    };
    let Some(copy) = work.check.pop().or_else(|| host.make_prelude()) else {
        return;
    };
    let (allocator, list) = &copy;
    let spans = pieces
        .iter()
        .map(|p| (&plan.ranges[p.range].shadow.buffer, p.lo, p.hi, p.offset));
    let recorded = record_copy(allocator, list, &buffer, spans);
    let Ok(list) = recorded else { return };
    plan.snapshot = Some(Snapshot {
        buffer,
        list,
        pieces,
        bytes,
        order: None,
        compact: None,
    });
    plan.lists.push((true, copy));
    work.snapshots.active += 1;
}

fn record_copy<'a>(
    allocator: &ID3D12CommandAllocator,
    list: &ID3D12GraphicsCommandList,
    buffer: &ID3D12Resource,
    spans: impl IntoIterator<Item = (&'a ID3D12Resource, usize, usize, u64)>,
) -> windows::core::Result<ID3D12CommandList> {
    // SAFETY: the callers give us an idle list and allocator, readback in
    // COPY_DEST, and live source buffers that can promote from COMMON.
    unsafe {
        allocator.Reset()?;
        list.Reset(allocator, None)?;
        copy_fetch_runs(spans, |source, lo, hi, offset| {
            list.CopyBufferRegion(buffer, offset, source, lo as u64, (hi - lo) as u64);
        });
        list.Close()?;
        list.cast()
    }
}

/// Submits the compare and its readback in queue order, before the usual
/// flight fence. A separate execution allows buffers to decay from the
/// states used by the game's lists and the compare.
pub(super) fn execute(
    queue: &ID3D12CommandQueue,
    fence: &ID3D12Fence,
    target: u64,
    lists: &[Option<ID3D12CommandList>],
    snapshot: &mut Snapshot,
) -> windows::core::Result<()> {
    let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
    snapshot.order = Some(Order {
        sent: *order,
        legacy: LEGACY_EPOCH.load(Ordering::Relaxed),
    });
    // SAFETY: the caller hands us closed lists with resources owned until
    // this fence. The lock keeps a legacy fetch out of these two executes.
    unsafe {
        queue.ExecuteCommandLists(lists);
        queue.ExecuteCommandLists(&[Some(snapshot.list.clone())]);
        queue.Signal(fence, target)?;
    }
    if snapshot.compact.is_some() {
        compact::submitted();
    } else {
        STATS[0].fetch_add(snapshot.bytes, Ordering::Relaxed);
    }
    Ok(())
}

/// Returns false to let the caller use the existing fetch for this plan.
pub(super) fn merge(plan: &CardPlan) -> bool {
    let Some(snapshot) = &plan.snapshot else {
        return false;
    };
    let Some(order) = snapshot.order else {
        return false;
    };
    if !order.still_valid(LEGACY_EPOCH.load(Ordering::Relaxed))
        || (mode() >= 2 && order.sent < crate::shadow::relay_retired())
    {
        if snapshot.compact.is_some() {
            compact::rejected(compact::Reject::Order);
        } else {
            STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        return false;
    }
    let Some(differs) = compare_words() else {
        return false;
    };
    let flags: Vec<_> = snapshot
        .pieces
        .iter()
        .map(|piece| {
            let r = &plan.ranges[piece.range];
            let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
            // A range without a compare must be treated as dirty, exactly
            // as in the ordinary post-fence fetch.
            let flag = if r.compared {
                differs.get((r.at + piece.index) as usize).copied()
            } else {
                None
            };
            let flag = flag != Some(0);
            (copy, r.span.first + piece.index, flag)
        })
        .collect();
    let fetched = flagged_pieces(&flags);
    if !fetched.is_empty() {
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: snapshot.bytes as usize,
        };
        // SAFETY: this readback's fence completed and it is still held by the flight.
        if unsafe { snapshot.buffer.Map(0, Some(&range), Some(&mut data)) }.is_err() {
            return false;
        }
        if data.is_null() {
            // SAFETY: Map succeeded but returned an unusable address.
            unsafe { snapshot.buffer.Unmap(0, Some(&D3D12_RANGE::default())) };
            return false;
        }
        let mut title = Vec::new();
        let mut written = 0;
        for (piece, (copy, index, _)) in snapshot.pieces.iter().zip(flags) {
            if !fetched.contains(&(copy, index)) {
                continue;
            }
            // SAFETY: this flight owns the mapped readback and its offsets.
            let gpu = unsafe {
                std::slice::from_raw_parts(
                    data.cast::<u8>().add(piece.offset as usize),
                    piece.hi - piece.lo,
                )
            };
            written += plan.ranges[piece.range]
                .shadow
                .card_merge_relay(piece.lo, gpu, order.sent, &mut title);
        }
        // SAFETY: the CPU only read the mapping.
        unsafe { snapshot.buffer.Unmap(0, Some(&D3D12_RANGE::default())) };
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
    }
    STATS[1].fetch_add(1, Ordering::Relaxed);
    STATS[2].fetch_add(snapshot.bytes, Ordering::Relaxed);
    crate::shadow::relay_fetched(order.sent);
    true
}

/// A landing either reapplies every captured declaration in queue order or
/// uses the legacy fetch for all of them. In particular, a fallback must
/// not move an older write behind a newer compact merge through alias VAs.
pub(super) fn merge_compact_batch(plans: &[(&CardPlan, Option<&AliasGuard>)]) -> bool {
    let active: Vec<_> = plans.iter().filter(|(plan, _)| !plan.ranges.is_empty()).collect();
    if active.is_empty() {
        let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
        retire_empty(*order);
        return true;
    }
    if active.iter().any(|(plan, _)| plan.snapshot.is_none()) {
        compact::rejected(compact::Reject::Landing);
        return false;
    }
    let Some(flags) = compare_words() else {
        compact::rejected(compact::Reject::Host);
        return false;
    };
    let merged = AliasGuard::with_all_clear(active.iter().map(|(_, guard)| *guard), || {
        let staged = active.iter().map(|(plan, _)| {
            let snapshot = plan.snapshot.as_ref().ok_or(compact::Reject::Landing)?;
            let order = snapshot.order.ok_or(compact::Reject::Order)?;
            if !order.still_valid(LEGACY_EPOCH.load(Ordering::Relaxed))
                || order.sent < crate::shadow::relay_retired()
            {
                return Err(compact::Reject::Order);
            }
            let capture = snapshot.compact.as_ref().ok_or(compact::Reject::Landing)?;
            let valid = compact::validate(capture, flags)?;
            Ok((*plan, capture, order, valid))
        }).collect::<Result<Vec<_>, compact::Reject>>();
        match staged {
            Ok(staged) => {
                for (plan, capture, order, valid) in staged {
                    compact::merge(capture, plan, order, valid);
                }
                true
            }
            Err(reason) => {
                compact::rejected(reason);
                false
            }
        }
    });
    merged.unwrap_or_else(|| {
        compact::rejected(compact::Reject::Alias);
        false
    })
}

fn compare_words() -> Option<&'static [u32]> {
    let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let work = work.as_ref()?;
    // SAFETY: CardWork keeps this mapping for the process. The caller only
    // examines its waited plans' flag slots before their retirement.
    Some(unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) })
}


pub(super) fn report(frames: f64) {
    if compact_active() {
        compact::report(frames);
        return;
    }
    let [sent, plans, consumed, stale] = STATS.each_ref().map(|s| s.swap(0, Ordering::Relaxed));
    tracing::info!(
        "same-flight readback per frame: {:.1} plans, {:.1} MB used of {:.1} MB copied; {:.1} plans overtaken by a legacy fetch",
        plans as f64 / frames,
        consumed as f64 / (1 << 20) as f64 / frames,
        sent as f64 / (1 << 20) as f64 / frames,
        stale as f64 / frames,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    #[test]
    fn snapshot_falls_back_only_after_a_legacy_fetch_or_retirement() {
        let before = Order { sent: 4, legacy: 7 };
        let after = Order { sent: 8, legacy: 8 };
        assert!(before.still_valid(7));
        assert!(!before.still_valid(8));
        assert!(after.still_valid(8));
    }

    #[test]
    fn empty_compares_keep_a_bounded_send_history() {
        assert!(can_defer_empty_retirement(10, 10));
        assert!(can_defer_empty_retirement(137, 10));
        assert!(!can_defer_empty_retirement(138, 10));
        assert!(!can_defer_empty_retirement(139, 10));
    }

    const BYTES: u64 = 64 * 1024;

    fn buffer(
        device: &ID3D12Device,
        kind: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: BYTES,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: the heap and resource descriptions match the states used below.
        let mut resource = None;
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

    pub(super) fn prelude(device: &ID3D12Device) -> super::super::Prelude {
        // SAFETY: these objects are new and live through the test.
        unsafe {
            let allocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap();
            list.Close().unwrap();
            (allocator, list)
        }
    }

    pub(super) fn fill(resource: &ID3D12Resource, bytes: &[u8]) {
        let mut data = std::ptr::null_mut();
        // SAFETY: this is an upload buffer and is not on the queue yet.
        unsafe {
            resource
                .Map(0, Some(&D3D12_RANGE::default()), Some(&mut data))
                .unwrap();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
            resource.Unmap(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: bytes.len(),
                }),
            );
        }
    }

    fn read(resource: &ID3D12Resource, bytes: usize) -> Vec<u8> {
        let mut data = std::ptr::null_mut();
        // SAFETY: after the fence, only the test reads this resource.
        unsafe {
            resource
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: bytes,
                    }),
                    Some(&mut data),
                )
                .unwrap();
            let result = std::slice::from_raw_parts(data.cast::<u8>(), bytes).to_vec();
            resource.Unmap(0, Some(&D3D12_RANGE::default()));
            result
        }
    }

    #[test]
    fn warp_snapshots_keep_the_bytes_of_their_own_flight() {
        let adapter = super::super::warp_adapter().expect("software DXGI adapter");
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain software device creation.
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.unwrap();
        let device = device.unwrap();
        let source = buffer(
            &device,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let earlier = buffer(
            &device,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let later = buffer(
            &device,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let first = (0..32).map(|v| v as u8).collect::<Vec<_>>();
        let second = (201..217).map(|v| v as u8).collect::<Vec<_>>();
        fill(&earlier, &first);
        fill(&later, &second);

        // SAFETY: a separate software queue and fence.
        let (queue, fence): (ID3D12CommandQueue, ID3D12Fence) = unsafe {
            (
                device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
                    .unwrap(),
                device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(),
            )
        };
        let make_snapshot = |prelude: &super::super::Prelude| {
            let dest = buffer(
                &device,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let spans = [
                (&source, 0, 6, 0),
                (&source, 6, 16, 6),
                (&source, 24, 32, 16),
            ];
            let list = record_copy(&prelude.0, &prelude.1, &dest, spans).unwrap();
            Snapshot {
                buffer: dest,
                list,
                pieces: Vec::new(),
                bytes: 24,
                order: None,
                compact: None,
            }
        };
        let record_write = |p: &super::super::Prelude, src, dest_at, bytes| -> ID3D12CommandList {
            // SAFETY: three distinct idle lists, live buffers, no reused allocator.
            unsafe {
                p.0.Reset().unwrap();
                p.1.Reset(&p.0, None).unwrap();
                p.1.CopyBufferRegion(&source, dest_at, src, 0, bytes);
                p.1.Close().unwrap();
                p.1.cast().unwrap()
            }
        };
        let capture_a_list = prelude(&device);
        let mut capture_a = make_snapshot(&capture_a_list);
        let capture_b_list = prelude(&device);
        let mut capture_b = make_snapshot(&capture_b_list);
        let a_list = prelude(&device);
        let b_list = prelude(&device);
        let a = record_write(&a_list, &earlier, 0, 32);
        let b = record_write(&b_list, &later, 8, 16);

        execute(&queue, &fence, 1, &[Some(a)], &mut capture_a).unwrap();
        execute(&queue, &fence, 2, &[Some(b)], &mut capture_b).unwrap();

        let legacy_list = prelude(&device);
        let legacy = make_snapshot(&legacy_list);
        // SAFETY: retain all command allocators, buffers and event through the fence.
        unsafe {
            queue.ExecuteCommandLists(&[Some(legacy.list.clone())]);
            queue.Signal(&fence, 3).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            fence.SetEventOnCompletion(3, event).unwrap();
            assert_eq!(WaitForSingleObject(event, 10_000).0, 0);
            windows::Win32::Foundation::CloseHandle(event).unwrap();
            device.GetDeviceRemovedReason().unwrap();
        }
        let first_expected = [&first[..16], &first[24..]].concat();
        let second_expected = [&first[..8], &second[..8], &first[24..]].concat();
        assert_eq!(read(&capture_a.buffer, 24), first_expected);
        assert_eq!(read(&capture_b.buffer, 24), second_expected);
        assert_eq!(read(&legacy.buffer, 24), second_expected);
    }
}
