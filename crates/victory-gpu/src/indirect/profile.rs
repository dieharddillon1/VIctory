//! Bounded timestamp samples from the indirect validator on one recording frame.
//! Queries resolve in the original command list and are read at its normal completion fence.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

const BIN_COUNT: usize = 16;
const EVERY: u32 = 16;
const MAX_SAMPLES: u32 = 384;
const PER_LIST: usize = 32;
const QUERIES_PER_SAMPLE: usize = 4;

enum When {
    Frame(u64),
    Scene(u64),
    Warm(u64),
}

impl When {
    fn parse(value: &str) -> Option<Self> {
        if let Some(offset) = value.strip_prefix("scene+") {
            let offset = offset.parse().ok()?;
            return (offset > 0).then_some(Self::Scene(offset));
        }
        if let Some(offset) = value.strip_prefix("warm+") {
            let offset = offset.parse().ok()?;
            return (offset > 0).then_some(Self::Warm(offset));
        }
        let frame = value.parse().ok()?;
        (frame > 0).then_some(Self::Frame(frame))
    }

    fn resolve(&self, scene: u64, warm: u64) -> Option<u64> {
        match *self {
            Self::Frame(frame) => Some(frame),
            Self::Scene(offset) if scene != 0 => scene.checked_add(offset),
            Self::Warm(offset) if warm != 0 => warm.checked_add(offset),
            _ => None,
        }
    }
}

fn when() -> Option<&'static When> {
    static WHEN: OnceLock<Option<When>> = OnceLock::new();
    WHEN.get_or_init(|| {
        let raw = std::env::var("VICTORY_INDIRECT_GPU_TIMING").ok()?;
        let parsed = When::parse(raw.trim());
        if parsed.is_some() {
            tracing::info!(
                selection = raw,
                every = EVERY,
                max_samples = MAX_SAMPLES,
                "indirect GPU profile armed for one recording frame"
            );
        } else {
            tracing::warn!(
                selection = raw,
                "indirect GPU profile expects a frame, scene+N or warm+N"
            );
        }
        parsed
    })
    .as_ref()
}

fn selected(when: &When, frame: u64) -> bool {
    when.resolve(crate::frame::startup_scene(), crate::capture::warm_scene()) == Some(frame)
}

fn bin(kind: u32, records: u32) -> Option<usize> {
    let kind = match kind {
        0..=2 => kind as usize,
        10 => 3,
        _ => return None,
    };
    let size = match records {
        1 => 0,
        2..=64 => 1,
        65..=1024 => 2,
        1025.. => 3,
        0 => return None,
    };
    Some(kind * 4 + size)
}

static OBSERVED: [AtomicU32; BIN_COUNT] = [const { AtomicU32::new(0) }; BIN_COUNT];
static RESERVED: AtomicU32 = AtomicU32::new(0);
static ADMITTED: AtomicU32 = AtomicU32::new(0);
static OMITTED: AtomicU32 = AtomicU32::new(0);

pub(crate) fn capturing() -> bool {
    ADMITTED.load(Relaxed) != 0
}

/// Lists may be recorded ahead. This event declares the associated
/// present and the observed sampling population, not completion of its lists.
pub(crate) fn present(frame: u64) {
    if when().is_none_or(|when| !selected(when, frame)) {
        return;
    }
    let observed = OBSERVED.each_ref().map(|count| count.load(Relaxed));
    tracing::info!(
        frame,
        ?observed,
        accepted = ADMITTED.load(Relaxed),
        omitted = OMITTED.load(Relaxed),
        "indirect GPU profile frame recorded (bins draw/indexed/dispatch/mesh by max records 1,2-64,65-1024,1025+)"
    );
}

#[derive(Clone, Copy)]
struct Command {
    bin: usize,
    records: u32,
}

pub(crate) struct ListProfile {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    frame: u64,
    samples: Vec<Command>,
    reported: bool,
}

pub(super) struct Sample {
    heap: ID3D12QueryHeap,
    first: u32,
}

impl Sample {
    pub(super) fn stamp(&self, list: &ID3D12GraphicsCommandList, phase: u32) {
        // SAFETY: a recording direct list and a reserved index in its heap.
        unsafe { list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, self.first + phase) };
    }
}

fn permit(index: u32) -> bool {
    index.is_multiple_of(EVERY)
}

impl ListProfile {
    /// Called only after the validator has decided to record a real dispatch.
    pub(super) fn sample(list: &crate::host::HostList, kind: u32, records: u32) -> Option<Sample> {
        let when = when()?;
        let frame = crate::queue::presented() + 1;
        if !selected(when, frame) {
            return None;
        }
        let bin = bin(kind, records)?;
        let sequence = OBSERVED[bin].fetch_add(1, Relaxed);
        if !permit(sequence) {
            return None;
        }
        if RESERVED
            .fetch_update(Relaxed, Relaxed, |n| (n < MAX_SAMPLES).then_some(n + 1))
            .is_err()
        {
            OMITTED.fetch_add(1, Relaxed);
            return None;
        }
        let mut profile = list
            .indirect_profile
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if profile.is_none() {
            *profile = Self::new(crate::host::host()?, frame);
        }
        let Some(ref mut profile) = *profile else {
            OMITTED.fetch_add(1, Relaxed);
            return None;
        };
        let Some(sample) = profile.push(frame, Command { bin, records }) else {
            OMITTED.fetch_add(1, Relaxed);
            return None;
        };
        ADMITTED.fetch_add(1, Relaxed);
        Some(sample)
    }

    fn new(host: &crate::host::Host, frame: u64) -> Option<Self> {
        // SAFETY: a valid type and bounded count for a direct-queue heap.
        let heap = unsafe {
            let mut heap = None;
            host.device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: (PER_LIST * QUERIES_PER_SAMPLE) as u32,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .ok()?;
            heap?
        };
        let readback = host.read_back_buffer((PER_LIST * QUERIES_PER_SAMPLE * 8) as u64)?;
        Some(Self {
            heap,
            readback,
            frame,
            samples: Vec::new(),
            reported: false,
        })
    }

    fn push(&mut self, frame: u64, command: Command) -> Option<Sample> {
        if self.frame != frame || self.samples.len() == PER_LIST {
            return None;
        }
        let first = (self.samples.len() * QUERIES_PER_SAMPLE) as u32;
        self.samples.push(command);
        Some(Sample {
            heap: self.heap.clone(),
            first,
        })
    }

    pub(crate) fn resolve(&self, list: &ID3D12GraphicsCommandList) {
        let count = (self.samples.len() * QUERIES_PER_SAMPLE) as u32;
        if count != 0 {
            // SAFETY: initialized queries on this recording list and a
            // sufficiently large readback buffer kept until reset.
            unsafe {
                list.ResolveQueryData(
                    &self.heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    0,
                    count,
                    &self.readback,
                    0,
                )
            };
        }
    }

    pub(crate) fn report(&mut self, list: usize, submission: u64, frequency: u64) {
        if self.reported || self.samples.is_empty() {
            return;
        }
        self.reported = true;
        let frame = self.frame;
        let Some((bins, invalid)) = self.read(frequency) else {
            tracing::warn!(
                frame,
                submission,
                list = format_args!("{list:#x}"),
                "indirect GPU profile readback unavailable"
            );
            return;
        };
        let rendered: Vec<_> = bins
            .iter()
            .enumerate()
            .filter(|(_, b)| b.count != 0)
            .map(|(i, b)| (i, b.count, b.records, b.micros))
            .collect();
        tracing::info!(frame, submission, list = format_args!("{list:#x}"), invalid,
            samples = self.samples.len(), phases_us_by_bin = ?rendered,
            "indirect GPU profile completed list (pre barrier and bind, validator dispatch, post barrier)");
    }

    fn read(&self, frequency: u64) -> Option<([Totals; BIN_COUNT], usize)> {
        let len = self.samples.len() * QUERIES_PER_SAMPLE;
        let range = D3D12_RANGE {
            Begin: 0,
            End: len * size_of::<u64>(),
        };
        let mut data = std::ptr::null_mut();
        // SAFETY: the caller observes the list at a completed normal fence.
        // The readback allocation contains at least len timestamp words.
        unsafe {
            self.readback.Map(0, Some(&range), Some(&mut data)).ok()?;
            let out = (!data.is_null()).then(|| {
                summarize(
                    &self.samples,
                    std::slice::from_raw_parts(data.cast::<u64>(), len),
                    frequency,
                )
            });
            self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
            out
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Totals {
    count: u32,
    records: u64,
    micros: [f64; 3],
}

fn summarize(commands: &[Command], ticks: &[u64], freq: u64) -> ([Totals; BIN_COUNT], usize) {
    let mut out = [Totals::default(); BIN_COUNT];
    let mut invalid = commands
        .len()
        .saturating_sub(ticks.len() / QUERIES_PER_SAMPLE);
    if freq == 0 {
        return (out, commands.len());
    }
    for (cmd, t) in commands.iter().zip(ticks.chunks_exact(QUERIES_PER_SAMPLE)) {
        if t[0] == 0 || t.windows(2).any(|s| s[0] > s[1]) {
            invalid += 1;
            continue;
        }
        let bin = &mut out[cmd.bin];
        bin.count += 1;
        bin.records += u64::from(cmd.records);
        for (dest, range) in bin.micros.iter_mut().zip(t.windows(2)) {
            *dest += (range[1] - range[0]) as f64 * 1e6 / freq as f64;
        }
    }
    (out, invalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_and_sampling_remain_bounded() {
        assert_eq!(When::parse("127").unwrap().resolve(0, 0), Some(127));
        assert_eq!(When::parse("scene+120").unwrap().resolve(300, 0), Some(420));
        assert_eq!(When::parse("warm+2").unwrap().resolve(300, 660), Some(662));
        assert_eq!(When::parse("warm+2").unwrap().resolve(300, 0), None);
        assert!(When::parse("scene+0").is_none());
        assert!(permit(0) && permit(16) && !permit(17));
        assert_eq!(bin(1, 190), Some(6));
        assert_eq!(bin(10, 1), Some(12));
        assert_eq!(bin(10, 1025), Some(15));
        assert_eq!(bin(10, 0), None);
    }

    #[test]
    fn phases_use_the_queue_frequency_and_reject_invalid_queries() {
        let commands = [
            Command { bin: 0, records: 1 },
            Command { bin: 0, records: 1 },
            Command {
                bin: 15,
                records: 2000,
            },
            Command { bin: 0, records: 1 },
        ];
        let ticks = [100, 105, 107, 108, 110, 115, 119, 125, 123, 122, 130, 130];
        let (stats, invalid) = summarize(&commands, &ticks, 1_000_000);
        assert_eq!(invalid, 2);
        assert_eq!(stats[0].count, 2);
        assert_eq!(stats[0].records, 2);
        assert_eq!(stats[0].micros, [10., 6., 7.]);
        assert_eq!(stats[15].count, 0);
    }

    #[test]
    fn warp_resolves_profile_queries_through_a_normal_host_list_fence() {
        if std::env::var_os("VICTORY_D3D12_WARP").is_none() {
            eprintln!("enable VICTORY_D3D12_WARP=1 to run the timestamp GPU test");
            return;
        }
        let host = crate::host::host().expect("WARP host");
        let list = host.new_list().unwrap();
        let mut profile = ListProfile::new(host, 101).unwrap();
        let sample = profile
            .push(
                101,
                Command {
                    bin: 12,
                    records: 1,
                },
            )
            .unwrap();
        let (upload, data) = host.upload_buffer(16).unwrap();
        let copy = host.read_back_buffer(16).unwrap();
        // SAFETY: one idle mapped buffer, an open list, and the copy fits.
        unsafe {
            std::ptr::write_bytes(data, 0xa5, 16);
            sample.stamp(&list.list, 0);
            sample.stamp(&list.list, 1);
            list.list.CopyBufferRegion(&copy, 0, &upload, 0, 16);
            sample.stamp(&list.list, 2);
            sample.stamp(&list.list, 3);
        }
        profile.resolve(&list.list);
        list.close();
        host.execute(&[&list]);
        let (bins, invalid) = profile.read(1_000_000).unwrap();
        assert_eq!(invalid, 0);
        assert_eq!(bins[12].count, 1);
        assert_eq!(bins[12].records, 1);
    }
}
