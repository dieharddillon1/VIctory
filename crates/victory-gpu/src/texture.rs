//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Original packed 16-bit format (86 or 115) when the host texture was
    /// widened to R8G8B8A8_UNORM, so detiled 2-byte texels expand into the
    /// 4-byte staging buffer.
    widened_from: Option<u32>,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch query it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

thread_local! {
    /// Texture memory read for hashing and detiling: one per submitting
    /// thread, as large as the largest texture it read.
    static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        if size == 0 {
            return 0;
        }
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        SCRATCH.with_borrow_mut(|tiled| {
            tiled.resize(size, 0);
            if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
                if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                    return 0;
                }
                last.asked = now;
                last.hash = None;
                // Read through the copy, which stands unmapped pages: memory of
                // big pools is too slow to ask page by page.
                crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, size, false);
                self.detile(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
                return size;
            }
            let was_watched = std::mem::take(&mut last.asked) != 0;
            match read_resident(self.memory, tiled) {
                Residency::Unmapped => {
                    if last.hash.take().is_some() || was_watched {
                        tiled.fill(0);
                        self.detile(tiled);
                    }
                    #[cfg(feature = "diagnostics")]
                    tracing::debug!(
                        memory = ?self.memory,
                        size,
                        desc = ?self.layout.desc,
                        "a sampled texture's memory is not committed"
                    );
                    0
                }
                Residency::Full => {
                    let hash = blake3::hash(tiled);
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                    if last.hash == Some(hash) {
                        return size;
                    }
                    last.hash = Some(hash);
                    self.detile(tiled);
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                    size
                }
                Residency::Partial(intervals) => {
                    let (hash, resident_bytes) = hash_intervals(tiled, &intervals);
                    #[cfg(feature = "diagnostics")]
                    {
                        static REPORTED: std::sync::Once = std::sync::Once::new();
                        REPORTED.call_once(|| {
                            tracing::info!(resident_bytes, size, mips = self.mips,
                                "uploading a partially resident texture");
                        });
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::TexHashed, resident_bytes as u64);
                    if last.hash == Some(hash) {
                        return resident_bytes;
                    }
                    zero_gaps(tiled, &intervals);
                    last.hash = Some(hash);
                    self.detile(tiled);
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::TexChanged, resident_bytes as u64);
                    resident_bytes
                }
            }
        })
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        let mut packed = Vec::new();
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let lines = rows as usize * f.Depth as usize;
            let len = if self.widened_from.is_some() && lines > 0 {
                // The last subresource allocation stops at the last row's used bytes.
                pitch * (lines - 1) + f.Width as usize * 4
            } else {
                pitch * lines
            };
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) = detile_subresource(
                &self.layout,
                tiled,
                mip,
                slice,
                linear,
                pitch,
                rows as usize,
                f.Depth as usize,
                self.widened_from,
                &mut packed,
            ) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn detile_subresource(
    layout: &victory_formats::texture::Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    row_pitch: usize,
    rows: usize,
    depth: usize,
    widened_from: Option<u32>,
    packed: &mut Vec<u8>,
) -> Result<(), victory_formats::texture::Error> {
    let Some(packed_format) = widened_from else {
        return layout.detile(tiled, mip, slice, linear, row_pitch, row_pitch * rows);
    };
    let width = layout
        .mips
        .get(mip as usize)
        .map_or(0, |m| m.width as usize);
    let packed_pitch = width * 2;
    let packed_plane = packed_pitch * rows;
    packed.resize(packed_plane * depth, 0);
    layout.detile(tiled, mip, slice, packed, packed_pitch, packed_plane)?;
    crate::resource::expand_packed16_rows(
        packed_format,
        width,
        rows * depth,
        packed,
        packed_pitch,
        linear,
        row_pitch,
    );
    Ok(())
}

/// Which byte intervals of a title texture are currently mapped and readable.
#[derive(Debug, PartialEq, Eq)]
enum Residency {
    Unmapped,
    Full,
    Partial(Vec<(usize, usize)>),
}

/// Copies committed, readable regions of `[src, src + dst.len())` into `dst`
/// without `VirtualQuery` overhead on the fully mapped fast path.
fn read_resident(src: *const u8, dst: &mut [u8]) -> Residency {
    if dst.is_empty() || crate::host::read_into(src, dst) {
        return Residency::Full;
    }
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    const PAGE_SIZE: usize = 4096;

    let base = src as usize;
    let end = base.saturating_add(dst.len());
    let mut p = base;
    let mut intervals: Vec<(usize, usize)> = Vec::new();
    let push_interval = |intervals: &mut Vec<(usize, usize)>, start: usize, stop: usize| {
        if start >= stop {
            return;
        }
        if let Some(last) = intervals.last_mut()
            && last.1 == start
        {
            last.1 = stop;
        } else {
            intervals.push((start, stop));
        }
    };

    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            break;
        }
        let region_end = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .min(end);
        let step = region_end.saturating_sub(p);
        if step == 0 {
            break;
        }
        let next_p = p.saturating_add(step);
        if info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 {
            p = next_p;
            continue;
        }
        let offset = p - base;
        let region_len = region_end - p;
        if crate::host::read_into(p as *const u8, &mut dst[offset..offset + region_len]) {
            push_interval(&mut intervals, offset, offset + region_len);
        } else {
            // Concurrent unmap inside a committed region: fall back to 4 KiB pages.
            let mut q = p;
            while q < region_end {
                let next_page = (q & !(PAGE_SIZE - 1)).saturating_add(PAGE_SIZE);
                let page_end = next_page.min(region_end);
                let page_len = page_end.saturating_sub(q);
                if page_len == 0 {
                    break;
                }
                let page_offset = q - base;
                if crate::host::read_into(
                    q as *const u8,
                    &mut dst[page_offset..page_offset + page_len],
                ) {
                    push_interval(&mut intervals, page_offset, page_offset + page_len);
                }
                q = q.saturating_add(page_len);
            }
        }
        p = next_p;
    }

    if intervals.is_empty() {
        Residency::Unmapped
    } else if intervals.as_slice() == [(0, dst.len())] {
        Residency::Full
    } else {
        Residency::Partial(intervals)
    }
}

/// Hashes only the resident byte intervals (and their bounds) and returns `(hash, resident_bytes)`.
fn hash_intervals(data: &[u8], intervals: &[(usize, usize)]) -> (blake3::Hash, usize) {
    let mut hasher = blake3::Hasher::new();
    let mut resident_bytes = 0usize;
    for &(start, end) in intervals {
        hasher.update(&(start as u64).to_le_bytes());
        hasher.update(&(end as u64).to_le_bytes());
        hasher.update(&data[start..end]);
        resident_bytes += end - start;
    }
    (hasher.finalize(), resident_bytes)
}

/// Zeroes unmapped gaps between sorted `intervals` in `data` so evicted tiles leave no stale bytes.
fn zero_gaps(data: &mut [u8], intervals: &[(usize, usize)]) {
    let mut cursor = 0usize;
    for &(start, end) in intervals {
        if cursor < start {
            data[cursor..start].fill(0);
        }
        cursor = end;
    }
    if cursor < data.len() {
        data[cursor..].fill(0);
    }
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let title_format = crate::resource::pc_format(desc.format);
    let widened_from = (d.Format.0 as u32 == crate::resource::RGBA8_UNORM
        && crate::resource::PACKED_16.contains(&title_format))
    .then_some(title_format);
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        widened_from,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

    #[test]
    fn detile_widened_packed16_expands_into_rgba8_staging() {
        for format in crate::resource::PACKED_16 {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 17,
                height: 9,
                depth_or_array: 1,
                mips: 2,
                format,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            let tex = Texture::new(&desc).unwrap();
            let layout = &tex.planes[0].layout;
            let mut tiled = vec![0u8; layout.size as usize];
            let mut expected_mips = Vec::new();

            for mip in 0..2u32 {
                let m = layout.mips[mip as usize];
                let row_bytes = m.width as usize * 2;
                let mut src_mip = vec![0u8; row_bytes * m.height as usize];
                for y in 0..m.height as usize {
                    for x in 0..m.width as usize {
                        let px = ((mip as u16 + 1) * 0x1111)
                            .wrapping_add((y as u16) * 0x0103 + x as u16);
                        let at = y * row_bytes + x * 2;
                        src_mip[at..at + 2].copy_from_slice(&px.to_le_bytes());
                    }
                }
                layout
                    .tile(&mut tiled, mip, 0, &src_mip, row_bytes, src_mip.len())
                    .unwrap();
                expected_mips.push(src_mip);
            }

            let mut packed = Vec::new();
            for (mip, src_mip) in expected_mips.iter().enumerate() {
                let m = layout.mips[mip];
                let width = m.width as usize;
                let rows = m.height as usize;
                let row_pitch = (width * 4).next_multiple_of(256);
                let exact_len = row_pitch * (rows - 1) + width * 4;
                let mut staging = vec![0x5au8; exact_len];
                detile_subresource(
                    layout,
                    &tiled,
                    mip as u32,
                    0,
                    &mut staging,
                    row_pitch,
                    rows,
                    1,
                    Some(format),
                    &mut packed,
                )
                .unwrap();
                for y in 0..rows {
                    let mut expected = vec![0u8; width * 4];
                    crate::resource::expand_packed16(
                        format,
                        &src_mip[y * width * 2..][..width * 2],
                        &mut expected,
                    );
                    assert_eq!(
                        &staging[y * row_pitch..][..width * 4],
                        &expected[..],
                        "format {format} mip {mip} row {y}"
                    );
                }
            }
        }
    }

    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualAlloc, VirtualFree,
    };

    /// RAII guard for a `VirtualAlloc(..., MEM_RESERVE, ...)` address range.
    struct VirtualRegion {
        ptr: *mut u8,
        len: usize,
    }

    impl VirtualRegion {
        fn reserve(len: usize) -> Self {
            // SAFETY: reserves uncommitted virtual address space of `len` bytes, released on drop.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) }
                .cast::<u8>();
            assert!(!ptr.is_null(), "VirtualAlloc MEM_RESERVE failed");
            Self { ptr, len }
        }

        fn commit(&self, offset: usize, len: usize) {
            assert!(offset.checked_add(len).is_some_and(|end| end <= self.len));
            // SAFETY: commits pages within our reserved virtual address range.
            let committed = unsafe {
                VirtualAlloc(self.ptr.add(offset).cast(), len, MEM_COMMIT, PAGE_READWRITE)
            };
            assert!(!committed.is_null(), "VirtualAlloc MEM_COMMIT failed");
        }

        fn decommit(&self, offset: usize, len: usize) {
            assert!(offset.checked_add(len).is_some_and(|end| end <= self.len));
            // SAFETY: decommits pages within our reserved virtual address range.
            let ok = unsafe { VirtualFree(self.ptr.add(offset).cast(), len, MEM_DECOMMIT) };
            assert_ne!(ok, 0, "VirtualFree MEM_DECOMMIT failed");
        }

        /// Writes `bytes` at `offset`; caller must ensure `[offset..offset + bytes.len())` is committed.
        unsafe fn write_slice(&self, offset: usize, bytes: &[u8]) {
            assert!(
                offset
                    .checked_add(bytes.len())
                    .is_some_and(|end| end <= self.len)
            );
            // SAFETY: caller guarantees the destination range is committed and writable.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
            }
        }
    }

    impl Drop for VirtualRegion {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` was allocated with `VirtualAlloc(..., MEM_RESERVE, ...)` and not yet freed.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    /// RAII guard releasing a console COM resource so `STAGING` entries are cleaned up on drop.
    struct ResourceGuard(This);

    impl Drop for ResourceGuard {
        fn drop(&mut self) {
            type Release = extern "system" fn(This) -> u32;
            crate::object::slot::<Release>(self.0, 0x10)(self.0);
        }
    }

    #[test]
    fn partial_residency_reads_only_mapped_regions_and_detects_new_mips() {
        const TILE: usize = 64 * 1024;
        const TOTAL: usize = 256 * 1024;
        let region = VirtualRegion::reserve(TOTAL);
        let mut dst = vec![0xAAu8; TOTAL];

        assert_eq!(read_resident(region.ptr, &mut dst), Residency::Unmapped);

        region.commit(0, TILE);
        let tile0: Vec<u8> = (0..TILE)
            .map(|i| ((i * 37 + 11) & 0xff) as u8 | 1)
            .collect();
        // SAFETY: `[0..TILE)` was committed above.
        unsafe { region.write_slice(0, &tile0) };

        assert_eq!(
            read_resident(region.ptr, &mut dst),
            Residency::Partial(vec![(0, TILE)])
        );
        assert_eq!(dst[..TILE], tile0[..]);
        let (hash0, bytes0) = hash_intervals(&dst, &[(0, TILE)]);
        assert_eq!(bytes0, TILE);

        // Mutating unmapped tail bytes in `dst` must not affect the resident hash.
        dst[TILE..].fill(0x55);
        assert_eq!(
            read_resident(region.ptr, &mut dst),
            Residency::Partial(vec![(0, TILE)])
        );
        let (hash0_repeat, bytes0_repeat) = hash_intervals(&dst, &[(0, TILE)]);
        assert_eq!((hash0_repeat, bytes0_repeat), (hash0, TILE));

        // Commit a disjoint second 64 KiB tile at `[128 KiB..192 KiB)`.
        region.commit(2 * TILE, TILE);
        let tile2: Vec<u8> = (0..TILE)
            .map(|i| ((i * 53 + 29) & 0xff) as u8 | 1)
            .collect();
        // SAFETY: `[2 * TILE..3 * TILE)` was committed above.
        unsafe { region.write_slice(2 * TILE, &tile2) };

        let res = read_resident(region.ptr, &mut dst);
        assert_eq!(
            res,
            Residency::Partial(vec![(0, TILE), (2 * TILE, 3 * TILE)])
        );
        assert_eq!(dst[..TILE], tile0[..]);
        assert_eq!(dst[2 * TILE..3 * TILE], tile2[..]);
        let (hash_two, bytes_two) = hash_intervals(&dst, &[(0, TILE), (2 * TILE, 3 * TILE)]);
        assert_eq!(bytes_two, 2 * TILE);
        assert_ne!(hash_two, hash0);

        // Decommit the second tile while leaving junk across unmapped gaps; `zero_gaps` must clear it.
        region.decommit(2 * TILE, TILE);
        dst[TILE..].fill(0xCC);
        let res_after_evict = read_resident(region.ptr, &mut dst);
        assert_eq!(res_after_evict, Residency::Partial(vec![(0, TILE)]));
        let Residency::Partial(intervals) = res_after_evict else {
            unreachable!();
        };
        let (hash_after_evict, bytes_after_evict) = hash_intervals(&dst, &intervals);
        assert_eq!((hash_after_evict, bytes_after_evict), (hash0, TILE));
        zero_gaps(&mut dst, &intervals);
        assert_eq!(dst[..TILE], tile0[..]);
        assert!(dst[TILE..].iter().all(|&b| b == 0));

        // Committing the remaining gaps yields `Residency::Full`.
        region.commit(TILE, TILE);
        region.commit(2 * TILE, 2 * TILE);
        assert_eq!(read_resident(region.ptr, &mut dst), Residency::Full);
    }

    fn readback_mip(
        host: &crate::host::Host,
        resource: This,
        host_tex: &ID3D12Resource,
        subresource: u32,
    ) -> Vec<u8> {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
        // SAFETY: a live texture and one output footprint.
        unsafe {
            host.device.GetCopyableFootprints(
                &host_tex.GetDesc(),
                subresource,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                Some(&mut row_bytes),
                Some(&mut total),
            );
        }
        let byte_len = total as usize;

        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: byte_len as u64,
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
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a D3D12 readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();

        let back = host.new_list().unwrap();
        crate::list::ensure_state(&back, resource, 0x800);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(host_tex) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource,
            },
        };
        // SAFETY: `back.list` is open and both resources are live.
        unsafe { back.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        back.close();
        host.execute(&[&back]);

        let mut mapped = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: byte_len,
        };
        // SAFETY: `host.execute` waited for GPU completion and the buffer spans `byte_len` bytes.
        unsafe {
            readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), byte_len);
            let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
            for y in 0..rows as usize {
                let at = footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize;
                out.extend_from_slice(&bytes[at..at + row_bytes as usize]);
            }
            readback.Unmap(0, None);
            out
        }
    }

    fn title_texture(region: &VirtualRegion, width: u32, mips: u16, format: i32) -> ResourceGuard {
        const _: () =
            assert!(size_of::<D3D12_RESOURCE_DESC>() == size_of::<crate::resource::ResourceDesc>());
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: u64::from(width),
            Height: width,
            DepthOrArraySize: 1,
            MipLevels: mips,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT(0x109),
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut resource = std::ptr::null();
        let hr = crate::resource::create_at_address(
            std::ptr::null(),
            region.ptr as u64,
            (&raw const desc).cast(),
            0,
            std::ptr::null(),
            &crate::resource::PAGEABLE_IIDS[0],
            &mut resource,
        );
        assert_eq!(hr, crate::object::S_OK);
        ResourceGuard(resource)
    }

    #[test]
    fn partially_resident_mip_tail_uploads_to_host_texture() {
        let Some(host) = crate::host::host() else {
            return;
        };
        use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

        let tex = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: 256,
            height: 256,
            depth_or_array: 1,
            mips: 3,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let layout = &tex.planes[0].layout;
        assert_eq!(layout.first_tail_mip, 2);
        assert_eq!(layout.mips[2].offset, 0);
        assert!(layout.mips[2].tail.is_some());
        assert_eq!(layout.mips[1].offset, 65536);
        assert_eq!(layout.mips[0].offset, 131072);
        assert_eq!(layout.size, 393216);

        let region = VirtualRegion::reserve(layout.size as usize);
        region.commit(0, 65536);

        let expected_mip2: Vec<u8> = (0..64 * 64 * 4)
            .map(|i| (((i * 13 + 7) % 251) as u8) | 1)
            .collect();
        let mut tiled = vec![0u8; layout.size as usize];
        layout
            .tile(&mut tiled, 2, 0, &expected_mip2, 256, 64 * 256)
            .unwrap();
        assert!(tiled[65536..].iter().all(|&b| b == 0));
        // SAFETY: tile 0 `[0..65536)` was committed above.
        unsafe { region.write_slice(0, &tiled[..65536]) };
        assert!(!committed(region.ptr, layout.size as usize));

        let resource_guard = title_texture(&region, 256, 3, DXGI_FORMAT_R8G8B8A8_UNORM.0);
        let host_tex = crate::resource::host_resource(resource_guard.0).unwrap();

        let list = host.new_list().unwrap();
        upload(&list, resource_guard.0, host_tex);
        list.close();
        host.execute(&[&list]);

        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 2),
            expected_mip2
        );
        assert!(
            readback_mip(host, resource_guard.0, host_tex, 1)
                .iter()
                .all(|&b| b == 0)
        );

        // Stream in mip 1 while leaving mip 0 `[131072..393216)` unmapped.
        region.commit(65536, 65536);
        let expected_mip1: Vec<u8> = (0..128 * 128 * 4)
            .map(|i| (((i * 29 + 19) % 241) as u8) | 1)
            .collect();
        tiled.fill(0);
        layout
            .tile(&mut tiled, 1, 0, &expected_mip1, 512, 128 * 512)
            .unwrap();
        // SAFETY: tile 1 `[65536..131072)` was committed above.
        unsafe { region.write_slice(65536, &tiled[65536..131072]) };
        assert!(!committed(region.ptr, layout.size as usize));

        list.reset();
        upload(&list, resource_guard.0, host_tex);
        list.close();
        host.execute(&[&list]);

        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 1),
            expected_mip1
        );
        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 2),
            expected_mip2
        );

        // Evict mip 1: `zero_gaps` must clear stale cached bytes so host mip 1 becomes zeroed.
        region.decommit(65536, 65536);
        list.reset();
        upload(&list, resource_guard.0, host_tex);
        list.close();
        host.execute(&[&list]);

        assert!(
            readback_mip(host, resource_guard.0, host_tex, 1)
                .iter()
                .all(|&b| b == 0)
        );
        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 2),
            expected_mip2
        );
    }
    #[test]
    fn partially_resident_bc7_mip_uploads_and_clears_after_eviction() {
        let Some(host) = crate::host::host() else {
            return;
        };
        use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC7_UNORM;

        let format = DXGI_FORMAT_BC7_UNORM.0;
        let texture = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: 512,
            height: 512,
            depth_or_array: 1,
            mips: 5,
            format: format as u32,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let layout = &texture.planes[0].layout;
        let mip = layout.first_tail_mip;
        assert!(mip < 5 && layout.size > 65536);
        let info = layout.mips[mip as usize];
        assert!(info.tail.is_some() && info.offset == 0);
        let row = info.width as usize * layout.desc.element_bytes as usize;
        let len = row * info.height as usize;
        let expected: Vec<u8> = (0..len).map(|n| ((n * 41 + 3) % 255 + 1) as u8).collect();
        let mut tiled = vec![0u8; layout.size as usize];
        layout.tile(&mut tiled, mip, 0, &expected, row, len).unwrap();
        assert!(tiled[65536..].iter().all(|&b| b == 0));
        let region = VirtualRegion::reserve(tiled.len());
        region.commit(0, 65536);
        // SAFETY: the first tile is committed.
        unsafe { region.write_slice(0, &tiled[..65536]) };

        let resource = title_texture(&region, 512, 5, format);
        let host_tex = crate::resource::host_resource(resource.0).unwrap();
        let list = host.new_list().unwrap();
        upload(&list, resource.0, host_tex);
        list.close();
        host.execute(&[&list]);

        assert_eq!(readback_mip(host, resource.0, host_tex, mip), expected);

        region.decommit(0, 65536);
        list.reset();
        upload(&list, resource.0, host_tex);
        list.close();
        host.execute(&[&list]);
        assert_eq!(readback_mip(host, resource.0, host_tex, mip), vec![0; len]);
    }
}
