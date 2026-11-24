//! Keep a console wave with zero EXEC present for the host group barriers.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg};
use crate::native::{NativeProgram, StageIo};
use crate::resource::inline;

/// A wave that would exit before the first barrier can stay in the physical
/// PC thread group as long as its logical lanes never become active again.
/// Rebuild the IR after inserting an empty-mask trampoline for that edge.
pub(super) fn pad(native: &NativeProgram, program: &Program) -> Option<(NativeProgram, usize)> {
    let StageIo::Cs(cs) = &native.io else {
        return None;
    };
    let [width, 1, 1] = cs.threads else {
        return None;
    };
    if !native.wave64 || width <= 64 || !width.is_multiple_of(64) {
        return None;
    }
    let instructions = program.blocks.iter().flat_map(|block| &block.insts);
    if instructions
        .clone()
        .any(|inst| inst.name() == "s_getpc_b64")
    {
        return None;
    }
    let barrier = instructions
        .filter(|inst| inst.name() == "s_barrier")
        .map(|inst| inst.machine.offset)
        .min()?;
    let at_barrier = program.blocks.iter().position(|block| {
        block
            .insts
            .iter()
            .any(|inst| inst.machine.offset == barrier)
    })?;
    let mut exits = program.blocks.iter().enumerate().filter_map(|(b, block)| {
        let Exit::Branch {
            taken,
            not_taken: _,
        } = block.exit
        else {
            return None;
        };
        let branch = block.insts.last()?;
        let target = &program.blocks[taken];
        (branch.name() == "s_cbranch_execz"
            && branch.machine.offset < barrier
            && target.exit == Exit::End
            && target.insts.iter().all(|inst| inst.name() == "s_endpgm"))
        .then_some((b, taken, branch.machine.offset))
    });
    let (early, done, at) = exits.next()?;
    if exits.next().is_some()
        || !prefix_reaches_barrier(program, 0, at_barrier, (early, done), &mut HashMap::new())
    {
        return None;
    }

    let mut n = native.clone();
    let trampoline = n.code.len();
    fn displacement(from: usize, to: usize) -> Option<u32> {
        let bytes = i64::try_from(to).ok()? - i64::try_from(from).ok()? - 4;
        (bytes % 4 == 0).then_some(())?;
        Some(u32::from(i16::try_from(bytes / 4).ok()? as u16))
    }
    let early_jump = 0xbf88_0000u32 | displacement(at, trampoline)?;
    n.code
        .get_mut(at..at + 4)?
        .copy_from_slice(&early_jump.to_le_bytes());
    // Make VCC a tracked empty lane mask, not a stale compare from the
    // prologue. Keep EXEC at zero and leave all scalar state otherwise intact.
    for word in [
        0xd4c2_006a, // v_cmp_eq_u32 vcc, 0, 1
        0x0001_0280,
        0xbf82_0000 | displacement(trampoline + 8, barrier)?,
    ] {
        n.code.extend_from_slice(&word.to_le_bytes());
    }
    Some((n, barrier))
}

/// Every path through the prefix takes the padded edge or reaches the first
/// barrier. Refuse cycles in this prefix rather than presuming they finish.
fn prefix_reaches_barrier(
    program: &Program,
    b: usize,
    barrier: usize,
    padded: (usize, usize),
    seen: &mut HashMap<usize, bool>,
) -> bool {
    if b == barrier {
        return true;
    }
    if let Some(&okay) = seen.get(&b) {
        return okay;
    }
    seen.insert(b, false);
    let successors = program.blocks[b].exit.successors();
    let okay = !successors.is_empty()
        && successors.iter().all(|&next| {
            (b, next) == padded || prefix_reaches_barrier(program, next, barrier, padded, seen)
        });
    seen.insert(b, okay);
    okay
}

/// Proves that a lane entering this phase with EXEC cleared cannot be
/// reactivated. Starting from the shared barrier, EXEC changes may narrow the
/// current mask, restore a mask saved after the barrier, or clear it.
pub(super) fn stays_inactive(program: &Program, barrier: usize) -> bool {
    let Some(entry) = program
        .blocks
        .iter()
        .position(|block| block.offset == barrier)
    else {
        return false;
    };
    let pair = |code| [Reg::S(code), Reg::S(code + 1)];
    let [lo, hi] = pair(EXEC);
    let initial = |reg| {
        program.blocks[entry]
            .phis
            .iter()
            .find(|phi| phi.reg == reg)
            .map(|phi| phi.value)
    };
    let (Some(lo_root), Some(hi_root)) = (initial(lo), initial(hi)) else {
        return false;
    };
    let mut phase = HashSet::new();
    let mut next = vec![entry];
    while let Some(b) = next.pop() {
        let block = &program.blocks[b];
        if block.offset < barrier || block.exit == Exit::Indirect {
            return false;
        }
        if phase.insert(b) {
            next.extend(block.exit.successors());
        }
    }
    let mut defs = HashMap::new();
    let mut phis = HashMap::new();
    let mut pending = Vec::new();
    for &b in &phase {
        let block = &program.blocks[b];
        phis.extend(block.phis.iter().map(|phi| (phi.value, (b, phi))));
        for inst in &block.insts {
            if matches!(inst.fields, Fields::Smem { .. })
                && (inst.name().contains("store") || inst.name().contains("atomic"))
            {
                return false;
            }
            defs.extend(inst.writes.iter().map(|&(_, value)| (value, inst)));
            for rw in [Inst::read, Inst::write] {
                match (rw(inst, lo), rw(inst, hi)) {
                    (Some(lo), Some(hi)) => pending.push([lo, hi]),
                    (None, None) => {}
                    _ => return false,
                }
            }
        }
    }
    let mut seen = HashSet::from([[lo_root, hi_root]]);
    while let Some(mask @ [lo, hi]) = pending.pop() {
        if !seen.insert(mask) {
            continue;
        }
        if let (Some(&(a, lo)), Some(&(b, hi))) = (phis.get(&lo), phis.get(&hi))
            && a == b
        {
            pending.extend(lo.args.iter().zip(&hi.args).map(|(&lo, &hi)| [lo, hi]));
            continue;
        }
        let Some(&inst) = defs.get(&lo) else {
            return false;
        };
        let scalar = |op: &Operand| match *op {
            Operand::Scalar { code, dwords: 2 } => Some(pair(code)),
            _ => None,
        };
        let writes = |[a, b]: [Reg; 2]| {
            inst.write(a)
                .zip(inst.write(b))
                .is_some_and(|(a, b)| [a, b] == mask)
        };
        let read_pair = |[a, b]: [Reg; 2]| inst.read(a).zip(inst.read(b)).map(|(a, b)| [a, b]);
        let source = if inst.name().starts_with("v_cmpx_") && writes(pair(EXEC)) {
            read_pair(pair(EXEC))
        } else {
            let Fields::Sop { dst, src, .. } = &inst.fields else {
                return false;
            };
            let output_pair = dst.as_ref().and_then(scalar);
            match inst.name() {
                "s_and_saveexec_b64" if writes(pair(EXEC)) || output_pair.is_some_and(writes) => {
                    read_pair(pair(EXEC))
                }
                "s_mov_b64" if output_pair.is_some_and(writes) => {
                    let [op] = src.as_slice() else { return false };
                    if matches!(*op, Operand::Const(c) if inline(c) == Some(0)) {
                        continue;
                    }
                    scalar(op).and_then(read_pair)
                }
                _ => None,
            }
        };
        let Some(source) = source else {
            return false;
        };
        pending.push(source);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{self, TranslateOptions, Wave64Mode};
    use crate::native::{ComputeIo, HwStage, ResourceType, RootParam, RootType, UsedResource};
    use crate::{dxbc, io, ir, resource, structure};

    fn counting_shader(reactivate: bool) -> NativeProgram {
        let mut words = vec![
            0xbe81_0380, // s_mov_b32 s1, 0
            0xf408_0100,
            0xfa00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
            0x7da8_00c0, // v_cmpx_gt_u32 exec, 64, v0
            0xbf88_0000, // s_cbranch_execz done (patched below)
            0xbe88_047e, // s_mov_b64 s[8:9], exec
            0xbefe_0408, // s_mov_b64 exec, s[8:9]
            0xbf8a_0000, // s_barrier
            0xbe8a_0380, // s_mov_b32 s10, 0
        ];
        if reactivate {
            words.push(0xbefe_04c1); // s_mov_b64 exec, -1
        }
        let loop_header = words.len();
        words.extend([
            0x8f0b_0a81, // s_lshl_b32 s11, 1, s10
            0x3602_000b, // v_and_b32 v1, s11, v0
            0xd4c5_000c,
            0x0002_0280, // v_cmp_ne_u32 s[12:13], 0, v1
            0xbe88_047e, // s_mov_b64 s[8:9], exec
            0x7da4_0080, // v_cmpx_eq_u32 exec, 0, v0
            0xbe8e_100c, // s_bcnt1_i32_b64 s14, s[12:13]
            0x810e_810e, // s_add_i32 s14, s14, 1 (distinguish stores from zero-filled UAV)
            0x7e04_020a, // v_mov_b32 v2, s10
            0x3404_0482, // v_lshlrev_b32 v2, 2, v2
            0x7e06_020e, // v_mov_b32 v3, s14
            0xe070_1000,
            0x8001_0302, // buffer_store_dword v3, v2, s[4:7], 0 offen
            0xbefe_0408, // s_mov_b64 exec, s[8:9]
            0x810a_810a, // s_add_i32 s10, s10, 1
            0xbf06_990a, // s_cmp_eq_u32 s10, 25
        ]);
        let delta = loop_header as i32 - words.len() as i32 - 1;
        words.push(0xbf84_0000 | u32::from(delta as i16 as u16)); // s_cbranch_scc0 loop
        words.push(0xbf8a_0000); // s_barrier
        words[4] |= u32::try_from(words.len() - 5).unwrap();
        words.push(0xbf81_0000); // done: s_endpgm
        NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [128, 1, 1],
            }),
            code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
        }
    }

    #[test]
    fn an_early_exit_can_wait_inactive_but_must_not_reenable_exec() {
        for reactivate in [false, true] {
            let n = counting_shader(reactivate);
            let original = ir::build(&n.code, true, n.vgprs).unwrap();
            let (n, barrier) = pad(&n, &original).expect("an early zero-EXEC edge");
            let p = ir::build(&n.code, true, n.vgprs).unwrap();
            assert_eq!(stays_inactive(&p, barrier), !reactivate);
        }
    }

    #[test]
    fn fixed_trip_ballot_loop_translates_after_a_wave_exits() {
        shader_for_readback();
    }

    fn shader_for_readback() -> emit::HostShader {
        let n = counting_shader(false);
        let p = ir::build(&n.code, true, n.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let io = io::raise(&p, &n).unwrap();
        let res = resource::track(&p, &n.layout, n.stage).unwrap();
        let container = dxbc::build(&[]);
        let container = dxbc::Container::parse(&container).unwrap();
        emit::emit_with(
            &container,
            &n,
            &p,
            &st,
            &io,
            &res,
            TranslateOptions {
                wave64: Wave64Mode::Wave32,
            },
        )
        .expect("fixed-trip wave64 ballot loop")
    }

    #[cfg(windows)]
    #[test]
    fn padded_wave_ballot_counts_match_hardware_readback() {
        use emit::host::{Dispatch, Host, Root, View};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        let description = |adapter: &IDXGIAdapter1| {
            let name = unsafe { adapter.GetDesc1() }.unwrap().Description;
            let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
            String::from_utf16_lossy(&name[..end])
        };
        let (host, name) = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            if let Ok(filter) = std::env::var("VICTORY_ADAPTER") {
                let adapter = (0..)
                    .map_while(|i| factory.EnumAdapters1(i).ok())
                    .find(|adapter| {
                        description(adapter)
                            .to_lowercase()
                            .contains(&filter.to_lowercase())
                    })
                    .expect("requested hardware adapter");
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).unwrap();
                (
                    Host {
                        device: device.unwrap(),
                    },
                    description(&adapter),
                )
            } else {
                let Ok(host) = Host::new(false) else { return };
                let adapter: IDXGIAdapter1 = factory
                    .EnumAdapterByLuid(host.device.GetAdapterLuid())
                    .unwrap();
                (host, description(&adapter))
            }
        };
        if !host.supports_wave32() {
            return;
        }
        let shader = shader_for_readback();
        let (root, pipeline) = host.compute_pipeline(&shader).unwrap();
        let counts = host
            .dispatch(
                &root,
                &pipeline,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                    )],
                    views: &[View::RawUav(25)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let expect: Vec<_> = (0..25)
            .map(|bit| 1 + (0..64).filter(|lane| lane & (1 << bit) != 0).count() as u32)
            .collect();
        assert_eq!(counts[0], expect);
        eprintln!("{name}: all 25 counts matched after both virtual waves reached the group sync");
    }
}
