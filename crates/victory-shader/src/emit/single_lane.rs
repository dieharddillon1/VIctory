//! First-lane reads in the single-thread tail of a compute reduction.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, Operand};
use crate::io::{Input, Io};
use crate::ir::{EXEC, Exit, Inst, M0, Program, Reg, Value};
use crate::resource::{Site, inline};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// Finds regions entered only by thread zero, while all other threads
/// either exit or immediately skip the region's consumers at its only join.
/// An unguarded read under the gate's unchanged EXEC reads the thread's
/// own value without exchanging it between divergent host waves.
pub(super) fn first_lane_reads(
    program: &Program,
    structure: &Structured,
    io: &Io,
    threads: [u32; 3],
) -> HashSet<Site> {
    let mut reads = HashSet::new();
    let [width, _, depth] = threads;
    if depth != 1 || !width.is_power_of_two() {
        return reads;
    }
    let defs: HashMap<_, _> = program
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block.insts.iter().enumerate().flat_map(move |(i, inst)| {
                inst.writes.iter().map(move |&(_, value)| (value, (b, i)))
            })
        })
        .collect();
    let definition = |v| {
        let &(b, i) = defs.get(&v)?;
        structure.guards[b][i]
            .is_none()
            .then_some((b, &program.blocks[b].insts[i]))
    };
    let flat_thread_zero = |compare| {
        let Some((id, 0)) = compare_const(compare) else {
            return false;
        };
        definition(id).is_some_and(|(_, flatten)| is_flat_thread_id(flatten, io, width))
    };
    for (gate, block) in program.blocks.iter().enumerate() {
        if threads[1] == 1
            && let Some((mask, entry, join, exec)) = saved_exec_gate(block)
            && thread_zero_mask(program, io, mask, &defs)
            && let Some(enclosed) = region(program, gate, entry, join)
        {
            let local = region_reads(program, structure, &enclosed, exec);
            if !reads_escape(program, structure, &enclosed, &local) {
                reads.extend(local);
            }
        }
        let Some((gate_exec, entry, inactive)) = compare_branch(structure, gate) else {
            continue;
        };
        let Some((gate_def, compare)) = definition(gate_exec) else {
            continue;
        };
        if compare.name() == "v_cmp_eq_u32" && gate_def == gate && flat_thread_zero(compare) {
            let Some(exec) = compare_exec(program, gate, compare, gate_exec) else {
                continue;
            };
            if terminal(program, inactive) {
                reads.extend(
                    region(program, gate, entry, inactive)
                        .map(|region| region_reads(program, structure, &region, exec))
                        .unwrap_or_default(),
                );
            } else {
                // Before reading values computed under the first gate,
                // converge and immediately reapply the same predicate.
                let region_exit = &program.blocks[inactive];
                let Some((again, second, end)) = compare_branch(structure, inactive) else {
                    continue;
                };
                if again != gate_exec
                    || !region_exit.insts.iter().all(|inst| match inst.name() {
                        "s_mov_b64" => inst.write(Reg::S(EXEC)).is_some(),
                        "s_and_saveexec_b64" | "s_cbranch_execz" | "s_cbranch_execnz" => true,
                        _ => false,
                    })
                {
                    continue;
                }
                let Some(second_exec) = compare_exec(program, inactive, compare, gate_exec) else {
                    continue;
                };
                let Some(mut enclosed) = region(program, gate, entry, inactive) else {
                    continue;
                };
                let Some(again) = region(program, inactive, second, end) else {
                    continue;
                };
                let mut local_reads = region_reads(program, structure, &enclosed, exec);
                local_reads.extend(region_reads(program, structure, &again, second_exec));
                enclosed.extend(again);
                if !reads_escape(program, structure, &enclosed, &local_reads) {
                    reads.extend(local_reads);
                }
            }
            continue;
        }
        if compare.name() != "v_cmpx_eq_u32" || !terminal(program, inactive) {
            continue;
        }
        let Some((marker, target)) = compare_const(compare).filter(|_| gate_def == gate) else {
            continue;
        };
        let Some(gate_high) = compare.write(Reg::S(EXEC + 1)) else {
            continue;
        };
        let Some(phi) = block.phis.iter().find(|phi| phi.value == marker) else {
            continue;
        };
        let [first, second] = block.preds.as_slice() else {
            continue;
        };
        let [a, b] = phi.args.as_slice() else {
            continue;
        };
        let mov_from = |p, value| {
            let (b, mov) = definition(value)?;
            if b != p || mov.name() != "v_mov_b32" {
                return None;
            }
            let [op] = operands(mov)? else { return None };
            constant(op).map(|v| v == target)
        };
        let selected = match (mov_from(*first, *a), mov_from(*second, *b)) {
            (Some(true), Some(false)) => *first,
            (Some(false), Some(true)) => *second,
            _ => continue,
        };
        let [pick] = program.blocks[selected].preds.as_slice() else {
            continue;
        };
        let Some((pick_exec, matched, missed)) = compare_branch(structure, *pick) else {
            continue;
        };
        if matched != selected
            || missed != gate
            || program.blocks[selected].exit != Exit::Jump(gate)
        {
            continue;
        }
        let Some((pick_def, compare)) = definition(pick_exec) else {
            continue;
        };
        if pick_def == *pick && compare.name() == "v_cmpx_eq_u32" && flat_thread_zero(compare) {
            reads.extend(
                region(program, gate, entry, inactive)
                    .map(|region| region_reads(program, structure, &region, (gate_exec, gate_high)))
                    .unwrap_or_default(),
            );
        }
    }
    reads
}

/// A branch on the EXEC pair formed by s_and_saveexec_b64.
fn saved_exec_gate(block: &crate::ir::Block) -> Option<([Value; 2], usize, usize, (Value, Value))> {
    let Exit::Branch { taken, not_taken } = block.exit else {
        return None;
    };
    let branch = block.insts.last()?;
    let (entry, join) = match branch.name() {
        "s_cbranch_execz" => (not_taken, taken),
        "s_cbranch_execnz" => (taken, not_taken),
        _ => return None,
    };
    let exec = (branch.read(Reg::S(EXEC))?, branch.read(Reg::S(EXEC + 1))?);
    let save = block.insts.iter().find(|inst| {
        inst.name() == "s_and_saveexec_b64"
            && inst.write(Reg::S(EXEC)) == Some(exec.0)
            && inst.write(Reg::S(EXEC + 1)) == Some(exec.1)
    })?;
    let Fields::Sop { src, .. } = &save.fields else {
        return None;
    };
    let [Operand::Scalar { code, dwords: 2 }] = src.as_slice() else {
        return None;
    };
    let mask = [save.read(Reg::S(*code))?, save.read(Reg::S(*code + 1))?];
    Some((mask, entry, join, exec))
}

/// The pair of dwords of a comparison of local X against zero, or a
/// join of that mask and empty masks left by other paths.
fn thread_zero_mask(
    program: &Program,
    io: &Io,
    mask: [Value; 2],
    defs: &HashMap<Value, Site>,
) -> bool {
    let mut pending = vec![mask];
    let mut seen = HashSet::new();
    let mut has_leaf = false;
    while let Some(pair @ [lo, hi]) = pending.pop() {
        if !seen.insert(pair) {
            continue;
        }
        let Some(&(b, i)) = defs.get(&lo) else {
            let Some((a, b)) = program.blocks.iter().find_map(|block| {
                let a = block.phis.iter().find(|phi| phi.value == lo)?;
                let b = block.phis.iter().find(|phi| phi.value == hi)?;
                Some((a, b))
            }) else {
                return false;
            };
            pending.extend(a.args.iter().zip(&b.args).map(|(&lo, &hi)| [lo, hi]));
            continue;
        };
        let inst = &program.blocks[b].insts[i];
        let is_thread_zero = inst.name() == "v_cmp_eq_u32"
            && compare_const(inst).is_some_and(|(id, target)| {
                target == 0 && io.inputs.contains(&(id, Input::ThreadIdInGroup(0)))
            });
        let compare_false = inst.name() == "v_cmp_eq_u32"
            && matches!(operands(inst), Some([a,b])
                if constant(a).zip(constant(b)).is_some_and(|(a,b)| a != b));
        let moves_zero = inst.name() == "s_mov_b64"
            && matches!(&inst.fields,
            Fields::Sop { src, .. } if matches!(src.as_slice(), [source] if constant(source) == Some(0)));
        if !(is_thread_zero || compare_false || moves_zero)
            || inst
                .writes
                .get(..2)
                .is_none_or(|w| [w[0].1, w[1].1] != pair)
        {
            return false;
        }
        // The compare's lane result intersects EXEC, including when it is
        // predicated. A restricted compare can only remove the selected lane.
        has_leaf = true;
    }
    has_leaf
}

/// The EXEC pair formed by AND'ing the named lane compare into the live mask.
fn compare_exec(
    program: &Program,
    gate: usize,
    compare: &Inst,
    value: Value,
) -> Option<(Value, Value)> {
    let Fields::Valu(cmp) = &compare.fields else {
        return None;
    };
    let Some(Operand::Scalar { code, dwords: 2 }) = cmp.dst else {
        return None;
    };
    let high = compare.write(Reg::S(code + 1))?;
    let insts = &program.blocks[gate].insts;
    let branch = insts.last()?;
    let (lo_exec, hi_exec) = (branch.read(Reg::S(EXEC))?, branch.read(Reg::S(EXEC + 1))?);
    insts
        .iter()
        .any(|inst| {
            let Fields::Sop { src, .. } = &inst.fields else {
                return false;
            };
            let [Operand::Scalar { code, dwords: 2 }] = src.as_slice() else {
                return false;
            };
            inst.name() == "s_and_saveexec_b64"
                && inst.read(Reg::S(*code)) == Some(value)
                && inst.read(Reg::S(*code + 1)) == Some(high)
                && inst.write(Reg::S(EXEC)) == Some(lo_exec)
                && inst.write(Reg::S(EXEC + 1)) == Some(hi_exec)
        })
        .then_some((lo_exec, hi_exec))
}

fn terminal(program: &Program, b: usize) -> bool {
    let block = &program.blocks[b];
    block.exit == Exit::End && block.insts.iter().all(|inst| inst.name() == "s_endpgm")
}

/// Accept the lane-restricted region only when it has no other entry edges.
/// Its only exits can reach the explicit join or immediately end.
fn region(program: &Program, gate: usize, entry: usize, join: usize) -> Option<HashSet<usize>> {
    let mut region = HashSet::new();
    let mut pending = vec![entry];
    while let Some(b) = pending.pop() {
        if b != join && !terminal(program, b) && region.insert(b) {
            pending.extend(program.blocks[b].exit.successors());
        }
    }
    if region.iter().any(|&b| {
        b == gate
            || program.blocks[b].exit == Exit::Indirect
            || program.blocks[b]
                .insts
                .iter()
                .any(|inst| inst.name() == "s_barrier")
            || program.blocks[b]
                .preds
                .iter()
                .any(|p| !region.contains(p) && !(b == entry && *p == gate))
    }) {
        return None;
    }
    Some(region)
}

fn region_reads(
    program: &Program,
    structure: &Structured,
    region: &HashSet<usize>,
    (exec, exec_hi): (Value, Value),
) -> Vec<Site> {
    let mut reads = Vec::new();
    for &b in region {
        for (i, inst) in program.blocks[b].insts.iter().enumerate() {
            if inst.name() == "v_readfirstlane_b32"
                && operands(inst).is_some()
                && structure.guards[b][i].is_none()
                && inst.read(Reg::S(EXEC)) == Some(exec)
                && inst.read(Reg::S(EXEC + 1)) == Some(exec_hi)
            {
                reads.push((b, i));
            }
        }
    }
    reads
}

/// A first-lane result may cross the connecting phi, so follow its users
/// through both gated regions. Any actual use in the shared suffix would
/// rely on lanes that did not run the replacement.
fn reads_escape(
    program: &Program,
    structure: &Structured,
    enclosed: &HashSet<usize>,
    reads: &[Site],
) -> bool {
    let mut dependent: HashSet<_> = reads
        .iter()
        .flat_map(|&(b, i)| program.blocks[b].insts[i].writes.iter().map(|w| w.1))
        .collect();
    let mut changed = !dependent.is_empty();
    while changed {
        changed = false;
        for (b, block) in program.blocks.iter().enumerate() {
            for phi in &block.phis {
                if phi.args.iter().any(|v| dependent.contains(v)) {
                    changed |= dependent.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let used = inst.reads.iter().any(|&(reg, value)| {
                    dependent.contains(&value)
                        && !overwritten_vector(inst, reg, structure.guards[b][i].is_none())
                        && !relative_read_outside_loop_range(program, b, i, inst, reg)
                });
                if !used {
                    continue;
                }
                if !enclosed.contains(&b) {
                    return true;
                }
                for &(_, value) in &inst.writes {
                    changed |= dependent.insert(value);
                }
            }
        }
    }
    false
}

/// The IR enumerates every possible relative VGPR source. On a proven
/// ascending finite loop, indices outside its live range are unreachable.
fn relative_read_outside_loop_range(
    program: &Program,
    b: usize,
    i: usize,
    inst: &Inst,
    reg: Reg,
) -> bool {
    if inst.name() != "v_movrels_b32" {
        return false;
    }
    let (Reg::V(reg), Fields::Valu(v)) = (reg, &inst.fields) else {
        return false;
    };
    let Some(
        [
            Operand::Vector {
                reg: base,
                dwords: 1,
            },
        ],
    ) = operands(inst)
    else {
        return false;
    };
    if matches!(v.dst, Some(Operand::Vector { reg: dst, .. }) if dst == reg) {
        return false;
    }
    relative_loop_limit(program, b, i, inst)
        .is_some_and(|limit| u32::from(reg) >= u32::from(*base) + limit)
}

fn relative_loop_limit(program: &Program, b: usize, i: usize, inst: &Inst) -> Option<u32> {
    let m0 = inst.read(Reg::S(M0))?;
    let set = program.blocks[b].insts[..i]
        .iter()
        .find(|inst| inst.write(Reg::S(M0)) == Some(m0))?;
    let [Operand::Scalar { code, dwords: 1 }] = scalar_operands(set, "s_mov_b32")? else {
        return None;
    };
    let reg = Reg::S(*code);
    let counter = set.read(reg)?;
    let (header, phi) = program.blocks.iter().enumerate().find_map(|(b, block)| {
        block
            .phis
            .iter()
            .find(|phi| phi.value == counter && phi.reg == reg)
            .map(|phi| (b, phi))
    })?;
    let [entry, latch] = program.blocks[header].preds.as_slice() else {
        return None;
    };
    let [initial, next] = phi.args.as_slice() else {
        return None;
    };
    let init = program.blocks[*entry]
        .insts
        .iter()
        .find(|inst| inst.write(reg) == Some(*initial))?;
    let [imm] = scalar_operands(init, "s_mov_b32")? else {
        return None;
    };
    if constant(imm) != Some(0) {
        return None;
    }
    let latch = &program.blocks[*latch];
    let Exit::Branch { taken, not_taken } = latch.exit else {
        return None;
    };
    let branch = latch.insts.last()?;
    if taken != header || not_taken == header || branch.name() != "s_cbranch_scc0" {
        return None;
    }
    let inc = latch
        .insts
        .iter()
        .find(|inst| inst.write(reg) == Some(*next))?;
    let [Operand::Scalar { code, dwords: 1 }, step] = scalar_operands(inc, "s_add_i32")? else {
        return None;
    };
    if Reg::S(*code) != reg || inc.read(reg) != Some(counter) || constant(step) != Some(1) {
        return None;
    }
    let scc = branch.read(Reg::Scc)?;
    let test = latch
        .insts
        .iter()
        .find(|inst| inst.write(Reg::Scc) == Some(scc))?;
    let [Operand::Scalar { code, dwords: 1 }, bound] = scalar_operands(test, "s_cmp_eq_u32")?
    else {
        return None;
    };
    if Reg::S(*code) != reg || test.read(reg) != Some(*next) {
        return None;
    }
    constant(bound).filter(|limit| (1..=256).contains(limit))
}

fn scalar_operands<'a>(inst: &'a Inst, name: &str) -> Option<&'a [Operand]> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    (inst.name() == name).then_some(src)
}

/// The IR preserves an old destination for possibly guarded writes. A full
/// move from a different register does not actually consume that old value.
fn overwritten_vector(inst: &Inst, reg: Reg, unguarded: bool) -> bool {
    let (Reg::V(reg), Fields::Valu(v)) = (reg, &inst.fields) else {
        return false;
    };
    inst.name() == "v_mov_b32"
        && unguarded
        && v.dpp.is_none()
        && v.sdwa.is_none()
        && matches!(v.dst, Some(Operand::Vector { reg: r, dwords: 1 }) if r == reg)
        && !matches!(v.src.as_slice(), [Operand::Vector { reg: r, .. }] if *r == reg)
}

fn is_flat_thread_id(flatten: &Inst, io: &Io, width: u32) -> bool {
    let Some([y, shift, x]) = operands(flatten) else {
        return false;
    };
    let thread_component = |o, c| {
        reg_value(flatten, o).is_some_and(|v| io.inputs.contains(&(v, Input::ThreadIdInGroup(c))))
    };
    flatten.name() == "v_lshl_add_u32"
        && constant(shift) == Some(width.trailing_zeros())
        && thread_component(x, 0)
        && thread_component(y, 1)
}

/// The positive and negative destinations of one proven per-lane compare.
fn compare_branch(s: &Structured, b: usize) -> Option<(Value, usize, usize)> {
    let LaneExit::Branch {
        when,
        taken,
        not_taken,
    } = s.exits[b]
    else {
        return None;
    };
    match s.bools.node(when)? {
        (Atom::Lane(v), Bool::FALSE, Bool::TRUE) => Some((v, taken, not_taken)),
        (Atom::Lane(v), Bool::TRUE, Bool::FALSE) => Some((v, not_taken, taken)),
        _ => None,
    }
}

fn compare_const(inst: &Inst) -> Option<(Value, u32)> {
    let [a, b] = operands(inst)? else { return None };
    reg_value(inst, a)
        .zip(constant(b))
        .or_else(|| reg_value(inst, b).zip(constant(a)))
}

fn operands(inst: &Inst) -> Option<&[Operand]> {
    let Fields::Valu(v) = &inst.fields else {
        return None;
    };
    (v.neg == 0
        && v.abs == 0
        && v.neg_hi == 0
        && v.opsel == 0
        && v.opsel_hi == 0
        && !v.clamp
        && v.omod == 0
        && v.sdwa.is_none()
        && v.dpp.is_none())
    .then_some(&v.src)
}

fn constant(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Const(c) => inline(c),
        Operand::Literal(v) => Some(v),
        _ => None,
    }
}

fn reg_value(inst: &Inst, op: &Operand) -> Option<Value> {
    let Operand::Vector { reg, dwords: 1 } = *op else {
        return None;
    };
    inst.read(Reg::V(reg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir, structure};

    fn analyze(threads: [u32; 3], changed_exec: bool) -> HashSet<Site> {
        let mut words = vec![
            0xd746_0002,
            0x0401_0901,                           // v_lshl_add_u32 v2, v1, 4, v0
            0xbe80_047e,                           // s_mov_b64 s[0:1], exec
            0x7e06_0280,                           // v_mov_b32 v3, 0
            0x7da4_0480,                           // v_cmpx_eq_u32 exec, 0, v2
            0xbf88_0001,                           // s_cbranch_execz join
            0x7e06_02c1,                           // v_mov_b32 v3, -1
            0xbefe_0400,                           // join: s_mov_b64 exec, s[0:1]
            0x7da4_06c1,                           // v_cmpx_eq_u32 exec, -1, v3
            0xbf88_0002 | u32::from(changed_exec), // s_cbranch_execz inactive
        ];
        if changed_exec {
            words.push(0xbefe_04c1); // s_mov_b64 exec, -1
        }
        words.extend([
            0x7e04_0502, // v_readfirstlane_b32 s2, v2
            0xbf81_0000, // s_endpgm
            0xbf81_0000, // inactive: s_endpgm
        ]);
        analyze_words(&words, threads)
    }

    fn analyze_words(words: &[u32], threads: [u32; 3]) -> HashSet<Site> {
        let code = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        let program = ir::build(&code, true, 8).unwrap();
        let structured = structure::structure(&program).unwrap();
        let io = Io {
            inputs: program
                .inputs
                .iter()
                .filter_map(|&(reg, v)| match reg {
                    Reg::V(r @ 0..=1) => Some((v, Input::ThreadIdInGroup(r))),
                    _ => None,
                })
                .collect(),
            ..Io::default()
        };
        first_lane_reads(&program, &structured, &io, threads)
    }

    fn direct_retest(shared_use: bool) -> HashSet<Site> {
        let shared = if shared_use {
            0x7e08_0303 // v_mov_b32 v4, v3: consumes the marked arm's value
        } else {
            0x7e06_0280 // v_mov_b32 v3, 0: overwrites it in every lane
        };
        analyze_words(
            &[
                0xd746_0002,
                0x0401_0901, // v_lshl_add_u32 v2, v1, 4, v0
                0xd4c2_0000,
                0x0002_0480, // v_cmp_eq_u32 s[0:1], 0, v2
                0xbe82_2400, // s_and_saveexec_b64 s[2:3], s[0:1]
                0xbf88_0002, // s_cbranch_execz join
                0x7e0c_0502, // v_readfirstlane_b32 s6, v2
                0x7e06_0206, // v_mov_b32 v3, s6
                0xbefe_0402, // join: s_mov_b64 exec, s[2:3]
                0xbe84_2400, // s_and_saveexec_b64 s[4:5], s[0:1]
                0xbf88_0001, // s_cbranch_execz common
                0x7e08_0303, // v_mov_b32 v4, v3
                0xbefe_0404, // common: s_mov_b64 exec, s[4:5]
                shared,
                0xbf81_0000, // s_endpgm
            ],
            [16, 16, 1],
        )
    }

    #[test]
    fn retest_keeps_the_value_on_the_same_thread() {
        assert_eq!(direct_retest(false).len(), 1);
    }

    #[test]
    fn shared_path_cannot_use_a_gated_first_lane_result() {
        assert!(direct_retest(true).is_empty());
    }

    #[test]
    fn only_the_marked_thread_reads_its_own_value() {
        assert_eq!(analyze([16, 16, 1], false).len(), 1);
    }

    #[test]
    fn the_same_marker_can_select_many_threads_in_a_3d_group() {
        assert!(analyze([16, 16, 2], false).is_empty());
    }

    #[test]
    fn replacing_the_gate_exec_mask_disables_the_rewrite() {
        assert!(analyze([16, 16, 1], true).is_empty());
    }
}
