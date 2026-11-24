//! First-lane reads in the single-thread tail of a compute reduction.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, Operand};
use crate::io::{Input, Io};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::resource::{Site, inline};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// Recognizes a thread-zero marker carried through a join, then tested at the
/// entrance to a suffix where all other threads exit. A read in that suffix
/// under the gate's unchanged EXEC reads the thread's own value. Executing it
/// locally avoids exchanging values between host waves after they diverge.
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
    let terminal = |b: usize| {
        let block = &program.blocks[b];
        block.exit == Exit::End && block.insts.iter().all(|inst| inst.name() == "s_endpgm")
    };
    for (gate, block) in program.blocks.iter().enumerate() {
        let Some((gate_exec, entry, inactive)) = compare_branch(structure, gate) else {
            continue;
        };
        if !terminal(inactive) {
            continue;
        }
        let Some((gate_def, compare)) = definition(gate_exec) else {
            continue;
        };
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
        let Some((id, 0)) = compare_const(compare).filter(|_| pick_def == *pick) else {
            continue;
        };
        let Some((_, flatten)) = definition(id) else {
            continue;
        };
        let Some([y, shift, x]) = operands(flatten) else {
            continue;
        };
        let thread_component = |o, c| {
            reg_value(flatten, o)
                .is_some_and(|v| io.inputs.contains(&(v, Input::ThreadIdInGroup(c))))
        };
        if flatten.name() != "v_lshl_add_u32"
            || constant(shift) != Some(width.trailing_zeros())
            || !thread_component(x, 0)
            || !thread_component(y, 1)
        {
            continue;
        }

        let mut suffix = HashSet::new();
        let mut pending = vec![entry];
        while let Some(b) = pending.pop() {
            if !terminal(b) && suffix.insert(b) {
                pending.extend(program.blocks[b].exit.successors());
            }
        }
        if suffix.iter().any(|&b| {
            b == gate
                || program.blocks[b].exit == Exit::Indirect
                || program.blocks[b]
                    .preds
                    .iter()
                    .any(|p| !suffix.contains(p) && !(b == entry && *p == gate))
        }) {
            continue;
        }
        for b in suffix {
            for (i, inst) in program.blocks[b].insts.iter().enumerate() {
                if inst.name() == "v_readfirstlane_b32"
                    && structure.guards[b][i].is_none()
                    && inst.read(Reg::S(EXEC)) == Some(gate_exec)
                    && inst.read(Reg::S(EXEC + 1)) == Some(gate_high)
                {
                    reads.insert((b, i));
                }
            }
        }
    }
    reads
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
    if inst.name() != "v_cmpx_eq_u32" {
        return None;
    }
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
