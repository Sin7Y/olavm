use core::{trace::trace::Step, types::GoldilocksField, vm::opcodes::OlaOpcode};
use std::cmp::max;

use plonky2::hash::hash_types::RichField;

use crate::{program::columns::*, stark::lookup::permuted_cols};

pub fn generate_prog_trace<F: RichField>(
    execs: &[Step],
    progs: Vec<([GoldilocksField; 4], Vec<GoldilocksField>)>,
    compress_challenger: F,
) -> [Vec<F>; NUM_PROG_COLS] {
    let main_lines: Vec<&Step> = execs.iter().filter(|e| e.is_ext_line.0 == 0).collect();
    let exec_len: usize = main_lines
        .iter()
        .map(|e| {
            if e.op1_imm.0 == 1
                || e.opcode.0 == OlaOpcode::MLOAD.binary_bit_mask()
                || e.opcode.0 == OlaOpcode::MSTORE.binary_bit_mask()
            {
                2
            } else {
                1
            }
        })
        .sum();
    let progs_total_len: usize = progs.iter().map(|v| v.1.len()).sum();
    let num_filled_row_len = max(exec_len, progs_total_len);
    let num_padded_rows = if !num_filled_row_len.is_power_of_two() || num_filled_row_len < 2 {
        if num_filled_row_len < 2 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        }
    } else {
        num_filled_row_len
    };

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; NUM_PROG_COLS];
    let mut exec_index = 0;
    for e in execs {
        for j in 0..4 {
            trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + j][exec_index] =
                F::from_canonical_u64(e.addr_code[j].0);
        }
        trace[COL_PROG_EXEC_PC][exec_index] = F::from_canonical_u64(e.pc);
        trace[COL_PROG_EXEC_INST][exec_index] = F::from_canonical_u64(e.instruction.0);
        trace[COL_PROG_FILTER_EXEC_OPERATION][exec_index] = F::ONE;
        trace[COL_PROG_EXEC_COMP_PROG][exec_index] = compress(
            [
                trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start][exec_index],
                trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 1][exec_index],
                trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 2][exec_index],
                trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 3][exec_index],
                trace[COL_PROG_EXEC_PC][exec_index],
                trace[COL_PROG_EXEC_INST][exec_index],
            ],
            compress_challenger,
        );
        exec_index += 1;

        // for immediate value
        if e.op1_imm.0 == 1
            || e.opcode.0 == OlaOpcode::MLOAD.binary_bit_mask()
            || e.opcode.0 == OlaOpcode::MSTORE.binary_bit_mask()
        {
            for j in 0..4 {
                trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + j][exec_index] =
                    F::from_canonical_u64(e.addr_code[j].0);
            }
            trace[COL_PROG_EXEC_PC][exec_index] = F::from_canonical_u64(e.pc + 1);
            trace[COL_PROG_EXEC_INST][exec_index] = F::from_canonical_u64(e.immediate_data.0);
            trace[COL_PROG_FILTER_EXEC_IMM_VALUE][exec_index] = F::ONE;
            trace[COL_PROG_EXEC_COMP_PROG][exec_index] = compress(
                [
                    trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start][exec_index],
                    trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 1][exec_index],
                    trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 2][exec_index],
                    trace[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 3][exec_index],
                    trace[COL_PROG_EXEC_PC][exec_index],
                    trace[COL_PROG_EXEC_INST][exec_index],
                ],
                compress_challenger,
            );
            exec_index += 1;
        }
    }

    let mut prog_index = 0;
    for (addr, prog) in progs {
        for (pc, inst) in prog.iter().enumerate() {
            for j in 0..4 {
                trace[COL_PROG_CODE_ADDR_RANGE.start + j][prog_index] =
                    F::from_canonical_u64(addr[j].0);
            }
            trace[COL_PROG_PC][prog_index] = F::from_canonical_u64(pc as u64);
            trace[COL_PROG_INST][prog_index] = F::from_canonical_u64(inst.0);
            trace[COL_PROG_FILTER_PROG_CHUNK][exec_index] = F::ONE;
            trace[COL_PROG_COMP_PROG][prog_index] = compress(
                [
                    trace[COL_PROG_CODE_ADDR_RANGE.start][prog_index],
                    trace[COL_PROG_CODE_ADDR_RANGE.start + 1][prog_index],
                    trace[COL_PROG_CODE_ADDR_RANGE.start + 2][prog_index],
                    trace[COL_PROG_CODE_ADDR_RANGE.start + 3][prog_index],
                    trace[COL_PROG_PC][prog_index],
                    trace[COL_PROG_INST][prog_index],
                ],
                compress_challenger,
            );
        }
        prog_index += 1;
    }
    let (permuted_inputs, permuted_table) =
        permuted_cols(&trace[COL_PROG_EXEC_COMP_PROG], &trace[COL_PROG_COMP_PROG]);
    trace[COL_PROG_EXEC_COMP_PROG_PERM] = permuted_inputs;
    trace[COL_PROG_COMP_PROG_PERM] = permuted_table;

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_PROG_COLS,
            v.len()
        )
    })
}

fn compress<F: RichField>(values: [F; 6], compress_challenger: F) -> F {
    values[0]
        + values[1] * compress_challenger
        + values[2] * compress_challenger * compress_challenger
        + values[3] * compress_challenger * compress_challenger * compress_challenger
        + values[4]
            * compress_challenger
            * compress_challenger
            * compress_challenger
            * compress_challenger
        + values[5]
            * compress_challenger
            * compress_challenger
            * compress_challenger
            * compress_challenger
            * compress_challenger
}
