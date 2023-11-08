use core::{
    crypto::poseidon_trace::calculate_poseidon,
    trace::trace::Step,
    types::{Field, GoldilocksField},
    vm::opcodes::OlaOpcode,
};
use std::cmp::max;

use itertools::Itertools;
use plonky2::{
    hash::hash_types::RichField,
    iop::challenger::Challenger,
    plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
};

use crate::{program::columns::*, stark::lookup::permuted_cols};

pub fn generate_prog_trace<F: RichField>(
    execs: &[Step],
    progs: Vec<([GoldilocksField; 4], Vec<GoldilocksField>)>,
    start_end_roots: ([GoldilocksField; 4], [GoldilocksField; 4]),
) -> ([Vec<F>; NUM_PROG_COLS], F) {
    let mut challenger =
        Challenger::<F, <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher>::new();
    for limb_idx in 0..4 {
        challenger.observe_element(F::from_canonical_u64(start_end_roots.0[limb_idx].0));
        challenger.observe_element(F::from_canonical_u64(start_end_roots.1[limb_idx].0))
    }
    let beta = challenger.get_challenge();

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
            beta,
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
                beta,
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
                beta,
            );
            prog_index += 1;
        }
    }
    let (permuted_inputs, permuted_table) =
        permuted_cols(&trace[COL_PROG_EXEC_COMP_PROG], &trace[COL_PROG_COMP_PROG]);
    trace[COL_PROG_EXEC_COMP_PROG_PERM] = permuted_inputs;
    trace[COL_PROG_COMP_PROG_PERM] = permuted_table;

    let trace_row_vecs = trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_PROG_COLS,
            v.len()
        )
    });
    (trace_row_vecs, beta)
}

fn compress<F: RichField>(values: [F; 6], beta: F) -> F {
    values[0]
        + values[1] * beta
        + values[2] * beta * beta
        + values[3] * beta * beta * beta
        + values[4] * beta * beta * beta * beta
        + values[5] * beta * beta * beta * beta * beta
}

pub fn generate_prog_chunk_trace<F: RichField>(
    progs: Vec<([GoldilocksField; 4], Vec<GoldilocksField>)>,
) -> [Vec<F>; NUM_PROG_CHUNK_COLS] {
    let vec_addr_pc_chunk = progs
        .into_iter()
        .map(|(addr, insts)| {
            let chunks: Vec<Vec<_>> = insts.chunks(8).map(|chunk| chunk.to_vec()).collect();
            (addr, chunks)
        })
        .flat_map(|(addr, chunks)| {
            let chunks_len = chunks.len();
            chunks
                .into_iter()
                .enumerate()
                .map(move |(chunk_idx, chunk)| {
                    let is_first_line = chunk_idx == 0;
                    let is_result_line = chunk_idx == chunks_len - 1;
                    (
                        addr.clone(),
                        chunk_idx * 8,
                        chunk,
                        is_first_line,
                        is_result_line,
                    )
                })
        })
        .collect_vec();

    let num_filled_row_len: usize = vec_addr_pc_chunk.len();
    let num_padded_rows = if !num_filled_row_len.is_power_of_two() || num_filled_row_len < 2 {
        if num_filled_row_len < 2 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        }
    } else {
        num_filled_row_len
    };
    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; NUM_PROG_CHUNK_COLS];

    let mut pre_hash = [F::ZERO; 12];
    for (i, (addr, start_pc, chunk, is_first_line, is_result_line)) in
        vec_addr_pc_chunk.iter().enumerate()
    {
        for j in 0..4 {
            trace[COL_PROG_CHUNK_CODE_ADDR_RANGE.start + j][i] = F::from_canonical_u64(addr[j].0);
        }
        trace[COL_PROG_CHUNK_START_PC][i] = F::from_canonical_u64(*start_pc as u64);
        let mut hash_input: [GoldilocksField; 12] = [GoldilocksField::ZERO; 12];
        let chunk_len = chunk.len();
        for j in 0..chunk_len {
            let v = F::from_canonical_u64(chunk[j].0);
            trace[COL_PROG_CHUNK_INST_RANGE.start + j][i] = v;
            hash_input[j] = GoldilocksField::from_canonical_u64(v.to_canonical_u64());
        }
        for j in chunk_len..8 {
            let v = pre_hash[j];
            trace[COL_PROG_CHUNK_INST_RANGE.start + j][i] = v;
            hash_input[j] = GoldilocksField::from_canonical_u64(v.to_canonical_u64());
        }
        for j in 0..4 {
            let v = pre_hash[j + 8];
            trace[COL_PROG_CHUNK_CAP_RANGE.start + j][i] = v;
            hash_input[j + 8] = GoldilocksField::from_canonical_u64(v.to_canonical_u64());
        }
        let hash = calculate_poseidon(hash_input);
        for j in 0..12 {
            let limb = F::from_canonical_u64(hash[j].0);
            trace[COL_PROG_CHUNK_HASH_RANGE.start + j][i] = limb;
            pre_hash[j] = limb;
        }
        trace[COL_PROG_CHUNK_IS_FIRST_LINE][i] = if *is_first_line { F::ONE } else { F::ZERO };
        trace[COL_PROG_CHUNK_IS_RESULT_LINE][i] = if *is_result_line { F::ONE } else { F::ZERO };
        for j in 0..chunk_len {
            trace[COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE.start + j][i] = F::ONE;
        }
    }

    if num_padded_rows != num_filled_row_len {
        for i in num_filled_row_len..num_padded_rows {
            trace[COL_PROG_CHUNK_IS_PADDING_LINE][i] = F::ONE;
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_PROG_COLS,
            v.len()
        )
    })
}
