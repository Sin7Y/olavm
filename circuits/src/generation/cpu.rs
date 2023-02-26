use core::{
    program::{instruction::Opcode, REGISTER_NUM},
    trace::trace::Step,
};

use crate::{cpu::columns as cpu, stark::lookup::permuted_cols};
use plonky2::{
    hash::hash_types::RichField,
    iop::challenger::Challenger,
    plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    util::transpose,
};

pub fn generate_cpu_trace<F: RichField>(
    steps: &[Step],
    raw_instructions: &[String],
) -> (Vec<[F; cpu::NUM_CPU_COLS]>, F) {
    let mut raw_insts: Vec<(usize, F)> = raw_instructions
        .iter()
        .enumerate()
        .map(|(i, ri)| {
            (
                i,
                F::from_canonical_u64(u64::from_str_radix(&ri[2..], 16).unwrap()),
            )
        })
        .collect();

    // make raw and steps has same length.
    let mut steps = steps.to_vec();
    if raw_instructions.len() < steps.len() {
        raw_insts.resize(steps.len(), raw_insts.last().unwrap().to_owned());
    } else if raw_instructions.len() > steps.len() {
        steps.resize(raw_instructions.len(), steps.last().unwrap().to_owned());
    }

    let mut trace: Vec<[F; cpu::NUM_CPU_COLS]> = steps
        .iter()
        .zip(raw_insts.iter())
        .map(|(s, r)| {
            let mut row: [F; cpu::NUM_CPU_COLS] = [F::default(); cpu::NUM_CPU_COLS];

            // Context related columns.
            row[cpu::COL_CLK] = F::from_canonical_u32(s.clk);
            row[cpu::COL_PC] = F::from_canonical_u64(s.pc);
            for i in 0..REGISTER_NUM {
                row[cpu::COL_START_REG + i] = F::from_canonical_u64(s.regs[i].0);
            }

            // Instruction related columns.
            row[cpu::COL_INST] = F::from_canonical_u64(s.instruction.0);
            row[cpu::COL_OP1_IMM] = F::from_canonical_u64(s.op1_imm.0);
            row[cpu::COL_OPCODE] = F::from_canonical_u64(s.opcode.0);
            row[cpu::COL_IMM_VAL] = F::from_canonical_u64(s.immediate_data.0);

            // Selectors of register related columns.
            row[cpu::COL_OP0] = F::from_canonical_u64(s.register_selector.op0.0);
            row[cpu::COL_OP1] = F::from_canonical_u64(s.register_selector.op1.0);
            row[cpu::COL_DST] = F::from_canonical_u64(s.register_selector.dst.0);
            row[cpu::COL_AUX0] = F::from_canonical_u64(s.register_selector.aux0.0);
            row[cpu::COL_AUX1] = F::from_canonical_u64(s.register_selector.aux1.0);
            for i in 0..REGISTER_NUM {
                row[cpu::COL_S_OP0_START + i] =
                    F::from_canonical_u64(s.register_selector.op0_reg_sel[i].0);
                row[cpu::COL_S_OP1_START + i] =
                    F::from_canonical_u64(s.register_selector.op1_reg_sel[i].0);
                row[cpu::COL_S_DST_START + i] =
                    F::from_canonical_u64(s.register_selector.dst_reg_sel[i].0);
            }

            // Selectors of opcode related columns.
            match s.opcode.0 {
                o if (1_u64 << Opcode::ADD as u8) == o => {
                    row[cpu::COL_S_ADD] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::MUL as u8) == o => {
                    row[cpu::COL_S_MUL] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::EQ as u8) == o => {
                    row[cpu::COL_S_EQ] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::ASSERT as u8) == o => {
                    row[cpu::COL_S_ASSERT] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::MOV as u8) == o => {
                    row[cpu::COL_S_MOV] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::JMP as u8) == o => {
                    row[cpu::COL_S_JMP] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::CJMP as u8) == o => {
                    row[cpu::COL_S_CJMP] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::CALL as u8) == o => {
                    row[cpu::COL_S_CALL] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::RET as u8) == o => {
                    row[cpu::COL_S_RET] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::MLOAD as u8) == o => {
                    row[cpu::COL_S_MLOAD] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::MSTORE as u8) == o => {
                    row[cpu::COL_S_MSTORE] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::END as u8) == o => {
                    row[cpu::COL_S_END] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::RC as u8) == o => {
                    row[cpu::COL_S_RC] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::AND as u8) == o => {
                    row[cpu::COL_S_AND] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::OR as u8) == o => {
                    row[cpu::COL_S_OR] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::XOR as u8) == o => {
                    row[cpu::COL_S_XOR] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::NOT as u8) == o => {
                    row[cpu::COL_S_NOT] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::NEQ as u8) == o => {
                    row[cpu::COL_S_NEQ] = F::from_canonical_u64(1)
                }
                o if (1_u64 << Opcode::GTE as u8) == o => {
                    row[cpu::COL_S_GTE] = F::from_canonical_u64(1)
                }
                // o if (1_u64 << Opcode::PSDN as u8) == o => row[cpu::COL_S_PSDN] =
                // F::from_canonical_u64(1), o (1_u64 << Opcode::ECDSA as u8) == o
                // => row[cpu::COL_S_ECDSA] = F::from_canonical_u64(1),
                _ => panic!("unspported opcode!"),
            }

            // Raw program
            row[cpu::COL_RAW_INST] = r.1;
            row[cpu::COL_RAW_PC] = F::from_canonical_usize(r.0);

            row
        })
        .collect();

    // We use our public (program) column to generate oracles.
    let mut challenger =
        Challenger::<F, <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher>::new();
    let mut raw_insts = vec![];
    trace.iter().for_each(|row| {
        raw_insts.push(row[cpu::COL_RAW_INST]);
    });
    challenger.observe_elements(&raw_insts);
    let beta = challenger.get_challenge();

    // Compress raw_pc and raw_inst columns into one column: COL_ZIP_RAW.
    // Compress pc + inst columns into one column: COL_ZIP_EXED.
    trace.iter_mut().for_each(|row| {
        row[cpu::COL_ZIP_RAW] = row[cpu::COL_RAW_INST] * beta + row[cpu::COL_RAW_PC];
        row[cpu::COL_ZIP_EXED] = row[cpu::COL_INST] * beta + row[cpu::COL_PC];
    });

    // Pad trace to power of two, we use last row `END` to do it.
    let row_len = trace.len();
    if !row_len.is_power_of_two() {
        let new_row_len = row_len.next_power_of_two();
        trace.resize(new_row_len, trace.last().unwrap().to_owned());
    }

    // Transpose to column-major form.
    let trace_row_vecs: Vec<_> = trace.into_iter().map(|row| row.to_vec()).collect();
    let mut trace_col_vecs = transpose(&trace_row_vecs);

    // Permuate zip_raw and zip_exed column.
    let (permuted_inputs, permuted_table) = permuted_cols(
        &trace_col_vecs[cpu::COL_ZIP_EXED],
        &trace_col_vecs[cpu::COL_ZIP_RAW],
    );
    trace_col_vecs[cpu::COL_PER_ZIP_EXED] = permuted_inputs;
    trace_col_vecs[cpu::COL_PER_ZIP_RAW] = permuted_table;

    let final_trace = transpose(&trace_col_vecs);
    let trace_row_vecs: Vec<[F; cpu::NUM_CPU_COLS]> = final_trace
        .into_iter()
        .map(|row| row.try_into().unwrap())
        .collect();

    (trace_row_vecs, beta)
}
