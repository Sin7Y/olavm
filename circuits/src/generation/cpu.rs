use core::{
    program::{instruction::Opcode, REGISTER_NUM},
    trace::trace::Step,
};

use crate::{cpu::columns as cpu, stark::lookup::permuted_cols};
use plonky2::{
    hash::hash_types::RichField,
    iop::challenger::Challenger,
    plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
};

pub fn generate_cpu_trace<F: RichField, C: GenericConfig<D, F = F>, const D: usize>(
    steps: &[Step],
    raw_instructions: &[String],
) -> ([Vec<F>; cpu::NUM_CPU_COLS], F) {
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
    let trace_len = std::cmp::max(steps.len(), raw_instructions.len());
    if raw_instructions.len() < steps.len() {
        raw_insts.resize(trace_len, raw_insts.last().unwrap().to_owned());
    } else if raw_instructions.len() > steps.len() {
        steps.resize(trace_len, steps.last().unwrap().to_owned());
    }

    let ext_trace_len = if !trace_len.is_power_of_two() {
        trace_len.next_power_of_two()
    } else {
        trace_len
    };
    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; ext_trace_len]; cpu::NUM_CPU_COLS];
    for (i, (s, r)) in steps.iter().zip(raw_insts.iter()).enumerate() {
        // Context related columns.
        trace[cpu::COL_CLK][i] = F::from_canonical_u32(s.clk);
        trace[cpu::COL_PC][i] = F::from_canonical_u64(s.pc);
        for j in 0..REGISTER_NUM {
            trace[cpu::COL_START_REG + j][i] = F::from_canonical_u64(s.regs[j].0);
        }

        // Instruction related columns.
        trace[cpu::COL_INST][i] = F::from_canonical_u64(s.instruction.0);
        trace[cpu::COL_OP1_IMM][i] = F::from_canonical_u64(s.op1_imm.0);
        trace[cpu::COL_OPCODE][i] = F::from_canonical_u64(s.opcode.0);
        trace[cpu::COL_IMM_VAL][i] = F::from_canonical_u64(s.immediate_data.0);

        // Selectors of register related columns.
        trace[cpu::COL_OP0][i] = F::from_canonical_u64(s.register_selector.op0.0);
        trace[cpu::COL_OP1][i] = F::from_canonical_u64(s.register_selector.op1.0);
        trace[cpu::COL_DST][i] = F::from_canonical_u64(s.register_selector.dst.0);
        trace[cpu::COL_AUX0][i] = F::from_canonical_u64(s.register_selector.aux0.0);
        trace[cpu::COL_AUX1][i] = F::from_canonical_u64(s.register_selector.aux1.0);
        for j in 0..REGISTER_NUM {
            trace[cpu::COL_S_OP0_START + j][i] =
                F::from_canonical_u64(s.register_selector.op0_reg_sel[j].0);
            trace[cpu::COL_S_OP1_START + j][i] =
                F::from_canonical_u64(s.register_selector.op1_reg_sel[j].0);
            trace[cpu::COL_S_DST_START + j][i] =
                F::from_canonical_u64(s.register_selector.dst_reg_sel[j].0);
        }

        // Selectors of opcode related columns.
        match s.opcode.0 {
            o if (1_u64 << Opcode::ADD as u8) == o => {
                trace[cpu::COL_S_ADD][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::MUL as u8) == o => {
                trace[cpu::COL_S_MUL][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::EQ as u8) == o => {
                trace[cpu::COL_S_EQ][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::ASSERT as u8) == o => {
                trace[cpu::COL_S_ASSERT][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::MOV as u8) == o => {
                trace[cpu::COL_S_MOV][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::JMP as u8) == o => {
                trace[cpu::COL_S_JMP][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::CJMP as u8) == o => {
                trace[cpu::COL_S_CJMP][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::CALL as u8) == o => {
                trace[cpu::COL_S_CALL][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::RET as u8) == o => {
                trace[cpu::COL_S_RET][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::MLOAD as u8) == o => {
                trace[cpu::COL_S_MLOAD][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::MSTORE as u8) == o => {
                trace[cpu::COL_S_MSTORE][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::END as u8) == o => {
                trace[cpu::COL_S_END][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::RC as u8) == o => {
                trace[cpu::COL_S_RC][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::AND as u8) == o => {
                trace[cpu::COL_S_AND][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::OR as u8) == o => {
                trace[cpu::COL_S_OR][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::XOR as u8) == o => {
                trace[cpu::COL_S_XOR][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::NOT as u8) == o => {
                trace[cpu::COL_S_NOT][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::NEQ as u8) == o => {
                trace[cpu::COL_S_NEQ][i] = F::from_canonical_u64(1)
            }
            o if (1_u64 << Opcode::GTE as u8) == o => {
                trace[cpu::COL_S_GTE][i] = F::from_canonical_u64(1)
            }
            // o if (1_u64 << Opcode::PSDN as u8) == o => {
            //     trace[cpu::COL_S_PSDN][i] = F::from_canonical_u64(1)
            // }
            // o if (1_u64 << Opcode::ECDSA as u8) == o  => {
            //     trace[cpu::COL_S_ECDSA][i] = F::from_canonical_u64(1)
            // }
            _ => panic!("unspported opcode!"),
        }

        // Raw program
        trace[cpu::COL_RAW_INST][i] = r.1;
        trace[cpu::COL_RAW_PC][i] = F::from_canonical_usize(r.0);
    }

    // For expanded trace from `trace_len` to `trace_len's power of two`,
    // we use last row `END` to pad them, and except for zipped and permuated rows.
    if trace_len != ext_trace_len {
        trace[cpu::COL_CLK..cpu::COL_ZIP_RAW]
            .iter_mut()
            .for_each(|row| {
                let last = row[trace_len - 1];
                row[trace_len..].fill(last);
            });
    }

    // We use our public (program) column to generate oracles.
    let mut challenger =
        Challenger::<F, C::Hasher>::new();
    challenger.observe_elements(&trace[cpu::COL_RAW_INST]);
    let beta = challenger.get_challenge();

    // Compress raw_pc and raw_inst columns into one column: COL_ZIP_RAW.
    // Compress pc + inst columns into one column: COL_ZIP_EXED.
    for i in 0..trace[0].len() {
        trace[cpu::COL_ZIP_RAW][i] = trace[cpu::COL_RAW_INST][i] * beta + trace[cpu::COL_RAW_PC][i];
        trace[cpu::COL_ZIP_EXED][i] = trace[cpu::COL_INST][i] * beta + trace[cpu::COL_PC][i];
    }

    // Permuate zip_raw and zip_exed column.
    let (permuted_inputs, permuted_table) =
        permuted_cols(&trace[cpu::COL_ZIP_EXED], &trace[cpu::COL_ZIP_RAW]);
    trace[cpu::COL_PER_ZIP_EXED] = permuted_inputs;
    trace[cpu::COL_PER_ZIP_RAW] = permuted_table;

    let trace_row_vecs = trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            cpu::NUM_CPU_COLS,
            v.len()
        )
    });

    (trace_row_vecs, beta)
}
