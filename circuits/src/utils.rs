use crate::columns::*;
use core::program::{instruction::Opcode, REGISTER_NUM};
use core::trace::trace::{MemoryOperation, MemoryTraceCell, Step};

use plonky2::hash::hash_types::RichField;


pub(crate) fn generate_cpu_trace<F: RichField>(
    steps: &Vec<Step>,
    memory: &Vec<MemoryTraceCell>,
) -> Vec<[F; NUM_CPU_COLS]> {
    let trace: Vec<[F; NUM_CPU_COLS]> = steps.iter().map(|s| {
        let mut row: [F; NUM_CPU_COLS] = [F::default(); NUM_CPU_COLS];

        // Context related columns.
        row[COL_CLK] = F::from_canonical_u32(s.clk);
        row[COL_CLK] = F::from_canonical_u64(s.pc);
        row[COL_FLAG] = F::from_canonical_u32(s.flag as u32);
        for i in 0..REGISTER_NUM {
            row[COL_START_REG + i] = F::from_canonical_u64(s.regs[i].0);
        }

        // Instruction related columns.
        row[COL_INST] = F::from_canonical_u64(s.instruction.0);
        row[COL_OP1_IMM] = F::from_canonical_u64(s.op1_imm.0);
        row[COL_OPCODE] = F::from_canonical_u64(s.opcode.0);
        row[COL_IMM_VAL] = F::from_canonical_u64(s.immediate_data.0);

        // Selectors of register related columns.
        row[COL_OP0] = F::from_canonical_u64(s.register_selector.op0.0);
        row[COL_OP1] = F::from_canonical_u64(s.register_selector.op1.0);
        row[COL_DST] = F::from_canonical_u64(s.register_selector.dst.0);
        row[COL_AUX0] = F::from_canonical_u64(s.register_selector.aux0.0);
        for i in 0..REGISTER_NUM {
            row[COL_S_OP0_START + i] = F::from_canonical_u64(s.register_selector.op0_reg_sel[i].0);
            row[COL_S_OP1_START + i] = F::from_canonical_u64(s.register_selector.op1_reg_sel[i].0);
            row[COL_S_DST_START + i] = F::from_canonical_u64(s.register_selector.dst_reg_sel[i].0);
        }

        // Selectors of opcode related columns.
        match s.opcode.0 {
            o if 1_u8 << Opcode::ADD as u8 == o => row[COL_S_ADD] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::MUL as u8 == o => row[COL_S_MUL] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::EQ as u8 == o => row[COL_S_EQ] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::ASSERT as u8 == o => row[COL_S_ASSERT] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::MOV as u8 == o => row[COL_S_MOV] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::JMP as u8 == o => row[COL_S_JMP] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::CJMP as u8 == o => row[COL_S_CJMP] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::CALL as u8 == o => row[COL_S_CALL] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::RET as u8 == o => row[COL_S_RET] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::MLOAD as u8 == o => row[COL_S_MLOAD] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::MSTORE as u8 == o => row[COL_S_MSTORE] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::END as u8 == o => row[COL_S_END] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::RANGE_CHECK as u8 == o => row[COL_S_RC] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::AND as u8 == o => row[COL_S_AND] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::OR as u8 == o => row[COL_S_OR] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::XOR as u8 == o => row[COL_S_XOR] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::NOT as u8 == o => row[COL_S_NOT] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::NEQ as u8 == o => row[COL_S_NEQ] = F::from_canonical_u64(1),
            o if 1_u8 << Opcode::GTE as u8 == o => row[COL_S_GTE] = F::from_canonical_u64(1),
            // o if 1_u8 << Opcode::PSDN as u8 == o => row[COL_S_PSDN] = F::from_canonical_u64(1),
            // o if 1_u8 << Opcode::ECDSA as u8 == o => row[COL_S_ECDSA] = F::from_canonical_u64(1),

            _ => panic!("unspported opcode!"),
        }

        row
    }).collect();
    trace
}
