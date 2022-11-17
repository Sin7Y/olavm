use crate::columns::*;
use vm_core::program::{instruction::*, REGISTER_NUM};
use vm_core::trace::trace::{MemoryTraceCell, Step};

use plonky2::hash::hash_types::RichField;

pub(crate) fn generate_inst_trace<F: RichField>(steps: &Vec<Step>) -> Vec<[F; NUM_INST_COLS]> {
    #[macro_export]
    macro_rules! handle_op2 {
        ( $a:expr, $row:expr ) => {
            match $a {
                ImmediateOrRegName::Immediate(input1) => {
                    $row[COL_OP_2] = F::from_canonical_u64(input1.0);
                    $row[COL_IOR] = F::from_canonical_u64(0);
                }
                ImmediateOrRegName::RegName(reg_index) => {
                    assert!(reg_index < REGISTER_NUM as u8);
                    $row[COL_OP_2] = F::from_canonical_u8(reg_index);
                    $row[COL_IOR] = F::from_canonical_u64(1);
                }
            };
        };
    }

    let trace: Vec<[F; NUM_INST_COLS]> = steps
        .iter()
        .map(|s| {
            let mut row: [F; NUM_INST_COLS] = [F::default(); NUM_INST_COLS];
            row[COL_CLK] = F::from_canonical_u32(s.clk);
            row[COL_PC] = F::from_canonical_u64(s.pc);
            row[COL_FLAG] = F::from_canonical_u32(s.flag as u32);
            for i in 0..REGISTER_NUM {
                row[COL_REG + i] = F::from_canonical_u64(s.regs[i].0);
            }
            match s.instruction {
                Instruction::ADD(Add { ri, rj, a }) => {
                    assert!(ri < REGISTER_NUM as u8);
                    assert!(rj < REGISTER_NUM as u8);
                    row[COL_S_ADD] = F::from_canonical_u64(1);
                    row[COL_OP_0] = F::from_canonical_u8(ri);
                    row[COL_OP_1] = F::from_canonical_u8(rj);
                    handle_op2!(a, row);
                }
                Instruction::MUL(Mul { ri, rj, a }) => {
                    assert!(ri < REGISTER_NUM as u8);
                    assert!(rj < REGISTER_NUM as u8);
                    row[COL_S_MUL] = F::from_canonical_u64(1);
                    row[COL_OP_0] = F::from_canonical_u8(ri);
                    row[COL_OP_1] = F::from_canonical_u8(rj);
                    handle_op2!(a, row);
                }
                Instruction::EQ(Equal { ri, a }) => {
                    row[COL_S_EQ] = F::from_canonical_u64(1);
                    row[COL_OP_0] = F::from_canonical_u8(ri);
                    handle_op2!(a, row);
                }
                Instruction::MOV(Mov { ri, a }) => {
                    row[COL_S_MOV] = F::from_canonical_u64(1);
                    row[COL_OP_0] = F::from_canonical_u8(ri);
                    handle_op2!(a, row);
                }
                Instruction::JMP(Jmp { a }) => {
                    row[COL_S_JMP] = F::from_canonical_u64(1);
                    handle_op2!(a, row);
                }
                Instruction::CJMP(CJmp { a }) => {
                    row[COL_S_CJMP] = F::from_canonical_u64(1);
                    handle_op2!(a, row);
                }
                Instruction::CALL(Call { ri }) => {
                    row[COL_S_CALL] = F::from_canonical_u64(1);
                    handle_op2!(ri, row);
                }
                Instruction::RET(..) => {
                    row[COL_S_RET] = F::from_canonical_u64(1);
                }
                Instruction::MLOAD(Mload { ri, rj }) => {
                    row[COL_S_MLOAD] = F::from_canonical_u64(1);
                    row[COL_OP_0] = F::from_canonical_u8(ri);
                    handle_op2!(rj, row);
                }
                Instruction::MSTORE(Mstore { a, ri }) => {
                    row[COL_S_MSTORE] = F::from_canonical_u64(1);
                    row[COL_OP_0] = F::from_canonical_u8(ri);
                    handle_op2!(a, row);
                }
                _ => panic!("Unsupported instruction!"),
            }
            row
        })
        .collect();

    trace
}

pub(crate) fn generate_mem_trace<F: RichField>(
    memory: &Vec<MemoryTraceCell>,
) -> Vec<[F; NUM_MEM_COLS]> {
    let mut trace: Vec<[F; NUM_MEM_COLS]> = memory
        .iter()
        .map(|m| {
            let mut mt: [F; NUM_MEM_COLS] = [F::default(); NUM_MEM_COLS];
            mt[COL_MEM_ADDR] = F::from_canonical_u64(m.addr);
            mt[COL_MEM_CLK] = F::from_canonical_u32(m.clk);
            mt[COL_MEM_PC] = F::from_canonical_u64(m.pc);
            mt[COL_MEM_RW] = F::from_canonical_u8(m.op as u8);
            mt[COL_MEM_VAL] = F::from_canonical_u64(m.value.0);
            mt
        })
        .collect();

    // We sort memory trace by address first, then clk, pc.
    trace.sort_unstable_by_key(|t| {
        (
            t[COL_MEM_ADDR].to_canonical_u64(),
            t[COL_MEM_CLK].to_canonical_u64(),
            t[COL_MEM_PC].to_canonical_u64(),
            t[COL_MEM_RW].to_canonical_u64(),
            t[COL_MEM_VAL].to_canonical_u64(),
        )
    });
    trace
}
