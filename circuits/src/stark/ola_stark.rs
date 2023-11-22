use std::iter;

use super::config::StarkConfig;
use super::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use super::stark::Stark;
use crate::builtins::bitwise::bitwise_stark::{self, BitwiseStark};
use crate::builtins::cmp::cmp_stark::{self, CmpStark};
use crate::builtins::poseidon::poseidon_chunk_stark::{self, PoseidonChunkStark};
use crate::builtins::poseidon::poseidon_stark::{self, PoseidonStark};
use crate::builtins::rangecheck::rangecheck_stark::{self, RangeCheckStark};
use crate::builtins::sccall::sccall_stark::{self, SCCallStark};
use crate::builtins::storage::storage_access_stark::{self, StorageAccessStark};
use crate::builtins::tape::tape_stark::{self, TapeStark};
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::memory::memory_stark::{
    self, ctl_data as mem_ctl_data, ctl_data_mem_rc_diff_cond, ctl_data_mem_sort_rc,
    ctl_filter as mem_ctl_filter, ctl_filter_mem_rc_diff_cond, ctl_filter_mem_sort_rc, MemoryStark,
};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

#[derive(Clone)]
pub struct OlaStark<F: RichField + Extendable<D>, const D: usize> {
    pub cpu_stark: CpuStark<F, D>,
    pub memory_stark: MemoryStark<F, D>,
    // builtins
    pub bitwise_stark: BitwiseStark<F, D>,
    pub cmp_stark: CmpStark<F, D>,
    pub rangecheck_stark: RangeCheckStark<F, D>,
    pub poseidon_stark: PoseidonStark<F, D>,
    pub poseidon_chunk_stark: PoseidonChunkStark<F, D>,
    pub storage_access_stark: StorageAccessStark<F, D>,
    pub tape_stark: TapeStark<F, D>,
    pub sccall_stark: SCCallStark<F, D>,

    pub cross_table_lookups: Vec<CrossTableLookup<F>>,
}

impl<F: RichField + Extendable<D>, const D: usize> Default for OlaStark<F, D> {
    fn default() -> Self {
        Self {
            cpu_stark: CpuStark::default(),
            memory_stark: MemoryStark::default(),
            bitwise_stark: BitwiseStark::default(),
            cmp_stark: CmpStark::default(),
            rangecheck_stark: RangeCheckStark::default(),
            poseidon_stark: PoseidonStark::default(),
            poseidon_chunk_stark: PoseidonChunkStark::default(),
            storage_access_stark: StorageAccessStark::default(),
            tape_stark: TapeStark::default(),
            sccall_stark: SCCallStark::default(),
            cross_table_lookups: all_cross_table_lookups(),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> OlaStark<F, D> {
    pub(crate) fn nums_permutation_zs(&self, config: &StarkConfig) -> [usize; NUM_TABLES] {
        [
            self.cpu_stark.num_permutation_batches(config),
            self.memory_stark.num_permutation_batches(config),
            self.bitwise_stark.num_permutation_batches(config),
            self.cmp_stark.num_permutation_batches(config),
            self.rangecheck_stark.num_permutation_batches(config),
            self.poseidon_stark.num_permutation_batches(config),
            self.poseidon_chunk_stark.num_permutation_batches(config),
            self.storage_access_stark.num_permutation_batches(config),
            self.tape_stark.num_permutation_batches(config),
            self.sccall_stark.num_permutation_batches(config),
        ]
    }

    pub(crate) fn permutation_batch_sizes(&self) -> [usize; NUM_TABLES] {
        [
            self.cpu_stark.permutation_batch_size(),
            self.memory_stark.permutation_batch_size(),
            self.bitwise_stark.permutation_batch_size(),
            self.cmp_stark.permutation_batch_size(),
            self.rangecheck_stark.permutation_batch_size(),
            self.poseidon_stark.permutation_batch_size(),
            self.poseidon_chunk_stark.permutation_batch_size(),
            self.storage_access_stark.permutation_batch_size(),
            self.tape_stark.permutation_batch_size(),
            self.sccall_stark.permutation_batch_size(),
        ]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Table {
    Cpu = 0,
    Memory = 1,
    // builtins
    Bitwise = 2,
    Cmp = 3,
    RangeCheck = 4,
    Poseidon = 5,
    PoseidonChunk = 6,
    StorageAccess = 7,
    Tape = 8,
    SCCall = 9,
    // program table
    // Program = 8,
}

pub(crate) const NUM_TABLES: usize = 10;

pub(crate) fn all_cross_table_lookups<F: Field>() -> Vec<CrossTableLookup<F>> {
    vec![
        ctl_cpu_memory(),
        ctl_memory_rc_sort(),
        ctl_memory_rc_region(),
        ctl_bitwise_cpu(),
        ctl_cmp_cpu(),
        ctl_cmp_rangecheck(),
        ctl_rangecheck_cpu(),
        ctl_cpu_poseidon_chunk(),
        ctl_poseidon_chunk_mem(),
        ctl_poseidon_chunk_poseidon(),
        ctl_cpu_poseidon_tree_key(),
        ctl_cpu_storage_access(),
        ctl_storage_access_poseidon(),
        ctl_cpu_tape(),
        ctl_cpu_sccall(),
        ctl_cpu_sccall_end(),
    ]
}

fn ctl_cpu_memory<F: Field>() -> CrossTableLookup<F> {
    let cpu_mem_store_load = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_store_load(),
        Some(cpu_stark::ctl_filter_cpu_mem_store_load()),
    );
    let cpu_mem_call_ret_pc = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_call_ret_pc(),
        Some(cpu_stark::ctl_filter_cpu_mem_call_ret()),
    );
    let cpu_mem_call_ret_fp = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_call_ret_fp(),
        Some(cpu_stark::ctl_filter_cpu_mem_call_ret()),
    );
    let cpu_mem_tload_tstore = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_tload_tstore(),
        Some(cpu_stark::ctl_filter_cpu_mem_tload_tstore()),
    );
    let cpu_sccall_mems = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_mem_sccall(i),
            Some(cpu_stark::ctl_filter_cpu_mem_sccall()),
        )
    });
    let cpu_storage_addr = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_mem_for_storage_addr(i),
            Some(cpu_stark::ctl_filter_cpu_storage_access()),
        )
    });
    let cpu_storage_value = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_mem_for_storage_value(i),
            Some(cpu_stark::ctl_filter_cpu_storage_access()),
        )
    });

    let mut all_cpu_lookers = vec![
        cpu_mem_store_load,
        cpu_mem_call_ret_pc,
        cpu_mem_call_ret_fp,
        cpu_mem_tload_tstore,
    ];
    all_cpu_lookers.extend(cpu_sccall_mems);
    all_cpu_lookers.extend(cpu_storage_addr);
    all_cpu_lookers.extend(cpu_storage_value);
    let memory_looked =
        TableWithColumns::new(Table::Memory, mem_ctl_data(), Some(mem_ctl_filter()));
    CrossTableLookup::new(all_cpu_lookers, memory_looked)
}

fn ctl_memory_rc_sort<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Memory,
            ctl_data_mem_sort_rc(),
            Some(ctl_filter_mem_sort_rc()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_memory(),
            Some(rangecheck_stark::ctl_filter_memory_sort()),
        ),
    )
}

fn ctl_memory_rc_region<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Memory,
            ctl_data_mem_rc_diff_cond(),
            Some(ctl_filter_mem_rc_diff_cond()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_memory(),
            Some(rangecheck_stark::ctl_filter_memory_region()),
        ),
    )
}

// add bitwise rangecheck instance
// Cpu table
// +-----+-----+-----+---------+--------+---------+-----+-----+-----+-----+----
// | clk | ins | ... | sel_and | sel_or | sel_xor | ... | op0 | op1 | dst | ...
// +-----+-----+-----+---------+--------+---------+-----+-----+----+----+----
//
// Bitwise table
// +-----+-----+-----+-----+------------+------------+-----------+------------+---
// | tag | op0 | op1 | res | op0_limb_0 | op0_limb_1 |res_limb_2 | op0_limb_3
// |...
// +-----+-----+-----+-----+------------+------------+-----------+------------+---
//
// Filter bitwise from CPU Table
// 1. (sel_add + sel_or + sel_xor) * (op0, op1, dst) = looking_table
// Filter bitwise from Bitwsie Table
// 1. (op0, op1, res) = looked_table

// Cross_Lookup_Table(looking_table, looked_table)
fn ctl_bitwise_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![
            TableWithColumns::new(
                Table::Cpu,
                cpu_stark::ctl_data_with_bitwise(),
                Some(cpu_stark::ctl_filter_with_bitwise()),
            ),
        ],
        TableWithColumns::new(
            Table::Bitwise,
            bitwise_stark::ctl_data_with_cpu(),
            Some(bitwise_stark::ctl_filter_with_cpu()),
        ),
    )
}

// add CMP cross lookup instance
fn ctl_cmp_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_cmp(),
            Some(cpu_stark::ctl_filter_with_cmp()),
        )],
        TableWithColumns::new(
            Table::Cmp,
            cmp_stark::ctl_data_with_cpu(),
            Some(cmp_stark::ctl_filter_with_cpu()),
        ),
    )
}

fn ctl_cmp_rangecheck<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_with_cmp(),
            Some(rangecheck_stark::ctl_filter_with_cmp()),
        )],
        TableWithColumns::new(
            Table::Cmp,
            cmp_stark::ctl_data_with_rangecheck(),
            Some(cmp_stark::ctl_filter_with_rangecheck()),
        ),
    )
}

// add Rangecheck cross lookup instance
fn ctl_rangecheck_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_rangecheck(),
            Some(cpu_stark::ctl_filter_with_rangecheck()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_with_cpu(),
            Some(rangecheck_stark::ctl_filter_with_cpu()),
        ),
    )
}

fn ctl_cpu_poseidon_chunk<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_poseidon_chunk(),
            Some(cpu_stark::ctl_filter_with_poseidon_chunk()),
        )],
        TableWithColumns::new(
            Table::PoseidonChunk,
            poseidon_chunk_stark::ctl_data_with_cpu(),
            Some(poseidon_chunk_stark::ctl_filter_with_cpu()),
        ),
    )
}

fn ctl_poseidon_chunk_mem<F: Field>() -> CrossTableLookup<F> {
    let looker_src = (0..8).map(|i: usize| {
        TableWithColumns::new(
            Table::PoseidonChunk,
            poseidon_chunk_stark::ctl_data_with_mem_src(i),
            Some(poseidon_chunk_stark::ctl_filter_with_mem_src(i)),
        )
    });
    let looker_dst = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::PoseidonChunk,
            poseidon_chunk_stark::ctl_data_with_mem_dst(i),
            Some(poseidon_chunk_stark::ctl_filter_with_mem_dst()),
        )
    });
    let all_lookers = looker_src.into_iter().chain(looker_dst).collect();
    let mem_looked = TableWithColumns::new(
        Table::Memory,
        memory_stark::ctl_data_with_poseidon_chunk(),
        Some(memory_stark::ctl_filter_with_poseidon_chunk()),
    );
    CrossTableLookup::new(all_lookers, mem_looked)
}

fn ctl_poseidon_chunk_poseidon<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::PoseidonChunk,
            poseidon_chunk_stark::ctl_data_with_poseidon(),
            Some(poseidon_chunk_stark::ctl_filter_with_poseidon()),
        )],
        TableWithColumns::new(
            Table::Poseidon,
            poseidon_stark::ctl_data_with_poseidon_chunk(),
            Some(poseidon_stark::ctl_filter_with_poseidon_chunk()),
        ),
    )
}

fn ctl_cpu_storage_access<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_storage_access(),
            Some(cpu_stark::ctl_filter_cpu_storage_access()),
        )],
        TableWithColumns::new(
            Table::StorageAccess,
            storage_access_stark::ctl_data_with_cpu(),
            Some(storage_access_stark::ctl_filter_with_cpu_sstore()),
        ),
    )
}

fn ctl_storage_access_poseidon<F: Field>() -> CrossTableLookup<F> {
    let looker_bit0 = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit0(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit0()),
    );
    let looker_bit0_pre = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit0_pre(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit0()),
    );
    let looker_bit1 = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit1(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit1()),
    );
    let looker_bit1_pre = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit1_pre(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit1()),
    );
    let all_lookers = vec![looker_bit0, looker_bit0_pre, looker_bit1, looker_bit1_pre];
    let looked = TableWithColumns::new(
        Table::Poseidon,
        poseidon_stark::ctl_data_with_storage(),
        Some(poseidon_stark::ctl_filter_with_storage()),
    );
    CrossTableLookup::new(all_lookers, looked)
}

fn ctl_cpu_poseidon_tree_key<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_poseidon_treekey(),
            Some(cpu_stark::ctl_filter_poseidon_treekey()),
        )],
        TableWithColumns::new(
            Table::Poseidon,
            poseidon_stark::ctl_data_cpu_tree_key(),
            Some(poseidon_stark::ctl_filter_cpu_tree_key()),
        ),
    )
}

fn ctl_cpu_tape<F: Field>() -> CrossTableLookup<F> {
    let cpu_tape_tload_tstore = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_tape_load_store(),
        Some(cpu_stark::ctl_filter_cpu_tape_load_store()),
    );
    let cpu_tape_sccall_caller = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_tape_sccall_caller(i),
            Some(cpu_stark::ctl_filter_cpu_is_sccall_ext()),
        )
    });
    let cpu_tape_sccall_callee_code = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_tape_sccall_callee_code(i),
            Some(cpu_stark::ctl_filter_cpu_is_sccall_ext()),
        )
    });
    let cpu_tape_sccall_callee_storage = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_tape_sccall_callee_storage(i),
            Some(cpu_stark::ctl_filter_cpu_is_sccall_ext()),
        )
    });

    let all_lookers = iter::once(cpu_tape_tload_tstore)
        .chain(cpu_tape_sccall_caller)
        .chain(cpu_tape_sccall_callee_code)
        .chain(cpu_tape_sccall_callee_storage)
        .collect();

    let tape_looked = TableWithColumns::new(
        Table::Tape,
        tape_stark::ctl_data_tape(),
        Some(tape_stark::ctl_filter_tape()),
    );
    CrossTableLookup::new(all_lookers, tape_looked)
}

fn ctl_cpu_sccall<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_sccall(),
            Some(cpu_stark::ctl_filter_cpu_sccall()),
        )],
        TableWithColumns::new(
            Table::SCCall,
            sccall_stark::ctl_data_sccall(),
            Some(sccall_stark::ctl_filter_sccall()),
        ),
    )
}

fn ctl_cpu_sccall_end<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_sccall_end(),
            Some(cpu_stark::ctl_filter_cpu_sccall_end()),
        )],
        TableWithColumns::new(
            Table::SCCall,
            sccall_stark::ctl_data_sccall_end(),
            Some(sccall_stark::ctl_filter_sccall_end()),
        ),
    )
}

// Cross_Lookup_Table(looking_table, looked_table)
/*fn ctl_bitwise_bitwise_fixed_table<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::BitwiseFixed,
            bitwise_fixed_stark::ctl_data_with_bitwise(),
            Some(bitwise_fixed_stark::ctl_filter_with_bitwise()),
        )],
        TableWithColumns::new(
            Table::Bitwise,
            bitwise_stark::ctl_data_with_bitwise_fixed(),
            Some(bitwise_stark::ctl_filter_with_bitwise_fixed()),
        ),
        None,
    )
}*/

/*fn ctl_rangecheck_rangecheck_fixed<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::RangecheckFixed,
            rangecheck_fixed_stark::ctl_data_with_rangecheck(),
            Some(rangecheck_fixed_stark::ctl_filter_with_rangecheck()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_with_rangecheck_fixed(),
            Some(rangecheck_stark::ctl_filter_with_rangecheck_fixed()),
        ),
        None,
    )
}*/

// check the correct program with lookup

// Program table
// +-----+--------------+-------+----------+
// | PC  |      INS     |  IMM  | COMPRESS |
// +-----+--------------+-------+----------+
// +-----+--------------+-------+----------+
// |  1  |  0x********  |  U32  |   Field  |
// +-----+--------------+-------+----------+
// +-----+--------------+-------+----------+
// |  2  |  0x********  |  U32  |   Field  |
// +-----+--------------+-------+----------++
// +-----+--------------+-------+----------+
// |  3  |  0x********  |  U32  |   Field  |
// +-----+--------------+-------+----------+

// CPU table
// +-----+-----+--------------+-------+----------+
// | ... | PC  |      INS     |  IMM  | COMPRESS |
// +-----+-----+--------------+-------+----------+
// +-----+-----+--------------+-------+----------+
// | ... |  1  |  0x********  |  U32  |   Field  |
// +-----+-----+--------------+-------+----------+
// +-----+-----+--------------+-------+----------+
// | ... |  2  |  0x********  |  U32  |   Field  |
// +-----+-----+--------------+-------+----------++
// +-----+-----+--------------+-------+----------+
// | ... |  3  |  0x********  |  U32  |   Field  |
// +-----+-----+--------------+-------+----------+

// Note that COMPRESS will be computed by vector lookup argument protocol
/*fn ctl_correct_program_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_program(),
            Some(cpu_stark::ctl_filter_with_program()),
        )],
        TableWithColumns::new(
            Table::Program,
            program_stark::ctl_data_with_cpu(),
            Some(program_stark::ctl_filter_with_cpu()),
        ),
        None,
    )
}*/

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use crate::generation::{generate_traces, GenerationInputs};
    use crate::stark::config::StarkConfig;
    use crate::stark::ola_stark::OlaStark;
    use crate::stark::proof::PublicValues;
    use crate::stark::prover::prove_with_traces;
    use crate::stark::serialization::Buffer;
    use crate::stark::stark::Stark;
    use crate::stark::util::trace_rows_to_poly_values;
    use crate::stark::verifier::verify_proof;
    use anyhow::Result;
    use assembler::encoder::encode_asm_from_json_file;
    use core::merkle_tree::tree::AccountTree;
    use core::program::binary_program::BinaryProgram;
    use core::program::Program;
    use core::types::account::Address;
    use core::types::{Field, GoldilocksField};
    use core::vm::transaction::init_tx_context;
    use executor::load_tx::init_tape;
    use executor::Process;
    use itertools::Itertools;
    use log::{debug, LevelFilter};
    use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::mem;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    #[allow(dead_code)]
    const D: usize = 2;
    #[allow(dead_code)]
    type C = Blake3GoldilocksConfig;
    #[allow(dead_code)]
    type F = <C as GenericConfig<D>>::F;
    #[allow(dead_code)]
    type S = dyn Stark<F, D>;

    #[test]
    fn fibo_loop_test() {
        let calldata = [10u64, 1u64, 2u64, 4185064725u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        test_by_asm_json("fib_asm.json".to_string(), Some(calldata), None)
    }

    #[test]
    fn fibo_recursive_decode() {
        test_by_asm_json("fibo_recursive.json".to_string(), None, None)
    }

    #[test]
    fn memory_test() {
        test_by_asm_json("memory.json".to_string(), None, None)
    }

    #[test]
    fn call_test() {
        test_by_asm_json("call.json".to_string(), None, None)
    }

    // #[test]
    // fn range_check_test() {
    //     test_by_asm_json("range_check.json".to_string(), None)
    // }

    // #[test]
    // fn bitwise_test() {
    //     test_by_asm_json("bitwise.json".to_string(), None)
    // }

    #[test]
    fn comparison_test() {
        test_by_asm_json("comparison.json".to_string(), None, None)
    }

    // #[test]
    // fn test_ola_prophet_hand_write() {
    //     test_by_asm_json("hand_write_prophet.json".to_string(), None);
    // }

    #[test]
    fn test_ola_prophet_sqrt() {
        let calldata = [144u64, 10u64, 2u64, 3509365327u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        test_by_asm_json("sqrt_prophet_asm.json".to_string(), Some(calldata), None);
    }

    // #[test]
    // fn test_ola_sqrt() {
    //     test_by_asm_json("sqrt.json".to_string(), None);
    // }

    #[test]
    fn test_ola_poseidon() {
        let call_data = vec![
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(1239976900),
        ];
        test_by_asm_json("poseidon_hash.json".to_string(), Some(call_data), None);
    }

    #[test]
    fn test_ola_storage() {
        let call_data = vec![
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(2364819430),
        ];
        test_by_asm_json("storage_u32.json".to_string(), Some(call_data), None);
    }

    #[test]
    fn test_ola_malloc() {
        test_by_asm_json("malloc.json".to_string(), None, None);
    }

    #[test]
    fn test_ola_vote() {
        let db_name = "vote_test".to_string();

        let init_calldata = [3u64, 1u64, 2u64, 3u64, 4u64, 2817135588u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        let vote_calldata = [2u64, 1u64, 2791810083u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        let winning_proposal_calldata = [0u64, 3186728800u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        let winning_name_calldata = [0u64, 363199787u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        test_by_asm_json(
            "vote.json".to_string(),
            Some(winning_proposal_calldata),
            Some(db_name),
        );
    }

    #[test]
    fn test_ola_mem_gep() {
        test_by_asm_json("mem_gep.json".to_string(), None, None);
    }

    #[test]
    fn test_ola_mem_gep_vector() {
        test_by_asm_json("mem_gep_vector.json".to_string(), None, None);
    }

    // #[test]
    // fn test_ola_string_assert() {
    //     test_by_asm_json("string_assert.json".to_string(), None);
    // }

    #[allow(unused)]
    pub fn test_by_asm_json(
        file_name: String,
        call_data: Option<Vec<GoldilocksField>>,
        db_name: Option<String>,
    ) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        let mut db = match db_name {
            Some(name) => {
                let mut db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                db_path.push("../db_test/");
                db_path.push(name);
                AccountTree::new_db_test(db_path.display().to_string())
            }
            _ => AccountTree::new_test(),
        };

        let program = encode_asm_from_json_file(program_path).unwrap();
        let instructions = program.bytecode.split("\n");
        let mut prophets = HashMap::new();
        for item in program.prophets {
            prophets.insert(item.host as u64, item);
        }

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
            debug_info: program.debug_info,
        };

        for inst in instructions {
            program.instructions.push(inst.to_string());
        }

        let mut process = Process::new();
        process.addr_storage = Address::default();
        if let Some(calldata) = call_data {
            process.tp = GoldilocksField::ZERO;

            init_tape(
                &mut process,
                calldata,
                Address::default(),
                Address::default(),
                Address::default(),
                &init_tx_context(),
            );
        }
        let res = process.execute(&mut program, &mut Some(prophets), &mut db);
        match res {
            Ok(_) => {}
            Err(e) => {
                println!("execute err:{:?}", e);
                return;
            }
        }

        let inputs = GenerationInputs::default();

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(program, &mut ola_stark, inputs);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        );

        if let Ok(proof) = proof {
            let ola_stark = OlaStark::default();
            let verify_res = verify_proof(ola_stark, proof, &config);
            println!("verify result:{:?}", verify_res);
        } else {
            println!("proof err:{:?}", proof);
        }
    }
}
