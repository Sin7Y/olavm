use super::config::StarkConfig;
use super::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use super::stark::Stark;
use crate::builtins::bitwise::bitwise_stark::{self, BitwiseStark};
use crate::builtins::cmp::cmp_stark::{self, CmpStark};
use crate::builtins::poseidon::poseidon_stark::{self, PoseidonStark};
use crate::builtins::rangecheck::rangecheck_stark::{self, RangeCheckStark};
use crate::builtins::storage::storage_hash::{self, StorageHashStark};
use crate::builtins::storage::storage_stark::{self, StorageStark};
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::memory::memory_stark::{
    ctl_data as mem_ctl_data, ctl_data_mem_rc, ctl_data_mem_rc_diff_cond,
    ctl_filter as mem_ctl_filter, ctl_filter_mem_rc, ctl_filter_mem_rc_diff_cond, MemoryStark,
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
    pub storage_stark: StorageStark<F, D>,
    pub storage_hash_stark: StorageHashStark<F, D>,

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
            storage_stark: StorageStark::default(),
            storage_hash_stark: StorageHashStark::default(),
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
            self.storage_stark.num_permutation_batches(config),
            self.storage_hash_stark.num_permutation_batches(config),
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
            self.storage_stark.permutation_batch_size(),
            self.storage_hash_stark.permutation_batch_size(),
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
    Storage = 6,
    StorageHash = 7,
    // program table
    // Program = 8,
}

pub(crate) const NUM_TABLES: usize = 8;

pub(crate) fn all_cross_table_lookups<F: Field>() -> Vec<CrossTableLookup<F>> {
    vec![
        ctl_cpu_memory(),
        ctl_memory_rc(),
        ctl_bitwise_cpu(),
        ctl_cmp_cpu(),
        ctl_cmp_rangecheck(),
        ctl_rangecheck_cpu(),
        ctl_cpu_poseidon(),
        ctl_cpu_poseidon_tree_key(),
        ctl_cpu_storage_sstore(),
        ctl_cpu_storage_sload(),
        ctl_storage_poseidon_tree_key(),
        ctl_storage_hash(),
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
    let all_cpu_lookers = vec![cpu_mem_store_load, cpu_mem_call_ret_pc, cpu_mem_call_ret_fp];
    let memory_looked =
        TableWithColumns::new(Table::Memory, mem_ctl_data(), Some(mem_ctl_filter()));
    CrossTableLookup::new(all_cpu_lookers, memory_looked, None)
}

fn ctl_memory_rc<F: Field>() -> CrossTableLookup<F> {
    let mem_rc_value =
        TableWithColumns::new(Table::Memory, ctl_data_mem_rc(), Some(ctl_filter_mem_rc()));
    let mem_diff_cond = TableWithColumns::new(
        Table::Memory,
        ctl_data_mem_rc_diff_cond(),
        Some(ctl_filter_mem_rc_diff_cond()),
    );
    let all_mem_lookers = vec![mem_rc_value, mem_diff_cond];
    let rc_looked = TableWithColumns::new(
        Table::RangeCheck,
        rangecheck_stark::ctl_data_memory(),
        Some(rangecheck_stark::ctl_filter_memory()),
    );
    CrossTableLookup::new(all_mem_lookers, rc_looked, None)
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
                Some(cpu_stark::ctl_filter_with_bitwise_and()),
            ),
            TableWithColumns::new(
                Table::Cpu,
                cpu_stark::ctl_data_with_bitwise(),
                Some(cpu_stark::ctl_filter_with_bitwise_or()),
            ),
            TableWithColumns::new(
                Table::Cpu,
                cpu_stark::ctl_data_with_bitwise(),
                Some(cpu_stark::ctl_filter_with_bitwise_xor()),
            ),
        ],
        TableWithColumns::new(
            Table::Bitwise,
            bitwise_stark::ctl_data_with_cpu(),
            Some(bitwise_stark::ctl_filter_with_cpu()),
        ),
        None,
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
        None,
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
        None,
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
        None,
    )
}

fn ctl_cpu_poseidon<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_poseidon(),
            Some(cpu_stark::ctl_filter_with_poseidon()),
        )],
        TableWithColumns::new(
            Table::Poseidon,
            poseidon_stark::ctl_data_with_cpu(),
            Some(poseidon_stark::ctl_filter_with_cpu()),
        ),
        None,
    )
}

fn ctl_cpu_poseidon_tree_key<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_poseidon_tree_key(),
            Some(cpu_stark::ctl_filter_with_poseidon_tree_key()),
        )],
        TableWithColumns::new(
            Table::Poseidon,
            poseidon_stark::ctl_data_with_cpu_tree_key(),
            Some(poseidon_stark::ctl_filter_with_cpu_tree_key()),
        ),
        None,
    )
}

fn ctl_cpu_storage_sstore<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_sstore(),
            Some(cpu_stark::ctl_filter_with_sstore()),
        )],
        TableWithColumns::new(
            Table::Storage,
            storage_stark::ctl_data_with_cpu(),
            Some(storage_stark::ctl_filter_with_cpu_sstore()),
        ),
        None,
    )
}

fn ctl_cpu_storage_sload<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_sload(),
            Some(cpu_stark::ctl_filter_with_sload()),
        )],
        TableWithColumns::new(
            Table::Storage,
            storage_stark::ctl_data_with_cpu(),
            Some(storage_stark::ctl_filter_with_cpu_sload()),
        ),
        None,
    )
}

fn ctl_storage_poseidon_tree_key<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Storage,
            storage_stark::ctl_data_with_poseidon(),
            Some(storage_stark::ctl_filter_with_hash()),
        )],
        TableWithColumns::new(
            Table::Poseidon,
            poseidon_stark::ctl_data_with_storage_tree_key(),
            Some(poseidon_stark::ctl_filter_with_cpu_tree_key()),
        ),
        None,
    )
}

fn ctl_storage_hash<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Storage,
            storage_stark::ctl_data_with_hash(),
            Some(storage_stark::ctl_filter_with_hash()),
        )],
        TableWithColumns::new(
            Table::StorageHash,
            storage_hash::ctl_data_with_storage(),
            Some(storage_hash::ctl_filter_with_storage()),
        ),
        None,
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
mod tests {
    use crate::generation::generate_traces;
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
    use executor::Process;
    use log::{debug, LevelFilter};
    use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::mem;
    use std::path::PathBuf;
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
        test_by_asm_json("fibo_loop.json".to_string())
    }

    #[test]
    fn fibo_recursive_decode() {
        test_by_asm_json("fibo_recursive.json".to_string())
    }

    #[test]
    fn memory_test() {
        test_by_asm_json("memory.json".to_string())
    }

    #[test]
    fn call_test() {
        test_by_asm_json("call.json".to_string())
    }

    #[test]
    fn range_check_test() {
        test_by_asm_json("range_check.json".to_string())
    }

    #[test]
    fn bitwise_test() {
        test_by_asm_json("bitwise.json".to_string())
    }

    #[test]
    fn comparison_test() {
        test_by_asm_json("comparison.json".to_string())
    }

    #[test]
    fn test_ola_prophet_hand_write() {
        test_by_asm_json("hand_write_prophet.json".to_string());
    }

    #[test]
    fn test_ola_prophet_sqrt() {
        test_by_asm_json("prophet_sqrt.json".to_string());
    }

    #[test]
    fn test_ola_sqrt() {
        test_by_asm_json("sqrt.json".to_string());
    }

    #[test]
    fn test_ola_poseidon() {
        test_by_asm_json("poseidon.json".to_string());
    }

    #[test]
    fn test_ola_storage() {
        test_by_asm_json("storage.json".to_string());
    }

    #[test]
    fn test_ola_malloc() {
        test_by_asm_json("malloc.json".to_string());
    }

    #[test]
    fn test_ola_vote() {
        test_by_asm_json("vote.json".to_string());
    }

    #[allow(unused)]
    pub fn test_by_asm_json(file_name: String) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        let program = encode_asm_from_json_file(program_path).unwrap();
        let instructions = program.bytecode.split("\n");
        let mut prophets = HashMap::new();
        for item in program.prophets {
            prophets.insert(item.host as u64, item);
        }

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };

        for inst in instructions {
            program.instructions.push(inst.to_string());
        }

        let mut process = Process::new();
        process.ctx_registers_stack.push(Address::default());
        let _ = process.execute(
            &mut program,
            &mut Some(prophets),
            &mut AccountTree::new_test(),
        );

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
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
