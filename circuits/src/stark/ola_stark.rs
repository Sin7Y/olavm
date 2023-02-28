use super::config::StarkConfig;
use super::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use super::stark::Stark;
use crate::builtins::bitwise::bitwise_stark::{self, BitwiseStark};
use crate::builtins::cmp::cmp_stark::{self, CmpStark};
use crate::builtins::rangecheck::rangecheck_stark::{self, RangeCheckStark};
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::memory::memory_stark::{
    ctl_data as mem_ctl_data, ctl_data_mem_rc, ctl_filter as mem_ctl_filter, ctl_filter_mem_rc,
    MemoryStark,
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
        ]
    }

    pub(crate) fn permutation_batch_sizes(&self) -> [usize; NUM_TABLES] {
        [
            self.cpu_stark.permutation_batch_size(),
            self.memory_stark.permutation_batch_size(),
            self.bitwise_stark.permutation_batch_size(),
            self.cmp_stark.permutation_batch_size(),
            self.rangecheck_stark.permutation_batch_size(),
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
    BitwiseFixed = 5,
    RangecheckFixed = 6,
    // program table
    Program = 7,
}

pub(crate) const NUM_TABLES: usize = 5;

pub(crate) fn all_cross_table_lookups<F: Field>() -> Vec<CrossTableLookup<F>> {
    vec![
        ctl_cpu_memory(),
        ctl_memory_rc(),
        ctl_bitwise_cpu(),
        ctl_cmp_cpu(),
        ctl_cmp_rangecheck(),
        ctl_rangecheck_cpu(),
    ]
}

fn ctl_cpu_memory<F: Field>() -> CrossTableLookup<F> {
    let cpu_mem_mstore = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_mstore(),
        Some(cpu_stark::ctl_filter_cpu_mem_mstore()),
    );
    let cpu_mem_mload = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_mload(),
        Some(cpu_stark::ctl_filter_cpu_mem_mload()),
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
    let all_cpu_lookers = vec![
        cpu_mem_mstore,
        cpu_mem_mload,
        cpu_mem_call_ret_pc,
        cpu_mem_call_ret_fp,
    ];
    let memory_looked =
        TableWithColumns::new(Table::Memory, mem_ctl_data(), Some(mem_ctl_filter()));
    CrossTableLookup::new(all_cpu_lookers, memory_looked, None)
}

fn ctl_memory_rc<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Memory,
            ctl_data_mem_rc(),
            Some(ctl_filter_mem_rc()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_memory(),
            Some(rangecheck_stark::ctl_filter_memory()),
        ),
        None,
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
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use crate::generation::builtin::{
        generate_bitwise_trace, generate_cmp_trace, generate_rc_trace,
    };
    use crate::generation::cpu::generate_cpu_trace;
    use crate::generation::generate_traces;
    use crate::generation::memory::generate_memory_trace;
    use crate::stark::config::StarkConfig;
    use crate::stark::ola_stark::OlaStark;
    use crate::stark::proof::PublicValues;
    use crate::stark::prover::prove_with_traces;
    use crate::stark::serialization::Buffer;
    use crate::stark::stark::Stark;
    use crate::stark::util::trace_rows_to_poly_values;
    use crate::stark::verifier::verify_proof;
    use anyhow::Result;
    use core::program::Program;
    use executor::Process;
    use log::debug;
    use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use std::mem;
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
    fn fibo_recursive_decode() -> Result<()> {
        let file = File::open("../assembler/testdata/fib_recursive.bin").unwrap();
        let mut instructions = BufReader::new(file).lines();

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions {
            program.instructions.push(inst.unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program);

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        println!("{}", mem::size_of_val(&proof));
        let ola_stark = OlaStark::default();
        verify_proof(ola_stark, proof, &config)
    }

    #[test]
    fn memory_test() -> Result<()> {
        let file = File::open("../assembler/testdata/memory.bin").unwrap();
        let mut instructions = BufReader::new(file).lines();
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program);

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        let ola_stark = OlaStark::default();
        verify_proof(ola_stark, proof, &config)
    }

    #[test]
    fn call_test() -> Result<()> {
        let file = File::open("../assembler/testdata/call.bin").unwrap();
        let mut instructions = BufReader::new(file).lines();

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program);

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        let ola_stark = OlaStark::default();
        verify_proof(ola_stark, proof, &config)
    }

    #[test]
    fn range_check_test() -> Result<()> {
        let file = File::open("../assembler/testdata/range_check.bin").unwrap();
        let mut instructions = BufReader::new(file).lines();

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions {
            program.instructions.push(inst.unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program);

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        let ola_stark = OlaStark::default();
        verify_proof(ola_stark, proof, &config)
    }

    #[test]
    fn bitwise_test() -> Result<()> {
        let file = File::open("../assembler/testdata/bitwise.bin").unwrap();
        let mut instructions = BufReader::new(file).lines();

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program);

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        let ola_stark = OlaStark::default();
        verify_proof(ola_stark, proof, &config)
    }

    #[test]
    fn comparison_test() -> Result<()> {
        // main:
        // .LBL0_0:
        //   add r8 r8 4
        //   mstore [r8,-2] r8
        //   mov r1 1
        //   call le
        //   add r8 r8 -4
        //   end
        // le:
        // .LBL1_0:
        //   mov r0 r1
        //   mov r7 1
        //   gte r0 r7 r0
        //   cjmp r0 .LBL1_1
        //   jmp .LBL1_2
        // .LBL1_1:
        //   mov r0 2
        //   ret
        // .LBL1_2:
        //   mov r0 3
        //   ret

        let file = File::open("../assembler/testdata/comparison.bin").unwrap();
        let mut instructions = BufReader::new(file).lines();

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program);

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        let ola_stark = OlaStark::default();
        verify_proof(ola_stark, proof, &config)
    }

    #[test]
    fn fibo_use_loop_memory_decode() -> Result<()> {
        let file = File::open("../assembler/testdata/fib_loop.bin").unwrap();
        let mut instructions = BufReader::new(file).lines();

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program);

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(&program, &mut ola_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        let ola_stark = OlaStark::default();
        verify_proof(ola_stark, proof, &config)
    }
}
