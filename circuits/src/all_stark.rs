use crate::builtins::bitwise::bitwise_stark::{self, BitwiseStark};
use crate::builtins::cmp::cmp_stark::{self, CmpStark};
use crate::builtins::rangecheck::rangecheck_stark::{self, RangeCheckStark};
use crate::config::StarkConfig;
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use crate::memory::memory_stark::{
    ctl_data as mem_ctl_data, ctl_data_mem_rc, ctl_filter as mem_ctl_filter, ctl_filter_mem_rc,
    MemoryStark,
};
use crate::stark::Stark;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

#[derive(Clone)]
pub struct AllStark<F: RichField + Extendable<D>, const D: usize> {
    pub cpu_stark: CpuStark<F, D>,
    pub memory_stark: MemoryStark<F, D>,
    // builtins
    pub bitwise_stark: BitwiseStark<F, D>,
    pub cmp_stark: CmpStark<F, D>,
    pub rangecheck_stark: RangeCheckStark<F, D>,

    pub cross_table_lookups: Vec<CrossTableLookup<F>>,
}

impl<F: RichField + Extendable<D>, const D: usize> Default for AllStark<F, D> {
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

impl<F: RichField + Extendable<D>, const D: usize> AllStark<F, D> {
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

#[allow(unused)] // TODO: Should be used soon.
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
// | tag | op0 | op1 | res | op0_limb_0 | op0_limb_1 |res_limb_2 | op0_limb_3 |...
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
    use crate::all_stark::AllStark;
    use crate::config::StarkConfig;
    use crate::generation::builtin::{
        generate_builtins_bitwise_trace, generate_builtins_cmp_trace,
        generate_builtins_rangecheck_trace,
    };
    use crate::generation::cpu::generate_cpu_trace;
    use crate::generation::generate_traces;
    use crate::generation::memory::generate_memory_trace;
    use crate::proof::PublicValues;
    use crate::prover::prove_with_traces;
    use crate::stark::Stark;
    use crate::util::trace_rows_to_poly_values;
    use crate::verifier::verify_proof;
    use anyhow::Result;
    use core::program::Program;
    use executor::Process;
    use log::debug;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[allow(dead_code)]
    const D: usize = 2;
    #[allow(dead_code)]
    type C = PoseidonGoldilocksConfig;
    #[allow(dead_code)]
    type F = <C as GenericConfig<D>>::F;
    #[allow(dead_code)]
    type S = dyn Stark<F, D>;

    #[test]
    fn add_mul_decode() -> Result<()> {
        //mov r0 8
        //mov r1 2
        //mov r2 3
        //add r3 r0 r1
        //mul r4 r3 r2
        //end
        let program_src = "0x4000000840000000
            0x8
            0x4000001040000000
            0x2
            0x4000002040000000
            0x3
            0x0020204400000000
            0x0100408200000000
            0x0000000000800000";

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }

    #[test]
    fn fibo_use_loop_decode() -> Result<()> {
        // mov r0 8
        // mov r1 1
        // mov r2 1
        // mov r3 0
        // EQ r0 r3
        // cjmp 19
        // add r4 r1 r2
        // mov r1 r2
        // mov r2 r4
        // mov r4 1
        // add r3 r3 r4
        // jmp 8
        // end
        let program_src = "0x4000000840000000
            0x8
            0x4000001040000000
            0x1
            0x4000002040000000
            0x1
            0x4000004040000000
            0x0
            0x0020800100000000
            0x4000000010000000
            0x13
            0x0040408400000000
            0x0000401040000000
            0x0001002040000000
            0x4000008040000000
            0x1
            0x0101004400000000
            0x4000000020000000
            0x8
            0x0000000000800000";

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }

    #[test]
    fn memory_test() -> Result<()> {
        // mov r0 8
        // mstore  0x100 r0
        // mov r1 2
        // mstore  0x200 r1
        // mov r0 20
        // mload r1 0x100
        // mload r2 0x200
        // mload r3 0x200
        // add r0 r1 r1
        // end
        let program_src = "0x4000000840000000
                            0x8
                            0x4020000001000000
                            0x100
                            0x4000001040000000
                            0x2
                            0x4040000001000000
                            0x200
                            0x4000000840000000
                            0x14
                            0x4000001002000000
                            0x100
                            0x4000002002000000
                            0x200
                            0x4000004002000000
                            0x200
                            0x0040200c00000000
                            0x0000000000800000";

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }

    #[test]
    fn call_test() -> Result<()> {
        //JMP 7
        //MUL r4 r0 10
        //ADD r4 r4 r1
        //MOV r0 r4
        //RET
        //MOV r0 8
        //MOV r1 2
        //mov r8 0x100010000
        //add r7 r8 -2
        //mov r6 0x100000000
        //mstore r7 r6
        //CALL 2
        //ADD r0 r0 r1
        //END
        let program_src = "0x4000000020000000
                                0x7
                            0x4020008200000000
                            0xa
                            0x0200208400000000
                            0x0001000840000000
                            0x0000000004000000
                            0x4000000840000000
                            0x8
                            0x4000001040000000
                            0x2
                            0x4000080040000000
                            0x100010000
                            0x6000040400000000
                            0xfffffffeffffffff
                            0x4000020040000000
                            0x100000000
                            0x0808000001000000
                            0x4000000008000000
                            0x2
                            0x0020200c00000000
                            0x0000000000800000";

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }

    #[test]
    fn range_check_test() -> Result<()> {
        //mov r0 8
        //mov r1 2
        //mov r2 3
        //add r3 r0 r1
        //mul r4 r3 r2
        //range_check r4
        //end
        let program_src = "0x4000000840000000
            0x8
            0x4000001040000000
            0x2
            0x4000002040000000
            0x3
            0x0020204400000000
            0x0100408200000000
            0x0001000000400000
            0x0000000000800000";

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }

    #[allow(unused)]
    fn timestamp() -> i64 {
        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        since_the_epoch.as_secs() as i64 * 1000i64
            + (since_the_epoch.subsec_nanos() as f64 / 1_000_000.0) as i64
    }

    #[test]
    fn bitwise_test() -> Result<()> {
        //mov r0 8
        //mov r1 2
        //mov r2 3
        //add r3 r0 r1
        //mul r4 r3 r2
        //and r5 r4 r3
        //end
        let program_src = "0x4000000840000000
            0x8
            0x4000001040000000
            0x2
            0x4000002040000000
            0x3
            0x0020204400000000
            0x0100408200000000
            0x0200810000200000
            0x0000000000800000";

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }

    #[test]
    fn comparison_test() -> Result<()> {
        //mov r0 8
        //mov r1 2
        //mov r2 3
        //add r3 r0 r1
        //mul r4 r3 r2
        //gte r4 r3
        //end
        let program_src = "0x4000000840000000
            0x8
            0x4000001040000000
            0x2
            0x4000002040000000
            0x3
            0x0020204400000000
            0x0100408200000000
            0x0200800000010000
            0x0000000000800000";

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }

    #[test]
    fn fibo_use_loop_memory_decode() -> Result<()> {
        //2 0 mov r0 1
        //2 2 mov r2 1
        //2 4 mstore 128 r0
        //2 6 mstore 135 r0
        //2 8 mov r0 test_loop
        //2 10 mov r3 0
        //1 12 EQ r0 r3
        //2 13 cjmp 30
        //2 15 mload  r1  128
        //1 17 assert r1  r2
        //2 18 mload  r2  135
        //1 20 add r4 r1 r2
        //2 21 mstore 128 r2
        //2 23 mstore 135 r4
        //2 25 mov r4 1
        //1 27 add r3 r3 r4
        //2 28 jmp 12
        //1 30 range_check r3
        //1 31 end
        let program_src = format!(
            "0x4000000840000000
        0x1
         0x4000002040000000
        0x1
        0x4020000001000000
        0x80
        0x4020000001000000
        0x87
        0x4000000840000000
        {:#x}
        0x4000004040000000
        0x0
        0x0020800100000000
        0x4000000010000000
        0x1e
        0x4000001002000000
        0x80
        0x0040400080000000
        0x4000002002000000
        0x87
        0x0040408400000000
        0x4080000001000000
        0x80
        0x4200000001000000
        0x87
        0x4000008040000000
        0x1
        0x0101004400000000
        0x4000000020000000
        0xc
        0x0000800000400000
        0x0000000000800000",
            8 // 200~300, 0x20000 == 100 w
        );

        debug!("program_src:{:?}", program_src);

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };
        debug!("instructions:{:?}", program.instructions);

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);
        process.gen_memory_table(&mut program);

        let mut all_stark = AllStark::default();
        let (traces, public_values) = generate_traces(&program, &mut all_stark);
        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;
        verify_proof(all_stark, proof, &config)
    }
}
