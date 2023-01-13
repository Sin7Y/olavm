use std::iter;

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use crate::builtins::bitwise::bitwise_stark::{self, BitwiseStark};
use crate::builtins::cmp::cmp_stark::{self, CmpStark};
use crate::builtins::rangecheck::rangecheck_stark::{self, RangeCheckStark};
use crate::config::StarkConfig;
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use crate::fixed_table::bitwise_fixed::bitwise_fixed_stark::{self, BitwiseFixedStark};
use crate::fixed_table::rangecheck_fixed::rangecheck_fixed_stark::{self, RangecheckFixedStark};
use crate::memory::{
    ctl_data as mem_ctl_data, ctl_data_mem_rc_diff_addr, ctl_data_mem_rc_diff_clk,
    ctl_data_mem_rc_diff_cond, ctl_filter as mem_ctl_filter, ctl_filter_mem_rc_diff_addr,
    ctl_filter_mem_rc_diff_clk, ctl_filter_mem_rc_diff_cond, MemoryStark,
};
use crate::program::program_stark::{self, ProgramStark};
use crate::stark::Stark;

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
            // builtins
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
    // fixed table
    BitwiseFixed = 5,
    RangecheckFixed = 6,
    // program table
    Program = 7,
}

pub(crate) const NUM_TABLES: usize = Table::RangeCheck as usize + 1;

#[allow(unused)] // TODO: Should be used soon.
pub(crate) fn all_cross_table_lookups<F: Field>() -> Vec<CrossTableLookup<F>> {
    // TODO:
    vec![]
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
    let mem_rc_diff_cond = TableWithColumns::new(
        Table::Memory,
        ctl_data_mem_rc_diff_cond(),
        Some(ctl_filter_mem_rc_diff_cond()),
    );
    let mem_rc_diff_addr = TableWithColumns::new(
        Table::Memory,
        ctl_data_mem_rc_diff_addr(),
        Some(ctl_filter_mem_rc_diff_addr()),
    );
    let mem_rc_diff_clk = TableWithColumns::new(
        Table::Memory,
        ctl_data_mem_rc_diff_clk(),
        Some(ctl_filter_mem_rc_diff_clk()),
    );
    let all_mem_rc_lookers = vec![mem_rc_diff_cond, mem_rc_diff_addr, mem_rc_diff_clk];
    let rc_looked = TableWithColumns::new(
        Table::RangeCheck,
        rangecheck_stark::ctl_data_memory(),
        Some(rangecheck_stark::ctl_filter_memory()),
    );
    CrossTableLookup::new(all_mem_rc_lookers, rc_looked, None)
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

// Cross_Lookup_Table(looking_table, looked_table)
/*fn ctl_bitwise_rangecheck<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::RangecheckFixed,
            rangecheck_fixed_stark::ctl_data_with_bitwise(),
            Some(rangecheck_fixed_stark::ctl_filter_with_bitwise()),
        )],
        TableWithColumns::new(
            Table::Bitwise,
            bitwise_stark::ctl_data_with_rangecheck_fixed(),
            Some(bitwise_stark::ctl_filter_with_rangecheck_fixed()),
        ),
        None,
    )
}*/

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
fn ctl_correct_program_cpu<F: Field>() -> CrossTableLookup<F> {
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
}

mod tests {
    use std::borrow::BorrowMut;

    use anyhow::Result;
    use ethereum_types::U256;
    use itertools::Itertools;
    // use crate::cross_table_lookup::testutils::check_ctls;
    use crate::verifier::verify_proof;
    use core::program::Program;
    use core::trace::trace::Trace;
    use executor::Process;
    use log::debug;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::{CircuitConfig, VerifierCircuitData};
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use rand::{thread_rng, Rng};
    // use serde_json::Value;
    use crate::all_stark::{AllStark, NUM_TABLES};
    use crate::config::StarkConfig;
    use crate::cpu::cpu_stark::CpuStark;
    use crate::proof::{AllProof, PublicValues};
    use crate::prover::prove_with_traces;
    use crate::util::{generate_cpu_trace, trace_rows_to_poly_values};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    fn fibo_use_loop() -> Vec<PolynomialValues<F>> {
        let program_src = "mov r0 8
            mov r1 1
            mov r2 1
            mov r3 0
            EQ r0 r3
            cjmp 12
            add r4 r1 r2
            mov r1 r2
            mov r2 r4
            mov r4 1
            sub r0 r0 r4
            jmp 4
            end
            ";

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
        process.execute(&mut program, false);

        println!("vm trace: {:?}", program.trace);

        let cpu_rows =
            generate_cpu_trace::<F>(&program.trace.exec, &program.trace.raw_binary_instructions);

        println!("cpu rows: {:?}", cpu_rows);

        trace_rows_to_poly_values(cpu_rows)
    }

    fn add_mul_decode() -> Vec<PolynomialValues<F>> {
        //mov r0 8
        //mov r1 2
        //mov r2 3
        //add r3 r0 r1
        //mul r4 r3 r2
        let program_src = "0x4000000840000000
            0x8
            0x4000001040000000
            0x2
            0x4000002040000000
            0x3
            0x0020204400000000
            0x0100408200000000
            ";

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
        process.execute(&mut program, true);

        println!("vm trace: {:?}", program.trace);

        let cpu_rows =
            generate_cpu_trace::<F>(&program.trace.exec, &program.trace.raw_binary_instructions);

        println!("cpu rows: {:?}", cpu_rows);
        trace_rows_to_poly_values(cpu_rows)
    }

    fn make_cpu_trace() -> Vec<PolynomialValues<F>> {
        // add_mul_decode()
        fibo_use_loop()
    }

    fn get_proof(config: &StarkConfig) -> Result<(AllStark<F, D>, AllProof<F, C, D>)> {
        let all_stark = AllStark::default();
        let cpu_trace = make_cpu_trace();
        let memory_rows: Vec<[F; 1]> = vec![];
        let memory_trace = trace_rows_to_poly_values(memory_rows);
        let bitwise_rows: Vec<[F; 1]> = vec![];
        let bitwise_trace = trace_rows_to_poly_values(bitwise_rows);
        let cmp_rows: Vec<[F; 1]> = vec![];
        let cmp_trace = trace_rows_to_poly_values(cmp_rows);
        let rangecheck_rows: Vec<[F; 1]> = vec![];
        let rangecheck_trace = trace_rows_to_poly_values(rangecheck_rows);
        let traces = [
            cpu_trace,
            memory_trace,
            bitwise_trace,
            cmp_trace,
            rangecheck_trace,
        ];
        // check_ctls(&traces, &all_stark.cross_table_lookups);

        let public_values = PublicValues::default();
        let proof = prove_with_traces::<F, C, D>(
            &all_stark,
            config,
            traces,
            public_values,
            &mut TimingTree::default(),
        )?;

        Ok((all_stark, proof))
    }

    #[test]
    #[ignore] // Ignoring but not deleting so the test can serve as an API usage example
    fn test_all_stark() -> Result<()> {
        let config = StarkConfig::standard_fast_config();
        let (all_stark, proof) = get_proof(&config)?;
        verify_proof(all_stark, proof, &config)
    }

    // #[test]
    // #[ignore] // Ignoring but not deleting so the test can serve as an API usage example
    // fn test_all_stark_recursive_verifier() -> Result<()> {
    //     init_logger();

    //     let config = StarkConfig::standard_fast_config();
    //     let (all_stark, proof) = get_proof(&config)?;
    //     verify_proof(all_stark.clone(), proof.clone(), &config)?;

    //     recursive_proof(all_stark, proof, &config)
    // }

    // fn recursive_proof(
    //     inner_all_stark: AllStark<F, D>,
    //     inner_proof: AllProof<F, C, D>,
    //     inner_config: &StarkConfig,
    // ) -> Result<()> {
    //     let circuit_config = CircuitConfig::standard_recursion_config();
    //     let recursive_all_proof = recursively_verify_all_proof(
    //         &inner_all_stark,
    //         &inner_proof,
    //         inner_config,
    //         &circuit_config,
    //     )?;

    //     let verifier_data: [VerifierCircuitData<F, C, D>; NUM_TABLES] =
    //         all_verifier_data_recursive_stark_proof(
    //             &inner_all_stark,
    //             inner_proof.degree_bits(inner_config),
    //             inner_config,
    //             &circuit_config,
    //         );
    //     let circuit_config = CircuitConfig::standard_recursion_config();
    //     let mut builder = CircuitBuilder::<F, D>::new(circuit_config);
    //     let mut pw = PartialWitness::new();
    //     let recursive_all_proof_target =
    //         add_virtual_recursive_all_proof(&mut builder, &verifier_data);
    //     set_recursive_all_proof_target(&mut pw, &recursive_all_proof_target, &recursive_all_proof);
    //     RecursiveAllProof::verify_circuit(
    //         &mut builder,
    //         recursive_all_proof_target,
    //         &verifier_data,
    //         inner_all_stark.cross_table_lookups,
    //         inner_config,
    //     );

    //     let data = builder.build::<C>();
    //     let proof = data.prove(pw)?;
    //     data.verify(proof)
    // }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }
}
