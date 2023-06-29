use crate::builtins::bitwise::columns::*;
use anyhow::Result;
use itertools::Itertools;
//use crate::var::{StarkEvaluationTargets, StarkEvaluationVars};
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::stark::cross_table_lookup::Column;
use crate::stark::lookup::*;
use crate::stark::permutation::*;
use crate::stark::stark::Stark;
use crate::stark::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
//use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::plonk_common::*;
use std::marker::PhantomData;
//use std::ops::*;

#[derive(Copy, Clone, Default)]
pub struct BitwiseStark<F, const D: usize> {
    compress_challenge: Option<F>,
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> BitwiseStark<F, D> {
    const BASE: usize = 1 << 8;

    pub fn set_compress_challenge(&mut self, challenge: F) -> Result<()> {
        assert!(self.compress_challenge.is_none(), "already set?");
        self.compress_challenge = Some(challenge);
        Ok(())
    }

    pub fn get_compress_challenge(&self) -> Option<F> {
        self.compress_challenge
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for BitwiseStark<F, D> {
    const COLUMNS: usize = COL_NUM_BITWISE;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { COL_NUM_BITWISE }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let op0 = lv[OP0];
        let op1 = lv[OP1];
        let res = lv[RES];

        // sumcheck for op0, op1, res
        // op0 = Sum(op0_limbs_i * 2^(8*i))
        let op0_limbs: Vec<_> = lv[OP0_LIMBS].to_vec();
        let computed_sum =
            reduce_with_powers(&op0_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - op0);

        // op1 = Sum(op1_limbs_i * 2^(8*i))
        let op1_limbs: Vec<_> = lv[OP1_LIMBS].to_vec();
        let computed_sum =
            reduce_with_powers(&op1_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - op1);

        // res = Sum(res_limbs_i * 2^(8*i))
        let res_limbs: Vec<_> = lv[RES_LIMBS].to_vec();
        let computed_sum =
            reduce_with_powers(&res_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - res);

        // Constrain compress logic.
        let beta = FE::from_basefield(self.get_compress_challenge().unwrap());
        for i in 0..4 {
            yield_constr.constraint(
                lv[TAG]
                    + lv[OP0_LIMBS.start + i] * beta
                    + lv[OP1_LIMBS.start + i] * beta * beta
                    + lv[RES_LIMBS.start + i] * beta * beta * beta
                    - lv[COMPRESS_LIMBS.start + i],
            );
        }

        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 1,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 2,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 3,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 4,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 5,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 6,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 7,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 8,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 9,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 10,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 11,
        );

        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start,
            FIX_COMPRESS_PERMUTED.start,
        );
        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 1,
            FIX_COMPRESS_PERMUTED.start + 1,
        );
        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 2,
            FIX_COMPRESS_PERMUTED.start + 2,
        );
        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 3,
            FIX_COMPRESS_PERMUTED.start + 3,
        );
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COL_NUM_BITWISE }>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let lv = vars.local_values;
        let op0 = lv[OP0];
        let op1 = lv[OP1];
        let res = lv[RES];

        // sumcheck for op0, op1, res
        // op0 = Sum(op0_limbs_i * 2^(8*i))
        let op0_limbs: Vec<_> = lv[OP0_LIMBS].to_vec();
        let alpha = builder.constant(F::from_canonical_usize(Self::BASE));
        let computed_sum = reduce_with_powers_ext_circuit(builder, &op0_limbs, alpha);
        let op0_sum_cs = builder.sub_extension(computed_sum, op0);
        yield_constr.constraint(builder, op0_sum_cs);

        // op1 = Sum(op1_limbs_i * 2^(8*i))
        let op1_limbs: Vec<_> = lv[OP1_LIMBS].to_vec();
        let computed_sum = reduce_with_powers_ext_circuit(builder, &op1_limbs, alpha);
        let op1_sum_cs = builder.sub_extension(computed_sum, op1);
        yield_constr.constraint(builder, op1_sum_cs);

        // res = Sum(res_limbs_i * 2^(8*i))
        let res_limbs: Vec<_> = lv[RES_LIMBS].to_vec();
        let computed_sum = reduce_with_powers_ext_circuit(builder, &res_limbs, alpha);
        let res_sum_cs = builder.sub_extension(computed_sum, res);
        yield_constr.constraint(builder, res_sum_cs);

        // Constrain compress logic.
        let beta = builder.constant_extension(F::Extension::from_basefield(
            self.get_compress_challenge().unwrap(),
        ));
        let beta_2 = builder.mul_extension(beta, beta);
        let beta_3 = builder.mul_extension(beta_2, beta);
        for i in 0..4 {
            let op0_cs = builder.mul_extension(lv[OP0_LIMBS.start + i], beta);
            let op1_cs = builder.mul_extension(lv[OP1_LIMBS.start + i], beta_2);
            let res_cs = builder.mul_extension(lv[RES_LIMBS.start + i], beta_3);
            let cs = builder.add_many_extension([lv[TAG], op0_cs, op1_cs, res_cs]);
            let cs = builder.sub_extension(cs, lv[COMPRESS_LIMBS.start + i]);
            yield_constr.constraint(builder, cs);
        }

        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 1,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 2,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 3,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 4,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 5,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 6,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 7,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 8,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 9,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 10,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 11,
        );

        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start,
            FIX_COMPRESS_PERMUTED.start,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 1,
            FIX_COMPRESS_PERMUTED.start + 1,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 2,
            FIX_COMPRESS_PERMUTED.start + 2,
        );
        eval_lookups_circuit(
            builder,
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 3,
            FIX_COMPRESS_PERMUTED.start + 3,
        );
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        vec![
            PermutationPair::singletons(COMPRESS_LIMBS.start, COMPRESS_PERMUTED.start),
            PermutationPair::singletons(COMPRESS_LIMBS.start + 1, COMPRESS_PERMUTED.start + 1),
            PermutationPair::singletons(COMPRESS_LIMBS.start + 2, COMPRESS_PERMUTED.start + 2),
            PermutationPair::singletons(COMPRESS_LIMBS.start + 3, COMPRESS_PERMUTED.start + 3),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start + 1),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start + 2),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start + 3),
        ]
    }
}

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {
    Column::singles([OP0, OP1, RES]).collect_vec()
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {
    Column::single(FILTER)
}

// Get the column info for Cross_Lookup<Rangecheck_Fixed_table, Bitwise_table>
/*pub fn ctl_data_with_rangecheck_fixed<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles(OP0_LIMBS).collect_vec();
    res.extend(Column::singles(OP1_LIMBS).collect_vec());
    res.extend(Column::singles(RES_LIMBS).collect_vec());
    res
}
pub fn ctl_filter_with_rangecheck_fixed<F: Field>() -> Column<F> {
    Column::one()
}
// Get the column info for Cross_Lookup<Bitwise_Fixed_table, Bitwise_table>
pub fn ctl_data_with_bitwise_fixed<F: Field>() -> Vec<Column<F>> {
    let mut res =
        Column::singles([OP0_LIMBS.start, OP1_LIMBS.start, RES_LIMBS.start]).collect_vec();
    res.extend(
        Column::singles([
            OP0_LIMBS.start.add(1),
            OP1_LIMBS.start.add(1),
            RES_LIMBS.start.add(1),
        ])
        .collect_vec(),
    );
    res.extend(
        Column::singles([
            OP0_LIMBS.start.add(2),
            OP1_LIMBS.start.add(2),
            RES_LIMBS.start.add(2),
        ])
        .collect_vec(),
    );
    res.extend(Column::singles([OP0_LIMBS.end, OP1_LIMBS.end, RES_LIMBS.end]).collect_vec());
    res.extend(Column::singles([TAG]));
    res
}
pub fn ctl_filter_with_bitwise_fixed<F: Field>() -> Column<F> {
    Column::one()
}*/

mod tests {
    use crate::builtins::bitwise::bitwise_stark::BitwiseStark;
    use crate::builtins::bitwise::columns::get_bitwise_col_name_map;
    use crate::generation::builtin::generate_bitwise_trace;
    use crate::stark::constraint_consumer::ConstraintConsumer;
    use crate::stark::stark::Stark;
    use crate::stark::vars::StarkEvaluationVars;
    use assembler::encoder::encode_asm_from_json_file;
    use core::merkle_tree::tree::AccountTree;
    use core::program::Program;
    use core::types::account::Address;
    use executor::Process;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2_util::log2_strict;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;

    #[allow(unused)]
    fn test_bitwise_stark(program_path: String) {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = BitwiseStark<F, D>;
        let mut stark = S::default();

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

        let (rows, bitwise_beta) =
            generate_bitwise_trace::<F>(&program.trace.builtin_bitwise_combined);
        let len = rows[0].len();
        println!(
            "raw trace len:{}, extended len: {}",
            program.trace.builtin_bitwise_combined.len(),
            len
        );
        stark.set_compress_challenge(bitwise_beta);

        let last = F::primitive_root_of_unity(log2_strict(len)).inverse();
        let subgroup =
            F::cyclic_subgroup_known_order(F::primitive_root_of_unity(log2_strict(len)), len);

        for i in 0..len {
            let local_values = rows
                .iter()
                .map(|row| row[i % len])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let next_values = rows
                .iter()
                .map(|row| row[(i + 1) % len])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let vars = StarkEvaluationVars {
                local_values: &local_values,
                next_values: &next_values,
            };

            let mut constraint_consumer = ConstraintConsumer::new(
                vec![F::rand()],
                subgroup[i] - last,
                if i == 0 {
                    GoldilocksField::ONE
                } else {
                    GoldilocksField::ZERO
                },
                if i == len - 1 {
                    GoldilocksField::ONE
                } else {
                    GoldilocksField::ZERO
                },
            );
            stark.eval_packed_generic(vars, &mut constraint_consumer);

            for &acc in &constraint_consumer.constraint_accs {
                if !acc.eq(&GoldilocksField::ZERO) {
                    println!("constraint error in line {}", i);
                    let m = get_bitwise_col_name_map();
                    println!("{:>32}\t{:>22}\t{:>22}", "name", "lv", "nv");
                    for col in m.keys() {
                        let name = m.get(col).unwrap();
                        let lv = vars.local_values[*col].0;
                        let nv = vars.next_values[*col].0;
                        println!("{:>32}\t{:>22}\t{:>22}", name, lv, nv);
                    }
                }
                assert_eq!(acc, GoldilocksField::ZERO);
            }
        }
    }

    #[test]
    fn test_bitwise_with_program() {
        let file_name = "bitwise.json";
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();
        test_bitwise_stark(program_path);
    }
}
