use std::matches;

use plonky2::filed;

use vm_core::trace::{trace::Step, instruction::Instruction::*};
use crate::columns::*;

#[derive(Copy, Clone, Default)]
pub struct AddStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField, const D: usize> ArithmeticStark<F, D> {
    pub fn generate_trace(&self, step: &Step) -> [F; NUM_ARITH_COLS] {
        assert!(matches!(step.instruction, Instruction::ADD));

        let mut lv = [F::default(); NUM_ARITH_COLS];
        lv[INST_COL] = columns::ADD_id;
        lv[CLK_COL] = step.clk;
        lv[PC_COL] = step.pc;
        lv[FLAG_COL] = step.flag;
        for (&c, w) in &step.regs[0..24].iter() {
            lv[REG_COL + c] = F::from_canonical_u64(w);
        }
        lv
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ArithmeticStark<F, D>  {
    const COLUMNS: usize = NUM_ARITH_COLS;
    const PUBLIC_INPUTS: usize = 3;

    fn eval_packed_generic<P: PackedField>(
        lv: &[P; NUM_ARITH_COLS],
        yield_constr: &mut ConstraintConsumer<P>,
    ) {
        // We use range_check to check each input/output limb in 32 bits.
        range_check_error!(ADD_INPUT_0, 32);
        range_check_error!(ADD_INPUT_1, 32);
        range_check_error!(ADD_OUTPUT, 32);

        // Get ADD data from trace.
        let is_add = if lv[INST_COL] == columns::ADD_id { 1 } else { 0 };
        let output_limbs = ADD_OUTPUT_COLS.iter().map(|&c| lv[c]);
        let input0_limbs = ADD_INPUT0_COLS.iter().map(|&c| lv[c]);
        let input1_limbs = ADD_INPUT1_COLS.iter().map(|&c| lv[c]);

        // Do a local addition.
        let unreduced_output = input0_limbs.zip(input1_limbs).map(|(a, b)| a + b);

        // Constraint each limb's addition.
        let overflow = P::Scalar::from_canonical_u64(1 << LIMB_BITS);
        let overflow_inv = overflow.inverse();
        let mut carry = P::ZEROS;
        for (a, b) in unreduced_output.zip(output_limbs) {
            // t should be either 0 or 2^LIMB_BITS
            let t = carry + a - b;
            yield_constr.constraint(is_add * t * (overflow - t));
            // carry = t / overflow
            carry = t * overflow_inv;
        }
    }

    fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        lv: &[ExtensionTarget<D>; NUM_ARITH_COLS],
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        // Get ADD data from trace.
        let is_add = if lv[INST_COL] == columns::ADD_id { 1 } else { 0 };
        let output_limbs = ADD_OUTPUT_COLS.iter().map(|&c| lv[c]);
        let input0_limbs = ADD_INPUT0_COLS.iter().map(|&c| lv[c]);
        let input1_limbs = ADD_INPUT1_COLS.iter().map(|&c| lv[c]);

        let output_computed = input0_limbs
            .zip(input1_limbs)
            .map(|(a, b)| builder.add_extension(a, b))
            .collect::<Vec<ExtensionTarget<D>>>();

        // 2^LIMB_BITS in the base field
        let overflow_base = F::from_canonical_u64(1 << LIMB_BITS);
        // 2^LIMB_BITS in the extension field as an ExtensionTarget
        let overflow = builder.constant_extension(F::Extension::from(overflow_base));
        // 2^-LIMB_BITS in the base field.
        let overflow_inv = F::inverse_2exp(LIMB_BITS);

        let mut carry = builder.zero_extension();
        for (a, b) in output_computed.zip(output_limbs) {
            // t0 = cy + a
            let t0 = builder.add_extension(cy, a);
            // t  = t0 - b
            let t = builder.sub_extension(t0, b);
            // t1 = overflow - t
            let t1 = builder.sub_extension(overflow, t);
            // t2 = t * t1
            let t2 = builder.mul_extension(t, t1);

            let filtered_limb_constraint = builder.mul_extension(is_add, t2);
            yield_constr.constraint(builder, filtered_limb_constraint);

            carry = builder.mul_const_extension(overflow_inv, t);
        }
    }

    fn constraint_degree(&self) -> usize {
        // FIXME!
        2
    }
}

mod tests {
    use anyhow::Result;
    use plonky2::field::extension::Extendable;
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::RichField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{
        AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig,
    };
    use plonky2::util::timing::TimingTree;

    use crate::config::StarkConfig;
    use crate::arithmetic_stark::AddStark;
    use crate::proof::StarkProofWithPublicInputs;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::stark::Stark;
    use crate::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_add_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = AddStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let step = Step {
            clk: 0,
            pc: 0,
            fp: 0,
            instruction: Instruction::ADD(),
        };
        let stark = S::default();
        let trace = stark.generate_trace(step);
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            public_inputs,
            &mut TimingTree::default(),
        )?;

        verify_stark_proof(stark, proof, &config)
    }
}
