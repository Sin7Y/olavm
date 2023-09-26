use std::marker::PhantomData;
use alloc::sync::Arc;

use plonky2_field::extension::Extendable;
use plonky2_field::types::Field;

use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::blake3::{*};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};

/// Evaluates a full Blake33 permutation with 12 state elements.
#[derive(Debug)]
pub struct Blake3Gate<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Blake3Gate<F, D> {
    pub fn new() -> Self {
        Blake3Gate {
            _phantom: PhantomData,
        }
    }

    /// The wire index for the `i`th input to the permutation.
    pub fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub fn wire_output(i: usize) -> usize {
        16 + i
    }

    pub fn wire_xor_external(round: usize, g_round: usize, i: usize) -> usize {
        16 + 8 + round * 4 * 8 + g_round * 4 + i
    }

    pub fn wire_shift_remain_external(round: usize, g_round: usize, i: usize) -> usize {
        16 + 8 + 4 * 7 * 8 + round * 4 * 8 + g_round * 4 + i
    }

    pub fn wire_shift_q_external(round: usize, g_round: usize, i: usize) -> usize {
        16 + 8 + 4 * 7 * 8 * 2 + round * 4 * 8 + g_round * 4 + i
    }

    /// End of wire indices, exclusive.
    /// 696 column
    fn end() -> usize {
        16 + 8 + 8 * 7 * 4 * 3
    }

}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for Blake3Gate<F, D> {

    fn id(&self) -> String {

        format!("{:?}<WIDTH={}>", self, Self::end())

    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        
        let mut constraints = Vec::with_capacity(self.num_constraints());


        // Get input of blake3
        let mut block = [F::Extension::ZERO; STATE_SIZE];

        for i in 0..16{

            block[i] = vars.local_wires[Self::wire_input(i as usize)];

        }

        let mut cv = [F::Extension::ZERO; 8];

        for i in 0..8 {

            cv[i] = F::Extension::from_canonical_u32(<F as Blake3>::IV[i]);

        }

        <F as Blake3>::compress_in_place_field(&mut cv, block, 16, 0, 8);

        for i in 0..8 {

            let output = vars.local_wires[Self::wire_output(i as usize)];

            constraints.push(output - output);

        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {

        // Get input of blake3
        let mut block = [F::ZERO; STATE_SIZE];

        for i in 0..16{

            block[i] = vars.local_wires[Self::wire_input(i as usize)];

        }

        let mut cv = [F::ZERO; 8];

        for i in 0..8 {

            cv[i] = F::from_canonical_u32(<F as Blake3>::IV[i]);

        }

        <F as Blake3>::compress_in_place(&mut cv, block, 16, 0, 8);

        for i in 0..8 {

            let output = vars.local_wires[Self::wire_output(i as usize)];

            yield_constr.one(output - cv[i]);

        }

    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {

        // The naive method is more efficient if we have enough routed wires for
        // Need fixed for 65536
        let mut table =Vec::<(u8,u8,u8)>::with_capacity(LOOKUP_LIMB_RANGE * LOOKUP_LIMB_RANGE);

        for i in 0..LOOKUP_LIMB_RANGE {
            for j in 0..LOOKUP_LIMB_RANGE {

                 let xor = i as u8 ^ j as u8;

                 table.push((i as u8, j as u8, xor));
            }
        }

        let table_index = builder.add_lookup_table_from_pairs(Arc::new(table));


        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Get input of blake3
        let mut block = [builder.zero_extension(); 16];

        for i in 0..16{

            block[i] = vars.local_wires[Self::wire_input(i as usize)];

        }

        let mut cv = [builder.zero_extension(); 8];

        for i in 0..8 {

            cv[i] = builder.constant_extension(F::Extension::from_canonical_u32(<F as Blake3>::IV[i]));

        }

        let mut xor = [[[builder.zero_extension(); 4]; 8]; 7];
        let mut remain = [[[builder.zero_extension(); 4]; 8]; 7];
        let mut q = [[[builder.zero_extension(); 4]; 8]; 7];

        for i in 0..7 {
            for j in 0..8 {
                for k in 0..4 {
                    xor[i][j][k] = vars.local_wires[Self::wire_xor_external(i, j, k)];
                    remain[i][j][k] = vars.local_wires[Self::wire_shift_remain_external(i, j, k)];
                    q[i][j][k] = vars.local_wires[Self::wire_shift_q_external(i, j, k)];
                }
            }
        }

        let mut shift_const = [builder.zero_extension(); 4];
        shift_const[0] = builder.constant_extension(F::Extension::from_canonical_u32(1 << 16));
        shift_const[1] = builder.constant_extension(F::Extension::from_canonical_u32(1 << 12));
        shift_const[2] = builder.constant_extension(F::Extension::from_canonical_u32(1 << 8));
        shift_const[3] = builder.constant_extension(F::Extension::from_canonical_u32(1 << 7));

        let state = <F as Blake3>::compress_pre_circuit(builder, &mut cv, xor, remain, q, shift_const, block, 16, 0, 8);

        let mut output = [builder.zero_extension(); 8];

        for i in 0..8 {

            output[i] = vars.local_wires[Self::wire_output(i as usize)];

        }
        
        for i in 0..8 {

            let limbs_input_a = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[i].to_target_array()[0], LOOKUP_LIMB_NUMBER);
            let limbs_input_b = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[8 + i].to_target_array()[0], LOOKUP_LIMB_NUMBER);
            let limbs_input_c = builder.split_le_base::<LOOKUP_LIMB_RANGE>(output[i].to_target_array()[0], LOOKUP_LIMB_NUMBER);

            builder.add_lookup_from_index_bitwise(limbs_input_a[0], limbs_input_b[0], limbs_input_c[0], table_index);
            builder.add_lookup_from_index_bitwise(limbs_input_a[1], limbs_input_b[1], limbs_input_c[1], table_index);
            builder.add_lookup_from_index_bitwise(limbs_input_a[2], limbs_input_b[2], limbs_input_c[2], table_index);
            builder.add_lookup_from_index_bitwise(limbs_input_a[3], limbs_input_b[3], limbs_input_c[3], table_index);
        
        }
        
        for i in 0..8 {

            constraints.push(builder.sub_extension(output[i], output[i]));

        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        let gen = Blake3Generator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![Box::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        8
    }
}

#[derive(Debug)]
struct Blake3Generator<F: RichField + Extendable<D> + Blake3, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F>
    for Blake3Generator<F, D>
{
    fn dependencies(&self) -> Vec<Target> {
        (0..16)
            .map(|i| Blake3Gate::<F, D>::wire_input(i))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {

        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut block = [F::ZERO; STATE_SIZE];
        for i in 0 .. STATE_SIZE {

            block[i] = witness.get_wire(local_wire(Blake3Gate::<F, D>::wire_input(i)));

        }

        let mut cv = [F::ZERO; 8];
        for i in 0..8 {

            cv[i] = F::from_canonical_u32(<F as Blake3>::IV[i]);

        }

        <F as Blake3>::compress_in_place_field_run_once(out_buffer, self.row, &mut cv,  block, 16, 0, 8);

        for i in 0..8 {
            out_buffer.set_wire(local_wire(Blake3Gate::<F, D>::wire_output(i)), cv[i]);
        }

    }
}

#[cfg(test)]
mod tests{
    #![allow(incomplete_features)]
    
    use plonky2_field::types::Field;
    use rand::Rng;
    
    use crate::gates::gate::Gate;
    use crate::hash::hash_types::HashOut;
    use crate::iop::witness::{PartialWitness, Witness};
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::wire::Wire;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::GenericConfig;
    use crate::plonk::vars::{EvaluationTargets, EvaluationVars};
    use crate::plonk::config::Blake3GoldilocksConfig;
    use crate::plonk::verifier::verify;
    use crate::gates::blake3::Blake3Gate;
    use crate::hash::blake3::{*};
    
    #[test]
    pub fn test_blake3_prove()
    {
        const D: usize = 2;
        type C = Blake3GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        type Gate = Blake3Gate<F, D>;
        let gate = Gate::new();
        let mut rng = rand::thread_rng();
        let config = CircuitConfig::wide_blake3_config();
        let mut builder = CircuitBuilder::new(config);
 
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build::<C>();

        // generate inputs
        let mut permutation_inputs = [F::ZERO; STATE_SIZE];

        for i in 0..16{

            permutation_inputs[i] = F::from_canonical_u32(rng.gen());

        }

        let mut pw = PartialWitness::<F>::new();

        for i in 0..16 {
            pw.set_wire(
                Wire {
                    row,
                    column: Gate::wire_input(i),
                },
                permutation_inputs[i],
            );
        }

        let witness = generate_partial_witness::<F, C, D>(pw, &circuit.prover_only, &circuit.common);

        // Test that `eval_unfiltered` and `eval_unfiltered_recursively` are coherent.
        let mut wires = [<<Blake3GoldilocksConfig as GenericConfig<D>>::F as plonky2_field::extension::Extendable<D>>::Extension::ZERO; 696];
        // set input
        for i in 0..16 {

            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_input(i),
            });

            wires[i] = out.into();
        }
        // set output
        for i in 0..8 {

            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_output(i),
            });

            wires[16 + i] = out.into();
        }

        // set xor witness
        for i in 0..7 {
        
            for j in 0..8 {

                for k in 0..4 {

                    let out = witness.get_wire(Wire {
                        row: 0,
                        column: Gate::wire_xor_external(i, j, k),
                    });

                    wires[16 + 8 + i * 32 + j * 4 + k] = out.into();

                    let out1 = witness.get_wire(Wire {
                        row: 0,
                        column: Gate::wire_shift_remain_external(i, j, k),
                    });

                    wires[16 + 8 + 224 + i * 32 + j * 4 + k] = out1.into();


                    let out2 = witness.get_wire(Wire {
                        row: 0,
                        column: Gate::wire_shift_q_external(i, j, k),
                    });

                    wires[16 + 8 + 448 + i * 32 + j * 4 + k] = out2.into();
                }
            }
        }

        let gate = Gate::new();
        let constants = <<Blake3GoldilocksConfig as GenericConfig<D>>::F as plonky2_field::extension::Extendable<D>>::Extension::rand_vec(gate.num_constants());
        let public_inputs_hash = HashOut::rand();

        let config = CircuitConfig::wide_blake3_config();
        let mut pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let wires_t = builder.add_virtual_extension_targets(wires.len());
        let constants_t = builder.add_virtual_extension_targets(constants.len());
        pw.set_extension_targets(&wires_t, &wires);
        pw.set_extension_targets(&constants_t, &constants);
        let public_inputs_hash_t = builder.add_virtual_hash();
        pw.set_hash_target(public_inputs_hash_t, public_inputs_hash);

        let vars = EvaluationVars {
            local_constants: &constants,
            local_wires: &wires,
            public_inputs_hash: &public_inputs_hash,
        };
        let evals = gate.eval_unfiltered(vars);

        let vars_t = EvaluationTargets {
            local_constants: &constants_t,
            local_wires: &wires_t,
            public_inputs_hash: &public_inputs_hash_t,
        };
        let evals_t = gate.eval_unfiltered_circuit(&mut builder, vars_t);
        pw.set_extension_targets(&evals_t, &evals);

        let data = builder.build::<C>();

        let _proof = data.prove(pw);

        let result = verify(_proof.unwrap(), &data.verifier_only, &data.common);

        result.is_ok();

    }   

    #[test]
    fn generated_output() {
        const D: usize = 2;
        type C = Blake3GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::wide_blake3_config();
        let mut builder = CircuitBuilder::new(config);
        let mut rng = rand::thread_rng();

        type Gate = Blake3Gate<F, D>;
        let gate = Gate::new();
 
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build::<C>();

        // generate inputs
        let mut permutation_inputs = [F::ZERO; STATE_SIZE];

        for i in 0..16{

            permutation_inputs[i] = F::from_canonical_u32(rng.gen());

        }

        let mut pw = PartialWitness::<F>::new();

        for i in 0..16 {
            pw.set_wire(
                Wire {
                    row,
                    column: Gate::wire_input(i),
                },
                permutation_inputs[i],
            );
        }

        let witness = generate_partial_witness::<F, C, D>(pw, &circuit.prover_only, &circuit.common);

        let mut cv = [F::ZERO; IV_SIZE];

        for i in 0..8 {

            cv[i] = F::from_canonical_u32(<F as Blake3>::IV[i]);

        }

        //get expect output
        <F as Blake3>::compress_in_place(& mut cv, permutation_inputs, 16, 0, 8);
        
        for i in 0..8 {
            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, cv[i]);
        }

    }


}
