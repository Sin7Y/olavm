use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::usize;

use itertools::Itertools;
use keccak_hash::keccak;

use super::lookup_table::{LookupTable,BitwiseLookupTable};
use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CircuitConfig;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};

pub type Lookup = Vec<(Target, Target)>;

pub type BitwiseLookup = Vec<(Target, Target, Target)>;

/// A gate which stores (input, output) lookup pairs made elsewhere in the trace. It doesn't check any constraints itself.
#[derive(Debug, Clone)]
pub struct LookupGate {
    /// Number of lookups per gate.
    pub num_slots: usize,
    /// LUT associated to the gate.
    lut: LookupTable,
    /// The Keccak hash of the lookup table.
    lut_hash: [u8; 32],
}

/// A gate which stores (input, output) lookup pairs made elsewhere in the trace. It doesn't check any constraints itself.
#[derive(Debug, Clone)]
pub struct BitwiseLookupGate {
    /// Number of lookups per gate.
    pub num_slots: usize,
    /// LUT associated to the gate.
    lut: BitwiseLookupTable,
    /// The Keccak hash of the lookup table.
    lut_hash: [u8; 32],
}

impl LookupGate {
    pub fn new_from_table(config: &CircuitConfig, lut: LookupTable) -> Self {
        let table_bytes = lut
            .iter()
            .flat_map(|(input, output)| [input.to_le_bytes(), output.to_le_bytes()].concat())
            .collect_vec();

        Self {
            num_slots: Self::num_slots(config),
            lut,
            lut_hash: keccak(table_bytes).0,
        }
    }
    pub(crate) fn num_slots(config: &CircuitConfig) -> usize {
        let wires_per_lookup = 2;
        config.num_routed_wires / wires_per_lookup
    }

    pub fn wire_ith_looking_inp(i: usize) -> usize {
        2 * i
    }

    pub fn wire_ith_looking_out(i: usize) -> usize {
        2 * i + 1
    }
}

impl BitwiseLookupGate {

    pub fn new_from_table(config: &CircuitConfig, lut: BitwiseLookupTable) -> Self {
        let table_bytes = lut
            .iter()
            .flat_map(|(input_0, input_1, output)| [input_0.to_le_bytes(), input_1.to_le_bytes(), output.to_le_bytes()].concat())
            .collect_vec();

        Self {
            num_slots: Self::num_slots(config),
            lut,
            lut_hash: keccak(table_bytes).0,
        }
    }
    pub(crate) fn num_slots(config: &CircuitConfig) -> usize {
        let wires_per_lookup = 3;
        config.num_routed_wires / wires_per_lookup
    }

    pub fn wire_ith_looking_inp0(i: usize) -> usize {
        3 * i + 0
    }

    pub fn wire_ith_looking_inp1(i: usize) -> usize {
        3 * i + 1
    }

    pub fn wire_ith_looking_out(i: usize) -> usize {
        3 * i + 2
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for LookupGate {
    fn id(&self) -> String {
        // Custom implementation to not have the entire lookup table
        format!(
            "LookupGate {{num_slots: {}, lut_hash: {:?}}}",
            self.num_slots, self.lut_hash
        )
    }

    fn eval_unfiltered(&self, _vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        // No main trace constraints for lookups.
        vec![]
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        // No main trace constraints for lookups.
        vec![]
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        (0..self.num_slots)
            .map(|i| {
                let g: Box<dyn WitnessGenerator<F>> = Box::new(
                    LookupGenerator {
                        row,
                        lut: self.lut.clone(),
                        slot_nb: i,
                    }
                    .adapter(),
                );
                g
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_slots * 2
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        0
    }

    fn num_constraints(&self) -> usize {
        0
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for BitwiseLookupGate {
    fn id(&self) -> String {
        // Custom implementation to not have the entire lookup table
        format!(
            "BitwiseLookupGate {{num_slots: {}, lut_hash: {:?}}}",
            self.num_slots, self.lut_hash
        )
    }

    fn eval_unfiltered(&self, _vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        // No main trace constraints for lookups.
        vec![]
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        // No main trace constraints for lookups.
        vec![]
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        (0..self.num_slots)
            .map(|i| {
                let g: Box<dyn WitnessGenerator<F>> = Box::new(
                    BitwiseLookupGenerator {
                        row,
                        lut: self.lut.clone(),
                        slot_nb: i,
                    }
                    .adapter(),
                );
                g
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_slots * 2
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        0
    }

    fn num_constraints(&self) -> usize {
        0
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for LookupGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        _vars: EvaluationVarsBasePacked<P>,
        mut _yield_constr: StridedConstraintConsumer<P>,
    ) {
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for BitwiseLookupGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        _vars: EvaluationVarsBasePacked<P>,
        mut _yield_constr: StridedConstraintConsumer<P>,
    ) {
    }
}


#[derive(Clone, Debug, Default)]
pub struct LookupGenerator<const D: usize> {
    row: usize,
    lut: LookupTable,
    slot_nb: usize,
}

#[derive(Clone, Debug, Default)]
pub struct BitwiseLookupGenerator<const D: usize> {
    row: usize,
    lut: BitwiseLookupTable,
    slot_nb: usize,
}


impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F> for LookupGenerator<D> {

    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(
            self.row,
            LookupGate::wire_ith_looking_inp(self.slot_nb),
        )]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };

        let input_val = get_wire(LookupGate::wire_ith_looking_inp(self.slot_nb));
        let (input, output) = self.lut[input_val.to_canonical_u64() as usize];
        if input_val == F::from_canonical_u16(input) {
            let output_val = F::from_canonical_u16(output);

            let out_wire = Target::wire(self.row, LookupGate::wire_ith_looking_out(self.slot_nb));
            out_buffer.set_target(out_wire, output_val);
        } else {
            for (input, output) in self.lut.iter() {
                if input_val == F::from_canonical_u16(*input) {
                    let output_val = F::from_canonical_u16(*output);

                    let out_wire =
                        Target::wire(self.row, LookupGate::wire_ith_looking_out(self.slot_nb));
                    out_buffer.set_target(out_wire, output_val);
                    return;
                }
            }
            panic!("Incorrect input value provided");
        };
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F> for BitwiseLookupGenerator<D> {

    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(
                self.row,
                BitwiseLookupGate::wire_ith_looking_inp0(self.slot_nb)),
            Target::wire(
                self.row,
                BitwiseLookupGate::wire_ith_looking_inp1(self.slot_nb))
            ]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };

        let input0_val = get_wire(BitwiseLookupGate::wire_ith_looking_inp0(self.slot_nb));
        let input1_val = get_wire(BitwiseLookupGate::wire_ith_looking_inp1(self.slot_nb));
        let (input0, input1, output) = self.lut[(input0_val.to_canonical_u64() * 16 +  input1_val.to_canonical_u64()) as usize];
        // find directly
        if input0_val == F::from_canonical_u8(input0) && input1_val == F::from_canonical_u8(input1){
            let output_val = F::from_canonical_u8(output);

            let out_wire = Target::wire(self.row, BitwiseLookupGate::wire_ith_looking_out(self.slot_nb));
            out_buffer.set_target(out_wire, output_val);
        } else {
            // loop all case
            for (input0, input1, output) in self.lut.iter() {

                if input0_val == F::from_canonical_u8(*input0) && input1_val == F::from_canonical_u8(*input1){

                    let output_val = F::from_canonical_u8(*output);

                    let out_wire =
                        Target::wire(self.row, BitwiseLookupGate::wire_ith_looking_out(self.slot_nb));
                    out_buffer.set_target(out_wire, output_val);
                    return;
                }
            }
            panic!("Incorrect input value provided");
        };
    }
}

#[cfg(test)]
mod tests {
    static LOGGER_INITIALIZED: Once = Once::new();

    use alloc::sync::Arc;
    use std::ops::Add;
    use std::sync::Once;

    use itertools::Itertools;
    use log::{Level, LevelFilter};

    use crate::gadgets::lookup::{OTHER_TABLE, SMALLER_TABLE, TIP5_TABLE};
    use crate::gates::lookup_table::LookupTable;
    use crate::gates::noop::NoopGate;
    use crate::plonk::prover::prove;
    use crate::util::timing::TimingTree;

    #[test]
    fn test_no_lookup() -> anyhow::Result<()> {
        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        use crate::iop::witness::PartialWitness;
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        builder.add_gate(NoopGate, vec![]);
        let pw = PartialWitness::new();

        let data = builder.build::<C>();
        let mut timing = TimingTree::new("prove first", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        timing.print();
        data.verify(proof)?;

        Ok(())
    }

    #[should_panic]
    #[test]
    fn test_lookup_table_not_used() {
        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        //let tip5_table = TIP5_TABLE.to_vec();

        let mut table =Vec::<(u8,u8,u8)>::with_capacity(65536 * 3);

        for i in 0..256 {
            for j in 0..256 {
                 let or = i as u8 | j as u8;
                 let and = i as u8 & j as u8;
                 let xor = i as u8 ^ j as u8;

                 table[i + j * 256] = (i as u8, j as u8, or);
                 table[i + j * 256 + 65536] = (i as u8, j as u8, and);
                 table[i + j * 256 + 65536 * 2] = (i as u8, j as u8, xor);
            }
        }

        builder.add_lookup_table_from_pairs(Arc::new(table));

        builder.build::<C>();

    }

    #[should_panic]
    #[test]
    fn test_lookup_without_table() {
        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let dummy = builder.add_virtual_target();
        builder.add_lookup_from_index(dummy, dummy, 0);

        builder.build::<C>();
    }

    // Tests two lookups in one lookup table.
    #[test]
    fn test_one_lookup_bitwise() -> anyhow::Result<()> {
        use crate::field::types::Field;
        use crate::iop::witness::{PartialWitness, Witness};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let mut table =Vec::<(u8,u8,u8)>::with_capacity(256);

        for i in 0..16 {
            for j in 0..16 {
                 //let or = i as u8 | j as u8;
                 let and = i as u8 & j as u8;
                 //let xor = i as u8 ^ j as u8;

                 //table.push((i as u8, j as u8, or));
                 table.push((i as u8, j as u8, and));
                 //table.push((i as u8, j as u8, xor));
            }
        }

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();
        let initial_a_and_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;
        let look_val_a_and_b = look_val_a & look_val_b;

        let table_index = builder.add_lookup_table_from_pairs(Arc::new(table));

        let output_a_and_b = builder.add_lookup_from_index(initial_a, initial_b, table_index);


        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(initial_a_and_b);

        builder.register_public_input(output_a_and_b);

        let mut pw = PartialWitness::new();

        pw.set_target(initial_a, F::from_canonical_usize(look_val_a));
        pw.set_target(initial_b, F::from_canonical_usize(look_val_b));
        pw.set_target(initial_a_and_b, F::from_canonical_usize(look_val_a_and_b));

        let data = builder.build::<C>();
        let mut timing = TimingTree::new("prove one lookup", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        timing.print();
        data.verify(proof.clone())?;

        Ok(())
    }

    /*
    // Tests two lookups in one lookup table.
    #[test]
    fn test_one_lookup() -> anyhow::Result<()> {
        use crate::field::types::Field;
        use crate::iop::witness::{PartialWitness, Witness};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        let tip5_table = TIP5_TABLE.to_vec();
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let out_a = table[look_val_a].1;
        let out_b = table[look_val_b].1;
        let table_index = builder.add_lookup_table_from_pairs(table);
        let output_a = builder.add_lookup_from_index(initial_a, table_index);

        let output_b = builder.add_lookup_from_index(initial_b, table_index);

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);

        let mut pw = PartialWitness::new();

        pw.set_target(initial_a, F::from_canonical_usize(look_val_a));
        pw.set_target(initial_b, F::from_canonical_usize(look_val_b));

        let data = builder.build::<C>();
        let mut timing = TimingTree::new("prove one lookup", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        timing.print();
        data.verify(proof.clone())?;

        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(out_a),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(out_b),
            "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );

        Ok(())
    }

    // Tests one lookup in two different lookup tables.
    #[test]
    pub fn test_two_luts() -> anyhow::Result<()> {
        use crate::field::types::Field;
        use crate::iop::witness::{PartialWitness, Witness};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();

        let first_out = tip5_table[look_val_a];
        let second_out = tip5_table[look_val_b];

        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let other_table = OTHER_TABLE.to_vec();

        let table_index = builder.add_lookup_table_from_pairs(table);
        let output_a = builder.add_lookup_from_index(initial_a, table_index);

        let output_b = builder.add_lookup_from_index(initial_b, table_index);
        let sum = builder.add(output_a, output_b);

        let s = first_out + second_out;
        let final_out = other_table[s as usize];

        let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());
        let table2_index = builder.add_lookup_table_from_pairs(table2);

        let output_final = builder.add_lookup_from_index(sum, table2_index);

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(sum);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);
        builder.register_public_input(output_final);

        let mut pw = PartialWitness::new();
        pw.set_target(initial_a, F::from_canonical_usize(look_val_a));
        pw.set_target(initial_b, F::from_canonical_usize(look_val_b));
        let data = builder.build::<C>();
        let mut timing = TimingTree::new("prove two_luts", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        data.verify(proof.clone())?;
        timing.print();

        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(first_out),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[4] == F::from_canonical_u16(second_out),
            "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );
        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(s),
            "Sum between the first two LUT outputs is incorrect."
        );
        assert!(
            proof.public_inputs[5] == F::from_canonical_u16(final_out),
            "Output of the second LUT at index {} is incorrect.",
            s
        );

        Ok(())
    }

    #[test]
    pub fn test_different_inputs() -> anyhow::Result<()> {
        use crate::field::types::Field;
        use crate::iop::witness::{PartialWitness, Witness};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let init_a = 1;
        let init_b = 2;

        let tab: Vec<u16> = SMALLER_TABLE.to_vec();
        let table: LookupTable = Arc::new((2..10).zip_eq(tab).collect());

        let other_table = OTHER_TABLE.to_vec();

        let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());
        let small_index = builder.add_lookup_table_from_pairs(table.clone());
        let output_a = builder.add_lookup_from_index(initial_a, small_index);

        let output_b = builder.add_lookup_from_index(initial_b, small_index);
        let sum = builder.add(output_a, output_b);

        let other_index = builder.add_lookup_table_from_pairs(table2.clone());
        let output_final = builder.add_lookup_from_index(sum, other_index);

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(sum);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);
        builder.register_public_input(output_final);

        let mut pw = PartialWitness::new();

        let look_val_a = table[init_a].0;
        let look_val_b = table[init_b].0;
        pw.set_target(initial_a, F::from_canonical_u16(look_val_a));
        pw.set_target(initial_b, F::from_canonical_u16(look_val_b));

        let data = builder.build::<C>();
        let mut timing = TimingTree::new("prove different lookups", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        data.verify(proof.clone())?;
        timing.print();

        let out_a = table[init_a].1;
        let out_b = table[init_b].1;
        let s = out_a + out_b;
        let out_final = table2[s as usize].1;

        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(out_a),
            "First lookup, at index {} in the smaller LUT gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[4] == F::from_canonical_u16(out_b),
            "Second lookup, at index {} in the smaller LUT gives an incorrect output.",
            proof.public_inputs[1]
        );
        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(s),
            "Sum between the first two LUT outputs is incorrect."
        );
        assert!(
            proof.public_inputs[5] == F::from_canonical_u16(out_final),
            "Output of the second LUT at index {} is incorrect.",
            s
        );

        Ok(())
    }

    // This test looks up over 514 values for one LookupTableGate, which means that several LookupGates are created.
    #[test]
    pub fn test_many_lookups() -> anyhow::Result<()> {
        use crate::field::types::Field;
        use crate::iop::witness::{PartialWitness, Witness};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let out_a = table[look_val_a].1;
        let out_b = table[look_val_b].1;

        let tip5_index = builder.add_lookup_table_from_pairs(table);
        let output_a = builder.add_lookup_from_index(initial_a, tip5_index);

        let output_b = builder.add_lookup_from_index(initial_b, tip5_index);
        let sum = builder.add(output_a, output_b);

        for _ in 0..514 {
            builder.add_lookup_from_index(initial_a, tip5_index);
        }

        let other_table = OTHER_TABLE.to_vec();

        let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());

        let s = out_a + out_b;
        let out_final = table2[s as usize].1;

        let other_index = builder.add_lookup_table_from_pairs(table2);
        let output_final = builder.add_lookup_from_index(sum, other_index);

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(sum);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);
        builder.register_public_input(output_final);

        let mut pw = PartialWitness::new();

        pw.set_target(initial_a, F::from_canonical_usize(look_val_a));
        pw.set_target(initial_b, F::from_canonical_usize(look_val_b));

        let data = builder.build::<C>();
        let mut timing = TimingTree::new("prove different lookups", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;

        data.verify(proof.clone())?;
        timing.print();

        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(out_a),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[4] == F::from_canonical_u16(out_b),
            "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );
        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(s),
            "Sum between the first two LUT outputs is incorrect."
        );
        assert!(
            proof.public_inputs[5] == F::from_canonical_u16(out_final),
            "Output of the second LUT at index {} is incorrect.",
            s
        );

        Ok(())
    }

    // Tests whether, when adding the same LUT to the circuit, the circuit only adds one copy, with the same index.
    #[test]
    pub fn test_same_luts() -> anyhow::Result<()> {
        use crate::field::types::Field;
        use crate::iop::witness::{PartialWitness, Witness};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        LOGGER_INITIALIZED.call_once(|| init_logger().unwrap());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let table_index = builder.add_lookup_table_from_pairs(table.clone());
        let output_a = builder.add_lookup_from_index(initial_a, table_index);

        let output_b = builder.add_lookup_from_index(initial_b, table_index);
        let sum = builder.add(output_a, output_b);

        let table2_index = builder.add_lookup_table_from_pairs(table);

        let output_final = builder.add_lookup_from_index(sum, table2_index);

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(sum);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);
        builder.register_public_input(output_final);

        let luts_length = builder.get_luts_length();

        assert!(
            luts_length == 1,
            "There are {} LUTs when there should be only one",
            luts_length
        );

        let mut pw = PartialWitness::new();

        pw.set_target(initial_a, F::from_canonical_usize(look_val_a));
        pw.set_target(initial_b, F::from_canonical_usize(look_val_b));

        let data = builder.build::<C>();
        let mut timing = TimingTree::new("prove two_luts", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        data.verify(proof)?;
        timing.print();

        Ok(())
    }
    */

    fn init_logger() -> anyhow::Result<()> {
        let mut builder = env_logger::Builder::from_default_env();
        builder.format_timestamp(None);
        builder.filter_level(LevelFilter::Debug);

        builder.try_init()?;
        Ok(())
    }
    
}

