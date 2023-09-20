use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::usize;

use itertools::Itertools;
use keccak_hash::keccak;
use plonky2_util::ceil_div_usize;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::witness::PartitionWitness;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CircuitConfig;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};

pub type LookupTable = Arc<Vec<(u16, u16)>>;
pub type BitwiseLookupTable = Arc<Vec<(u8, u8, u8)>>;

/// A gate which stores the set of (input, output) value pairs of a lookup table, and their multiplicities.
#[derive(Debug, Clone)]
pub struct LookupTableGate {
    /// Number of lookup entries per gate.
    pub num_slots: usize,
    /// Lookup table associated to the gate.
    pub lut: LookupTable,
    /// The Keccak hash of the lookup table.
    lut_hash: [u8; 32],
    /// First row of the lookup table.
    last_lut_row: usize,
}

impl LookupTableGate {
    pub fn new_from_table(config: &CircuitConfig, lut: LookupTable, last_lut_row: usize) -> Self {
        let table_bytes = lut
            .iter()
            .flat_map(|(input, output)| [input.to_le_bytes(), output.to_le_bytes()].concat())
            .collect_vec();

        Self {
            num_slots: Self::num_slots(config),
            lut,
            lut_hash: keccak(table_bytes).0,
            last_lut_row,
        }
    }

    pub(crate) fn num_slots(config: &CircuitConfig) -> usize {
        let wires_per_entry = 3;
        config.num_routed_wires / wires_per_entry
    }

    /// Wire for the looked input.
    pub fn wire_ith_looked_inp(i: usize) -> usize {
        3 * i
    }

    // Wire for the looked output.
    pub fn wire_ith_looked_out(i: usize) -> usize {
        3 * i + 1
    }

    /// Wire for the multiplicity. Set after the trace has been generated.
    pub fn wire_ith_multiplicity(i: usize) -> usize {
        3 * i + 2
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for LookupTableGate {
    fn id(&self) -> String {
        // Custom implementation to not have the entire lookup table
        format!(
            "LookupTableGate {{num_slots: {}, lut_hash: {:?}, last_lut_row: {}}}",
            self.num_slots, self.lut_hash, self.last_lut_row
        )
    }

    fn eval_unfiltered(&self, _vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        // No main trace constraints for the lookup table.
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
        // No main trace constraints for the lookup table.
        vec![]
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        (0..self.num_slots)
            .map(|i| {
                let g: Box<dyn WitnessGenerator<F>> = Box::new(
                    LookupTableGenerator {
                        row,
                        lut: self.lut.clone(),
                        slot_nb: i,
                        num_slots: self.num_slots,
                        last_lut_row: self.last_lut_row,
                    }
                    .adapter(),
                );
                g
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_slots * 3
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

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for LookupTableGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        _vars: EvaluationVarsBasePacked<P>,
        mut _yield_constr: StridedConstraintConsumer<P>,
    ) {
    }
}

#[derive(Clone, Debug, Default)]
pub struct LookupTableGenerator<const D: usize> {
    row: usize,
    lut: LookupTable,
    slot_nb: usize,
    num_slots: usize,
    last_lut_row: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F> for LookupTableGenerator<D> {

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(&self, _witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let first_row = self.last_lut_row + ceil_div_usize(self.lut.len(), self.num_slots) - 1;
        let slot = (first_row - self.row) * self.num_slots + self.slot_nb;

        let slot_input_target =
            Target::wire(self.row, LookupTableGate::wire_ith_looked_inp(self.slot_nb));
        let slot_output_target =
            Target::wire(self.row, LookupTableGate::wire_ith_looked_out(self.slot_nb));

        if slot < self.lut.len() {
            let (input, output) = self.lut[slot];
            out_buffer.set_target(slot_input_target, F::from_canonical_usize(input as usize));
            out_buffer.set_target(slot_output_target, F::from_canonical_usize(output as usize));
        } else {
            // Pad with zeros.
            out_buffer.set_target(slot_input_target, F::ZERO);
            out_buffer.set_target(slot_output_target, F::ZERO);
        }
    }
}


/// A gate which stores the set of (input, output) value pairs of a lookup table, and their multiplicities.
#[derive(Debug, Clone)]
pub struct BitwiseLookupTableGate {
    /// Number of lookup entries per gate.
    pub num_slots: usize,
    /// Lookup table associated to the gate.
    pub lut: BitwiseLookupTable,
    /// The Keccak hash of the lookup table.
    lut_hash: [u8; 32],
    /// First row of the lookup table.
    last_lut_row: usize,
}

impl BitwiseLookupTableGate {

    pub fn new_from_table(config: &CircuitConfig, lut: BitwiseLookupTable, last_lut_row: usize) -> Self {
        let table_bytes = lut
            .iter()
            .flat_map(|(input0, input1, output)| [input0.to_le_bytes(), input1.to_le_bytes(), output.to_le_bytes()].concat())
            .collect_vec();

        Self {
            num_slots: Self::num_slots(config),
            lut,
            lut_hash: keccak(table_bytes).0,
            last_lut_row,
        }
    }

    pub(crate) fn num_slots(config: &CircuitConfig) -> usize {
        let wires_per_entry = 4;
        config.num_routed_wires / wires_per_entry
    }

    /// Wire for the looked input.
    pub fn wire_ith_looked_inp0(i: usize) -> usize {
        4 * i
    }

    /// Wire for the looked input.
    pub fn wire_ith_looked_inp1(i: usize) -> usize {
        4 * i + 1
    }

    // Wire for the looked output.
    pub fn wire_ith_looked_out(i: usize) -> usize {
        4 * i + 2
    }

    /// Wire for the multiplicity. Set after the trace has been generated.
    pub fn wire_ith_multiplicity(i: usize) -> usize {
        4 * i + 3
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for BitwiseLookupTableGate {
    fn id(&self) -> String {
        // Custom implementation to not have the entire lookup table
        format!(
            "BitwiseLookupTableGate {{num_slots: {}, lut_hash: {:?}, last_lut_row: {}}}",
            self.num_slots, self.lut_hash, self.last_lut_row
        )
    }

    fn eval_unfiltered(&self, _vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        // No main trace constraints for the lookup table.
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
        // No main trace constraints for the lookup table.
        vec![]
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        (0..self.num_slots)
            .map(|i| {
                let g: Box<dyn WitnessGenerator<F>> = Box::new(
                    BitwiseLookupTableGenerator {
                        row,
                        lut: self.lut.clone(),
                        slot_nb: i,
                        num_slots: self.num_slots,
                        last_lut_row: self.last_lut_row,
                    }
                    .adapter(),
                );
                g
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_slots * 3
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

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for BitwiseLookupTableGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        _vars: EvaluationVarsBasePacked<P>,
        mut _yield_constr: StridedConstraintConsumer<P>,
    ) {
    }
}

#[derive(Clone, Debug, Default)]
pub struct BitwiseLookupTableGenerator<const D: usize> {
    row: usize,
    lut: BitwiseLookupTable,
    slot_nb: usize,
    num_slots: usize,
    last_lut_row: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F> for BitwiseLookupTableGenerator<D> {

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(&self, _witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let first_row = self.last_lut_row + ceil_div_usize(self.lut.len(), self.num_slots) - 1;
        let slot = (first_row - self.row) * self.num_slots + self.slot_nb;

        let slot_input0_target =
            Target::wire(self.row, BitwiseLookupTableGate::wire_ith_looked_inp0(self.slot_nb));
        let slot_input1_target =
            Target::wire(self.row, BitwiseLookupTableGate::wire_ith_looked_inp1(self.slot_nb));
        let slot_output_target =
            Target::wire(self.row, BitwiseLookupTableGate::wire_ith_looked_out(self.slot_nb));

        if slot < self.lut.len() {
            let (input0, input1, output) = self.lut[slot];
            out_buffer.set_target(slot_input0_target, F::from_canonical_usize(input0 as usize));
            out_buffer.set_target(slot_input1_target, F::from_canonical_usize(input1 as usize));
            out_buffer.set_target(slot_output_target, F::from_canonical_usize(output as usize));
        } else {
            // Pad with zeros.
            out_buffer.set_target(slot_input0_target, F::ZERO);
            out_buffer.set_target(slot_input1_target, F::ZERO);
            out_buffer.set_target(slot_output_target, F::ZERO);
        }
    }
}
