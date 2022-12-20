use {
    super::*,
    crate::columns::*,
    itertools::izip,
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::hash::hash_types::RichField,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    starky::stark::Stark,
    starky::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    std::marker::PhantomData,
};

#[derive(Copy, Clone, Default)]
pub struct BuiltinStark<F, const D: usize> {
    pub f: PhantomData<F>,
}
