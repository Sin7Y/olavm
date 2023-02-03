// use plonky2::plonk::circuit_data::CircuitData;


/// The recursion threshold. We end a chain of recursive proofs once we reach this size.
const THRESHOLD_DEGREE_BITS: usize = 13;

// /// Contains all recursive circuits used in the system. For each STARK and each initial
// /// `degree_bits`, this contains a chain of recursive circuits for shrinking that STARK from
// /// `degree_bits` to a constant `THRESHOLD_DEGREE_BITS`. It also contains a special root circuit
// /// for combining each STARK's shrunk wrapper proof into a single proof.
// pub struct AllRecursiveCircuits<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     pub root: RootCircuitData<F, C, D>,
// }

pub struct RootCircuitData<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    circuit: CircuitData<F, C, D>,
}