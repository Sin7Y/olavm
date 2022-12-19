
use crate::bitwise::columns;


#[derive(Copy, Clone, Default)]
pub struct AndOrXorStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}


impl<F: RichField, const D: usize> AndOrXorStark<F, D> {

    const BASE: usize = 1 << 8;

}


impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for AndOrXorStark<F, D> {

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { columns.COL_NUM_AND }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let op0 = vars.local_values[OP0];
        let op1 = vars.local_values[OP1];
        let res = vars.local_values[RES];

        // sumcheck for op0, op1, res
        // op0 = Sum(op0_limbs_i * 2^(8*i))
        let op0_limbs: Vec<_> = vars.local_values[OP0_LIMBS].to_vec();
        let computed_sum = reduce_with_powers(&op0_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - op0);

        // op1 = Sum(op1_limbs_i * 2^(8*i))       
        let op1_limbs: Vec<_> = vars.local_values[OP1_LIMBS].to_vec();
        let computed_sum = reduce_with_powers(&op1_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - op1);

        // res = Sum(res_limbs_i * 2^(8*i))
        let res_limbs: Vec<_> = vars.local_values[RES_LIMBS].to_vec();
        let computed_sum = reduce_with_powers(&res_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - res);

    }

}