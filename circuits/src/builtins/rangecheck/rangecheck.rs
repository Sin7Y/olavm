
use crate::cmp::columns;

#[derive(Copy, Clone, Default)]
pub struct RcStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> RcStark<F, D> {

    const BASE: usize = 1 << 16;

}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RcStark<F, D> {

    // Split U32 into 2 16bit limbs
    // Sumcheck between  Val and limbs
    // RC for limbs
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { columns.COL_NUM_AND }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let val = vars.local_values[VAL];
        let limb_lo = vars.local_values[LIMB_LO];
        let limb_hi = vars.local_values[LIMB_HI];

        // Addition checl for op0, op1, diff
        yield_constr.constraint(val - (limb_lo + limb_hi * 1 << BASE));

    }

}