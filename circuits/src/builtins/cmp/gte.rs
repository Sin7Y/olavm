
use crate::cmp::columns;

#[derive(Copy, Clone, Default)]
pub struct CmpStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for CmpStark<F, D> {
    // Since op0 is in [0, U32), op1 is in [0, U32)
    // op0, op1 are all field elements
    // if op0 >= op1 is true
    //    diff = op0 - op1  is in [0, U32)
    // if op0 >= op1 is false
    //    diff = op0 - op1 < 0; as this is in finite field, so diff = P + (op0 - op1)
    // As P =  2^64 - 2^32 + 1; op0 - op1 in (-U32, 0)
    // So P + (op0 - op1) > U32
    // so if we Constraint the diff is U32, RC(diff), we could get the GTE relation between op0, op1
    // The constraints is should be:
    // 1. addition check
    //       op0 = diff + op1
    // 2. rangecheck for diff
    //      RC(diff)
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
        let diff = vars.local_values[RES];

        // Addition checl for op0, op1, diff
        yield_constr.constraint(op0 - (op1 + diff));

    }

}