use std::marker::PhantomData;

use itertools::izip;
use plonky2::{
    field::{
        extension::{Extendable, FieldExtension},
        packed::PackedField,
    },
    hash::hash_types::RichField,
    plonk::circuit_builder::CircuitBuilder,
};

use crate::stark::{
    constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    stark::Stark,
    vars::{StarkEvaluationTargets, StarkEvaluationVars},
};

use super::columns::*;

#[derive(Copy, Clone, Default)]
pub struct StorageAccessStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for StorageAccessStark<F, D> {
    const COLUMNS: usize = NUM_COL_ST;
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;
        let lv_is_padding = lv[COL_ST_IS_PADDING];
        let nv_is_padding = nv[COL_ST_IS_PADDING];
        let lv_st_access_idx = lv[COL_ST_ACCESS_IDX];
        let nv_st_access_idx = nv[COL_ST_ACCESS_IDX];
        let lv_layer = lv[COL_ST_LAYER];
        let nv_layer = nv[COL_ST_LAYER];
        // is_padding binary and change from 0 to 1 once.
        yield_constr.constraint((P::ONES - lv_is_padding) * lv_is_padding);
        yield_constr.constraint_transition(
            (nv_is_padding - lv_is_padding) * (nv_is_padding - lv_is_padding - P::ONES),
        );
        // st_access_idx: from 1, donnot change or increase by 1
        yield_constr.constraint_first_row((P::ONES - lv_is_padding) * (lv_st_access_idx - P::ONES));
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (nv_st_access_idx - lv_st_access_idx)
                * (nv_st_access_idx - lv_st_access_idx - P::ONES),
        );

        // layer: from 1 to 256
        // first line layer is 1
        yield_constr.constraint_first_row((P::ONES - lv_is_padding) * (P::ONES - lv_layer));
        // if st_access_idx not change, layer increase by 1
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (P::ONES - (nv_st_access_idx - lv_st_access_idx))
                * (nv_layer - lv_layer - P::ONES),
        );
        // if st_access_idx increase by 1, current layer is 256, next layer is 1
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (nv_st_access_idx - lv_st_access_idx)
                * (lv_layer - P::Scalar::from_canonical_u64(256)),
        );
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (nv_st_access_idx - lv_st_access_idx)
                * (nv_layer - P::ONES),
        );
        // for lv_layer not 256, nv_layer increase by 1
        yield_constr.constraint(
            (P::ONES - nv_is_padding)
                * (lv_layer - P::Scalar::from_canonical_u64(256))
                * (nv_layer - lv_layer - P::ONES),
        );

        // is_layer_n constraints
        // binary
        yield_constr.constraint(lv[COL_ST_IS_LAYER_1] * (P::ONES - lv[COL_ST_IS_LAYER_1]));
        yield_constr.constraint(lv[COL_ST_IS_LAYER_64] * (P::ONES - lv[COL_ST_IS_LAYER_64]));
        yield_constr.constraint(lv[COL_ST_IS_LAYER_128] * (P::ONES - lv[COL_ST_IS_LAYER_128]));
        yield_constr.constraint(lv[COL_ST_IS_LAYER_192] * (P::ONES - lv[COL_ST_IS_LAYER_192]));
        yield_constr.constraint(lv[COL_ST_IS_LAYER_256] * (P::ONES - lv[COL_ST_IS_LAYER_256]));
        // for first line and st_access_idx increased line, is_layer_1 is 1
        yield_constr
            .constraint_first_row((P::ONES - lv_is_padding) * (P::ONES - lv[COL_ST_IS_LAYER_1]));
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (nv_st_access_idx - lv_st_access_idx)
                * (P::ONES - nv[COL_ST_IS_LAYER_1]),
        );
        // for not layer n, is_layer_n is 0
        yield_constr.constraint((lv[COL_ST_LAYER] - P::ONES) * lv[COL_ST_IS_LAYER_1]);
        yield_constr.constraint(
            (lv[COL_ST_LAYER] - P::Scalar::from_canonical_u64(64)) * lv[COL_ST_IS_LAYER_64],
        );
        yield_constr.constraint(
            (lv[COL_ST_LAYER] - P::Scalar::from_canonical_u64(128)) * lv[COL_ST_IS_LAYER_128],
        );
        yield_constr.constraint(
            (lv[COL_ST_LAYER] - P::Scalar::from_canonical_u64(192)) * lv[COL_ST_IS_LAYER_192],
        );
        yield_constr.constraint(
            (lv[COL_ST_LAYER] - P::Scalar::from_canonical_u64(256)) * lv[COL_ST_IS_LAYER_256],
        );
        // if st_access_idx not change, nv_acc_layer_marker =
        // lv_acc_layer_marker + sum(markers)
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (P::ONES - (nv_st_access_idx - lv_st_access_idx))
                * (nv[COL_ST_ACC_LAYER_MARKER]
                    - lv[COL_ST_ACC_LAYER_MARKER]
                    - (nv[COL_ST_IS_LAYER_1]
                        + nv[COL_ST_IS_LAYER_64]
                        + nv[COL_ST_IS_LAYER_128]
                        + nv[COL_ST_IS_LAYER_192]
                        + nv[COL_ST_IS_LAYER_256])),
        );
        // if st_access_idx increased, acc_layer_marker must be 5
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (nv_st_access_idx - lv_st_access_idx)
                * (lv[COL_ST_ACC_LAYER_MARKER] - P::Scalar::from_canonical_u64(5)),
        );

        // hash_type: layer 256 hash_type = 1, others hash_type = 0
        // if st_access_idx increased, hash_type = 1
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (nv_st_access_idx - lv_st_access_idx)
                * (lv[COL_ST_HASH_TYPE] - P::ONES),
        );
        // if st_access_idx not change, hash_type = 0
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (P::ONES - (nv_st_access_idx - lv_st_access_idx))
                * lv[COL_ST_HASH_TYPE],
        );

        // pre_root and root constraints:
        // in padding line, root not change
        COL_ST_ROOT_RANGE.for_each(|col| {
            yield_constr.constraint(nv_is_padding * (nv[col] - lv[col]));
        });

        for (col_pre_root_limb, col_root_limb, col_pre_hash_limb, col_hash_limb) in izip!(
            COL_ST_PRE_ROOT_RANGE,
            COL_ST_ROOT_RANGE,
            COL_ST_PRE_HASH_RANGE,
            COL_ST_HASH_RANGE
        ) {
            // when st_accesss_idx increased, nv_pre_root = lv_root
            yield_constr.constraint_transition(
                (P::ONES - nv_is_padding)
                    * (nv_st_access_idx - lv_st_access_idx)
                    * (nv[col_pre_root_limb] - lv[col_root_limb]),
            );
            // when st_access_idx not change, pre_root and root not change
            yield_constr.constraint_transition(
                (P::ONES - nv_is_padding)
                    * (P::ONES - (nv_st_access_idx - lv_st_access_idx))
                    * (nv[col_pre_root_limb] - lv[col_pre_root_limb]),
            );
            yield_constr.constraint_transition(
                (P::ONES - nv_is_padding)
                    * (P::ONES - (nv_st_access_idx - lv_st_access_idx))
                    * (nv[col_root_limb] - lv[col_root_limb]),
            );
            // in layer_1 line, root equals related hash
            yield_constr.constraint(
                lv[COL_ST_IS_LAYER_1] * (lv[col_pre_root_limb] - lv[col_pre_hash_limb]),
            );
            yield_constr
                .constraint(lv[COL_ST_IS_LAYER_1] * (lv[col_root_limb] - lv[col_hash_limb]));
        }

        // addr_acc constraints:
        // layer_bit is binary
        yield_constr.constraint(lv[COL_ST_LAYER_BIT] * (P::ONES - lv[COL_ST_LAYER_BIT]));
        // in lines other than 64, 128, 192, 256, nv_addr_acc = acc_acc * 2 +
        // nv_layer_bit
        yield_constr.constraint_transition(
            (P::ONES
                - lv[COL_ST_IS_LAYER_64]
                - lv[COL_ST_IS_LAYER_128]
                - lv[COL_ST_IS_LAYER_192]
                - lv[COL_ST_IS_LAYER_256])
                * (nv[COL_ST_ADDR_ACC]
                    - lv[COL_ST_ADDR_ACC] * P::Scalar::from_canonical_u64(2)
                    - nv[COL_ST_LAYER_BIT]),
        );
        // in line 64, 128, 192 or 256, addr_addr equals related addr limb
        yield_constr.constraint(
            lv[COL_ST_IS_LAYER_64] * (lv[COL_ST_ADDR_ACC] - lv[COL_ST_ADDR_RANGE.start]),
        );
        yield_constr.constraint(
            lv[COL_ST_IS_LAYER_128] * (lv[COL_ST_ADDR_ACC] - lv[COL_ST_ADDR_RANGE.start + 1]),
        );
        yield_constr.constraint(
            lv[COL_ST_IS_LAYER_192] * (lv[COL_ST_ADDR_ACC] - lv[COL_ST_ADDR_RANGE.start + 2]),
        );
        yield_constr.constraint(
            lv[COL_ST_IS_LAYER_256] * (lv[COL_ST_ADDR_ACC] - lv[COL_ST_ADDR_RANGE.start + 3]),
        );

        // path constraint: when st_access_idx not change, next hash equals path
        COL_ST_HASH_RANGE
            .zip(COL_ST_PATH_RANGE)
            .for_each(|(col_hash, col_path)| {
                yield_constr.constraint_transition(
                    (P::ONES - nv_is_padding)
                        * (P::ONES - (nv_st_access_idx - lv_st_access_idx))
                        * (lv[col_hash] - nv[col_path]),
                );
            });

        // filter constraints:
        yield_constr.constraint(
            (P::ONES - lv_is_padding)
                * (lv[COL_ST_FILTER_IS_HASH_BIT_0] + lv[COL_ST_LAYER_BIT] - P::ONES),
        );
        yield_constr.constraint(
            (P::ONES - lv_is_padding) * (lv[COL_ST_FILTER_IS_HASH_BIT_1] - lv[COL_ST_LAYER_BIT]),
        );
        yield_constr.constraint(lv_is_padding * lv[COL_ST_FILTER_IS_HASH_BIT_0]);
        yield_constr.constraint(lv_is_padding * lv[COL_ST_FILTER_IS_HASH_BIT_1]);
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { Self::COLUMNS }>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        4
    }
}
