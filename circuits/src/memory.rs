use core::program::instruction;
use std::ops::Sub;

use plonky2::field::types::Field;

use crate::cross_table_lookup::Column;
use {
    crate::columns::*,
    crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::stark::Stark,
    crate::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::hash::hash_types::RichField,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    std::marker::PhantomData,
};

pub fn ctl_data_mem_rc_diff_cond<F: Field>() -> Vec<Column<F>> {
    vec![Column::single(COL_MEM_DIFF_ADDR_COND)]
}

pub fn ctl_filter_mem_rc_diff_cond<F: Field>() -> Column<F> {
    Column::sum([
        COL_MEM_REGION_PROPHET,
        COL_MEM_REGION_POSEIDON,
        COL_MEM_REGION_ECDSA,
    ])
}

pub fn ctl_data_mem_rc_diff_addr<F: Field>() -> Vec<Column<F>> {
    vec![Column::single(COL_MEM_DIFF_ADDR)]
}

pub fn ctl_filter_mem_rc_diff_addr<F: Field>() -> Column<F> {
    Column::single(COL_MEM_IS_RW)
}

pub fn ctl_data_mem_rc_diff_clk<F: Field>() -> Vec<Column<F>> {
    vec![Column::single(COL_MEM_DIFF_CLK)]
}

pub fn ctl_filter_mem_rc_diff_clk<F: Field>() -> Column<F> {
    Column::single(COL_MEM_RW_ADDR_UNCHANGED)
}

// todo ctl for poseidon and ecdsa

pub fn ctl_data<F: Field>() -> Vec<Column<F>> {
    let mut cols: Vec<_> =
        Column::singles([COL_MEM_CLK, COL_MEM_OP, COL_MEM_ADDR, COL_MEM_VALUE]).collect();
    cols
}

pub fn ctl_filter<F: Field>() -> Column<F> {
    Column::single(COL_MEM_FILTER_LOOKED_FOR_MAIN)
}

#[derive(Copy, Clone, Default)]
pub struct MemoryStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MemoryStark<F, D> {
    const COLUMNS: usize = NUM_MEM_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_MEM_COLS>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;

        let p = P::ZEROS - P::ONES;
        let span = P::Scalar::from_canonical_u64(2_u64.pow(32).sub(1));

        let op = lv[COL_MEM_OP];
        let is_rw = lv[COL_MEM_IS_RW];
        let region_prophet = lv[COL_MEM_REGION_PROPHET];
        let region_poseidon = lv[COL_MEM_REGION_POSEIDON];
        let region_ecdsa = lv[COL_MEM_REGION_ECDSA];
        let nv_region_prophet = nv[COL_MEM_REGION_PROPHET];
        let nv_region_poseidon = nv[COL_MEM_REGION_POSEIDON];
        let is_write = lv[COL_MEM_IS_WRITE];
        let nv_is_write = nv[COL_MEM_IS_WRITE];
        let filter_looked_for_main = lv[COL_MEM_FILTER_LOOKED_FOR_MAIN];
        let addr = lv[COL_MEM_ADDR];
        let diff_addr = nv[COL_MEM_DIFF_ADDR];
        let diff_addr_inv = nv[COL_MEM_DIFF_ADDR_INV];
        let nv_addr = nv[COL_MEM_ADDR];
        let nv_diff_addr = nv[COL_MEM_DIFF_ADDR];
        let rw_addr_unchanged = lv[COL_MEM_RW_ADDR_UNCHANGED];
        let diff_addr_cond = lv[COL_MEM_DIFF_ADDR_COND];
        let value = lv[COL_MEM_VALUE];
        let nv_value = lv[COL_MEM_VALUE];

        let op_mload = P::Scalar::from_canonical_u64(2_u64.pow(25));
        let op_mstore = P::Scalar::from_canonical_u64(2_u64.pow(24));
        let op_call = P::Scalar::from_canonical_u64(2_u64.pow(27));
        let op_ret = P::Scalar::from_canonical_u64(2_u64.pow(26));
        // op is one of mload, mstore, call, ret or prophet write 0.
        yield_constr
            .constraint(op * (op - op_mload) * (op - op_mstore) * (op - op_call) * (op - op_ret));
        // when op is 0, is_rw must be zero.
        yield_constr.constraint(
            (op - op_mload) * (op - op_mstore) * (op - op_call) * (op - op_ret) * is_rw,
        );

        // constraint is_write and op. When write, op can be mstore, call and 0; When read, op can be mload, call, ret.
        // call can both write and read, does not need a constraint rule.
        yield_constr
            .constraint((op - op_mload) * (op - op_call) * (op - op_ret) * (P::ONES - is_write));
        yield_constr.constraint(op * (op - op_mstore) * (op - op_call) * is_write);

        // when op is not 0, filter_looked_for_main need to be enabled.
        yield_constr.constraint(op * (P::ONES - filter_looked_for_main));

        // addr'-addr-diff_addr'= 0
        yield_constr.constraint(nv_addr - addr - nv_diff_addr);
        // for rw_addr_unchanged
        yield_constr.constraint_first_row(rw_addr_unchanged);
        yield_constr.constraint(is_rw * (P::ONES - rw_addr_unchanged - diff_addr * diff_addr_inv));

        // for region division: 1. one of four region is selected; 2. binary constraints; 3. diff_addr_cond in different region.
        yield_constr.constraint(is_rw + region_prophet + region_poseidon + region_ecdsa - P::ONES);
        yield_constr.constraint(is_rw * (P::ONES - is_rw));
        yield_constr.constraint(region_prophet * (P::ONES - region_prophet));
        yield_constr.constraint(region_poseidon * (P::ONES - region_poseidon));
        yield_constr.constraint(region_ecdsa * (P::ONES - region_ecdsa));
        yield_constr.constraint(region_prophet * (p - addr - diff_addr_cond));
        yield_constr.constraint(region_poseidon * (p - span - addr - diff_addr_cond));
        yield_constr.constraint(
            region_ecdsa * (p - span.mul(P::Scalar::from_canonical_u64(2)) - addr - diff_addr_cond),
        );

        // for write once:
        // 1. addr doesn't change or increase by 1 when not cross region;
        // 2. when addr not increase, must be read.
        yield_constr.constraint(
            (P::ONES - is_rw)
                * (P::ONES - nv_region_prophet + region_prophet)
                * (P::ONES - nv_region_poseidon + region_poseidon)
                * (nv_addr - addr)
                * (P::ONES - nv_addr + addr),
        );
        yield_constr.constraint(
            (P::ONES - is_rw)
                * (P::ONES - nv_region_prophet + region_prophet)
                * (P::ONES - nv_region_poseidon + region_poseidon)
                * (P::ONES - nv_addr + addr)
                * nv_is_write,
        );

        // read/write constraint:
        // 1. first operation for each addr must be write;
        // 2. next value does not change if it is read.
        yield_constr.constraint_first_row(P::ONES - is_write);
        yield_constr.constraint((nv_addr - addr) * (P::ONES - nv_is_write));
        yield_constr.constraint((P::ONES - nv_is_write) * (nv_value - value));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_MEM_COLS>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let one = builder.one_extension();

        let lv = vars.local_values;
        let nv = vars.next_values;

        let p = builder.constant_extension(
            F::Extension::from_canonical_u64(0).sub(F::Extension::from_canonical_u64(1)),
        );
        let span =
            builder.constant_extension(F::Extension::from_canonical_u64(2_u64.pow(32).sub(1)));

        let op = lv[COL_MEM_OP];
        let is_rw = lv[COL_MEM_IS_RW];
        let region_prophet = lv[COL_MEM_REGION_PROPHET];
        let region_poseidon = lv[COL_MEM_REGION_POSEIDON];
        let region_ecdsa = lv[COL_MEM_REGION_ECDSA];
        let nv_region_prophet = nv[COL_MEM_REGION_PROPHET];
        let nv_region_poseidon = nv[COL_MEM_REGION_POSEIDON];
        let is_write = lv[COL_MEM_IS_WRITE];
        let nv_is_write = nv[COL_MEM_IS_WRITE];
        let filter_looked_for_main = lv[COL_MEM_FILTER_LOOKED_FOR_MAIN];
        let addr = lv[COL_MEM_ADDR];
        let diff_addr = nv[COL_MEM_DIFF_ADDR];
        let diff_addr_inv = nv[COL_MEM_DIFF_ADDR_INV];
        let nv_addr = nv[COL_MEM_ADDR];
        let nv_diff_addr = nv[COL_MEM_DIFF_ADDR];
        let rw_addr_unchanged = lv[COL_MEM_RW_ADDR_UNCHANGED];
        let diff_addr_cond = lv[COL_MEM_DIFF_ADDR_COND];
        let value = lv[COL_MEM_VALUE];
        let nv_value = lv[COL_MEM_VALUE];

        let op_mload =
            builder.constant_extension(F::Extension::from_canonical_usize(2_usize.pow(25)));
        let op_mstore =
            builder.constant_extension(F::Extension::from_canonical_usize(2_usize.pow(24)));
        let op_call =
            builder.constant_extension(F::Extension::from_canonical_usize(2_usize.pow(27)));
        let op_ret =
            builder.constant_extension(F::Extension::from_canonical_usize(2_usize.pow(26)));
        // op is one of mload, mstore, call, ret or prophet write 0.
        let d_op_mload = builder.sub_extension(op, op_mload);
        let d_op_mstore = builder.sub_extension(op, op_mstore);
        let d_op_call = builder.sub_extension(op, op_call);
        let d_op_ret = builder.sub_extension(op, op_ret);
        let one_m_is_write = builder.sub_extension(one, is_write);
        let op_inter_1 = builder.mul_extension(d_op_mload, d_op_mstore);
        let op_inter_2 = builder.mul_extension(op_inter_1, d_op_call);
        let op_inter_3 = builder.mul_extension(op_inter_2, d_op_ret);
        let op_inter_4 = builder.mul_extension(op_inter_3, op);
        yield_constr.constraint(builder, op_inter_4);
        // when op is 0, is_rw must be zero.
        let op_inter_rw = builder.mul_extension(op_inter_3, is_rw);
        yield_constr.constraint(builder, op_inter_rw);

        // constraint is_write and op. When write, op can be mstore, call and 0; When read, op can be mload, call, ret.
        // call can both write and read, does not need a constraint rule.
        let is_write_inter_1 = builder.mul_extension(d_op_mload, d_op_call);
        let is_write_inter_2 = builder.mul_extension(is_write_inter_1, d_op_ret);
        let is_write_inter_3 = builder.mul_extension(is_write_inter_2, one_m_is_write);
        yield_constr.constraint(builder, is_write_inter_3);
        let is_write_inter_2_1 = builder.mul_extension(op, d_op_mstore);
        let is_write_inter_2_2 = builder.mul_extension(is_write_inter_2_1, d_op_call);
        let is_write_inter_2_3 = builder.mul_extension(is_write_inter_2_2, is_write);
        yield_constr.constraint(builder, is_write_inter_2_3);

        // when op is not 0, filter_looked_for_main need to be enabled.
        let one_m_filter_looked_for_main = builder.sub_extension(one, filter_looked_for_main);
        let constraint_filter_main = builder.mul_extension(op, one_m_filter_looked_for_main);
        yield_constr.constraint(builder, constraint_filter_main);

        // addr'-addr-diff_addr'= 0
        let calculated_diff_addr = builder.sub_extension(nv_addr, addr);
        let constraint_diff_addr = builder.sub_extension(calculated_diff_addr, nv_diff_addr);
        yield_constr.constraint(builder, constraint_diff_addr);
        // for rw_addr_unchanged
        yield_constr.constraint_first_row(builder, rw_addr_unchanged);
        let diff_mul_inv = builder.mul_extension(diff_addr, diff_addr_inv);
        let one_m_unchanged = builder.sub_extension(one, rw_addr_unchanged);
        let constraint_unchanged = builder.sub_extension(one_m_unchanged, diff_mul_inv);
        yield_constr.constraint(builder, constraint_unchanged);

        // for region division: 1. one of four region is selected; 2. binary constraints; 3. diff_addr_cond in different region.
        let sum_rw_prophet = builder.add_extension(is_rw, region_prophet);
        let sum_rw_prophet_poseidon = builder.add_extension(sum_rw_prophet, region_poseidon);
        let sum_rw_prophet_poseidon_ecdsa =
            builder.add_extension(sum_rw_prophet_poseidon, region_ecdsa);
        let constraint_region = builder.sub_extension(sum_rw_prophet_poseidon_ecdsa, one);
        yield_constr.constraint(builder, constraint_region);
        let one_m_rw = builder.sub_extension(one, is_rw);
        let one_m_prophet = builder.sub_extension(one, region_prophet);
        let one_m_poseidon = builder.sub_extension(one, region_poseidon);
        let one_m_ecdsa = builder.sub_extension(one, region_ecdsa);
        let binary_rw = builder.mul_extension(is_rw, one_m_rw);
        let binary_prophet = builder.mul_extension(region_prophet, one_m_prophet);
        let binary_poseidon = builder.mul_extension(region_poseidon, one_m_poseidon);
        let binary_ecdsa = builder.mul_extension(region_ecdsa, one_m_ecdsa);
        yield_constr.constraint(builder, binary_rw);
        yield_constr.constraint(builder, binary_prophet);
        yield_constr.constraint(builder, binary_poseidon);
        yield_constr.constraint(builder, binary_ecdsa);
        let calculated_diff_prophet = builder.sub_extension(p, addr);
        let d_diff_prophet = builder.sub_extension(calculated_diff_prophet, diff_addr_cond);
        let constraint_prophet_cond = builder.mul_extension(region_prophet, d_diff_prophet);
        yield_constr.constraint(builder, constraint_prophet_cond);
        let cell_poseidon = builder.sub_extension(p, span.clone());
        let calculated_diff_poseidon = builder.sub_extension(cell_poseidon, addr);
        let d_diff_poseidon = builder.sub_extension(calculated_diff_poseidon, diff_addr_cond);
        let constraint_poseidon_cond = builder.mul_extension(region_poseidon, d_diff_poseidon);
        yield_constr.constraint(builder, constraint_poseidon_cond);
        let cell_ecdsa = builder.sub_extension(cell_poseidon, span.clone());
        let calculated_diff_ecdsa = builder.sub_extension(cell_ecdsa, addr);
        let d_diff_ecdsa = builder.sub_extension(calculated_diff_ecdsa, diff_addr_cond);
        let constraint_ecdsa_cond = builder.mul_extension(region_ecdsa, d_diff_ecdsa);
        yield_constr.constraint(builder, constraint_ecdsa_cond);

        // for write once:
        // 1. addr doesn't change or increase by 1 when not cross region;
        // 2. when addr not increase, must be read.
        let inc_prophet = builder.sub_extension(nv_region_prophet, region_prophet);
        let one_m_inc_prophet = builder.sub_extension(one, inc_prophet);
        let inc_poseidon = builder.sub_extension(nv_region_poseidon, region_poseidon);
        let one_m_inc_poseidon = builder.sub_extension(one, inc_poseidon);
        let inc_addr = builder.sub_extension(nv_addr, addr);
        let one_m_inc_addr = builder.sub_extension(one, inc_addr);
        let write_once_inter_1 = builder.mul_extension(one_m_rw, one_m_inc_prophet);
        let write_once_inter_2 = builder.mul_extension(write_once_inter_1, one_m_inc_poseidon);
        let write_once_inter_3 = builder.mul_extension(write_once_inter_2, one_m_inc_addr);
        let constraint_write_once_addr_change = builder.mul_extension(write_once_inter_3, inc_addr);
        yield_constr.constraint(builder, constraint_write_once_addr_change);
        let constraint_write_once = builder.mul_extension(write_once_inter_3, nv_is_write);
        yield_constr.constraint(builder, constraint_write_once);

        // read/write constraint:
        // 1. first operation for each addr must be write;
        // 2. next value does not change if it is read.
        yield_constr.constraint_first_row(builder, one_m_is_write);
        let one_m_nv_is_write = builder.sub_extension(one, nv_is_write);
        let constraint_addr_write_for_first = builder.mul_extension(inc_addr, one_m_nv_is_write);
        yield_constr.constraint(builder, constraint_addr_write_for_first);
        let inc_value = builder.sub_extension(nv_value, value);
        let constraint_value_read = builder.mul_extension(one_m_nv_is_write, inc_value);
        yield_constr.constraint(builder, constraint_value_read);
    }

    fn constraint_degree(&self) -> usize {
        5
    }
}

mod tests {
    use crate::config::StarkConfig;
    use crate::memory::MemoryStark;
    use crate::prover::prove;
    use crate::util::trace_rows_to_poly_values;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::fri::reduction_strategies::FriReductionStrategy;
    use plonky2::fri::FriConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use std::ops::{Add, Div, Sub};

    #[test]
    fn test_memory_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = MemoryStark<F, D>;

        let stark = S::default();
        let config = StarkConfig {
            security_bits: 100,
            num_challenges: 2,
            fri_config: FriConfig {
                rate_bits: 2,
                cap_height: 4,
                proof_of_work_bits: 16,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
                num_query_rounds: 84,
            },
        };
        let one = GoldilocksField::ONE;
        let zero = GoldilocksField::ZERO;
        let op_mload = GoldilocksField::from_canonical_u64(2_u64.pow(24));
        let op_mstore = GoldilocksField::from_canonical_u64(2_u64.pow(24));
        let op_call = GoldilocksField::from_canonical_u64(2_u64.pow(27));
        let op_ret = GoldilocksField::from_canonical_u64(2_u64.pow(26));

        let rw_row0: [F; 15] = [
            one,
            GoldilocksField(100),
            GoldilocksField(5),
            op_mstore,
            one,
            GoldilocksField(55),
            zero,
            zero,
            zero,
            zero,
            one,
            zero,
            zero,
            zero,
            zero,
        ];
        let rw_row1: [F; 15] = [
            one,
            GoldilocksField(100),
            GoldilocksField(12),
            op_mload,
            zero,
            GoldilocksField(55),
            zero,
            zero,
            GoldilocksField(7),
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];
        let rw_row2: [F; 15] = [
            one,
            GoldilocksField(100),
            GoldilocksField(17),
            op_mstore,
            one,
            GoldilocksField(300),
            zero,
            zero,
            GoldilocksField(5),
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];
        let rw_row3: [F; 15] = [
            one,
            GoldilocksField(100),
            GoldilocksField(26),
            op_mload,
            zero,
            GoldilocksField(300),
            zero,
            zero,
            GoldilocksField(9),
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];
        let rw_row4: [F; 15] = [
            one,
            GoldilocksField(124),
            GoldilocksField(36),
            op_mstore,
            one,
            GoldilocksField(20000),
            GoldilocksField(24),
            one.div(GoldilocksField(24)),
            zero,
            zero,
            one,
            zero,
            zero,
            zero,
            zero,
        ];
        let rw_row5: [F; 15] = [
            one,
            GoldilocksField(124),
            GoldilocksField(37),
            op_call,
            zero,
            GoldilocksField(20000),
            zero,
            zero,
            one,
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];
        let rw_row6: [F; 15] = [
            one,
            GoldilocksField(124),
            GoldilocksField(50),
            op_ret,
            zero,
            GoldilocksField(20000),
            zero,
            zero,
            GoldilocksField(13),
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];
        let rw_row7: [F; 15] = [
            one,
            GoldilocksField(125),
            GoldilocksField(37),
            op_call,
            one,
            GoldilocksField(30000),
            one,
            one,
            zero,
            zero,
            one,
            zero,
            zero,
            zero,
            zero,
        ];
        let rw_row8: [F; 15] = [
            one,
            GoldilocksField(125),
            GoldilocksField(50),
            op_ret,
            zero,
            GoldilocksField(30000),
            zero,
            zero,
            GoldilocksField(13),
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];
        let rw_row9: [F; 15] = [
            one,
            GoldilocksField(150),
            GoldilocksField(20),
            op_mstore,
            one,
            GoldilocksField(78),
            GoldilocksField(25),
            one.div(GoldilocksField(25)),
            zero,
            zero,
            one,
            zero,
            zero,
            zero,
            zero,
        ];
        let rw_row10: [F; 15] = [
            one,
            GoldilocksField(150),
            GoldilocksField(23),
            op_mload,
            zero,
            GoldilocksField(78),
            zero,
            zero,
            GoldilocksField(3),
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];
        let rw_row11: [F; 15] = [
            one,
            GoldilocksField(200),
            GoldilocksField(8),
            op_mstore,
            one,
            GoldilocksField(99),
            GoldilocksField(50),
            one.div(GoldilocksField(50)),
            zero,
            zero,
            one,
            zero,
            zero,
            zero,
            zero,
        ];
        let rw_row12: [F; 15] = [
            one,
            GoldilocksField(200),
            GoldilocksField(15),
            op_mload,
            zero,
            GoldilocksField(99),
            zero,
            zero,
            GoldilocksField(7),
            zero,
            one,
            one,
            zero,
            zero,
            zero,
        ];

        let p = GoldilocksField::order();
        let span = GoldilocksField::from_canonical_u64(2_u64.pow(32).sub(1));
        let prophet_row0: [F; 15] = [
            zero,
            p.sub(span),
            zero,
            zero,
            one,
            GoldilocksField(123),
            p.sub(span.add(GoldilocksField(200))),
            zero,
            zero,
            p.sub(p.sub(span)),
            zero,
            zero,
            one,
            zero,
            zero,
        ];
        let prophet_row1: [F; 15] = [
            zero,
            p.sub(span),
            GoldilocksField(7),
            op_mload,
            zero,
            GoldilocksField(123),
            zero,
            zero,
            zero,
            p.sub(p.sub(span)),
            one,
            zero,
            one,
            zero,
            zero,
        ];
        let prophet_row2: [F; 15] = [
            zero,
            p.sub(span),
            GoldilocksField(9),
            op_mload,
            zero,
            GoldilocksField(123),
            zero,
            zero,
            zero,
            p.sub(p.sub(span)),
            one,
            zero,
            one,
            zero,
            zero,
        ];
        let prophet_row3: [F; 15] = [
            zero,
            p.sub(span.sub(one)),
            zero,
            zero,
            one,
            GoldilocksField(456),
            one,
            zero,
            zero,
            p.sub(p.sub(span.sub(one))),
            zero,
            zero,
            one,
            zero,
            zero,
        ];
        let prophet_row4: [F; 15] = [
            zero,
            p.sub(span.sub(one)),
            GoldilocksField(27),
            op_mload,
            zero,
            GoldilocksField(456),
            zero,
            zero,
            zero,
            p.sub(p.sub(span.sub(one))),
            one,
            zero,
            one,
            zero,
            zero,
        ];
        let trace_rows = vec![
            rw_row0,
            rw_row1,
            rw_row2,
            rw_row3,
            rw_row4,
            rw_row5,
            rw_row6,
            rw_row7,
            rw_row8,
            rw_row9,
            rw_row10,
            rw_row11,
            rw_row12,
            prophet_row0,
            prophet_row1,
            prophet_row2,
            prophet_row3,
            prophet_row4,
        ];

        let trace = trace_rows_to_poly_values(trace_rows);
        // let proof = prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default())?;
        //
        // verify_stark_proof(stark, proof, &config)
    }
}
