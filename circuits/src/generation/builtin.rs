use core::program::instruction::Opcode;
use core::trace::trace::{BitwiseCombinedRow, CmpRow, RangeCheckRow};
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::Challenger;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::transpose;

use crate::builtins::bitwise::columns as bitwise;
use crate::builtins::cmp::columns as cmp;
use crate::builtins::cmp::columns::{
    COL_CMP_ABS_DIFF, COL_CMP_ABS_DIFF_INV, COL_CMP_FILTER_LOOKING_RC, COL_CMP_GTE, COL_CMP_OP0,
    COL_CMP_OP1,
};
use crate::builtins::rangecheck::columns as rangecheck;
use crate::stark::lookup::permuted_cols;

// add by xb 2023-1-5
// case 1:
// looking_table:
// <0,1,2,3,4,5>
// looked_table:
// <0,1,2,3,4,5,6,7>
// Extend:
//      looking_table: <0,1,2,3,4,5,5,5>
//      looked_table: <0,1,2,3,4,5,6,7>

// case 2:
// looking_table:
// <0,1,2,3,4,5>
// looked_table:
// <0,1,2,3,4,5,6,7,8,9>
// Extend:
//      looking_table: <0,1,2,3,4,5,5,5,5,5,5,5,5,5,5,5>
//      looked_table: <0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15>
pub fn generate_bitwise_trace<F: RichField>(
    cells: &[BitwiseCombinedRow],
) -> ([Vec<F>; bitwise::COL_NUM_BITWISE], F) {
    if cells.is_empty() {
        let trace: Vec<Vec<F>> = vec![vec![F::default(); 2]; bitwise::COL_NUM_BITWISE];
        let trace_row_vecs = trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
            panic!(
                "Expected a Vec of length {} but it was {}",
                bitwise::COL_NUM_BITWISE,
                v.len()
            )
        });
        return (trace_row_vecs, F::default());
    }

    // Ensure the max rows number.
    let trace_len = cells.len();
    let max_trace_len = trace_len
        .max(bitwise::RANGE_CHECK_U8_SIZE)
        .max(bitwise::BITWISE_U8_SIZE);

    let ext_trace_len = if !max_trace_len.is_power_of_two() {
        max_trace_len.next_power_of_two()
    } else {
        max_trace_len
    };

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; ext_trace_len]; bitwise::COL_NUM_BITWISE];
    for (i, c) in cells.iter().enumerate() {
        trace[bitwise::FILTER][i] = F::from_canonical_usize(1);
        trace[bitwise::TAG][i] = F::from_canonical_u32(c.opcode);
        trace[bitwise::OP0][i] = F::from_canonical_u64(c.op0.to_canonical_u64());
        trace[bitwise::OP1][i] = F::from_canonical_u64(c.op1.to_canonical_u64());
        trace[bitwise::RES][i] = F::from_canonical_u64(c.res.to_canonical_u64());

        trace[bitwise::OP0_LIMBS.start][i] = F::from_canonical_u64(c.op0_0.to_canonical_u64());
        trace[bitwise::OP0_LIMBS.start + 1][i] = F::from_canonical_u64(c.op0_1.to_canonical_u64());
        trace[bitwise::OP0_LIMBS.start + 2][i] = F::from_canonical_u64(c.op0_2.to_canonical_u64());
        trace[bitwise::OP0_LIMBS.end][i] = F::from_canonical_u64(c.op0_3.to_canonical_u64());

        trace[bitwise::OP1_LIMBS.start][i] = F::from_canonical_u64(c.op1_0.to_canonical_u64());
        trace[bitwise::OP1_LIMBS.start + 1][i] = F::from_canonical_u64(c.op1_1.to_canonical_u64());
        trace[bitwise::OP1_LIMBS.start + 2][i] = F::from_canonical_u64(c.op1_2.to_canonical_u64());
        trace[bitwise::OP1_LIMBS.end][i] = F::from_canonical_u64(c.op1_3.to_canonical_u64());

        trace[bitwise::RES_LIMBS.start][i] = F::from_canonical_u64(c.res_0.to_canonical_u64());
        trace[bitwise::RES_LIMBS.start + 1][i] = F::from_canonical_u64(c.res_1.to_canonical_u64());
        trace[bitwise::RES_LIMBS.start + 2][i] = F::from_canonical_u64(c.res_2.to_canonical_u64());
        trace[bitwise::RES_LIMBS.end][i] = F::from_canonical_u64(c.res_3.to_canonical_u64());
    }

    // add fix bitwise info
    // for 2^8 case, the row is 2^15 + 2^7
    // fixed at 2023-1-16, for 2^8 case, row number is 2^16
    let mut index = 0;
    for op0 in 0..bitwise::RANGE_CHECK_U8_SIZE {
        // add fix rangecheck info
        trace[bitwise::FIX_RANGE_CHECK_U8][op0] = F::from_canonical_usize(op0);

        //for op1 in op0..bitwise::RANGE_CHECK_U8_SIZE {
        for op1 in 0..bitwise::RANGE_CHECK_U8_SIZE {
            // exe the AND OPE ration
            let res_and = op0 & op1;
            trace[bitwise::FIX_BITWSIE_OP0][index] = F::from_canonical_usize(op0);
            trace[bitwise::FIX_BITWSIE_OP1][index] = F::from_canonical_usize(op1);
            trace[bitwise::FIX_BITWSIE_RES][index] = F::from_canonical_usize(res_and);
            trace[bitwise::FIX_TAG][index] = F::from_canonical_u64(1_u64 << Opcode::AND as u8);

            let res_or = op0 | op1;
            trace[bitwise::FIX_BITWSIE_OP0][bitwise::BITWISE_U8_SIZE_PER + index] =
                F::from_canonical_usize(op0);
            trace[bitwise::FIX_BITWSIE_OP1][bitwise::BITWISE_U8_SIZE_PER + index] =
                F::from_canonical_usize(op1);
            trace[bitwise::FIX_BITWSIE_RES][bitwise::BITWISE_U8_SIZE_PER + index] =
                F::from_canonical_usize(res_or);
            trace[bitwise::FIX_TAG][bitwise::BITWISE_U8_SIZE_PER + index] =
                F::from_canonical_u64(1_u64 << Opcode::OR as u8);

            let res_xor = op0 ^ op1;
            trace[bitwise::FIX_BITWSIE_OP0][bitwise::BITWISE_U8_SIZE_PER * 2 + index] =
                F::from_canonical_usize(op0);
            trace[bitwise::FIX_BITWSIE_OP1][bitwise::BITWISE_U8_SIZE_PER * 2 + index] =
                F::from_canonical_usize(op1);
            trace[bitwise::FIX_BITWSIE_RES][bitwise::BITWISE_U8_SIZE_PER * 2 + index] =
                F::from_canonical_usize(res_xor);
            trace[bitwise::FIX_TAG][bitwise::BITWISE_U8_SIZE_PER * 2 + index] =
                F::from_canonical_u64(1_u64 << Opcode::XOR as u8);

            index += 1;
        }
    }

    // TODO: We should choose proper columns for oracle.
    let mut challenger =
        Challenger::<F, <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher>::new();
    for i in 0..bitwise::OP0_LIMBS.len() {
        challenger.observe_elements(&trace[bitwise::OP0_LIMBS.start + i]);
    }
    for i in 0..bitwise::OP1_LIMBS.len() {
        challenger.observe_elements(&trace[bitwise::OP1_LIMBS.start + i]);
    }
    for i in 0..bitwise::RES_LIMBS.len() {
        challenger.observe_elements(&trace[bitwise::RES_LIMBS.start + i]);
    }
    let beta = challenger.get_challenge();

    for i in 0..trace[0].len() {
        trace[bitwise::COMPRESS_LIMBS.start][i] = trace[bitwise::TAG][i]
            + trace[bitwise::OP0_LIMBS.start][i] * beta
            + trace[bitwise::OP1_LIMBS.start][i] * beta * beta
            + trace[bitwise::RES_LIMBS.start][i] * beta * beta * beta;

        trace[bitwise::COMPRESS_LIMBS.start + 1][i] = trace[bitwise::TAG][i]
            + trace[bitwise::OP0_LIMBS.start + 1][i] * beta
            + trace[bitwise::OP1_LIMBS.start + 1][i] * beta * beta
            + trace[bitwise::RES_LIMBS.start + 1][i] * beta * beta * beta;

        trace[bitwise::COMPRESS_LIMBS.start + 2][i] = trace[bitwise::TAG][i]
            + trace[bitwise::OP0_LIMBS.start + 2][i] * beta
            + trace[bitwise::OP1_LIMBS.start + 2][i] * beta * beta
            + trace[bitwise::RES_LIMBS.start + 2][i] * beta * beta * beta;

        trace[bitwise::COMPRESS_LIMBS.start + 3][i] = trace[bitwise::TAG][i]
            + trace[bitwise::OP0_LIMBS.start + 3][i] * beta
            + trace[bitwise::OP1_LIMBS.start + 3][i] * beta * beta
            + trace[bitwise::RES_LIMBS.start + 3][i] * beta * beta * beta;

        trace[bitwise::FIX_COMPRESS][i] = trace[bitwise::FIX_TAG][i]
            + trace[bitwise::FIX_BITWSIE_OP0][i] * beta
            + trace[bitwise::FIX_BITWSIE_OP1][i] * beta * beta
            + trace[bitwise::FIX_BITWSIE_RES][i] * beta * beta * beta;
    }

    // add the permutation information
    for i in 0..4 {
        // permuted for rangecheck
        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace[bitwise::OP0_LIMBS.start + i],
            &trace[bitwise::FIX_RANGE_CHECK_U8],
        );

        trace[bitwise::OP0_LIMBS_PERMUTED.start + i] = permuted_inputs;
        trace[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + i] = permuted_table;

        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace[bitwise::OP1_LIMBS.start + i],
            &trace[bitwise::FIX_RANGE_CHECK_U8],
        );

        trace[bitwise::OP1_LIMBS_PERMUTED.start + i] = permuted_inputs;
        trace[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 4 + i] = permuted_table;

        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace[bitwise::RES_LIMBS.start + i],
            &trace[bitwise::FIX_RANGE_CHECK_U8],
        );

        trace[bitwise::RES_LIMBS_PERMUTED.start + i] = permuted_inputs;
        trace[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 8 + i] = permuted_table;

        // permutation for bitwise
        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace[bitwise::COMPRESS_LIMBS.start + i],
            &trace[bitwise::FIX_COMPRESS],
        );

        trace[bitwise::COMPRESS_PERMUTED.start + i] = permuted_inputs;
        trace[bitwise::FIX_COMPRESS_PERMUTED.start + i] = permuted_table;
    }

    let trace_row_vecs = trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            bitwise::COL_NUM_BITWISE,
            v.len()
        )
    });

    (trace_row_vecs, beta)
}

pub fn vec_to_ary_bitwise<F: RichField>(input: Vec<F>) -> [F; bitwise::COL_NUM_BITWISE] {
    let mut ary = [F::ZERO; bitwise::COL_NUM_BITWISE];

    ary[bitwise::FILTER] = input[bitwise::FILTER];
    ary[bitwise::TAG] = input[bitwise::TAG];
    ary[bitwise::OP0] = input[bitwise::OP0];
    ary[bitwise::OP1] = input[bitwise::OP1];
    ary[bitwise::RES] = input[bitwise::RES];
    ary[bitwise::OP0_LIMBS.start] = input[bitwise::OP0_LIMBS.start];
    ary[bitwise::OP0_LIMBS.start + 1] = input[bitwise::OP0_LIMBS.start + 1];
    ary[bitwise::OP0_LIMBS.start + 2] = input[bitwise::OP0_LIMBS.start + 2];
    ary[bitwise::OP0_LIMBS.start + 3] = input[bitwise::OP0_LIMBS.start + 3];
    ary[bitwise::OP1_LIMBS.start] = input[bitwise::OP1_LIMBS.start];
    ary[bitwise::OP1_LIMBS.start + 1] = input[bitwise::OP1_LIMBS.start + 1];
    ary[bitwise::OP1_LIMBS.start + 2] = input[bitwise::OP1_LIMBS.start + 2];
    ary[bitwise::OP1_LIMBS.start + 3] = input[bitwise::OP1_LIMBS.start + 3];
    ary[bitwise::RES_LIMBS.start] = input[bitwise::RES_LIMBS.start];
    ary[bitwise::RES_LIMBS.start + 1] = input[bitwise::RES_LIMBS.start + 1];
    ary[bitwise::RES_LIMBS.start + 2] = input[bitwise::RES_LIMBS.start + 2];
    ary[bitwise::RES_LIMBS.start + 3] = input[bitwise::RES_LIMBS.start + 3];
    ary[bitwise::OP0_LIMBS_PERMUTED.start] = input[bitwise::OP0_LIMBS_PERMUTED.start];
    ary[bitwise::OP0_LIMBS_PERMUTED.start + 1] = input[bitwise::OP0_LIMBS_PERMUTED.start + 1];
    ary[bitwise::OP0_LIMBS_PERMUTED.start + 2] = input[bitwise::OP0_LIMBS_PERMUTED.start + 2];
    ary[bitwise::OP0_LIMBS_PERMUTED.start + 3] = input[bitwise::OP0_LIMBS_PERMUTED.start + 3];
    ary[bitwise::OP1_LIMBS_PERMUTED.start] = input[bitwise::OP1_LIMBS_PERMUTED.start];
    ary[bitwise::OP1_LIMBS_PERMUTED.start + 1] = input[bitwise::OP1_LIMBS_PERMUTED.start + 1];
    ary[bitwise::OP1_LIMBS_PERMUTED.start + 2] = input[bitwise::OP1_LIMBS_PERMUTED.start + 2];
    ary[bitwise::OP1_LIMBS_PERMUTED.start + 3] = input[bitwise::OP1_LIMBS_PERMUTED.start + 3];
    ary[bitwise::RES_LIMBS_PERMUTED.start] = input[bitwise::RES_LIMBS_PERMUTED.start];
    ary[bitwise::RES_LIMBS_PERMUTED.start + 1] = input[bitwise::RES_LIMBS_PERMUTED.start + 1];
    ary[bitwise::RES_LIMBS_PERMUTED.start + 2] = input[bitwise::RES_LIMBS_PERMUTED.start + 2];
    ary[bitwise::RES_LIMBS_PERMUTED.start + 3] = input[bitwise::RES_LIMBS_PERMUTED.start + 3];
    ary[bitwise::COMPRESS_LIMBS.start] = input[bitwise::COMPRESS_LIMBS.start];
    ary[bitwise::COMPRESS_LIMBS.start + 1] = input[bitwise::COMPRESS_LIMBS.start + 1];
    ary[bitwise::COMPRESS_LIMBS.start + 2] = input[bitwise::COMPRESS_LIMBS.start + 2];
    ary[bitwise::COMPRESS_LIMBS.start + 3] = input[bitwise::COMPRESS_LIMBS.start + 3];
    ary[bitwise::COMPRESS_PERMUTED.start] = input[bitwise::COMPRESS_PERMUTED.start];
    ary[bitwise::COMPRESS_PERMUTED.start + 1] = input[bitwise::COMPRESS_PERMUTED.start + 1];
    ary[bitwise::COMPRESS_PERMUTED.start + 2] = input[bitwise::COMPRESS_PERMUTED.start + 2];
    ary[bitwise::COMPRESS_PERMUTED.start + 3] = input[bitwise::COMPRESS_PERMUTED.start + 3];
    ary[bitwise::FIX_RANGE_CHECK_U8] = input[bitwise::FIX_RANGE_CHECK_U8];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 1] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 1];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 2] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 2];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 3] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 3];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 4] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 4];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 5] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 5];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 6] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 6];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 7] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 7];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 8] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 8];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 9] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 9];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 10] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 10];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 11] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 11];
    ary[bitwise::FIX_TAG] = input[bitwise::FIX_TAG];
    ary[bitwise::FIX_BITWSIE_OP0] = input[bitwise::FIX_BITWSIE_OP0];
    ary[bitwise::FIX_BITWSIE_OP1] = input[bitwise::FIX_BITWSIE_OP1];
    ary[bitwise::FIX_BITWSIE_RES] = input[bitwise::FIX_BITWSIE_RES];
    ary[bitwise::FIX_COMPRESS] = input[bitwise::FIX_COMPRESS];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start] = input[bitwise::FIX_COMPRESS_PERMUTED.start];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start + 1] = input[bitwise::FIX_COMPRESS_PERMUTED.start + 1];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start + 2] = input[bitwise::FIX_COMPRESS_PERMUTED.start + 2];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start + 3] = input[bitwise::FIX_COMPRESS_PERMUTED.start + 3];

    ary
}

pub fn generate_cmp_trace<F: RichField>(cells: &[CmpRow]) -> [Vec<F>; cmp::COL_NUM_CMP] {
    let trace_len = cells.len();
    let ext_trace_len = if !trace_len.is_power_of_two() || trace_len < 2 {
        if trace_len < 2 {
            2
        } else {
            trace_len.next_power_of_two()
        }
    } else {
        trace_len
    };

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; ext_trace_len]; cmp::COL_NUM_CMP];
    for (i, c) in cells.iter().enumerate() {
        trace[COL_CMP_OP0][i] = F::from_canonical_u64(c.op0.to_canonical_u64());
        trace[COL_CMP_OP1][i] = F::from_canonical_u64(c.op1.to_canonical_u64());
        trace[COL_CMP_GTE][i] = F::from_canonical_u64(c.gte.to_canonical_u64());
        trace[COL_CMP_ABS_DIFF][i] = F::from_canonical_u64(c.abs_diff.to_canonical_u64());
        trace[COL_CMP_ABS_DIFF_INV][i] = F::from_canonical_u64(c.abs_diff_inv.to_canonical_u64());
        trace[COL_CMP_FILTER_LOOKING_RC][i] =
            F::from_canonical_u64(c.filter_looking_rc.to_canonical_u64());
    }

    // Pad trace to power of two.
    if trace_len != ext_trace_len {
        for i in trace_len..ext_trace_len {
            trace[COL_CMP_OP0][i] = F::ONE;
            trace[COL_CMP_GTE][i] = F::ONE;
            trace[COL_CMP_ABS_DIFF][i] = F::ONE;
            trace[COL_CMP_ABS_DIFF_INV][i] = F::ONE;
        }
    }
    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            bitwise::COL_NUM_BITWISE,
            v.len()
        )
    })
}

pub fn generate_rc_trace<F: RichField>(
    cells: &[RangeCheckRow],
) -> [Vec<F>; rangecheck::COL_NUM_RC] {
    if cells.is_empty() {
        let trace: Vec<Vec<F>> = vec![vec![F::default(); 2]; bitwise::COL_NUM_BITWISE];
        return trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
            panic!(
                "Expected a Vec of length {} but it was {}",
                bitwise::COL_NUM_BITWISE,
                v.len()
            )
        });
    }

    let trace_len = cells.len();
    let max_trace_len = trace_len.max(rangecheck::RANGE_CHECK_U16_SIZE);
    let ext_trace_len = if !max_trace_len.is_power_of_two() {
        max_trace_len.next_power_of_two()
    } else {
        max_trace_len
    };

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; ext_trace_len]; rangecheck::COL_NUM_RC];
    for (i, c) in cells.iter().enumerate() {
        trace[rangecheck::CPU_FILTER][i] =
            F::from_canonical_u64(c.filter_looked_for_cpu.to_canonical_u64());
        trace[rangecheck::MEMORY_FILTER][i] =
            F::from_canonical_u64(c.filter_looked_for_memory.to_canonical_u64());
        trace[rangecheck::CMP_FILTER][i] =
            F::from_canonical_u64(c.filter_looked_for_comparison.to_canonical_u64());
        trace[rangecheck::VAL][i] = F::from_canonical_u64(c.val.to_canonical_u64());
        trace[rangecheck::LIMB_LO][i] = F::from_canonical_u64(c.limb_lo.to_canonical_u64());
        trace[rangecheck::LIMB_HI][i] = F::from_canonical_u64(c.limb_hi.to_canonical_u64());
    }

    // add fix rangecheck info
    trace[rangecheck::FIX_RANGE_CHECK_U16] = (0..rangecheck::RANGE_CHECK_U16_SIZE)
        .map(|i| F::from_canonical_usize(i))
        .collect();

    let (permuted_inputs, permuted_table) = permuted_cols(
        &trace[rangecheck::LIMB_LO],
        &trace[rangecheck::FIX_RANGE_CHECK_U16],
    );

    trace[rangecheck::LIMB_LO_PERMUTED] = permuted_inputs;
    trace[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED_LO] = permuted_table;

    let (permuted_inputs, permuted_table) = permuted_cols(
        &trace[rangecheck::LIMB_HI],
        &trace[rangecheck::FIX_RANGE_CHECK_U16],
    );

    trace[rangecheck::LIMB_HI_PERMUTED] = permuted_inputs;
    trace[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED_HI] = permuted_table;

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            bitwise::COL_NUM_BITWISE,
            v.len()
        )
    })
}

pub fn vec_to_ary_rc<F: RichField>(input: Vec<F>) -> [F; rangecheck::COL_NUM_RC] {
    let mut ary = [F::ZERO; rangecheck::COL_NUM_RC];

    ary[rangecheck::CPU_FILTER] = input[rangecheck::CPU_FILTER];
    ary[rangecheck::MEMORY_FILTER] = input[rangecheck::MEMORY_FILTER];
    ary[rangecheck::CMP_FILTER] = input[rangecheck::CMP_FILTER];
    ary[rangecheck::VAL] = input[rangecheck::VAL];
    ary[rangecheck::LIMB_LO] = input[rangecheck::LIMB_LO];
    ary[rangecheck::LIMB_HI] = input[rangecheck::LIMB_HI];
    ary[rangecheck::LIMB_LO_PERMUTED] = input[rangecheck::LIMB_LO_PERMUTED];
    ary[rangecheck::LIMB_HI_PERMUTED] = input[rangecheck::LIMB_HI_PERMUTED];
    ary[rangecheck::FIX_RANGE_CHECK_U16] = input[rangecheck::FIX_RANGE_CHECK_U16];
    ary[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED_LO] =
        input[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED_LO];
    ary[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED_HI] =
        input[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED_HI];
    ary
}
