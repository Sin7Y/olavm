use core::vm::opcodes::OlaOpcode;
use std::ops::Sub;

use plonky2::field::types::Field;

use crate::stark::cross_table_lookup::Column;
use {
    super::columns::*,
    crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::stark::stark::Stark,
    crate::stark::vars::{StarkEvaluationTargets, StarkEvaluationVars},
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
    Column::single(COL_MEM_FILTER_LOOKING_RC_COND)
}

pub fn ctl_data_mem_sort_rc<F: Field>() -> Vec<Column<F>> {
    vec![Column::single(COL_MEM_RC_VALUE)]
}

pub fn ctl_filter_mem_sort_rc<F: Field>() -> Column<F> {
    Column::single(COL_MEM_FILTER_LOOKING_RC)
}

pub fn ctl_data<F: Field>() -> Vec<Column<F>> {
    let cols: Vec<_> = Column::singles([
        COL_MEM_TX_IDX,
        COL_MEM_ENV_IDX,
        COL_MEM_CLK,
        COL_MEM_OP,
        COL_MEM_ADDR,
        COL_MEM_VALUE,
    ])
    .collect();
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
        // tx_idx not change or increase by one
        yield_constr.constraint_transition(
            (nv[COL_MEM_TX_IDX] - lv[COL_MEM_TX_IDX])
                * (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX]),
        );
        // in same tx, env_idx not change or increase by one
        yield_constr.constraint_transition(
            (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX])
                * (nv[COL_MEM_ENV_IDX] - lv[COL_MEM_ENV_IDX])
                * (P::ONES - nv[COL_MEM_ENV_IDX] + lv[COL_MEM_ENV_IDX]),
        );

        let p = P::ZEROS;
        let span = P::Scalar::from_canonical_u64(2_u64.pow(32).sub(1));

        let is_rw = lv[COL_MEM_IS_RW];
        let region_prophet = lv[COL_MEM_REGION_PROPHET];
        let nv_region_prophet = nv[COL_MEM_REGION_PROPHET];
        let region_heap = lv[COL_MEM_REGION_HEAP];
        let nv_region_heap = nv[COL_MEM_REGION_HEAP];
        let region_stack = P::ONES - lv[COL_MEM_REGION_HEAP] - lv[COL_MEM_REGION_PROPHET];
        let nv_region_stack = P::ONES - nv[COL_MEM_REGION_HEAP] - nv[COL_MEM_REGION_PROPHET];
        let is_write = lv[COL_MEM_IS_WRITE];
        let nv_is_write = nv[COL_MEM_IS_WRITE];
        let filter_looked_for_main = lv[COL_MEM_FILTER_LOOKED_FOR_MAIN];
        let addr = lv[COL_MEM_ADDR];
        let nv_diff_addr_inv = nv[COL_MEM_DIFF_ADDR_INV];
        let nv_addr = nv[COL_MEM_ADDR];
        let diff_addr = lv[COL_MEM_DIFF_ADDR];
        let nv_diff_addr = nv[COL_MEM_DIFF_ADDR];
        let rw_addr_unchanged = lv[COL_MEM_RW_ADDR_UNCHANGED];
        let nv_rw_addr_unchanged = nv[COL_MEM_RW_ADDR_UNCHANGED];
        let diff_addr_cond = lv[COL_MEM_DIFF_ADDR_COND];
        let value = lv[COL_MEM_VALUE];
        let nv_value = lv[COL_MEM_VALUE];
        let diff_clk = lv[COL_MEM_DIFF_CLK];
        let rc_value = lv[COL_MEM_RC_VALUE];
        let filter_looking_rc = lv[COL_MEM_FILTER_LOOKING_RC];
        let lv_filter_looking_rc_cond = lv[COL_MEM_FILTER_LOOKING_RC_COND];

        let op_mload = P::Scalar::from_canonical_u64(OlaOpcode::MLOAD.binary_bit_mask());
        let op_mstore = P::Scalar::from_canonical_u64(OlaOpcode::MSTORE.binary_bit_mask());
        let op_call = P::Scalar::from_canonical_u64(OlaOpcode::CALL.binary_bit_mask());
        let op_ret = P::Scalar::from_canonical_u64(OlaOpcode::RET.binary_bit_mask());
        let op_tload = P::Scalar::from_canonical_u64(OlaOpcode::TLOAD.binary_bit_mask());
        let op_tstore = P::Scalar::from_canonical_u64(OlaOpcode::TSTORE.binary_bit_mask());
        let op_sc_call = P::Scalar::from_canonical_u64(OlaOpcode::SCCALL.binary_bit_mask());
        let op_prophet = P::ZEROS;

        // constraint opcode and selector matches, selector is binary and only one is
        // selected.
        yield_constr.constraint((lv[COL_MEM_OP] - op_mload) * lv[COL_MEM_S_MLOAD]);
        yield_constr.constraint((lv[COL_MEM_OP] - op_mstore) * lv[COL_MEM_S_MSTORE]);
        yield_constr.constraint((lv[COL_MEM_OP] - op_call) * lv[COL_MEM_S_CALL]);
        yield_constr.constraint((lv[COL_MEM_OP] - op_ret) * lv[COL_MEM_S_RET]);
        yield_constr.constraint((lv[COL_MEM_OP] - op_tload) * lv[COL_MEM_S_TLOAD]);
        yield_constr.constraint((lv[COL_MEM_OP] - op_tstore) * lv[COL_MEM_S_TSTORE]);
        yield_constr.constraint((lv[COL_MEM_OP] - op_sc_call) * lv[COL_MEM_S_SCCALL]);
        yield_constr.constraint((lv[COL_MEM_OP] - op_prophet) * lv[COL_MEM_S_PROPHET]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_MLOAD]) * lv[COL_MEM_S_MLOAD]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_MSTORE]) * lv[COL_MEM_S_MSTORE]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_CALL]) * lv[COL_MEM_S_CALL]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_RET]) * lv[COL_MEM_S_RET]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_TLOAD]) * lv[COL_MEM_S_TLOAD]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_TSTORE]) * lv[COL_MEM_S_TSTORE]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_SCCALL]) * lv[COL_MEM_S_SCCALL]);
        yield_constr.constraint((P::ONES - lv[COL_MEM_S_PROPHET]) * lv[COL_MEM_S_PROPHET]);
        yield_constr.constraint(
            P::ONES
                - lv[COL_MEM_S_MLOAD]
                - lv[COL_MEM_S_MSTORE]
                - lv[COL_MEM_S_CALL]
                - lv[COL_MEM_S_RET]
                - lv[COL_MEM_S_TLOAD]
                - lv[COL_MEM_S_TSTORE]
                - lv[COL_MEM_S_SCCALL]
                - lv[COL_MEM_S_PROPHET],
        );

        // constraint is_rw region
        yield_constr.constraint(is_rw * (P::ONES - is_rw));
        yield_constr.constraint(lv[COL_MEM_IS_RW] * lv[COL_MEM_S_PROPHET]);
        yield_constr.constraint(
            (P::ONES - lv[COL_MEM_IS_RW]) * (P::ONES - lv[COL_MEM_S_PROPHET] - lv[COL_MEM_S_MLOAD]),
        );
        // constraint is_write
        yield_constr.constraint(
            lv[COL_MEM_IS_WRITE]
                * (P::ONES
                    - lv[COL_MEM_S_MSTORE]
                    - lv[COL_MEM_S_CALL]
                    - lv[COL_MEM_S_TLOAD]
                    - lv[COL_MEM_S_PROPHET]),
        );
        yield_constr.constraint(
            (P::ONES - lv[COL_MEM_IS_WRITE])
                * (P::ONES
                    - lv[COL_MEM_S_MLOAD]
                    - lv[COL_MEM_S_CALL]
                    - lv[COL_MEM_S_RET]
                    - lv[COL_MEM_S_TSTORE]
                    - lv[COL_MEM_S_SCCALL]),
        );

        // region is one of stack, heap and prophet
        yield_constr.constraint(P::ONES - region_stack - region_heap - region_prophet);
        yield_constr.constraint(region_stack * (P::ONES - region_stack));
        yield_constr.constraint(region_heap * (P::ONES - region_heap));
        yield_constr.constraint(region_prophet * (P::ONES - region_prophet));
        // make sure region addr is right
        yield_constr.constraint(region_prophet * (p - addr - diff_addr_cond));
        yield_constr.constraint(region_heap * (p - span - addr - diff_addr_cond));

        // if not prophet write, must be looked up.
        yield_constr
            .constraint((P::ONES - lv[COL_MEM_S_PROPHET]) * (P::ONES - filter_looked_for_main));

        // addr'-addr-diff_addr'= 0
        yield_constr.constraint_transition(
            (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX])
                * (P::ONES - nv[COL_MEM_ENV_IDX] + lv[COL_MEM_ENV_IDX])
                * (nv_region_heap - region_heap - P::ONES)
                * (nv_addr - addr - nv_diff_addr),
        );
        // constaint rw_addr_unchanged, for stack and heap
        yield_constr.constraint_transition(
            (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX])
                * (P::ONES - nv[COL_MEM_ENV_IDX] + lv[COL_MEM_ENV_IDX])
                * region_stack
                * nv_region_stack
                * (P::ONES - nv_rw_addr_unchanged - nv_diff_addr * nv_diff_addr_inv),
        );
        yield_constr.constraint_transition(
            (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX])
                * (P::ONES - nv[COL_MEM_ENV_IDX] + lv[COL_MEM_ENV_IDX])
                * region_heap
                * nv_region_heap
                * (P::ONES - nv_rw_addr_unchanged - nv_diff_addr * nv_diff_addr_inv),
        );

        // for write once:
        // 1. addr doesn't change or increase by 1 in prophet region;
        // 2. when addr not increase, must be read.
        yield_constr.constraint(
            region_prophet * nv_region_prophet * (nv_addr - addr) * (nv_addr - addr - P::ONES),
        );
        yield_constr.constraint(
            region_prophet * nv_region_prophet * (nv_addr - addr - P::ONES) * nv_is_write,
        );

        // (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX])
        // * (nv[COL_MEM_ENV_IDX] - lv[COL_MEM_ENV_IDX]

        // read/write constraint:
        // 1. first operation for each addr must be write;
        // 2. next value does not change if it is read.
        yield_constr.constraint_first_row(P::ONES - is_write);
        yield_constr.constraint(
            (nv[COL_MEM_TX_IDX] - lv[COL_MEM_TX_IDX])
                * (nv[COL_MEM_ENV_IDX] - lv[COL_MEM_ENV_IDX])
                * (P::ONES - nv_is_write),
        );
        yield_constr.constraint((nv_addr - addr) * (P::ONES - nv_is_write));
        yield_constr.constraint((P::ONES - nv_is_write) * (nv_value - value));

        // rc_value constraint:
        yield_constr.constraint_transition(
            (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX])
                * (P::ONES - nv[COL_MEM_ENV_IDX] + lv[COL_MEM_ENV_IDX])
                * is_rw
                * (nv_region_heap - region_heap - P::ONES)
                * (rc_value - rw_addr_unchanged * diff_clk)
                * (rc_value - (P::ONES - rw_addr_unchanged) * diff_addr),
        );
        yield_constr.constraint_transition(
            (P::ONES - nv[COL_MEM_TX_IDX] + lv[COL_MEM_TX_IDX])
                * (P::ONES - nv[COL_MEM_ENV_IDX] + lv[COL_MEM_ENV_IDX])
                * is_rw
                * rc_value
                * (nv_region_heap - region_heap - P::ONES)
                * (P::ONES - filter_looking_rc),
        );

        // heap and prophet read, diff_cond must rc
        yield_constr.constraint((P::ONES - lv_filter_looking_rc_cond) * region_heap);
        yield_constr.constraint(
            (P::ONES - lv_filter_looking_rc_cond) * region_prophet * (P::ONES - is_write),
        );
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, NUM_MEM_COLS>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        9
    }
}

mod tests {
    use crate::generation::memory::generate_memory_trace;
    use crate::memory::columns::{get_memory_col_name_map, NUM_MEM_COLS};
    use crate::memory::memory_stark::MemoryStark;
    use crate::stark::constraint_consumer::ConstraintConsumer;
    use crate::stark::stark::Stark;
    use crate::stark::vars::StarkEvaluationVars;
    use crate::test_utils::test_stark_with_asm_path;
    use core::trace::trace::{MemoryTraceCell, Trace};
    use plonky2::{
        field::goldilocks_field::GoldilocksField,
        plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    };
    use std::path::PathBuf;

    #[test]
    fn test_memory_with_program() {
        let program_path = "memory.json";
        test_memory_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_memory_fib_loop() {
        let program_path = "fibo_loop.json";
        test_memory_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_memory_sqrt() {
        let program_path = "sqrt.json";
        test_memory_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_memory_malloc() {
        let program_path = "malloc.json";
        test_memory_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_memory_vote() {
        let program_path = "vote.json";
        test_memory_with_asm_file_name(program_path.to_string());
    }

    #[allow(unused)]
    fn test_memory_with_asm_file_name(file_name: String) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = MemoryStark<F, D>;
        let stark = S::default();

        let get_trace_rows = |trace: Trace| trace.memory;
        let generate_trace = |rows: &[MemoryTraceCell]| generate_memory_trace(rows);
        let eval_packed_generic =
            |vars: StarkEvaluationVars<GoldilocksField, GoldilocksField, NUM_MEM_COLS>,
             constraint_consumer: &mut ConstraintConsumer<GoldilocksField>| {
                stark.eval_packed_generic(vars, constraint_consumer);
            };
        let error_hook = |i: usize,
                          vars: StarkEvaluationVars<
            GoldilocksField,
            GoldilocksField,
            NUM_MEM_COLS,
        >| {
            println!("constraint error in line {}", i);
            let m = get_memory_col_name_map();
            println!("{:>32}\t{:>22}\t{:>22}", "name", "lv", "nv");
            for col in m.keys() {
                let name = m.get(col).unwrap();
                let lv = vars.local_values[*col].0;
                let nv = vars.next_values[*col].0;
                println!("{:>32}\t{:>22}\t{:>22}", name, lv, nv);
            }
        };
        test_stark_with_asm_path(
            program_path.to_string(),
            get_trace_rows,
            generate_trace,
            eval_packed_generic,
            Some(error_hook),
        );
    }
}
