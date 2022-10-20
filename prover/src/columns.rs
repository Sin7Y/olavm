// we use 8 limbs, each handle 32 bit.
pub const LIMB_BITS: usize = 32;
pub const N_LIMBS: usize = 8;

// columns bumber of trace.
pub const INST_COL: usize = 0;
pub const CLK_COL: usize = INST_COL + 1;
pub const PC_COL: usize = CLK_COL + 1;
pub const FLAG_COL: usize = PC_COL + 1;

// start column of register array.
pub const REG_COL: usize = FLAG_COL + 1;

pub const NUM_ARITH_COLS: usize = REG_COL + 3 * N_LIMBS;

const fn gen_reg_cols<const N: usize>(start: usize) -> [usize; N] {
    let mut cols = [0usize; N];
    let mut i = 0;
    while i < N {
        cols[i] = REG_COL + start + i;
        i += 1;
    }
    cols
}

// ADD register column
pub const ADD_OUTPUT_COLS: [usize; N_LIMBS] = gen_reg_cols<N_LIMBS>(0);
pub const ADD_INPUT0_COLS: [usize; N_LIMBS] = gen_reg_cols<N_LIMBS>(N_LIMBS);
pub const ADD_INPUT1_COLS: [usize; N_LIMBS] = gen_reg_cols<N_LIMBS>(2 * N_LIMBS);

// usize type of instruction.
pub const ADD_id: usize = 0;