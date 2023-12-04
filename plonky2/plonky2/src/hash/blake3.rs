use itertools::Itertools;
use std::iter;
use std::mem::size_of;

use crate::hash::hash_types::RichField;
use crate::hash::hashing::{PlonkyPermutation, SPONGE_WIDTH};
use crate::plonk::config::Hasher;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::GeneratedValues;
use crate::iop::wire::Wire;
use core::slice;

use blake3;

use super::hash_types::BytesHash;
use plonky2_field::types::{Field, PrimeField64};
use plonky2_field::extension::{Extendable, FieldExtension};

pub const ROUND: usize = 7;
pub const STATE_SIZE: usize = 16;
pub const IV_SIZE: usize = 8;
pub const BLOCK_LEN: usize = 64;
pub const LOOKUP_LIMB_RANGE: usize = 16;
pub const LOOKUP_LIMB_NUMBER: usize = 16;

pub trait Blake3: PrimeField64 {

    const MSG_SCHEDULE: [[usize; STATE_SIZE]; ROUND];
    const IV: [u32; IV_SIZE];

    #[inline]
    fn g(
        state: &mut [Self; STATE_SIZE], a: usize, b: usize, c: usize, d: usize, x_field: Self, y_field: Self) {

        
        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] = Self::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        let x = Self::to_noncanonical_u64(&x_field.to_basefield_array()[0]) as u32;
        let y = Self::to_noncanonical_u64(&y_field.to_basefield_array()[0]) as u32;

        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(x);
        state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(16);
        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(12);
        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(y);
        state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(8);
        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(7);


        for i in 0..STATE_SIZE {
            state[i] = Self::from_canonical_u32(state_tmp[i]);
        }

    }

    #[inline(always)]
    fn round(
        state: &mut [Self; STATE_SIZE], msg: [Self; STATE_SIZE], round: usize) {
        // Select the message schedule based on the round.
        let schedule = Self::MSG_SCHEDULE[round];

        // Mix the columns.
        Self::g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
        Self::g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
        Self::g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
        Self::g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

        // Mix the diagonals.
        Self::g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
        Self::g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
        Self::g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
        Self::g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
    }

    #[inline(always)]
    fn compress_pre(
        cv: &mut [Self; IV_SIZE],
        block_words: [Self; 16],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) -> [Self; 16] {

        let mut state = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            Self::from_canonical_u32(Self::IV[0]),
            Self::from_canonical_u32(Self::IV[1]),
            Self::from_canonical_u32(Self::IV[2]),
            Self::from_canonical_u32(Self::IV[3]),
            Self::from_canonical_u32(counter as u32),
            Self::from_canonical_u32((counter >> 32) as u32),
            Self::from_canonical_u32(block_len as u32),
            Self::from_canonical_u32(flags as u32),
        ];

        Self::round(&mut state, block_words, 0);
        Self::round(&mut state, block_words, 1);
        Self::round(&mut state, block_words, 2);
        Self::round(&mut state, block_words, 3);
        Self::round(&mut state, block_words, 4);
        Self::round(&mut state, block_words, 5);
        Self::round(&mut state, block_words, 6);

        state
    }


    fn compress_in_place(
        cv: &mut [Self; IV_SIZE],
        block: [Self; STATE_SIZE],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) {
        let state = Self::compress_pre(cv, block, block_len, counter, flags);

        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] = Self::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        cv[0] = Self::from_canonical_u32(state_tmp[0] ^ state_tmp[8]);
        cv[1] = Self::from_canonical_u32(state_tmp[1] ^ state_tmp[9]);
        cv[2] = Self::from_canonical_u32(state_tmp[2] ^ state_tmp[10]);
        cv[3] = Self::from_canonical_u32(state_tmp[3] ^ state_tmp[11]);
        cv[4] = Self::from_canonical_u32(state_tmp[4] ^ state_tmp[12]);
        cv[5] = Self::from_canonical_u32(state_tmp[5] ^ state_tmp[13]);
        cv[6] = Self::from_canonical_u32(state_tmp[6] ^ state_tmp[14]);
        cv[7] = Self::from_canonical_u32(state_tmp[7] ^ state_tmp[15]);
    }

    // -------------------------------------- field ------------------------------------
    #[inline]
    fn g_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; STATE_SIZE], a: usize, b: usize, c: usize, d: usize, x_field: F, y_field: F) {

        
        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] = F::BaseField::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        let x = F::BaseField::to_noncanonical_u64(&x_field.to_basefield_array()[0]) as u32;
        let y = F::BaseField::to_noncanonical_u64(&y_field.to_basefield_array()[0]) as u32;

        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(x);
        state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(16);
        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(12);
        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(y);
        state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(8);
        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(7);


        for i in 0..STATE_SIZE {
            state[i] = F::from_canonical_u32(state_tmp[i]);
        }

    }

    #[inline(always)]
    fn round_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; STATE_SIZE], msg: [F; STATE_SIZE], round: usize) {
        // Select the message schedule based on the round.
        let schedule = Self::MSG_SCHEDULE[round];

        // Mix the columns.
        Self::g_field(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
        Self::g_field(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
        Self::g_field(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
        Self::g_field(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

        // Mix the diagonals.
        Self::g_field(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
        Self::g_field(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
        Self::g_field(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
        Self::g_field(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
    }

    #[inline(always)]
    fn compress_pre_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        cv: &mut [F; IV_SIZE],
        block_words: [F; 16],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) -> [F; 16] {

        let mut state = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            F::from_canonical_u32(Self::IV[0]),
            F::from_canonical_u32(Self::IV[1]),
            F::from_canonical_u32(Self::IV[2]),
            F::from_canonical_u32(Self::IV[3]),
            F::from_canonical_u32(counter as u32),
            F::from_canonical_u32((counter >> 32) as u32),
            F::from_canonical_u32(block_len as u32),
            F::from_canonical_u32(flags as u32),
        ];

        Self::round_field(&mut state, block_words, 0);
        Self::round_field(&mut state, block_words, 1);
        Self::round_field(&mut state, block_words, 2);
        Self::round_field(&mut state, block_words, 3);
        Self::round_field(&mut state, block_words, 4);
        Self::round_field(&mut state, block_words, 5);
        Self::round_field(&mut state, block_words, 6);

        state
    }


    fn compress_in_place_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        cv: &mut [F; IV_SIZE],
        block: [F; STATE_SIZE],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) {
        let state = Self::compress_pre_field(cv, block, block_len, counter, flags);

        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] = F::BaseField::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        cv[0] = F::from_canonical_u32(state_tmp[0] ^ state_tmp[8]);
        cv[1] = F::from_canonical_u32(state_tmp[1] ^ state_tmp[9]);
        cv[2] = F::from_canonical_u32(state_tmp[2] ^ state_tmp[10]);
        cv[3] = F::from_canonical_u32(state_tmp[3] ^ state_tmp[11]);
        cv[4] = F::from_canonical_u32(state_tmp[4] ^ state_tmp[12]);
        cv[5] = F::from_canonical_u32(state_tmp[5] ^ state_tmp[13]);
        cv[6] = F::from_canonical_u32(state_tmp[6] ^ state_tmp[14]);
        cv[7] = F::from_canonical_u32(state_tmp[7] ^ state_tmp[15]);
    }

    // ---------------------------------- circuit --------------------------------------

    // g_circuit
    //#[inline(always)]
    fn g_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; STATE_SIZE],
        input_xor: [ExtensionTarget<D>; 4],
        shift_constant: [ExtensionTarget<D>; 4],
        remain: [ExtensionTarget<D>; 4],
        q: [ExtensionTarget<D>; 4],
        a: usize, 
        b: usize, 
        c: usize, 
        d: usize, 
        x_et: ExtensionTarget<D>, 
        y_et: ExtensionTarget<D>,
    ) where
        Self: RichField + Extendable<D>,
    {

        // state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(x);
        state[a] = builder.add_many_extension([state[a], state[b], x_et]);

        let limbs_input_a = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[a].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_input_d = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[d].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_xor_d_a = builder.split_le_base::<LOOKUP_LIMB_RANGE>(input_xor[0].to_target_array()[0], LOOKUP_LIMB_NUMBER);

        builder.add_lookup_from_index_bitwise(limbs_input_a[0], limbs_input_d[0], limbs_xor_d_a[0], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_a[1], limbs_input_d[1], limbs_xor_d_a[1], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_a[2], limbs_input_d[2], limbs_xor_d_a[2], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_a[3], limbs_input_d[3], limbs_xor_d_a[3], 0);

        // state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(16);
        //state[d] = builder.div_extension(input_xor[0], shift_constant[0]);

        let input_xor_real = builder.mul_add_extension(remain[0], shift_constant[0], q[0]);
        builder.connect_extension(input_xor_real, input_xor[0]);
        state[d] = remain[0];
        // state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state[c] = builder.add_extension(state[c], state[d]);

        let limbs_input_b = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[b].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_input_c = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[c].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_xor_b_c = builder.split_le_base::<LOOKUP_LIMB_RANGE>(input_xor[1].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        
        builder.add_lookup_from_index_bitwise(limbs_input_b[0], limbs_input_c[0], limbs_xor_b_c[0], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_b[1], limbs_input_c[1], limbs_xor_b_c[1], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_b[2], limbs_input_c[2], limbs_xor_b_c[2], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_b[3], limbs_input_c[3], limbs_xor_b_c[3], 0);

        
        // state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(12);
        //state[b] = builder.div_extension(input_xor[1], shift_constant[1]);

        let input_xor_real = builder.mul_add_extension(remain[1], shift_constant[1], q[1]);
        builder.connect_extension(input_xor_real, input_xor[1]);
        state[b] = remain[1];
        // state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(y);
        state[a] = builder.add_many_extension([state[a], state[b], y_et]);

        let limbs_input_a = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[a].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_input_d = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[d].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_xor_d_a = builder.split_le_base::<LOOKUP_LIMB_RANGE>(input_xor[2].to_target_array()[0], LOOKUP_LIMB_NUMBER);

        builder.add_lookup_from_index_bitwise(limbs_input_a[0], limbs_input_d[0], limbs_xor_d_a[0], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_a[1], limbs_input_d[1], limbs_xor_d_a[1], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_a[2], limbs_input_d[2], limbs_xor_d_a[2], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_a[3], limbs_input_d[3], limbs_xor_d_a[3], 0);

        // state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(8);
        //state[d] = builder.div_extension(input_xor[2], shift_constant[2]);

        let input_xor_real = builder.mul_add_extension(remain[2], shift_constant[2], q[2]);
        builder.connect_extension(input_xor_real, input_xor[2]);
        state[d] = remain[2];
        // state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state[c] = builder.add_extension(state[c], state[d]);

        let limbs_input_b = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[b].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_input_c = builder.split_le_base::<LOOKUP_LIMB_RANGE>(state[c].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        let limbs_xor_b_c = builder.split_le_base::<LOOKUP_LIMB_RANGE>(input_xor[3].to_target_array()[0], LOOKUP_LIMB_NUMBER);
        
        builder.add_lookup_from_index_bitwise(limbs_input_b[0], limbs_input_c[0], limbs_xor_b_c[0], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_b[1], limbs_input_c[1], limbs_xor_b_c[1], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_b[2], limbs_input_c[2], limbs_xor_b_c[2], 0);
        builder.add_lookup_from_index_bitwise(limbs_input_b[3], limbs_input_c[3], limbs_xor_b_c[3], 0);

  
        // state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(7);
        //state[b] = builder.div_extension(input_xor[3], shift_constant[3]);

        let input_xor_real = builder.mul_add_extension(remain[3], shift_constant[3], q[3]);
        builder.connect_extension(input_xor_real, input_xor[3]);
        state[b] = remain[3];

    }

    // g_circuit
    //#[inline(always)]
    fn round_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; STATE_SIZE],
        input_xor: [[ExtensionTarget<D>; 4]; 8],
        remain: [[ExtensionTarget<D>; 4]; 8],
        q: [[ExtensionTarget<D>; 4]; 8],
        shift_constant: [ExtensionTarget<D>; 4],
        msg: [ExtensionTarget<D>; STATE_SIZE],
        round: usize
    ) where
        Self: RichField + Extendable<D>,
    {
        // Select the message schedule based on the round.
        let schedule = Self::MSG_SCHEDULE[round];

        // Mix the columns.
        Self::g_circuit(builder, state, input_xor[0], shift_constant, remain[0], q[0], 
             0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
        Self::g_circuit(builder, state, input_xor[1], shift_constant, remain[1], q[1], 
             1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
        Self::g_circuit(builder, state, input_xor[2], shift_constant, remain[2], q[2], 
             2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
        Self::g_circuit(builder, state, input_xor[3], shift_constant, remain[3], q[3], 
             3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

        // Mix the diagonals.
        Self::g_circuit(builder, state, input_xor[4], shift_constant, remain[4], q[4], 
             0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
        Self::g_circuit(builder, state, input_xor[5], shift_constant, remain[5], q[5], 
             1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
        Self::g_circuit(builder, state, input_xor[6], shift_constant, remain[6], q[6], 
             2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
        Self::g_circuit(builder, state, input_xor[7], shift_constant, remain[7], q[7], 
             3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);

    }

    fn compress_pre_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        cv: &mut [ExtensionTarget<D>; IV_SIZE],
        input_xor: [[[ExtensionTarget<D>; 4]; 8]; 7],
        remain: [[[ExtensionTarget<D>; 4]; 8]; 7],
        q: [[[ExtensionTarget<D>; 4]; 8]; 7],
        shift_constant: [ExtensionTarget<D>; 4],
        block_words: [ExtensionTarget<D>; STATE_SIZE],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) -> [ExtensionTarget<D>; STATE_SIZE]
    where
        Self: RichField + Extendable<D>,
    {

        let mut state = [builder.zero_extension(); STATE_SIZE];

        state[0] =  cv[0];
        state[1] =  cv[1];
        state[2] =  cv[2];
        state[3] =  cv[3];
        state[4] =  cv[4];
        state[5] =  cv[5];
        state[6] =  cv[6];
        state[7] =  cv[7];
        state[8] =  builder.constant_extension(Self::Extension::from_canonical_u32(Self::IV[0]));
        state[9] =  builder.constant_extension(Self::Extension::from_canonical_u32(Self::IV[1]));
        state[10] =  builder.constant_extension(Self::Extension::from_canonical_u32(Self::IV[2]));
        state[11] =  builder.constant_extension(Self::Extension::from_canonical_u32(Self::IV[3]));
        state[12] =  builder.constant_extension(Self::Extension::from_canonical_u32(counter as u32));
        state[13] =  builder.constant_extension(Self::Extension::from_canonical_u32((counter >> 32) as u32));
        state[14] =  builder.constant_extension(Self::Extension::from_canonical_u8(block_len));
        state[15] =  builder.constant_extension(Self::Extension::from_canonical_u8(flags));


        Self::round_circuit(builder, &mut state, input_xor[0], remain[0],
            q[0], shift_constant, block_words, 0);
        Self::round_circuit(builder, &mut state, input_xor[1], remain[1],
            q[1], shift_constant, block_words, 1);
        Self::round_circuit(builder, &mut state, input_xor[2], remain[2],
            q[2], shift_constant, block_words, 2);
        Self::round_circuit(builder, &mut state, input_xor[3], remain[3],
            q[3], shift_constant, block_words, 3);
        Self::round_circuit(builder, &mut state, input_xor[4], remain[4],
            q[4], shift_constant, block_words, 4);
        Self::round_circuit(builder, &mut state, input_xor[5], remain[5],
            q[5], shift_constant, block_words, 5);
        Self::round_circuit(builder, &mut state, input_xor[6], remain[6],
            q[6], shift_constant, block_words, 6);
    
        state

    }

    // ---------------------------- run once ----------------------------------
    #[inline]
    fn g_field_run_once<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        out_buffer: &mut GeneratedValues<F>,
        row_num: usize,
        xor_index: usize,
        state: &mut [F; STATE_SIZE], 
        a: usize, 
        b: usize, 
        c: usize, 
        d: usize, 
        x_field: F, 
        y_field: F) {

        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] = F::BaseField::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        let x = F::BaseField::to_noncanonical_u64(&x_field.to_basefield_array()[0]) as u32;
        let y = F::BaseField::to_noncanonical_u64(&y_field.to_basefield_array()[0]) as u32;

        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(x);
        let tmp = state_tmp[d] ^ state_tmp[a];
        out_buffer.set_wire(Wire{row: row_num, column: xor_index}, F::from_canonical_u32(tmp)); 

        state_tmp[d] = tmp.rotate_right(16);
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 448}, F::from_canonical_u32(state_tmp[d] >> 16)); 
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 224}, 
                F::from_canonical_u32(state_tmp[d] - ((state_tmp[d] >> 16) << 16))); 
        
        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        let tmp = state_tmp[b] ^ state_tmp[c];
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 1}, F::from_canonical_u32(tmp)); 

        state_tmp[b] = tmp.rotate_right(12);
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 448 + 1}, F::from_canonical_u32(state_tmp[b] >> 20)); 
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 224 + 1}, 
                F::from_canonical_u32(state_tmp[b] - ((state_tmp[b] >> 20) << 20))); 

        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(y);
        let tmp = state_tmp[d] ^ state_tmp[a];
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 2}, F::from_canonical_u32(tmp)); 

        state_tmp[d] = tmp.rotate_right(8);
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 448 + 2}, F::from_canonical_u32(state_tmp[d] >> 24)); 
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 224 + 2}, 
                F::from_canonical_u32(state_tmp[d] - ((state_tmp[d] >> 24) << 24))); 

        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        let tmp = state_tmp[b] ^ state_tmp[c];
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 3}, F::from_canonical_u32(tmp)); 

        state_tmp[b] = tmp.rotate_right(7);
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 448 + 3}, F::from_canonical_u32(state_tmp[b] >> 25)); 
        out_buffer.set_wire(Wire{row: row_num, column: xor_index + 224 + 3}, 
                F::from_canonical_u32(state_tmp[b] - ((state_tmp[b] >> 25) << 25))); 

        for i in 0..STATE_SIZE {
            state[i] = F::from_canonical_u32(state_tmp[i]);
        }

    }

    #[inline(always)]
    fn round_field_run_once<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        out_buffer: &mut GeneratedValues<F>,
        row: usize,
        state: &mut [F; STATE_SIZE],
        msg: [F; STATE_SIZE], 
        round: usize) {
        // Select the message schedule based on the round.
        let schedule = Self::MSG_SCHEDULE[round];

        // Mix the columns.
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 0 * 4, state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 1 * 4, state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 2 * 4, state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 3 * 4, state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

        // Mix the diagonals.
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 4 * 4, state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 5 * 4, state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 6 * 4, state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
        Self::g_field_run_once(out_buffer, row, 24 + round * 8 * 4 + 7 * 4, state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
    }

    #[inline(always)]
    fn compress_pre_field_run_once<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        out_buffer: &mut GeneratedValues<F>,
        row: usize,
        cv: &mut [F; IV_SIZE],
        block_words: [F; 16],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) -> [F; 16] {

        let mut state = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            F::from_canonical_u32(Self::IV[0]),
            F::from_canonical_u32(Self::IV[1]),
            F::from_canonical_u32(Self::IV[2]),
            F::from_canonical_u32(Self::IV[3]),
            F::from_canonical_u32(counter as u32),
            F::from_canonical_u32((counter >> 32) as u32),
            F::from_canonical_u32(block_len as u32),
            F::from_canonical_u32(flags as u32),
        ];

        Self::round_field_run_once(out_buffer, row, &mut state, block_words, 0);
        Self::round_field_run_once(out_buffer, row, &mut state, block_words, 1);
        Self::round_field_run_once(out_buffer, row, &mut state, block_words, 2);
        Self::round_field_run_once(out_buffer, row, &mut state, block_words, 3);
        Self::round_field_run_once(out_buffer, row, &mut state, block_words, 4);
        Self::round_field_run_once(out_buffer, row, &mut state, block_words, 5);
        Self::round_field_run_once(out_buffer, row, &mut state, block_words, 6);

        state
    }


    fn compress_in_place_field_run_once<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        out_buffer: &mut GeneratedValues<F>,
        row: usize,
        cv: &mut [F; IV_SIZE],
        block: [F; STATE_SIZE],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) {
        let state = Self::compress_pre_field_run_once(out_buffer, row, cv, block, block_len, counter, flags);

        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] = F::BaseField::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        cv[0] = F::from_canonical_u32(state_tmp[0] ^ state_tmp[8]);
        cv[1] = F::from_canonical_u32(state_tmp[1] ^ state_tmp[9]);
        cv[2] = F::from_canonical_u32(state_tmp[2] ^ state_tmp[10]);
        cv[3] = F::from_canonical_u32(state_tmp[3] ^ state_tmp[11]);
        cv[4] = F::from_canonical_u32(state_tmp[4] ^ state_tmp[12]);
        cv[5] = F::from_canonical_u32(state_tmp[5] ^ state_tmp[13]);
        cv[6] = F::from_canonical_u32(state_tmp[6] ^ state_tmp[14]);
        cv[7] = F::from_canonical_u32(state_tmp[7] ^ state_tmp[15]);
    }
   

}

pub struct Blake3Permutation;
impl<F: RichField> PlonkyPermutation<F> for Blake3Permutation {
    fn permute(input: [F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH] {
        let mut state = vec![0u8; SPONGE_WIDTH * size_of::<u64>()];
        for i in 0..SPONGE_WIDTH {
            state[i * size_of::<u64>()..(i + 1) * size_of::<u64>()]
                .copy_from_slice(&input[i].to_canonical_u64().to_le_bytes());
        }

        let hash_onion = iter::repeat_with(|| {
            let output = blake3::hash(&state);
            state = output.as_bytes().to_vec();
            output.as_bytes().to_owned()
        });

        let hash_onion_u64s = hash_onion.flat_map(|output| {
            output
                .chunks_exact(size_of::<u64>())
                .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
                .collect_vec()
        });

        // Parse field elements from u64 stream, using rejection sampling such that
        // words that don't fit in F are ignored.
        let hash_onion_elems = hash_onion_u64s
            .filter(|&word| word < F::ORDER)
            .map(F::from_canonical_u64);

        hash_onion_elems
            .take(SPONGE_WIDTH)
            .collect_vec()
            .try_into()
            .unwrap()
    }
}

/// Blake3-256 hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Blake3_256<const N: usize>;
impl<F: RichField, const N: usize> Hasher<F> for Blake3_256<N> {
    const HASH_SIZE: usize = N;
    type Hash = BytesHash<N>;
    type Permutation = Blake3Permutation;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let buffer = unsafe {
            slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * F::BITS >> 3)
        };

        let mut arr = [0; N];
        let hash_bytes = blake3::hash(buffer);
        arr.copy_from_slice(hash_bytes.as_bytes());
        BytesHash(arr)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let left = unsafe { slice::from_raw_parts(left.0.as_ptr() as *const u8, 32) };

        let right = unsafe { slice::from_raw_parts(right.0.as_ptr() as *const u8, 32) };

        let mut v = vec![0; N * 2];
        v[0..N].copy_from_slice(left);
        v[N..].copy_from_slice(right);
        let mut arr = [0; N];
        let hash_bytes = blake3::hash(&v);
        arr.copy_from_slice(hash_bytes.as_bytes());
        BytesHash(arr)
    }
}
