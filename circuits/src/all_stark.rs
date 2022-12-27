use std::iter;

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use crate::config::StarkConfig;
use crate::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use crate::stark::Stark;
use crate::builtins::bitwise::bitwise_stark::{BitwiseStark, self};
use crate::builtins::cmp::cmp_stark::{CmpStark, self};
use crate::builtins::rangecheck::rangecheck_stark::{RangeCheckStark, self};
use crate::fixed_table::bitwise_fixed::bitwise_fixed_stark::{BitwiseFixedStark, self};
use crate::fixed_table::rangecheck_fixed::rangecheck_fixed_stark::{RangecheckFixedStark, self};
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::memory::{MemoryStark, ctl_data as mem_ctl_data, ctl_filter as mem_ctl_filter};

#[derive(Clone)]
pub struct AllStark<F: RichField + Extendable<D>, const D: usize> {
    pub cpu_stark: CpuStark<F, D>,
    pub memory_stark: MemoryStark<F, D>,
    // builtins
    pub bitwise_stark: BitwiseStark<F,D>,
    pub cmp_stark: CmpStark<F,D>,
    pub rangecheck_stark: RangeCheckStark<F, D>,

    pub cross_table_lookups: Vec<CrossTableLookup<F>>,
}

impl<F: RichField + Extendable<D>, const D: usize> Default for AllStark<F, D> {
    fn default() -> Self {
        Self {
            cpu_stark: CpuStark::default(),
            memory_stark: MemoryStark::default(),
            // builtins
            bitwise_stark: BitwiseStark::default(),
            cmp_stark: CmpStark::default(),
            rangecheck_stark: RangeCheckStark::default(),

            cross_table_lookups: all_cross_table_lookups(),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> AllStark<F, D> {
    pub(crate) fn nums_permutation_zs(&self, config: &StarkConfig) -> [usize; NUM_TABLES] {
        [
            self.cpu_stark.num_permutation_batches(config),
            self.memory_stark.num_permutation_batches(config),
            self.bitwise_stark.num_permutation_batches(config),
            self.cmp_stark.num_permutation_batches(config),
            self.rangecheck_stark.num_permutation_batches(config),
        ]
    }

    pub(crate) fn permutation_batch_sizes(&self) -> [usize; NUM_TABLES] {
        [
            self.cpu_stark.permutation_batch_size(),
            self.memory_stark.permutation_batch_size(),
            self.bitwise_stark.permutation_batch_size(),
            self.cmp_stark.permutation_batch_size(),
            self.rangecheck_stark.permutation_batch_size(),
        ]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Table {
    Cpu = 0,
    Memory = 1,
    Bitwise = 2,
    Cmp = 3,
    RangeCheck = 4,
    BitwiseFixed = 5,
    RangecheckFixed = 6,
}

pub(crate) const NUM_TABLES: usize = Table::RangeCheck as usize + 1;

#[allow(unused)] // TODO: Should be used soon.
pub(crate) fn all_cross_table_lookups<F: Field>() -> Vec<CrossTableLookup<F>> {
    // TODO:
    vec![ctl_memory()]
}

fn ctl_memory<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_memory(),
            Some(cpu_stark::ctl_filter_memory()),
        )],
        TableWithColumns::new(Table::Memory, mem_ctl_data(), Some(mem_ctl_filter())),
        None,
    )
}


// add bitwise rangecheck instance
// Cpu table
// +-----+-----+-----+---------+--------+---------+-----+-----+-----+-----+----
// | clk | ins | ... | sel_and | sel_or | sel_xor | ... | op0 | op1 | dst | ...
// +-----+-----+-----+---------+--------+---------+-----+-----+----+----+----
// 
// Bitwise table
// +-----+-----+-----+-----+------------+------------+-----------+------------+---
// | tag | op0 | op1 | res | op0_limb_0 | op0_limb_1 |res_limb_2 | op0_limb_3 |...
// +-----+-----+-----+-----+------------+------------+-----------+------------+---
//
// Filter bitwise from CPU Table
// 1. (sel_add + sel_or + sel_xor) * (op0, op1, dst) = looking_table
// Filter bitwise from Bitwsie Table
// 1. (op0, op1, res) = looked_table

// Cross_Lookup_Table(looking_table, looked_table)
fn ctl_bitwise_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_bitwise(),
            Some(cpu_stark::ctl_filter_with_bitwise_and()),
        ),
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_bitwise(),
            Some(cpu_stark::ctl_filter_with_bitwise_or()),
        ),
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_bitwise(),
            Some(cpu_stark::ctl_filter_with_bitwise_xor()),
        )],
        TableWithColumns::new(
            Table::Bitwise, 
            bitwise_stark::ctl_data_with_bitwise_fixed(), 
            Some(bitwise_stark::ctl_filter_with_bitwise_fixed())),
        None,
    )
}


// Cross_Lookup_Table(looking_table, looked_table)
fn ctl_bitwise_rangecheck<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::RangecheckFixed,
            rangecheck_fixed_stark::ctl_data_with_bitwise(),
            Some(rangecheck_fixed_stark::ctl_filter_with_bitwise()),
        )],
        TableWithColumns::new(
            Table::Bitwise, 
            bitwise_stark::ctl_data_with_rangecheck_fixed(), 
            Some(bitwise_stark::ctl_filter_with_rangecheck_fixed())),
        None,
    )
}

// Cross_Lookup_Table(looking_table, looked_table)
fn ctl_bitwise_bitwise_fixed_table<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::BitwiseFixed,
            bitwise_fixed_stark::ctl_data_with_bitwise(),
            Some(bitwise_fixed_stark::ctl_filter_with_bitwise()),
        )],
        TableWithColumns::new(
            Table::Bitwise, 
            bitwise_stark::ctl_data_with_bitwise_fixed(), 
            Some(bitwise_stark::ctl_filter_with_bitwise_fixed())),
        None,
    )
}


// add CMP cross lookup instance
fn ctl_cmp_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_cmp(),
            Some(cpu_stark::ctl_filter_with_cmp()),
        )],
        TableWithColumns::new(
            Table::Cmp, 
            cmp_stark::ctl_data_with_cpu(),
            Some(cmp_stark::ctl_filter_with_cpu())),
        None,
    )
}

fn ctl_cmp_rangecheck<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_with_cmp(),
            Some(rangecheck_stark::ctl_filter_with_cmp()),
        )],
        TableWithColumns::new(
            Table::Cmp, 
            cmp_stark::ctl_data_with_rangecheck(),
            Some(cmp_stark::ctl_filter_with_rangecheck())),
        None,
    )
}


// add Rangecheck cross lookup instance
fn ctl_rangecheck_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_rangecheck(),
            Some(cpu_stark::ctl_filter_with_rangecheck()),
        )],
        TableWithColumns::new(
            Table::RangeCheck, 
            rangecheck_stark::ctl_data_with_cpu(),
            Some(rangecheck_stark::ctl_filter_with_cpu())),
        None,
    )
}

fn ctl_rangecheck_rangecheck_fixed<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::RangecheckFixed,
            rangecheck_fixed_stark::ctl_data_with_rangecheck(),
            Some(rangecheck_fixed_stark::ctl_filter_with_rangecheck()),
        )],
        TableWithColumns::new(
            Table::RangeCheck, 
            rangecheck_stark::ctl_data_with_rangecheck_fixed(), 
            Some(rangecheck_stark::ctl_filter_with_rangecheck_fixed())),
        None,
    )
}