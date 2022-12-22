use std::iter;

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use crate::config::StarkConfig;
use crate::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use crate::stark::Stark;

use crate::builtins::bitwise::bitwise_stark::BitwiseStark;
use crate::builtins::cmp::cmp_stark::CmpStark;
use crate::builtins::rangecheck::rangecheck_stark::RangeCheckStark;
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::memory;
use crate::memory::MemoryStark;

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
        // [
        //     self.cpu_stark.num_permutation_batches(config),
        //     self.memory_stark.num_permutation_batches(config),
        //     self.builtin_stark.num_permutation_batches(config),
        // ]
        [0, 0, 0, 0, 0]
    }

    pub(crate) fn permutation_batch_sizes(&self) -> [usize; NUM_TABLES] {
        // [
        //     self.cpu_stark.permutation_batch_size(),
        //     self.memory_stark.permutation_batch_size(),
        //     self.builtin_stark.permutation_batch_size(),
        // ]
        [0, 0, 0, 0, 0]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Table {
    Cpu = 0,
    Memory = 1,
    Bitwise = 2,
    Cmp = 3,
    RangeCheck = 4,
}

pub(crate) const NUM_TABLES: usize = Table::RangeCheck as usize + 1;

#[allow(unused)] // TODO: Should be used soon.
pub(crate) fn all_cross_table_lookups<F: Field>() -> Vec<CrossTableLookup<F>> {
    vec![]
}

// fn ctl_memory<F: Field>() -> CrossTableLookup<F> {
//     CrossTableLookup::new(
//         vec![TableWithColumns::new(
//             Table::Cpu,
//             [],
//             None,
//         )],
//         TableWithColumns::new(Table::Memory, logic::ctl_data(), Some(logic::ctl_filter())),
//         None,
//     )
// }

// fn ctl_builtin<F: Field>() -> CrossTableLookup<F> {
//     CrossTableLookup::new(TableWithColumns::new(Table, columns, filter_column), looked_table, None)
// }
