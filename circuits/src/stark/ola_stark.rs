use core::types::GoldilocksField;
use std::iter;

use super::config::StarkConfig;
use super::cross_table_lookup::{CrossTableLookup, TableWithColumns};
use super::stark::Stark;
use crate::builtins::bitwise::bitwise_stark::{self, BitwiseStark};
use crate::builtins::cmp::cmp_stark::{self, CmpStark};
use crate::builtins::poseidon::poseidon_chunk_stark::{self, PoseidonChunkStark};
use crate::builtins::poseidon::poseidon_stark::{self, PoseidonStark};
use crate::builtins::rangecheck::rangecheck_stark::{self, RangeCheckStark};
use crate::builtins::sccall::sccall_stark::{self, SCCallStark};
use crate::builtins::storage::storage_access_stark::{self, StorageAccessStark};
use crate::builtins::tape::tape_stark::{self, TapeStark};
use crate::cpu::cpu_stark;
use crate::cpu::cpu_stark::CpuStark;
use crate::memory::memory_stark::{
    self, ctl_data as mem_ctl_data, ctl_data_mem_rc_diff_cond, ctl_data_mem_sort_rc,
    ctl_filter as mem_ctl_filter, ctl_filter_mem_rc_diff_cond, ctl_filter_mem_sort_rc, MemoryStark,
};
use crate::program::prog_chunk_stark::{self, ProgChunkStark};
use crate::program::program_stark::{self, ProgramStark};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug)]
pub struct OlaStark<F: RichField + Extendable<D>, const D: usize> {
    pub cpu_stark: CpuStark<F, D>,
    pub memory_stark: MemoryStark<F, D>,
    // builtins
    pub bitwise_stark: BitwiseStark<F, D>,
    pub cmp_stark: CmpStark<F, D>,
    pub rangecheck_stark: RangeCheckStark<F, D>,
    pub poseidon_stark: PoseidonStark<F, D>,
    pub poseidon_chunk_stark: PoseidonChunkStark<F, D>,
    pub storage_access_stark: StorageAccessStark<F, D>,
    pub tape_stark: TapeStark<F, D>,
    pub sccall_stark: SCCallStark<F, D>,
    pub program_stark: ProgramStark<F, D>,
    pub prog_chunk_stark: ProgChunkStark<F, D>,

    pub cross_table_lookups: Vec<CrossTableLookup<F>>,
}

impl Serialize for OlaStark<GoldilocksField, 2> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("OlaStark", 13)?;
        state.serialize_field("cpu_stark", &self.cpu_stark)?;
        state.serialize_field("memory_stark", &self.memory_stark)?;
        state.serialize_field("bitwise_stark", &self.bitwise_stark)?;
        state.serialize_field("cmp_stark", &self.cmp_stark)?;
        state.serialize_field("rangecheck_stark", &self.rangecheck_stark)?;
        state.serialize_field("poseidon_stark", &self.poseidon_stark)?;
        state.serialize_field("poseidon_chunk_stark", &self.poseidon_chunk_stark)?;
        state.serialize_field("storage_access_stark", &self.storage_access_stark)?;
        state.serialize_field("tape_stark", &self.tape_stark)?;
        state.serialize_field("sccall_stark", &self.sccall_stark)?;
        state.serialize_field("program_stark", &self.program_stark)?;
        state.serialize_field("prog_chunk_stark", &self.prog_chunk_stark)?;
        state.serialize_field("cross_table_lookups", &self.cross_table_lookups)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for OlaStark<GoldilocksField, 2> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            CpuStark,
            MemoryStark,
            BitwiseStark,
            CmpStark,
            RangeCheckStark,
            PoseidonStark,
            PoseidonChunkStark,
            StorageAccessStark,
            TapeStark,
            SccallStark,
            ProgramStark,
            ProgChunkStark,
            CrossTableLookups,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("a valid field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "cpu_stark" => Ok(Field::CpuStark),
                            "memory_stark" => Ok(Field::MemoryStark),
                            "bitwise_stark" => Ok(Field::BitwiseStark),
                            "cmp_stark" => Ok(Field::CmpStark),
                            "rangecheck_stark" => Ok(Field::RangeCheckStark),
                            "poseidon_stark" => Ok(Field::PoseidonStark),
                            "poseidon_chunk_stark" => Ok(Field::PoseidonChunkStark),
                            "storage_access_stark" => Ok(Field::StorageAccessStark),
                            "tape_stark" => Ok(Field::TapeStark),
                            "sccall_stark" => Ok(Field::SccallStark),
                            "program_stark" => Ok(Field::ProgramStark),
                            "prog_chunk_stark" => Ok(Field::ProgChunkStark),
                            "cross_table_lookups" => Ok(Field::CrossTableLookups),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct OlaStarkVisitor {
            marker: std::marker::PhantomData<GoldilocksField>,
        }

        impl<'de> Visitor<'de> for OlaStarkVisitor {
            type Value = OlaStark<GoldilocksField, 2>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct OlaStark")
            }

            fn visit_map<V>(self, mut map: V) -> Result<OlaStark<GoldilocksField, 2>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut cpu_stark = None;
                let mut memory_stark = None;
                let mut bitwise_stark = None;
                let mut cmp_stark = None;
                let mut rangecheck_stark = None;
                let mut poseidon_stark = None;
                let mut poseidon_chunk_stark = None;
                let mut storage_access_stark = None;
                let mut tape_stark = None;
                let mut sccall_stark = None;
                let mut program_stark = None;
                let mut prog_chunk_stark = None;
                let mut cross_table_lookups = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::CpuStark => {
                            if cpu_stark.is_some() {
                                return Err(de::Error::duplicate_field("cpu_stark"));
                            }
                            cpu_stark = Some(map.next_value()?);
                        }
                        Field::MemoryStark => {
                            if memory_stark.is_some() {
                                return Err(de::Error::duplicate_field("memory_stark"));
                            }
                            memory_stark = Some(map.next_value()?);
                        }
                        Field::BitwiseStark => {
                            if bitwise_stark.is_some() {
                                return Err(de::Error::duplicate_field("bitwise_stark"));
                            }
                            bitwise_stark = Some(map.next_value()?);
                        }
                        Field::CmpStark => {
                            if cmp_stark.is_some() {
                                return Err(de::Error::duplicate_field("cmp_stark"));
                            }
                            cmp_stark = Some(map.next_value()?);
                        }
                        Field::RangeCheckStark => {
                            if rangecheck_stark.is_some() {
                                return Err(de::Error::duplicate_field("rangecheck_stark"));
                            }
                            rangecheck_stark = Some(map.next_value()?);
                        }
                        Field::PoseidonStark => {
                            if poseidon_stark.is_some() {
                                return Err(de::Error::duplicate_field("poseidon_stark"));
                            }
                            poseidon_stark = Some(map.next_value()?);
                        }
                        Field::PoseidonChunkStark => {
                            if poseidon_chunk_stark.is_some() {
                                return Err(de::Error::duplicate_field("poseidon_chunk_stark"));
                            }
                            poseidon_chunk_stark = Some(map.next_value()?);
                        }
                        Field::StorageAccessStark => {
                            if storage_access_stark.is_some() {
                                return Err(de::Error::duplicate_field("storage_access_stark"));
                            }
                            storage_access_stark = Some(map.next_value()?);
                        }
                        Field::TapeStark => {
                            if tape_stark.is_some() {
                                return Err(de::Error::duplicate_field("tape_stark"));
                            }
                            tape_stark = Some(map.next_value()?);
                        }
                        Field::SccallStark => {
                            if sccall_stark.is_some() {
                                return Err(de::Error::duplicate_field("sccall_stark"));
                            }
                            sccall_stark = Some(map.next_value()?);
                        }
                        Field::ProgramStark => {
                            if program_stark.is_some() {
                                return Err(de::Error::duplicate_field("program_stark"));
                            }
                            program_stark = Some(map.next_value()?);
                        }
                        Field::ProgChunkStark => {
                            if prog_chunk_stark.is_some() {
                                return Err(de::Error::duplicate_field("prog_chunk_stark"));
                            }
                            prog_chunk_stark = Some(map.next_value()?);
                        }
                        Field::CrossTableLookups => {
                            if cross_table_lookups.is_some() {
                                return Err(de::Error::duplicate_field("cross_table_lookups"));
                            }
                            cross_table_lookups = Some(map.next_value()?);
                        }
                    }
                }
                // let cpu_stark = cpu_stark.ok_or_else(||
                // de::Error::missing_field("cpu_stark"))?;
                let cpu_stark = cpu_stark.ok_or_else(|| de::Error::missing_field("cpu_stark"))?;
                let memory_stark =
                    memory_stark.ok_or_else(|| de::Error::missing_field("memory_stark"))?;
                let bitwise_stark =
                    bitwise_stark.ok_or_else(|| de::Error::missing_field("bitwise_stark"))?;
                let cmp_stark = cmp_stark.ok_or_else(|| de::Error::missing_field("cmp_stark"))?;
                let rangecheck_stark =
                    rangecheck_stark.ok_or_else(|| de::Error::missing_field("rangecheck_stark"))?;
                let poseidon_stark =
                    poseidon_stark.ok_or_else(|| de::Error::missing_field("poseidon_stark"))?;
                let poseidon_chunk_stark = poseidon_chunk_stark
                    .ok_or_else(|| de::Error::missing_field("poseidon_chunk_stark"))?;
                let storage_access_stark = storage_access_stark
                    .ok_or_else(|| de::Error::missing_field("storage_access_stark"))?;
                let tape_stark =
                    tape_stark.ok_or_else(|| de::Error::missing_field("tape_stark"))?;
                let sccall_stark =
                    sccall_stark.ok_or_else(|| de::Error::missing_field("sccall_stark"))?;
                let program_stark =
                    program_stark.ok_or_else(|| de::Error::missing_field("program_stark"))?;
                let prog_chunk_stark =
                    prog_chunk_stark.ok_or_else(|| de::Error::missing_field("prog_chunk_stark"))?;
                let cross_table_lookups = cross_table_lookups
                    .ok_or_else(|| de::Error::missing_field("cross_table_lookups"))?;
                Ok(OlaStark {
                    cpu_stark,
                    memory_stark,
                    bitwise_stark,
                    cmp_stark,
                    rangecheck_stark,
                    poseidon_stark,
                    poseidon_chunk_stark,
                    storage_access_stark,
                    tape_stark,
                    sccall_stark,
                    program_stark,
                    prog_chunk_stark,
                    cross_table_lookups,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &[
            "cpu_stark",
            "memory_stark",
            "bitwise_stark",
            "cmp_stark",
            "rangecheck_stark",
            "poseidon_stark",
            "poseidon_chunk_stark",
            "storage_access_stark",
            "tape_stark",
            "sccall_stark",
            "program_stark",
            "prog_chunk_stark",
            "cross_table_lookups",
        ];
        deserializer.deserialize_struct("OlaStark", FIELDS, OlaStarkVisitor { marker: std::marker::PhantomData })
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Default for OlaStark<F, D> {
    fn default() -> Self {
        plonky2::field::cfft::ntt::init_gpu();

        Self {
            cpu_stark: CpuStark::default(),
            memory_stark: MemoryStark::default(),
            bitwise_stark: BitwiseStark::default(),
            cmp_stark: CmpStark::default(),
            rangecheck_stark: RangeCheckStark::default(),
            poseidon_stark: PoseidonStark::default(),
            poseidon_chunk_stark: PoseidonChunkStark::default(),
            storage_access_stark: StorageAccessStark::default(),
            tape_stark: TapeStark::default(),
            sccall_stark: SCCallStark::default(),
            program_stark: ProgramStark::default(),
            prog_chunk_stark: ProgChunkStark::default(),
            cross_table_lookups: all_cross_table_lookups(),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> OlaStark<F, D> {
    pub(crate) fn nums_permutation_zs(&self, config: &StarkConfig) -> [usize; NUM_TABLES] {
        [
            self.cpu_stark.num_permutation_batches(config),
            self.memory_stark.num_permutation_batches(config),
            self.bitwise_stark.num_permutation_batches(config),
            self.cmp_stark.num_permutation_batches(config),
            self.rangecheck_stark.num_permutation_batches(config),
            self.poseidon_stark.num_permutation_batches(config),
            self.poseidon_chunk_stark.num_permutation_batches(config),
            self.storage_access_stark.num_permutation_batches(config),
            self.tape_stark.num_permutation_batches(config),
            self.sccall_stark.num_permutation_batches(config),
            self.program_stark.num_permutation_batches(config),
            self.prog_chunk_stark.num_permutation_batches(config),
        ]
    }

    pub(crate) fn permutation_batch_sizes(&self) -> [usize; NUM_TABLES] {
        [
            self.cpu_stark.permutation_batch_size(),
            self.memory_stark.permutation_batch_size(),
            self.bitwise_stark.permutation_batch_size(),
            self.cmp_stark.permutation_batch_size(),
            self.rangecheck_stark.permutation_batch_size(),
            self.poseidon_stark.permutation_batch_size(),
            self.poseidon_chunk_stark.permutation_batch_size(),
            self.storage_access_stark.permutation_batch_size(),
            self.tape_stark.permutation_batch_size(),
            self.sccall_stark.permutation_batch_size(),
            self.program_stark.permutation_batch_size(),
            self.prog_chunk_stark.permutation_batch_size(),
        ]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum Table {
    Cpu = 0,
    Memory = 1,
    // builtins
    Bitwise = 2,
    Cmp = 3,
    RangeCheck = 4,
    Poseidon = 5,
    PoseidonChunk = 6,
    StorageAccess = 7,
    Tape = 8,
    SCCall = 9,
    Program = 10,
    ProgChunk = 11,
}

pub(crate) const NUM_TABLES: usize = 12;

pub(crate) fn all_cross_table_lookups<F: Field>() -> Vec<CrossTableLookup<F>> {
    vec![
        ctl_cpu_memory(),
        ctl_memory_rc_sort(),
        ctl_memory_rc_region(),
        ctl_bitwise_cpu(),
        ctl_cmp_cpu(),
        ctl_cmp_rangecheck(),
        ctl_rangecheck_cpu(),
        ctl_cpu_poseidon_chunk(),
        ctl_poseidon_chunk_mem(),
        ctl_chunk_poseidon(),
        ctl_cpu_poseidon_tree_key(),
        ctl_cpu_storage_access(),
        ctl_storage_access_poseidon(),
        ctl_cpu_tape(),
        ctl_cpu_sccall(),
        ctl_cpu_sccall_end(),
        ctl_cpu_program(),
        ctl_prog_chunk_prog(),
        ctl_prog_chunk_storage(),
    ]
}

fn ctl_cpu_memory<F: Field>() -> CrossTableLookup<F> {
    let cpu_mem_store_load = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_store_load(),
        Some(cpu_stark::ctl_filter_cpu_mem_store_load()),
    );
    let cpu_mem_call_ret_pc = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_call_ret_pc(),
        Some(cpu_stark::ctl_filter_cpu_mem_call_ret()),
    );
    let cpu_mem_call_ret_fp = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_call_ret_fp(),
        Some(cpu_stark::ctl_filter_cpu_mem_call_ret()),
    );
    let cpu_mem_tload_tstore = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_mem_tload_tstore(),
        Some(cpu_stark::ctl_filter_cpu_mem_tload_tstore()),
    );
    let cpu_sccall_mems = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_mem_sccall(i),
            Some(cpu_stark::ctl_filter_cpu_mem_sccall()),
        )
    });
    let cpu_storage_addr = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_mem_for_storage_addr(i),
            Some(cpu_stark::ctl_filter_cpu_storage_access()),
        )
    });
    let cpu_storage_value = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_mem_for_storage_value(i),
            Some(cpu_stark::ctl_filter_cpu_storage_access()),
        )
    });

    let mut all_cpu_lookers = vec![
        cpu_mem_store_load,
        cpu_mem_call_ret_pc,
        cpu_mem_call_ret_fp,
        cpu_mem_tload_tstore,
    ];
    all_cpu_lookers.extend(cpu_sccall_mems);
    all_cpu_lookers.extend(cpu_storage_addr);
    all_cpu_lookers.extend(cpu_storage_value);
    let memory_looked =
        TableWithColumns::new(Table::Memory, mem_ctl_data(), Some(mem_ctl_filter()));
    CrossTableLookup::new(all_cpu_lookers, memory_looked)
}

fn ctl_memory_rc_sort<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Memory,
            ctl_data_mem_sort_rc(),
            Some(ctl_filter_mem_sort_rc()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_memory(),
            Some(rangecheck_stark::ctl_filter_memory_sort()),
        ),
    )
}

fn ctl_memory_rc_region<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Memory,
            ctl_data_mem_rc_diff_cond(),
            Some(ctl_filter_mem_rc_diff_cond()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_memory(),
            Some(rangecheck_stark::ctl_filter_memory_region()),
        ),
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
// | tag | op0 | op1 | res | op0_limb_0 | op0_limb_1 |res_limb_2 | op0_limb_3
// |...
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
            Some(cpu_stark::ctl_filter_with_bitwise()),
        )],
        TableWithColumns::new(
            Table::Bitwise,
            bitwise_stark::ctl_data_with_cpu(),
            Some(bitwise_stark::ctl_filter_with_cpu()),
        ),
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
            Some(cmp_stark::ctl_filter_with_cpu()),
        ),
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
            Some(cmp_stark::ctl_filter_with_rangecheck()),
        ),
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
            Some(rangecheck_stark::ctl_filter_with_cpu()),
        ),
    )
}

fn ctl_cpu_poseidon_chunk<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_poseidon_chunk(),
            Some(cpu_stark::ctl_filter_with_poseidon_chunk()),
        )],
        TableWithColumns::new(
            Table::PoseidonChunk,
            poseidon_chunk_stark::ctl_data_with_cpu(),
            Some(poseidon_chunk_stark::ctl_filter_with_cpu()),
        ),
    )
}

fn ctl_poseidon_chunk_mem<F: Field>() -> CrossTableLookup<F> {
    let looker_src = (0..8).map(|i: usize| {
        TableWithColumns::new(
            Table::PoseidonChunk,
            poseidon_chunk_stark::ctl_data_with_mem_src(i),
            Some(poseidon_chunk_stark::ctl_filter_with_mem_src(i)),
        )
    });
    let looker_dst = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::PoseidonChunk,
            poseidon_chunk_stark::ctl_data_with_mem_dst(i),
            Some(poseidon_chunk_stark::ctl_filter_with_mem_dst()),
        )
    });
    let all_lookers = looker_src.into_iter().chain(looker_dst).collect();
    let mem_looked = TableWithColumns::new(
        Table::Memory,
        memory_stark::ctl_data_with_poseidon_chunk(),
        Some(memory_stark::ctl_filter_with_poseidon_chunk()),
    );
    CrossTableLookup::new(all_lookers, mem_looked)
}

fn ctl_chunk_poseidon<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![
            TableWithColumns::new(
                Table::PoseidonChunk,
                poseidon_chunk_stark::ctl_data_with_poseidon(),
                Some(poseidon_chunk_stark::ctl_filter_with_poseidon()),
            ),
            TableWithColumns::new(
                Table::ProgChunk,
                prog_chunk_stark::ctl_data_to_poseidon(),
                Some(prog_chunk_stark::ctl_filter_to_poseidon()),
            ),
        ],
        TableWithColumns::new(
            Table::Poseidon,
            poseidon_stark::ctl_data_with_poseidon_chunk(),
            Some(poseidon_stark::ctl_filter_with_poseidon_chunk()),
        ),
    )
}

fn ctl_cpu_storage_access<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_storage_access(),
            Some(cpu_stark::ctl_filter_cpu_storage_access()),
        )],
        TableWithColumns::new(
            Table::StorageAccess,
            storage_access_stark::ctl_data_with_cpu(),
            Some(storage_access_stark::ctl_filter_with_cpu_sstore()),
        ),
    )
}

fn ctl_storage_access_poseidon<F: Field>() -> CrossTableLookup<F> {
    let looker_bit0 = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit0(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit0()),
    );
    let looker_bit0_pre = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit0_pre(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit0()),
    );
    let looker_bit1 = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit1(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit1()),
    );
    let looker_bit1_pre = TableWithColumns::new(
        Table::StorageAccess,
        storage_access_stark::ctl_data_with_poseidon_bit1_pre(),
        Some(storage_access_stark::ctl_filter_with_poseidon_bit1()),
    );
    let all_lookers = vec![looker_bit0, looker_bit0_pre, looker_bit1, looker_bit1_pre];
    let looked = TableWithColumns::new(
        Table::Poseidon,
        poseidon_stark::ctl_data_with_storage(),
        Some(poseidon_stark::ctl_filter_with_storage()),
    );
    CrossTableLookup::new(all_lookers, looked)
}

fn ctl_cpu_poseidon_tree_key<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_poseidon_treekey(),
            Some(cpu_stark::ctl_filter_poseidon_treekey()),
        )],
        TableWithColumns::new(
            Table::Poseidon,
            poseidon_stark::ctl_data_cpu_tree_key(),
            Some(poseidon_stark::ctl_filter_cpu_tree_key()),
        ),
    )
}

fn ctl_cpu_tape<F: Field>() -> CrossTableLookup<F> {
    let cpu_tape_tload_tstore = TableWithColumns::new(
        Table::Cpu,
        cpu_stark::ctl_data_cpu_tape_load_store(),
        Some(cpu_stark::ctl_filter_cpu_tape_load_store()),
    );
    let cpu_tape_sccall_caller = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_tape_sccall_caller(i),
            Some(cpu_stark::ctl_filter_cpu_is_sccall_ext()),
        )
    });
    let cpu_tape_sccall_callee_code = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_tape_sccall_callee_code(i),
            Some(cpu_stark::ctl_filter_cpu_is_sccall_ext()),
        )
    });
    let cpu_tape_sccall_callee_storage = (0..4).map(|i: usize| {
        TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_tape_sccall_callee_storage(i),
            Some(cpu_stark::ctl_filter_cpu_is_sccall_ext()),
        )
    });

    let all_lookers = iter::once(cpu_tape_tload_tstore)
        .chain(cpu_tape_sccall_caller)
        .chain(cpu_tape_sccall_callee_code)
        .chain(cpu_tape_sccall_callee_storage)
        .collect();

    let tape_looked = TableWithColumns::new(
        Table::Tape,
        tape_stark::ctl_data_tape(),
        Some(tape_stark::ctl_filter_tape()),
    );
    CrossTableLookup::new(all_lookers, tape_looked)
}

fn ctl_cpu_sccall<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_sccall(),
            Some(cpu_stark::ctl_filter_cpu_sccall()),
        )],
        TableWithColumns::new(
            Table::SCCall,
            sccall_stark::ctl_data_sccall(),
            Some(sccall_stark::ctl_filter_sccall()),
        ),
    )
}

fn ctl_cpu_sccall_end<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_cpu_sccall_end(),
            Some(cpu_stark::ctl_filter_cpu_sccall_end()),
        )],
        TableWithColumns::new(
            Table::SCCall,
            sccall_stark::ctl_data_sccall_end(),
            Some(sccall_stark::ctl_filter_sccall_end()),
        ),
    )
}

fn ctl_cpu_program<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![
            TableWithColumns::new(
                Table::Cpu,
                cpu_stark::ctl_data_inst_to_program(),
                Some(cpu_stark::ctl_filter_with_program_inst()),
            ),
            TableWithColumns::new(
                Table::Cpu,
                cpu_stark::ctl_data_imm_to_program(),
                Some(cpu_stark::ctl_filter_with_program_imm()),
            ),
        ],
        TableWithColumns::new(
            Table::Program,
            program_stark::ctl_data_by_cpu(),
            Some(program_stark::ctl_filter_by_cpu()),
        ),
    )
}

fn ctl_prog_chunk_prog<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        (0..8)
            .map(|i: usize| {
                TableWithColumns::new(
                    Table::ProgChunk,
                    prog_chunk_stark::ctl_data_to_program(i),
                    Some(prog_chunk_stark::ctl_filter_to_program(i)),
                )
            })
            .collect(),
        TableWithColumns::new(
            Table::Program,
            program_stark::ctl_data_by_program_chunk(),
            Some(program_stark::ctl_filter_by_program_chunk()),
        ),
    )
}

fn ctl_prog_chunk_storage<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::ProgChunk,
            prog_chunk_stark::ctl_data_to_storage_access(),
            Some(prog_chunk_stark::ctl_filter_to_storage_access()),
        )],
        TableWithColumns::new(
            Table::StorageAccess,
            storage_access_stark::ctl_data_for_prog_chunk(),
            Some(storage_access_stark::ctl_filter_for_prog_chunk()),
        ),
    )
}

// Cross_Lookup_Table(looking_table, looked_table)
/*fn ctl_bitwise_bitwise_fixed_table<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::BitwiseFixed,
            bitwise_fixed_stark::ctl_data_with_bitwise(),
            Some(bitwise_fixed_stark::ctl_filter_with_bitwise()),
        )],
        TableWithColumns::new(
            Table::Bitwise,
            bitwise_stark::ctl_data_with_bitwise_fixed(),
            Some(bitwise_stark::ctl_filter_with_bitwise_fixed()),
        ),
        None,
    )
}*/

/*fn ctl_rangecheck_rangecheck_fixed<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::RangecheckFixed,
            rangecheck_fixed_stark::ctl_data_with_rangecheck(),
            Some(rangecheck_fixed_stark::ctl_filter_with_rangecheck()),
        )],
        TableWithColumns::new(
            Table::RangeCheck,
            rangecheck_stark::ctl_data_with_rangecheck_fixed(),
            Some(rangecheck_stark::ctl_filter_with_rangecheck_fixed()),
        ),
        None,
    )
}*/

// check the correct program with lookup

// Program table
// +-----+--------------+-------+----------+
// | PC  |      INS     |  IMM  | COMPRESS |
// +-----+--------------+-------+----------+
// +-----+--------------+-------+----------+
// |  1  |  0x********  |  U32  |   Field  |
// +-----+--------------+-------+----------+
// +-----+--------------+-------+----------+
// |  2  |  0x********  |  U32  |   Field  |
// +-----+--------------+-------+----------++
// +-----+--------------+-------+----------+
// |  3  |  0x********  |  U32  |   Field  |
// +-----+--------------+-------+----------+

// CPU table
// +-----+-----+--------------+-------+----------+
// | ... | PC  |      INS     |  IMM  | COMPRESS |
// +-----+-----+--------------+-------+----------+
// +-----+-----+--------------+-------+----------+
// | ... |  1  |  0x********  |  U32  |   Field  |
// +-----+-----+--------------+-------+----------+
// +-----+-----+--------------+-------+----------+
// | ... |  2  |  0x********  |  U32  |   Field  |
// +-----+-----+--------------+-------+----------++
// +-----+-----+--------------+-------+----------+
// | ... |  3  |  0x********  |  U32  |   Field  |
// +-----+-----+--------------+-------+----------+

// Note that COMPRESS will be computed by vector lookup argument protocol
/*fn ctl_correct_program_cpu<F: Field>() -> CrossTableLookup<F> {
    CrossTableLookup::new(
        vec![TableWithColumns::new(
            Table::Cpu,
            cpu_stark::ctl_data_with_program(),
            Some(cpu_stark::ctl_filter_with_program()),
        )],
        TableWithColumns::new(
            Table::Program,
            program_stark::ctl_data_with_cpu(),
            Some(program_stark::ctl_filter_with_cpu()),
        ),
        None,
    )
}*/

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use crate::generation::{generate_traces, GenerationInputs};
    use crate::stark::config::StarkConfig;
    use crate::stark::ola_stark::OlaStark;
    use crate::stark::proof::PublicValues;
    use crate::stark::prover::prove_with_traces;
    use crate::stark::serialization::Buffer;
    use crate::stark::stark::Stark;
    use crate::stark::util::trace_rows_to_poly_values;
    use crate::stark::verifier::verify_proof;
    use anyhow::Result;
    use assembler::encoder::encode_asm_from_json_file;
    use core::crypto::hash::Hasher;
    use core::crypto::ZkHasher;
    use core::merkle_tree::log::{StorageLog, WitnessStorageLog};
    use core::merkle_tree::tree::AccountTree;
    use core::program::binary_program::BinaryProgram;
    use core::program::Program;
    use core::state::state_storage::StateStorage;
    use core::types::account::Address;
    use core::types::merkle_tree::{encode_addr, tree_key_default};
    use core::types::{Field, GoldilocksField};
    use core::vm::transaction::init_tx_context_mock;
    use executor::load_tx::init_tape;
    use executor::trace::{gen_storage_hash_table, gen_storage_table};
    use executor::{BatchCacheManager, Process};
    use itertools::Itertools;
    use log::{debug, LevelFilter};
    use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::mem;
    use std::path::PathBuf;
    use std::process::exit;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    #[allow(dead_code)]
    const D: usize = 2;
    #[allow(dead_code)]
    type C = Blake3GoldilocksConfig;
    #[allow(dead_code)]
    type F = <C as GenericConfig<D>>::F;
    #[allow(dead_code)]
    type S = dyn Stark<F, D>;

    #[test]
    fn test_serialize() {
        let mut ola_stark = OlaStark::default();
        assert_eq!(ola_stark.bitwise_stark.get_compress_challenge(), None);
        assert_eq!(ola_stark.program_stark.get_compress_challenge(), None);
        let challenge1 = GoldilocksField::rand();
        let challenge2 = GoldilocksField::rand();
        ola_stark.bitwise_stark.set_compress_challenge(challenge1);
        ola_stark.program_stark.set_compress_challenge(challenge2);
        assert_eq!(ola_stark.bitwise_stark.get_compress_challenge(), Some(challenge1));
        assert_eq!(ola_stark.program_stark.get_compress_challenge(), Some(challenge2));
        let data = serde_json::to_string(&ola_stark).unwrap();
        let stark: OlaStark<GoldilocksField, 2> = serde_json::from_str(&data).unwrap();
        assert_eq!(ola_stark.bitwise_stark.get_compress_challenge(), stark.bitwise_stark.get_compress_challenge());
        assert_eq!(ola_stark.program_stark.get_compress_challenge(), stark.program_stark.get_compress_challenge());
    }

    #[test]
    fn fibo_loop_test() {
        let calldata = [10u64, 1u64, 2, 4185064725u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        test_by_asm_json("fib_asm.json".to_string(), Some(calldata), None)
    }

    #[test]
    fn fibo_recursive_decode() {
        test_by_asm_json("fibo_recursive.json".to_string(), None, None)
    }

    #[test]
    fn memory_test() {
        test_by_asm_json("memory.json".to_string(), None, None)
    }

    #[test]
    fn call_test() {
        test_by_asm_json("call.json".to_string(), None, None)
    }

    // #[test]
    // fn range_check_test() {
    //     test_by_asm_json("range_check.json".to_string(), None)
    // }

    // #[test]
    // fn bitwise_test() {
    //     test_by_asm_json("bitwise.json".to_string(), None)
    // }

    #[test]
    fn comparison_test() {
        test_by_asm_json("comparison.json".to_string(), None, None)
    }

    // #[test]
    // fn test_ola_prophet_hand_write() {
    //     test_by_asm_json("hand_write_prophet.json".to_string(), None);
    // }

    #[test]
    fn test_ola_prophet_sqrt() {
        let calldata = [144u64, 10u64, 2, 3509365327u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        test_by_asm_json("sqrt_prophet_asm.json".to_string(), Some(calldata), None);
    }

    // #[test]
    // fn test_ola_sqrt() {
    //     test_by_asm_json("sqrt.json".to_string(), None);
    // }

    #[test]
    fn test_ola_poseidon() {
        let call_data = vec![
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(1239976900),
        ];
        test_by_asm_json("poseidon_hash.json".to_string(), Some(call_data), None);
    }

    #[test]
    fn test_ola_storage() {
        let call_data = vec![
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(2364819430),
        ];
        test_by_asm_json("storage_u32.json".to_string(), Some(call_data), None);
    }

    #[test]
    fn test_ola_global() {
        let call_data = vec![
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(4171824493),
        ];
        test_by_asm_json("global.json".to_string(), Some(call_data), None);
    }

    #[test]
    fn test_ola_malloc() {
        test_by_asm_json("malloc.json".to_string(), None, None);
    }

    #[test]
    fn test_ola_vote() {
        let db_name = "vote_test".to_string();

        let init_calldata = [3u64, 1u64, 2u64, 3u64, 4, 2817135588u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        let vote_calldata = [2u64, 1, 2791810083u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        let winning_proposal_calldata = [0u64, 3186728800u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        let winning_name_calldata = [0u64, 363199787u64]
            .iter()
            .map(|v| GoldilocksField::from_canonical_u64(*v))
            .collect_vec();
        test_by_asm_json("vote.json".to_string(), Some(init_calldata), Some(db_name));
    }

    #[test]
    fn test_ola_mem_gep() {
        test_by_asm_json("mem_gep.json".to_string(), None, None);
    }

    #[test]
    fn test_ola_mem_gep_vector() {
        test_by_asm_json("mem_gep_vector.json".to_string(), None, None);
    }

    // #[test]
    // fn test_ola_string_assert() {
    //     test_by_asm_json("string_assert.json".to_string(), None);
    // }

    #[allow(unused)]
    pub fn test_by_asm_json(
        file_name: String,
        call_data: Option<Vec<GoldilocksField>>,
        db_name: Option<String>,
    ) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        let mut db = match db_name {
            Some(name) => {
                let mut db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                db_path.push("../executor/db_test/");
                db_path.push("../executor/db_test/");
                db_path.push(name);
                AccountTree::new_db_test(db_path.display().to_string())
            }
            _ => AccountTree::new_test(),
        };

        let program = encode_asm_from_json_file(program_path).unwrap();
        let hash = ZkHasher::default();
        let instructions = program.bytecode.split("\n");
        let code: Vec<_> = instructions
            .clone()
            .map(|e| GoldilocksField::from_canonical_u64(u64::from_str_radix(&e[2..], 16).unwrap()))
            .collect();
        let code_hash = hash.hash_bytes(&code);
        let mut prophets = HashMap::new();
        for item in program.prophets {
            prophets.insert(item.host as u64, item);
        }

        let mut program: Program = Program::default();

        for inst in instructions {
            program.instructions.push(inst.to_string());
        }

        let mut process = Process::new();
        process.addr_storage = Address::default();

        let tp_start = 0;

        let callee: Address = [
            GoldilocksField::from_canonical_u64(9),
            GoldilocksField::from_canonical_u64(10),
            GoldilocksField::from_canonical_u64(11),
            GoldilocksField::from_canonical_u64(12),
        ];
        let caller_addr = [
            GoldilocksField::from_canonical_u64(17),
            GoldilocksField::from_canonical_u64(18),
            GoldilocksField::from_canonical_u64(19),
            GoldilocksField::from_canonical_u64(20),
        ];
        let callee_exe_addr = [
            GoldilocksField::from_canonical_u64(13),
            GoldilocksField::from_canonical_u64(14),
            GoldilocksField::from_canonical_u64(15),
            GoldilocksField::from_canonical_u64(16),
        ];

        if let Some(calldata) = call_data {
            process.tp = GoldilocksField::from_canonical_u64(tp_start as u64);

            init_tape(
                &mut process,
                calldata,
                caller_addr,
                callee,
                callee_exe_addr,
                &init_tx_context_mock(),
            );
        }

        process.addr_code = callee_exe_addr;
        process.addr_storage = callee;
        program
            .trace
            .addr_program_hash
            .insert(encode_addr(&callee_exe_addr), code);

        db.process_block(vec![WitnessStorageLog {
            storage_log: StorageLog::new_write_log(callee_exe_addr, code_hash),
            previous_value: tree_key_default(),
        }]);
        let _ = db.save();

        let start = db.root_hash();

        process.program_log.push(WitnessStorageLog {
            storage_log: StorageLog::new_read_log(callee_exe_addr, code_hash),
            previous_value: tree_key_default(),
        });

        program.prophets = prophets;
        // FIXME: account tree is not used, merkle root cannot update.
        let res = process.execute(
            &mut program,
            &StateStorage::new_test(),
            &mut BatchCacheManager::default(),
        );
        match res {
            Ok(_) => {}
            Err(e) => {
                println!("execute err:{:?}", e);
                panic!("execute program failed");
            }
        }
        let hash_roots = gen_storage_hash_table(&mut process, &mut program, &mut db);
        gen_storage_table(&mut process, &mut program, hash_roots.unwrap()).unwrap();
        program.trace.start_end_roots = (start, db.root_hash());

        let inputs = GenerationInputs::default();

        let mut ola_stark = OlaStark::default();
        let (traces, public_values) = generate_traces(program, &mut ola_stark, inputs);

        let config = StarkConfig::standard_fast_config();
        let proof = prove_with_traces::<F, C, D>(
            &ola_stark,
            &config,
            traces,
            public_values,
            &mut TimingTree::default(),
        );

        if let Ok(proof) = proof {
            let ola_stark = OlaStark::default();
            let verify_res = verify_proof(ola_stark, proof, &config);
            println!("verify result:{:?}", verify_res);
        } else {
            println!("proof err:{:?}", proof);
        }
    }
}
