use {
    crate::{
        builtins::builtin_stark::BuiltinStark, cpu_proof::cpu_stark::CpuStark,
        memory_proof::MemoryStark,
    },
    plonky2::{field::extension::Extendable, hash::hash_types::RichField},
};

#[derive(Clone)]
pub struct AllStark<F: RichField + Extendable<D>, const D: usize> {
    pub cpu_stark: CpuStark<F, D>,
    pub memory_stark: MemoryStark<F, D>,
    pub builtin_stark: BuiltinStark<F, D>,
    // pub cross_table_lookups: Vec<CrossTableLookup<F>>,
}
