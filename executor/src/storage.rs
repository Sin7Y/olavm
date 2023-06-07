use core::types::merkle_tree::TreeKey;
use core::types::merkle_tree::TreeValue;
use core::types::merkle_tree::ZkHash;
use plonky2::field::goldilocks_field::GoldilocksField;
use std::cmp::Ordering;
use std::collections::{HashMap};
use std::marker::Destruct;

use core::types::merkle_tree::{tree_key_default};
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct StorageCell {
    pub clk: u32,
    pub op: GoldilocksField,
    pub root: ZkHash,
    pub addr: TreeKey,
    pub value: TreeValue,
}

impl Ord for StorageCell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.clk.cmp(&other.clk)
    }

    fn max(self, _other: Self) -> Self
    where
        Self: Sized,
        Self: Destruct,
    {
        todo!()
    }

    fn min(self, _other: Self) -> Self
    where
        Self: Sized,
        Self: Destruct,
    {
        todo!()
    }

    fn clamp(self, _min: Self, _max: Self) -> Self
    where
        Self: Sized,
        Self: Destruct,
        Self: PartialOrd,
    {
        todo!()
    }
}

impl PartialOrd for StorageCell {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.clk.cmp(&rhs.clk))
    }

    fn lt(&self, rhs: &Self) -> bool {
        self.clk < rhs.clk
    }

    fn le(&self, rhs: &Self) -> bool {
        self.clk <= rhs.clk
    }

    fn gt(&self, rhs: &Self) -> bool {
        self.clk > rhs.clk
    }

    fn ge(&self, rhs: &Self) -> bool {
        self.clk >= rhs.clk
    }
}

#[derive(Debug, Default)]
pub struct StorageTree {
    pub trace: HashMap<TreeKey, Vec<StorageCell>>,
}

impl StorageTree {
    pub fn read(
        &mut self,
        clk: u32,
        op: GoldilocksField,
        addr: TreeKey,
        root: ZkHash,
    ) -> TreeValue {
        // look up the previous value in the appropriate address trace and add (clk,
        // prev_value) to it; if this is the first time we access this address,
        // create address trace for it with entry (clk, [ZERO, 4]). in both
        // cases, return the last value in the address trace.
        self.trace
            .entry(addr)
            .and_modify(|addr_trace| {
                let last_value = addr_trace.last().expect("empty address trace").value;
                let new_value = StorageCell {
                    clk,
                    op,
                    addr,
                    root,
                    value: last_value,
                };
                addr_trace.push(new_value);
            })
            .or_insert_with(|| {
                let new_value = StorageCell {
                    clk,
                    op,
                    addr,
                    root,
                    value: tree_key_default(),
                };
                vec![new_value]
            })
            .last()
            .expect("empty address trace")
            .value
    }

    pub fn write(
        &mut self,
        clk: u32,
        op: GoldilocksField,
        addr: TreeKey,
        value: TreeValue,
        root: ZkHash,
    ) {
        // add a memory access to the appropriate address trace; if this is the first
        // time we access this address, initialize address trace.
        let new_cell = StorageCell {
            clk,
            op,
            addr,
            value,
            root,
        };
        self.trace
            .entry(addr)
            .and_modify(|addr_trace| addr_trace.push(new_cell))
            .or_insert_with(|| vec![new_cell]);
    }
}
