use crate::error::ProcessorError;
use crate::{GoldilocksField, Process, MemRangeType, MEM_SPAN_SIZE};
use core::merkle_tree::tree::AccountTree;
use core::program::Program;
use core::trace::trace::{MemoryTraceCell, StorageHashRow};
use core::types::merkle_tree::constant::ROOT_TREE_DEPTH;
use core::types::merkle_tree::{tree_key_to_u256, TreeKeyU256, TREE_VALUE_LEN};
use log::debug;
use plonky2::field::types::{Field, Field64, PrimeField64};
use std::collections::HashMap;
pub fn gen_memory_table(
    process: &mut Process,
    program: &mut Program,
) -> Result<(), ProcessorError> {
    let mut origin_addr = 0;
    let mut origin_clk = 0;
    let mut diff_addr;
    let mut diff_addr_inv;
    let mut diff_clk;
    let mut diff_addr_cond;
    let mut first_row_flag = true;
    let mut first_heap_row_flag = true;

    for (field_addr, cells) in process.memory.trace.iter() {
        let mut new_addr_flag = true;

        let canonical_addr = GoldilocksField::from_noncanonical_u64(*field_addr).to_canonical_u64();
        for cell in cells {
            let mut rc_insert = Vec::new();
            let mut write_once_region_flag = false;
            debug!(
                "canonical_addr:{}, addr:{}, cell:{:?}",
                canonical_addr, field_addr, cell
            );

            if cell.region_prophet.is_one() {
                diff_addr_cond =
                    GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - canonical_addr);
                write_once_region_flag = true;
            } else if cell.region_heap.is_one() {
                diff_addr_cond = GoldilocksField::from_canonical_u64(
                    GoldilocksField::ORDER - MEM_SPAN_SIZE - canonical_addr,
                );
            } else {
                diff_addr_cond = GoldilocksField::ZERO;
            }
            if first_row_flag {
                let rc_value = GoldilocksField::ZERO;
                let trace_cell = MemoryTraceCell {
                    addr: GoldilocksField::from_canonical_u64(canonical_addr),
                    clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                    is_rw: cell.is_rw,
                    op: cell.op,
                    is_write: cell.is_write,
                    diff_addr: GoldilocksField::from_canonical_u64(0_u64),
                    diff_addr_inv: GoldilocksField::from_canonical_u64(0_u64),
                    diff_clk: GoldilocksField::from_canonical_u64(0_u64),
                    diff_addr_cond,
                    filter_looked_for_main: cell.filter_looked_for_main,
                    rw_addr_unchanged: GoldilocksField::from_canonical_u64(0_u64),
                    region_prophet: cell.region_prophet,
                    region_heap: cell.region_heap,
                    value: cell.value,
                    rc_value,
                };
                program.trace.memory.push(trace_cell);
                first_row_flag = false;
                new_addr_flag = false;
                if cell.region_heap == GoldilocksField::ONE {
                    first_heap_row_flag = false;
                }
            } else if new_addr_flag {
                debug!(
                        "canonical_addr:{}, origin_addr:{}, spec_region_flag:{}, diff_addr_cond:{}, first_heap_row_flag:{}",
                        canonical_addr, origin_addr, write_once_region_flag, diff_addr_cond, first_heap_row_flag
                    );

                diff_addr = GoldilocksField::from_canonical_u64(canonical_addr - origin_addr);
                let rc_value;

                if write_once_region_flag {
                    diff_addr_inv = GoldilocksField::ZERO;
                    rc_value = diff_addr_cond;
                    rc_insert.push((diff_addr_cond, MemRangeType::MemRegion));
                } else if cell.region_heap == GoldilocksField::ONE && first_heap_row_flag {
                    diff_addr = GoldilocksField::ZERO;
                    diff_addr_inv = GoldilocksField::ZERO;
                    rc_value = GoldilocksField::ZERO;
                    rc_insert.push((diff_addr_cond, MemRangeType::MemRegion));
                    first_heap_row_flag = false;
                } else {
                    diff_addr_inv = diff_addr.inverse();
                    rc_value = diff_addr;
                    rc_insert.push((rc_value, MemRangeType::MemSort));
                    if cell.region_heap == GoldilocksField::ONE {
                        rc_insert.push((diff_addr_cond, MemRangeType::MemRegion));
                    }
                }
                diff_clk = GoldilocksField::ZERO;
                let trace_cell = MemoryTraceCell {
                    addr: GoldilocksField::from_canonical_u64(canonical_addr),
                    clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                    is_rw: cell.is_rw,
                    op: cell.op,
                    is_write: cell.is_write,
                    diff_addr,
                    diff_addr_inv,
                    diff_clk,
                    diff_addr_cond,
                    filter_looked_for_main: cell.filter_looked_for_main,
                    rw_addr_unchanged: GoldilocksField::from_canonical_u64(0_u64),
                    region_prophet: cell.region_prophet,
                    region_heap: cell.region_heap,
                    value: cell.value,
                    rc_value,
                };
                program.trace.memory.push(trace_cell);
                new_addr_flag = false;
            } else {
                diff_addr = GoldilocksField::ZERO;
                diff_addr_inv = GoldilocksField::ZERO;
                diff_clk = GoldilocksField::from_canonical_u64(cell.clk as u64 - origin_clk);
                let mut rw_addr_unchanged = GoldilocksField::ONE;
                let rc_value;
                let mem_filter_type;
                if cell.is_rw == GoldilocksField::ZERO {
                    rw_addr_unchanged = GoldilocksField::ZERO;
                    rc_value = diff_addr_cond;
                    mem_filter_type = MemRangeType::MemRegion;
                } else {
                    rc_value = diff_clk;
                    mem_filter_type = MemRangeType::MemSort;
                }
                rc_insert.push((rc_value, mem_filter_type));
                if cell.region_heap == GoldilocksField::ONE {
                    rc_insert.push((diff_addr_cond, MemRangeType::MemRegion));
                }

                let trace_cell = MemoryTraceCell {
                    addr: GoldilocksField::from_canonical_u64(canonical_addr),
                    clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                    is_rw: cell.is_rw,
                    op: cell.op,
                    is_write: cell.is_write,
                    diff_addr,
                    diff_addr_inv,
                    diff_clk,
                    diff_addr_cond,
                    filter_looked_for_main: cell.filter_looked_for_main,
                    rw_addr_unchanged,
                    region_prophet: cell.region_prophet,
                    region_heap: cell.region_heap,
                    value: cell.value,
                    rc_value,
                };
                program.trace.memory.push(trace_cell);
            }
            for item in &rc_insert {
                if item.0.to_canonical_u64() > u32::MAX as u64 {
                    return Err(ProcessorError::U32RangeCheckFail);
                }
            }
            rc_insert.iter_mut().for_each(|e| {
                program.trace.insert_rangecheck(
                    e.0,
                    (
                        GoldilocksField::ONE
                            * GoldilocksField::from_canonical_u8(1 - e.1.clone() as u8),
                        GoldilocksField::ZERO,
                        GoldilocksField::ZERO,
                        GoldilocksField::ZERO,
                        GoldilocksField::ONE
                            * GoldilocksField::from_canonical_u8(e.1.clone() as u8),
                    ),
                )
            });

            origin_clk = cell.clk as u64;
        }
        origin_addr = canonical_addr;
    }
    Ok(())
}

pub fn gen_storage_hash_table(
    process: &mut Process,
    program: &mut Program,
    account_tree: &mut AccountTree,
) -> Vec<[GoldilocksField; TREE_VALUE_LEN]> {
    let trace = std::mem::replace(&mut process.storage_log, Vec::new());

    let hash_traces = account_tree.process_block(trace.iter());
    let _ = account_tree.save();

    let mut root_hashes = Vec::new();

    for (chunk, log) in hash_traces.chunks(ROOT_TREE_DEPTH).enumerate().zip(trace) {
        let mut root_hash = [GoldilocksField::ZERO; TREE_VALUE_LEN];
        root_hash.clone_from_slice(&chunk.1.last().unwrap().0.output[0..4]);
        root_hashes.push(root_hash);
        let mut acc = GoldilocksField::ZERO;
        let key = tree_key_to_u256(&log.storage_log.key);

        let rows: Vec<_> = chunk
            .1
            .iter()
            .rev()
            .enumerate()
            .map(|item| {
                let layer_bit = ((key >> (255 - item.0)) & TreeKeyU256::one()).as_u64();
                let layer = (item.0 + 1) as u64;

                acc = acc * GoldilocksField::from_canonical_u64(2)
                    + GoldilocksField::from_canonical_u64(layer_bit);
                let mut deltas = [GoldilocksField::ZERO; TREE_VALUE_LEN];
                if layer_bit == 1 {
                    for i in 0..TREE_VALUE_LEN {
                        deltas[i] = item.1 .2[i] - item.1 .1[i]
                    }
                } else if layer_bit == 0 {
                } else {
                    panic!("layer_bit is 0 or 1");
                }
                let row = StorageHashRow {
                    idx_storage: (chunk.0 + 1) as u64,
                    layer,
                    layer_bit,
                    addr_acc: acc,
                    is_layer64: (layer == 64),
                    is_layer128: (layer == 128),
                    is_layer192: (layer == 192),
                    is_layer256: (layer == 256),
                    addr: log.storage_log.key,
                    caps: [
                        GoldilocksField::ONE,
                        GoldilocksField::ZERO,
                        GoldilocksField::ZERO,
                        GoldilocksField::ZERO,
                    ],
                    paths: item.1 .1,
                    siblings: item.1 .2,
                    deltas,
                    full_0_1: item.1 .0.full_0_1,
                    full_0_2: item.1 .0.full_0_2,
                    full_0_3: item.1 .0.full_0_3,
                    partial: item.1 .0.partial,
                    full_1_0: item.1 .0.full_1_0,
                    full_1_1: item.1 .0.full_1_1,
                    full_1_2: item.1 .0.full_1_2,
                    full_1_3: item.1 .0.full_1_3,
                    output: item.1 .0.output,
                };
                if layer % 64 == 0 {
                    acc = GoldilocksField::ZERO;
                }
                row
            })
            .collect();
        program.trace.builtin_storage_hash.extend(rows);
    }
    root_hashes
}

pub fn gen_storage_table(
    process: &mut Process,
    program: &mut Program,
    hash_roots: Vec<[GoldilocksField; 4]>,
) -> Result<(), ProcessorError> {
    if hash_roots.is_empty() {
        return Ok(());
    }

    let trace = std::mem::replace(&mut process.storage.trace, HashMap::new());

    let mut traces: Vec<_> = trace.into_iter().flat_map(|e| e.1).collect();
    traces.sort_by(|a, b| a.cmp(b));
    let mut pre_clk = 0;
    for (item, root) in traces.iter().enumerate().zip(hash_roots) {
        let mut diff_clk = 0;
        if item.0 != 0 {
            diff_clk = item.1.clk - pre_clk;
        }
        program.trace.insert_storage(
            item.1.clk,
            diff_clk,
            item.1.op,
            root,
            item.1.addr,
            item.1.value,
        );

        program.trace.insert_rangecheck(
            GoldilocksField::from_canonical_u64(diff_clk as u64),
            (
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
                GoldilocksField::ONE,
                GoldilocksField::ZERO,
            ),
        );
        pre_clk = item.1.clk;
    }
    Ok(())
}
