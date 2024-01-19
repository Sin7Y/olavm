use super::iter_ext::IteratorExt;
use super::TreeError;
use crate::crypto::hash::Hasher;

use crate::trace::trace::{HashTrace, PoseidonRow};
use crate::types::merkle_tree::constant::ROOT_TREE_DEPTH;
use crate::types::merkle_tree::{
    tree_key_to_u256, u256_to_tree_key, NodeEntry, TreeKey, TreeKeyU256, TreeValue,
};
use core::iter;
use itertools::Itertools;
use log::debug;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::crypto::poseidon_trace::PoseidonType;
use crate::mutex_data;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[macro_export]
macro_rules! compress_node {
    ($hasher: tt, $nei: expr, $current: expr, $odd: tt, $depth: tt) => {{
        let (left_hash, right_hash) = if $odd {
            ($nei, $current)
        } else {
            ($current, $nei)
        };

        let hash = if $depth == 0 {
            $hasher.compress(left_hash, right_hash, PoseidonType::Leaf)
        } else {
            $hasher.compress(left_hash, right_hash, PoseidonType::Branch)
        };

        let branch = NodeEntry::Branch {
            hash: hash.0,
            left_hash: left_hash.clone(),
            right_hash: right_hash.clone(),
        };
        (hash, branch)
    }};
}

/// Represents set of prepared updates to be applied to the given tree in batch.
/// To calculate actual patch, use [crate::UpdatesMap::calculate].
pub struct UpdatesBatch {
    pub(crate) updates: HashMap<
        // key affected by given storage update on respective tree level
        TreeKeyU256,
        Vec<Update>,
    >,
    pub(crate) pre_updates: HashMap<
        // key affected by given storage update on respective tree level
        TreeKeyU256,
        Vec<Update>,
    >,
}

/// Set of patches combined into one.
/// Each element represents changes from a single slot update.
pub type TreePatch = Vec<Vec<(TreeKey, NodeEntry)>>;

#[derive(Clone, Debug)]
pub struct Update {
    // operation index in a batch
    index: usize,
    // hashes of neighbour nodes on the path from root to the leaf
    uncles: Vec<TreeKey>,
    // all branch nodes that changed due to given update
    // empty initially; populated level-by-level during path calculate phase
    changes: Vec<(TreeKey, NodeEntry)>,
}

impl Update {
    pub fn new(index: usize, uncles: Vec<TreeKey>, key: TreeKey) -> Result<Self, TreeError> {
        let mut update = Self {
            index,
            uncles,
            changes: Vec::with_capacity(ROOT_TREE_DEPTH + 1),
        };
        update.changes.push((
            key,
            NodeEntry::Leaf {
                hash: update
                    .uncles
                    .pop()
                    .ok_or(TreeError::EmptyPatch(String::from(
                        "Empty unclues array in Update::new",
                    )))?,
            },
        ));
        Ok(update)
    }
}

impl UpdatesBatch {
    /// Instantiates new set of batch updates.
    pub(crate) fn new(
        updates: HashMap<TreeKeyU256, Vec<Update>>,
        pre_updates: HashMap<TreeKeyU256, Vec<Update>>,
    ) -> Self {
        Self {
            updates,
            pre_updates,
        }
    }

    /// Calculates new set of Merkle Trees produced by applying map of updates
    /// to the current tree. This calculation is parallelized over
    /// operations - all trees will be calculated in parallel.
    ///
    /// Memory and time: O(M * log2(N)), where
    /// - N - count of all leaf nodes (basically 2 in power of depth)
    /// - M - count of updates being applied
    pub fn calculate<H>(
        self,
        hasher: H,
    ) -> Result<(TreePatch, Arc<Mutex<Vec<(usize, HashTrace)>>>), TreeError>
    where
        H: Hasher<TreeValue> + Send + Sync,
    {
        let hash_trace = Arc::new(Mutex::new(Vec::new()));
        let mut cur_lvl_updates_map = self.updates;
        let mut cur_path_map = Arc::new(Mutex::new(self.pre_updates));

        for depth in 0..ROOT_TREE_DEPTH {
            let cur_changes: HashMap<_, _> = mutex_data!(cur_path_map)
                .iter()
                .map(|(key, updates)| {
                    let changes: Vec<_> = updates
                        .iter()
                        .map(|e| {
                            Ok(e.changes
                                .last()
                                .ok_or(TreeError::EmptyPatch(String::from(
                                    "Empty update array in pre_updates",
                                )))?
                                .1
                                .hash()
                                .clone())
                        })
                        .collect::<Result<Vec<_>, TreeError>>()?;
                    Ok((key.clone(), changes))
                })
                .collect::<Result<HashMap<_, _>, TreeError>>()?;

            let res = cur_lvl_updates_map
                .into_iter()
                .into_grouping_map_by(|(key, _)| key >> 1)
                .fold((None, None), |acc, _key, item| {
                    if item.0 % 2 == 1.into() {
                        (acc.0, Some(item.1))
                    } else {
                        (Some(item.1), acc.1)
                    }
                });

            cur_lvl_updates_map =
                res.into_par_iter()
                    .map(|(next_idx, (left_updates, right_updates))| {
                        let left_updates = left_updates
                            .into_iter()
                            .flat_map(|items| iter::repeat(false).zip(items));
                        debug!(
                            "left_updates len: {}",
                            left_updates.clone().collect::<Vec<_>>().len()
                        );

                        let right_updates = right_updates
                            .into_iter()
                            .flat_map(|items| iter::repeat(true).zip(items));
                        debug!(
                            "right_updates len: {}",
                            right_updates.clone().collect::<Vec<_>>().len()
                        );

                        let merged_ops: Vec<_> = left_updates
                            .merge_join_with_max_predecessor(
                                right_updates,
                                |(_, left), (_, right)| left.index.cmp(&right.index),
                                |(_, update)| update.changes.last().map(|(_, node)| node).cloned(),
                            )
                            .collect();

                        let ops_iter = merged_ops
                            // Parallel by operation index use
                            .into_par_iter()
                            .map(|((odd, mut update), nei)| {
                                let Update {
                                    uncles,
                                    changes,
                                    index,
                                } = &mut update;

                                let (cur_key, current_hash) = changes
                                    .last()
                                    .map(|(key, node)| (key, node.hash().clone()))
                                    .ok_or(TreeError::EmptyPatch(String::from(
                                        "Empty changes array in merged_ops",
                                    )))?;

                                let sibling_hash = uncles.pop().ok_or(TreeError::EmptyPatch(
                                    String::from("Empty uncles array in merged_ops"),
                                ))?;
                                let nei_hash = nei
                                    .flatten()
                                    .map(NodeEntry::into_hash)
                                    .unwrap_or(sibling_hash);

                                let cur_key = tree_key_to_u256(cur_key);

                                let update_index = mutex_data!(cur_path_map)
                                    .get(&cur_key)
                                    .ok_or(TreeError::EmptyPatch(format!(
                                        "No such key {} in cur_path_map",
                                        cur_key
                                    )))?
                                    .iter()
                                    .enumerate()
                                    .filter(|e| e.1.index == *index)
                                    .map(|e| (e.0))
                                    .collect::<Vec<_>>()[0];

                                let cur_change =
                                    cur_changes.get(&cur_key).ok_or(TreeError::EmptyPatch(
                                        format!("No such key {} in cur_changes", cur_key),
                                    ))?;
                                let pre_path = if update_index == 0 {
                                    mutex_data!(cur_path_map)
                                        .get(&cur_key)
                                        .ok_or(TreeError::EmptyPatch(format!(
                                            "No such key {} in cur_path_map",
                                            cur_key
                                        )))?
                                        .get(update_index)
                                        .ok_or(TreeError::EmptyPatch(format!(
                                            "No such data at index {} in cur_path_map",
                                            update_index
                                        )))?
                                        .uncles
                                        .last()
                                        .ok_or(TreeError::EmptyPatch(format!(
                                            "Empty uncles array in cur_path_map"
                                        )))?
                                        .clone()
                                } else {
                                    cur_change
                                        .get(update_index - 1)
                                        .ok_or(TreeError::EmptyPatch(format!(
                                            "No such data at index {} in cur_path_map",
                                            update_index - 1
                                        )))?
                                        .clone()
                                };

                                let (hash, branch) =
                                    compress_node!(hasher, &nei_hash, &current_hash, odd, depth);
                                let (pre_hash, _) =
                                    compress_node!(hasher, &nei_hash, &pre_path, odd, depth);

                                changes.push((u256_to_tree_key(&next_idx), branch.clone()));
                                mutex_data!(cur_path_map)
                                    .get_mut(&cur_key)
                                    .ok_or(TreeError::EmptyPatch(format!(
                                        "No such key {} in cur_path_map",
                                        cur_key
                                    )))?
                                    .get_mut(update_index)
                                    .ok_or(TreeError::EmptyPatch(format!(
                                        "No such data at index {} in cur_path_map",
                                        update_index
                                    )))?
                                    .changes
                                    .push((u256_to_tree_key(&next_idx), branch));

                                hash_trace
                                    .lock()
                                    .map_err(|err| {
                                        TreeError::MutexLockError(format!(
                                            "hash_trace lock failed with err: {}",
                                            err
                                        ))
                                    })?
                                    .push((
                                        *index,
                                        HashTrace(
                                            hash.1,
                                            current_hash.clone(),
                                            nei_hash.clone(),
                                            pre_hash.0,
                                            pre_path,
                                            pre_hash.1,
                                        ),
                                    ));
                                Ok(update)
                            })
                            .collect::<Result<Vec<_>, TreeError>>()?;

                        Ok((next_idx, ops_iter))
                    })
                    .collect::<Result<HashMap<_, _>, TreeError>>()?;

            let cur_path_map_raw: HashMap<_, _> = mutex_data!(cur_path_map)
                .iter_mut()
                .into_group_map_by(|e| e.0 >> 1)
                .iter_mut()
                .map(|e| {
                    let updates: Vec<_> =
                        e.1.iter_mut()
                            .map(|(_, ref mut updates)| {
                                let updates: Vec<_> = updates
                                    .iter_mut()
                                    .map(|update| {
                                        update.uncles.pop();
                                        update.clone()
                                    })
                                    .collect();
                                updates
                            })
                            .flat_map(|e| e)
                            .sorted_by(|a, b| Ord::cmp(&a.index, &b.index))
                            .collect();
                    (*e.0, updates)
                })
                .collect();
            cur_path_map = Arc::new(Mutex::new(cur_path_map_raw));
        }

        // Transforms map of leaf keys into an iterator of Merkle paths which produces
        // items sorted by operation index in increasing order.
        let patch: TreePatch = cur_lvl_updates_map
            .into_iter()
            .flat_map(|(_, updates)| updates.into_iter().map(|update| update.changes))
            .collect();
        debug!("tree patch:{:?}, len:{}", patch, patch.len());
        Ok((patch, hash_trace))
    }
}
