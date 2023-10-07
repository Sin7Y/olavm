use super::iter_ext::IteratorExt;
use super::TreeError;
use crate::crypto::hash::Hasher;

use crate::trace::trace::PoseidonRow;
use crate::types::merkle_tree::constant::ROOT_TREE_DEPTH;
use crate::types::merkle_tree::{u256_to_tree_key, NodeEntry, TreeKey, TreeKeyU256, TreeValue};
use core::iter;
use itertools::Itertools;
use log::debug;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::crypto::poseidon_trace::PoseidonType;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Represents set of prepared updates to be applied to the given tree in batch.
/// To calculate actual patch, use [crate::UpdatesMap::calculate].
pub struct UpdatesBatch {
    pub(crate) updates: HashMap<
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
    // empty initially; populated level-by-level during path calculate phasE
    changes: Vec<(TreeKey, NodeEntry)>,
}

impl Update {
    pub fn new(index: usize, uncles: Vec<TreeKey>, key: TreeKey) -> Self {
        let mut update = Self {
            index,
            uncles,
            changes: Vec::with_capacity(ROOT_TREE_DEPTH + 1),
        };
        update.changes.push((
            key,
            NodeEntry::Leaf {
                hash: update.uncles.pop().unwrap(),
            },
        ));
        update
    }
}

impl UpdatesBatch {
    /// Instantiates new set of batch updates.
    pub(crate) fn new(updates: HashMap<TreeKeyU256, Vec<Update>>) -> Self {
        Self { updates }
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
    ) -> Result<
        (
            TreePatch,
            Arc<Mutex<Vec<(usize, (PoseidonRow, TreeValue, TreeValue))>>>,
        ),
        TreeError,
    >
    where
        H: Hasher<TreeValue> + Send + Sync,
    {
        let hash_trace = Arc::new(Mutex::new(Vec::new()));
        let res_map = (0..ROOT_TREE_DEPTH).fold(self.updates, |cur_lvl_updates_map, depth| {
            // Calculate next level map based on current in parallel
            let res = cur_lvl_updates_map
                .into_iter()
                .into_grouping_map_by(|(key, _)| key >> 1)
                .fold((None, None), |acc, _key, item| {
                    if item.0 % 2 == 1.into() {
                        (acc.0, Some(item.1))
                    } else {
                        (Some(item.1), acc.1)
                    }
                })
                // Parallel by vertex key family
                .into_par_iter()
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

                            let current_hash =
                                changes.last().map(|(_, node)| node.hash()).unwrap().clone();

                            let sibling_hash = uncles.pop().unwrap();
                            let nei_hash = nei
                                .flatten()
                                .map(NodeEntry::into_hash)
                                .unwrap_or(sibling_hash);

                            // Hash current node with its neighbor
                            let (left_hash, right_hash) = if odd {
                                (&nei_hash, &current_hash)
                            } else {
                                (&current_hash, &nei_hash)
                            };

                            let hash = if depth == 0 {
                                hasher.compress(left_hash, right_hash, PoseidonType::Leaf)
                            } else {
                                hasher.compress(left_hash, right_hash, PoseidonType::Branch)
                            };
                            let branch = NodeEntry::Branch {
                                hash: hash.0,
                                left_hash: left_hash.clone(),
                                right_hash: right_hash.clone(),
                            };
                            changes.push((u256_to_tree_key(&next_idx), branch));
                            hash_trace
                                .lock()
                                .unwrap()
                                .push((*index, (hash.1, current_hash.clone(), nei_hash.clone())));
                            update
                        });

                    (next_idx, ops_iter.collect())
                })
                .collect();
            res
        });
        // Transforms map of leaf keys into an iterator of Merkle paths which produces
        // items sorted by operation index in increasing order.
        let patch: TreePatch = res_map
            .into_iter()
            .flat_map(|(_, updates)| updates.into_iter().map(|update| update.changes))
            .collect();
        debug!("tree patch:{:?}, len:{}", patch, patch.len());
        Ok((patch, hash_trace))
    }
}
