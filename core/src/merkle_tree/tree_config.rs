use crate::crypto::hash::Hasher;
use crate::types::merkle_tree::constant::ROOT_TREE_DEPTH;
use crate::types::merkle_tree::{NodeEntry, TreeKey, TreeValue, ZkHash, TREE_VALUE_LEN};

use crate::crypto::poseidon_trace::PoseidonType::{Branch, Leaf};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use std::iter::once;
use std::sync::Arc;

use super::TreeError;

#[derive(Debug)]
struct TreeConfigInner<H> {
    /// Hash generator used to hash entries.
    pub(crate) hasher: H,
    /// Precalculated empty leaf tree hashes. Start from leaf.
    pub(crate) empty_tree: Vec<NodeEntry>,
}

/// Shared configuration for Sparse Merkle Tree.
#[derive(Clone, Debug)]
pub struct TreeConfig<H> {
    inner: Arc<TreeConfigInner<H>>,
}

impl<H> TreeConfig<H>
where
    H: Hasher<TreeValue>,
{
    /// Creates new shared config with supplied params.
    pub fn new(hasher: H) -> Result<Self, TreeError> {
        let empty_hashes = Self::calc_default_hashes(ROOT_TREE_DEPTH, &hasher)
            .map_err(|err| TreeError::EmptyPatch(err))?;
        Ok(Self {
            inner: Arc::new(TreeConfigInner {
                empty_tree: Self::calc_empty_tree(&empty_hashes),
                hasher,
            }),
        })
    }

    /// Produces tree with all leaves having default value (empty).
    fn calc_empty_tree(hashes: &[TreeKey]) -> Vec<NodeEntry> {
        let mut empty_tree: Vec<_> = hashes
            .iter()
            .rev()
            .zip(once(None).chain(hashes.iter().rev().map(Some)))
            .map(|(hash, prev_hash)| match prev_hash {
                None => NodeEntry::Leaf { hash: hash.clone() },
                Some(prev_hash) => NodeEntry::Branch {
                    hash: hash.clone(),
                    left_hash: prev_hash.clone(),
                    right_hash: prev_hash.clone(),
                },
            })
            .collect();

        empty_tree.reverse();

        empty_tree
    }

    /// Returns reference to precalculated empty Merkle Tree hashes starting
    /// from leaf.
    pub fn empty_tree(&self) -> &[NodeEntry] {
        &self.inner.empty_tree
    }

    pub fn empty_leaf(_hasher: &H) -> ZkHash {
        [GoldilocksField::ZERO; TREE_VALUE_LEN]
    }

    pub fn default_root_hash(&self) -> ZkHash {
        self.empty_tree().first().cloned().unwrap().into_hash()
    }

    /// Returns current hasher.
    pub fn hasher(&self) -> &H {
        &self.inner.hasher
    }

    /// Calculates default empty leaf hashes for given types.
    fn calc_default_hashes(depth: usize, hasher: &H) -> Result<Vec<ZkHash>, String> {
        let mut def_hashes = Vec::with_capacity(depth + 1);
        def_hashes.push(Self::empty_leaf(hasher));
        for index in 0..depth {
            let last_hash = def_hashes.last().ok_or(format!("Empty hash arry"))?;
            let hash = if index == 0 {
                hasher.compress(last_hash, last_hash, Leaf)
            } else {
                hasher.compress(last_hash, last_hash, Branch)
            };

            def_hashes.push(hash.0);
        }
        def_hashes.reverse();

        Ok(def_hashes)
    }
}
