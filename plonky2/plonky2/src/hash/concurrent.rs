use core::slice;
use maybe_rayon::{MaybeParIterMut, MaybeParIter, IndexedParallelIterator, ParallelIterator};
use plonky2_field::cfft::uninit_vector;

use crate::plonk::config::Hasher;

use super::hash_types::RichField;

pub const MIN_CONCURRENT_LEAVES: usize = 1024;

/// Builds a all internal nodes of the Merkle using all available threads and stores the
/// results in a single vector such that root of the tree is at position 1, nodes immediately
/// under the root is at positions 2 and 3 etc.
pub fn build_merkle_nodes<F: RichField, H: Hasher<F>>(leaves: &[H::Hash]) -> Vec<H::Hash>
where 
    [(); H::HASH_SIZE]: {
    let n = leaves.len() / 2;

    // create un-initialized array to hold all intermediate nodes
    let mut nodes = unsafe { uninit_vector::<H::Hash>(2 * n) };
    nodes[0] = H::zero_hash();

    // re-interpret leaves as an array of two leaves fused together and use it to
    // build first row of internal nodes (parents of leaves)
    let two_leaves = unsafe { slice::from_raw_parts(leaves.as_ptr() as *const [H::Hash; 2], n) };
    nodes[n..]
        .par_iter_mut()
        .zip(two_leaves.par_iter())
        .for_each(|(target, source)| *target = H::two_to_one(source[0], source[1]));

    // calculate all other tree nodes, we can't use regular iterators  here because
    // access patterns are rather complicated - so, we use regular threads instead

    // number of sub-trees must always be a power of 2
    let num_subtrees = rayon::current_num_threads().next_power_of_two();
    let batch_size = n / num_subtrees;

    // re-interpret nodes as an array of two nodes fused together
    let two_nodes = unsafe { slice::from_raw_parts(nodes.as_ptr() as *const [H::Hash; 2], n) };

    // process each subtree in a separate thread
    rayon::scope(|s| {
        for i in 0..num_subtrees {
            let nodes = unsafe { &mut *(&mut nodes[..] as *mut [H::Hash]) };
            s.spawn(move |_| {
                let mut batch_size = batch_size / 2;
                let mut start_idx = n / 2 + batch_size * i;
                while start_idx >= num_subtrees {
                    for k in (start_idx..(start_idx + batch_size)).rev() {
                        nodes[k] = H::two_to_one(two_nodes[k][0], two_nodes[k][1]);
                    }
                    start_idx /= 2;
                    batch_size /= 2;
                }
            });
        }
    });

    // finish the tip of the tree
    for i in (1..num_subtrees).rev() {
        nodes[i] = H::two_to_one(two_nodes[i][0], two_nodes[i][1]);
    }

    nodes
}
