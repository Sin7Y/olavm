use crate::types::merkle_tree::constant::ROOT_TREE_DEPTH;
use crate::types::merkle_tree::LevelIndex;
use web3::types::U256;

/// Calculates neighbor index for given index to have complete pair.
fn neighbor_idx(idx: U256, nei_flag: bool) -> U256 {
    if nei_flag {
        let mask = 1.into();
        idx ^ mask
    } else {
        idx
    }
}

/// Produces a full merkle path of neighbors for given leaf (including given
/// leaf itself) Used to calculate hash changes for branch nodes caused by leaf
/// update
pub fn idx_to_merkle_path(
    idx: U256,
    nei_flag: bool,
) -> impl DoubleEndedIterator<Item = LevelIndex> + Clone {
    let level_index: Vec<LevelIndex> = (1..=ROOT_TREE_DEPTH)
        .map(move |cur_depth| {
            (
                cur_depth as u16,
                neighbor_idx(idx >> (ROOT_TREE_DEPTH - cur_depth), nei_flag),
            )
        })
        .map(Into::into)
        .collect();
    level_index.into_iter()
}

#[allow(unused)]
pub(crate) fn children_idxs(level_idx: &LevelIndex) -> (LevelIndex, LevelIndex) {
    (
        (level_idx.0 .0 + 1, level_idx.0 .1 << 1).into(),
        (level_idx.0 .0 + 1, (level_idx.0 .1 << 1) + 1).into(),
    )
}
