use crate::types::merkle_tree::TreeKey;
use plonky2::field::goldilocks_field::GoldilocksField;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Contracts {
    pub contracts: HashMap<TreeKey, Vec<GoldilocksField>>,
}
