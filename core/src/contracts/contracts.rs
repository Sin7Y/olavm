use std::collections::HashMap;
use plonky2::field::goldilocks_field::GoldilocksField;
use crate::types::merkle_tree::TreeKey;

#[derive(Debug)]
pub struct Contracts {
    pub contracts: HashMap<TreeKey, Vec<GoldilocksField>>,
}

