// add by xb 2022-12-27
// This columns designs refer to /plonky2/gates/poseidon.rs 
// 135 columns number

// INPUT DATA (12)
pub(crate) const INPUT_0: usize = 0;
pub(crate) const INPUT_1: usize = INPUT_0 + 1;
pub(crate) const INPUT_2: usize = INPUT_1 + 1;
pub(crate) const INPUT_3: usize = INPUT_2 + 1;
pub(crate) const INPUT_4: usize = INPUT_3 + 1;
pub(crate) const INPUT_5: usize = INPUT_4 + 1;
pub(crate) const INPUT_6: usize = INPUT_5 + 1;
pub(crate) const INPUT_7: usize = INPUT_6 + 1;
pub(crate) const INPUT_8: usize = INPUT_7 + 1;
pub(crate) const INPUT_9: usize = INPUT_8 + 1;
pub(crate) const INPUT_10: usize = INPUT_9 + 1;
pub(crate) const INPUT_11: usize = INPUT_10 + 1; // 11
// OUTPUT DATA (12)
pub(crate) const OUTPUT_0: usize = INPUT_11 + 1; // 12
pub(crate) const OUTPUT_1: usize = OUTPUT_0 + 1;
pub(crate) const OUTPUT_2: usize = OUTPUT_1 + 1;
pub(crate) const OUTPUT_3: usize = OUTPUT_2 + 1;
pub(crate) const OUTPUT_4: usize = OUTPUT_3 + 1;
pub(crate) const OUTPUT_5: usize = OUTPUT_4 + 1;
pub(crate) const OUTPUT_6: usize = OUTPUT_5 + 1;
pub(crate) const OUTPUT_7: usize = OUTPUT_6 + 1;
pub(crate) const OUTPUT_8: usize = OUTPUT_7 + 1;
pub(crate) const OUTPUT_9: usize = OUTPUT_8 + 1;
pub(crate) const OUTPUT_10: usize = OUTPUT_9 + 1;
pub(crate) const OUTPUT_11: usize = OUTPUT_10 + 1; // 23
// SWAP FLAG (1)
pub(crate) const SWAP: usize = OUTPUT_11 + 1; // 24
// DELTA DATA (4)
pub(crate) const DELTA_0: usize = SWAP + 1; // 25
pub(crate) const DELTA_1: usize = DELTA_0 + 1; 
pub(crate) const DELTA_2: usize = DELTA_1 + 1; 
pub(crate) const DELTA_3: usize = DELTA_2 + 1; // 28
// (FFR)FIRST FULL ROUND STATE (36 = 12 * 3)
// ROUND 0
pub(crate) const FFR_0_0: usize = DELTA_3 + 1; // 29
pub(crate) const FFR_0_1: usize = FFR_0_0 + 1;
pub(crate) const FFR_0_2: usize = FFR_0_1 + 1;
pub(crate) const FFR_0_3: usize = FFR_0_2 + 1;
pub(crate) const FFR_0_4: usize = FFR_0_3 + 1;
pub(crate) const FFR_0_5: usize = FFR_0_4 + 1;
pub(crate) const FFR_0_6: usize = FFR_0_5 + 1;
pub(crate) const FFR_0_7: usize = FFR_0_6 + 1;
pub(crate) const FFR_0_8: usize = FFR_0_7 + 1;
pub(crate) const FFR_0_9: usize = FFR_0_8 + 1;
pub(crate) const FFR_0_10: usize = FFR_0_9 + 1;
pub(crate) const FFR_0_11: usize = FFR_0_10 + 1; // 40
// ROUND 1
pub(crate) const FFR_1_0: usize = FFR_0_11 + 1; // 41
pub(crate) const FFR_1_1: usize = FFR_1_0 + 1;
pub(crate) const FFR_1_2: usize = FFR_1_1 + 1;
pub(crate) const FFR_1_3: usize = FFR_1_2 + 1;
pub(crate) const FFR_1_4: usize = FFR_1_3 + 1;
pub(crate) const FFR_1_5: usize = FFR_1_4 + 1;
pub(crate) const FFR_1_6: usize = FFR_1_5 + 1;
pub(crate) const FFR_1_7: usize = FFR_1_6 + 1;
pub(crate) const FFR_1_8: usize = FFR_1_7 + 1;
pub(crate) const FFR_1_9: usize = FFR_1_8 + 1;
pub(crate) const FFR_1_10: usize = FFR_1_9 + 1;
pub(crate) const FFR_1_11: usize = FFR_1_10 + 1; // 52
// ROUND 2
pub(crate) const FFR_2_0: usize = FFR_1_11 + 1; // 23
pub(crate) const FFR_2_1: usize = FFR_2_0 + 1;
pub(crate) const FFR_2_2: usize = FFR_2_1 + 1;
pub(crate) const FFR_2_3: usize = FFR_2_2 + 1;
pub(crate) const FFR_2_4: usize = FFR_2_3 + 1;
pub(crate) const FFR_2_5: usize = FFR_2_4 + 1;
pub(crate) const FFR_2_6: usize = FFR_2_5 + 1;
pub(crate) const FFR_2_7: usize = FFR_2_6 + 1;
pub(crate) const FFR_2_8: usize = FFR_2_7 + 1;
pub(crate) const FFR_2_9: usize = FFR_2_8 + 1;
pub(crate) const FFR_2_10: usize = FFR_2_9 + 1;
pub(crate) const FFR_2_11: usize = FFR_2_10 + 1; // 64
// (PR)PARITIAL ROUND STATE (22)
pub(crate) const PR_0: usize = DELTA_3 + 1; // 65
pub(crate) const PR_1: usize = PR_0 + 1;
pub(crate) const PR_2: usize = PR_1 + 1;
pub(crate) const PR_3: usize = PR_2 + 1;
pub(crate) const PR_4: usize = PR_3 + 1;
pub(crate) const PR_5: usize = PR_4 + 1;
pub(crate) const PR_6: usize = PR_5 + 1;
pub(crate) const PR_7: usize = PR_6 + 1;
pub(crate) const PR_8: usize = PR_7 + 1;
pub(crate) const PR_9: usize = PR_8 + 1;
pub(crate) const PR_10: usize = PR_9 + 1;
pub(crate) const PR_11: usize = PR_10 + 1; 
pub(crate) const PR_12: usize = PR_11 + 1; 
pub(crate) const PR_13: usize = PR_12 + 1;
pub(crate) const PR_14: usize = PR_13 + 1;
pub(crate) const PR_15: usize = PR_14 + 1;
pub(crate) const PR_16: usize = PR_15 + 1;
pub(crate) const PR_17: usize = PR_16 + 1;
pub(crate) const PR_18: usize = PR_17 + 1;
pub(crate) const PR_19: usize = PR_18 + 1;
pub(crate) const PR_20: usize = PR_19 + 1;
pub(crate) const PR_21: usize = PR_20 + 1; // 86
// (SFR)SECOND FULL ROUND STATE (48 = 12 * 4)
// ROUND 0
pub(crate) const SFR_0_0: usize = PR_21 + 1; //87
pub(crate) const SFR_0_1: usize = SFR_0_0 + 1;
pub(crate) const SFR_0_2: usize = SFR_0_1 + 1;
pub(crate) const SFR_0_3: usize = SFR_0_2 + 1;
pub(crate) const SFR_0_4: usize = SFR_0_3 + 1;
pub(crate) const SFR_0_5: usize = SFR_0_4 + 1;
pub(crate) const SFR_0_6: usize = SFR_0_5 + 1;
pub(crate) const SFR_0_7: usize = SFR_0_6 + 1;
pub(crate) const SFR_0_8: usize = SFR_0_7 + 1;
pub(crate) const SFR_0_9: usize = SFR_0_8 + 1;
pub(crate) const SFR_0_10: usize = SFR_0_9 + 1;
pub(crate) const SFR_0_11: usize = SFR_0_10 + 1; 
// ROUND 1
pub(crate) const SFR_1_0: usize = SFR_0_11 + 1; 
pub(crate) const SFR_1_1: usize = SFR_1_0 + 1;
pub(crate) const SFR_1_2: usize = SFR_1_1 + 1;
pub(crate) const SFR_1_3: usize = SFR_1_2 + 1;
pub(crate) const SFR_1_4: usize = SFR_1_3 + 1;
pub(crate) const SFR_1_5: usize = SFR_1_4 + 1;
pub(crate) const SFR_1_6: usize = SFR_1_5 + 1;
pub(crate) const SFR_1_7: usize = SFR_1_6 + 1;
pub(crate) const SFR_1_8: usize = SFR_1_7 + 1;
pub(crate) const SFR_1_9: usize = SFR_1_8 + 1;
pub(crate) const SFR_1_10: usize = SFR_1_9 + 1;
pub(crate) const SFR_1_11: usize = SFR_1_10 + 1; 
// ROUND 2
pub(crate) const SFR_2_0: usize = SFR_1_11 + 1; 
pub(crate) const SFR_2_1: usize = SFR_2_0 + 1;
pub(crate) const SFR_2_2: usize = SFR_2_1 + 1;
pub(crate) const SFR_2_3: usize = SFR_2_2 + 1;
pub(crate) const SFR_2_4: usize = SFR_2_3 + 1;
pub(crate) const SFR_2_5: usize = SFR_2_4 + 1;
pub(crate) const SFR_2_6: usize = SFR_2_5 + 1;
pub(crate) const SFR_2_7: usize = SFR_2_6 + 1;
pub(crate) const SFR_2_8: usize = SFR_2_7 + 1;
pub(crate) const SFR_2_9: usize = SFR_2_8 + 1;
pub(crate) const SFR_2_10: usize = SFR_2_9 + 1;
pub(crate) const SFR_2_11: usize = SFR_2_10 + 1;
// ROUND 3
pub(crate) const SFR_3_0: usize = SFR_2_11 + 1; 
pub(crate) const SFR_3_1: usize = SFR_3_0 + 1;
pub(crate) const SFR_3_2: usize = SFR_3_1 + 1;
pub(crate) const SFR_3_3: usize = SFR_3_2 + 1;
pub(crate) const SFR_3_4: usize = SFR_3_3 + 1;
pub(crate) const SFR_3_5: usize = SFR_3_4 + 1;
pub(crate) const SFR_3_6: usize = SFR_3_5 + 1;
pub(crate) const SFR_3_7: usize = SFR_3_6 + 1;
pub(crate) const SFR_3_8: usize = SFR_3_7 + 1;
pub(crate) const SFR_3_9: usize = SFR_3_8 + 1;
pub(crate) const SFR_3_10: usize = SFR_3_9 + 1;
pub(crate) const SFR_3_11: usize = SFR_3_10 + 1; // 134

// columns numbers
pub(crate) const COL_NUM: usize = SFR_3_11 + 1; // 135
