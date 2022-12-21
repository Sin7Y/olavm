// 2022-12-19: written by xb

/* CMP_Table construction as follows:
+-----+-----+-----+------+
| TAG | op0 | op1 | diff | 
+-----+-----+-----+------+
+-----+-----+-----+------+
|  1  |  a  |  b  | a - b | 
+-----+-----+-----+------+
+-----+-----+-----+------+
|  1  |  a  |  b  | a - b | 
+-----+-----+-----+------+
+-----+-----+-----+------+
|  1  |  a  |  b  | a - b | 
+-----+-----+-----+------+

Constraints as follows:
1. addition relation
    op0 - (op1 + diff) = 0
2. Cross Lookup for diff , assume that the input in U32 type
    Lookup {<diff>; rangecheck}
*/
//Identify different Rangecheck TABLE
// 0 => Main TABLE
// 1 => GTE  TABLE
pub(crate) const TAG: usize  = 0;

pub(crate) const OP0: usize  = TAG + 1;
pub(crate) const OP1: usize  = OP0 + 1;
pub(crate) const DIFF: usize  = OP1 + 1;

pub(crate) const COL_NUM_CMP: usize  = DIFF + 1; //4