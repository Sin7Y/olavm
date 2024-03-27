use core::{
    crypto::poseidon_trace::{
        calculate_arbitrary_poseidon_and_generate_intermediate_trace,
        calculate_poseidon_and_generate_intermediate_trace,
    },
    trace::{
        exe_trace::{
            BitwiseExePiece, CmpExePiece, CpuExePiece, MemExePiece, PoseidonPiece, RcExePiece,
            SCCallPiece, StorageExePiece, TapeExePiece, TxExeTrace,
        },
        trace::{
            BitwiseCombinedRow, CmpRow, MemoryTraceCell, PoseidonChunkRow, PoseidonRow,
            RangeCheckRow, RegisterSelector, SCCallRow, Step, TapeRow, Trace,
        },
    },
    types::{merkle_tree::encode_addr, Field, GoldilocksField},
    utils::split_u16_limbs_from_field,
    vm::{
        hardware::{
            ContractAddress, ExeContext, MEM_HEAP_REGION, MEM_PROPHET_REGION, MEM_STACK_REGION,
        },
        opcodes::OlaOpcode,
    },
};
use std::{cmp::min, collections::HashMap};

// #[derive(Debug, Clone)]
// pub struct TxExeTrace {
//     pub programs: Vec<(ContractAddress, Vec<u64>)>, // contract address to
// bytecode     pub cpu: Vec<(u64, u64, ExeContext, Vec<CpuExePiece>)>, /*
// call_sc_cnt, env_idx, context, trace.
//                                                      * Sorted by execution
//                                                        env. */
//     pub env_mem: HashMap<u64, Vec<MemExePiece>>, // env_id to mem, mem not
// sorted yet.     pub rc: Vec<RcExePiece>,                     /* rc only
// triggered by range_check
//                                                   * opcode. */
//     pub bitwise: Vec<BitwiseExePiece>,
//     pub cmp: Vec<CmpExePiece>,
//     pub poseidon: Vec<PoseidonPiece>, // poseidon only triggered by poseidon
// opcode.     pub storage: Vec<StorageExePiece>,
//     pub tape: Vec<TapeExePiece>,
// }

// pub struct Trace {
//     //(inst_asm_str, imm_flag, step, inst_encode, imm_val)
//     pub instructions: HashMap<u64, (String, u8, u64, GoldilocksField,
// GoldilocksField)>,     // pub raw_instructions: HashMap<u64, Instruction>,
//     pub raw_instructions: HashMap<u64, String>,
//     pub raw_binary_instructions: Vec<String>,
//     pub addr_program_hash: HashMap<String, Vec<GoldilocksField>>,
//     pub start_end_roots: (TreeValue, TreeValue),
//     // todo need limit the trace size
//     pub exec: Vec<Step>,
//     pub memory: Vec<MemoryTraceCell>,
//     pub builtin_rangecheck: Vec<RangeCheckRow>,
//     pub builtin_bitwise_combined: Vec<BitwiseCombinedRow>,
//     pub builtin_cmp: Vec<CmpRow>,
//     pub builtin_poseidon: Vec<PoseidonRow>,
//     pub builtin_poseidon_chunk: Vec<PoseidonChunkRow>,
//     pub builtin_storage: Vec<StorageRow>,
//     pub builtin_storage_hash: Vec<StorageHashRow>,
//     pub builtin_program_hash: Vec<StorageHashRow>,
//     pub tape: Vec<TapeRow>,
//     pub sc_call: Vec<SCCallRow>,
//     pub ret: Vec<GoldilocksField>,
// }

fn pre_process(block_exe_trace: Vec<TxExeTrace>) -> Vec<Trace> {
    let mut traces: Vec<Trace> = Vec::new();
    let mut next_storage_access_idx = 0;
    for tx_trace in block_exe_trace {
        let trace = gen_tx_trace(&mut next_storage_access_idx, tx_trace);
        traces.push(trace);
    }
    todo!()
}

fn gen_tx_trace(next_storage_access_idx: &mut u64, mut tx: TxExeTrace) -> Trace {
    let mut trace = Trace::default();
    process_program(&mut trace, tx.programs);
    process_cpu(&mut trace, tx.cpu, next_storage_access_idx);
    process_mem(&mut trace, &mut tx.env_mem);
    process_rc(&mut trace, tx.rc);
    process_bitwise(&mut trace, tx.bitwise);
    process_cmp(&mut trace, tx.cmp);
    process_poseidon(&mut trace, tx.poseidon);
    process_tape(&mut trace, tx.tape);
    process_sccall(&mut trace, tx.sccall);
    add_treekey_poseidon(&mut trace, tx.storage);
    todo!()
}

fn process_program(trace: &mut Trace, addr_to_bytecodes: Vec<(ContractAddress, Vec<u64>)>) {
    addr_to_bytecodes.iter().for_each(|(addr, bytecodes)| {
        let addr_str = encode_addr(&[fe(addr[0]), fe(addr[1]), fe(addr[2]), fe(addr[3])]);
        let program: Vec<GoldilocksField> = bytecodes.iter().map(|inst| fe(*inst)).collect();
        trace.addr_program_hash.insert(addr_str, program);
    })
}

fn process_cpu(
    trace: &mut Trace,
    cpu: Vec<(u64, u64, ExeContext, Vec<CpuExePiece>)>,
    next_storage_access_idx: &mut u64,
) {
    for (call_sc_cnt, env_idx, context, pieces) in cpu {
        for p in pieces {
            let step = Step {
                env_idx: fe(env_idx),
                call_sc_cnt: fe(call_sc_cnt),
                clk: p.clk as u32,
                pc: p.clk,
                tp: fe(p.tp),
                addr_storage: fe_4(context.storage_addr),
                addr_code: fe_4(context.code_addr),
                instruction: fe(p.instruction),
                immediate_data: fe(p.imm.unwrap_or_default()),
                opcode: fe(p.opcode),
                op1_imm: fe((p.instruction >> 62) & 1),
                regs: fe_10(p.registers),
                register_selector: RegisterSelector {
                    op0: fe(p.op0.unwrap_or_default()),
                    op1: fe(p.op1.unwrap_or_default()),
                    dst: fe(p.dst.unwrap_or_default()),
                    aux0: fe(p.aux0.unwrap_or_default()),
                    aux1: fe(p.aux1.unwrap_or_default()),
                    op0_reg_sel: fe_10(p.op0_reg_sel),
                    op1_reg_sel: fe_10(p.op1_reg_sel),
                    dst_reg_sel: fe_10(p.dst_reg_sel),
                },
                is_ext_line: fe_bool(p.is_ext_line),
                ext_cnt: fe(p.ext_cnt),
                filter_tape_looking: if p.is_ext_line
                    && (p.opcode == OlaOpcode::TLOAD.binary_bit_mask()
                        || p.opcode == OlaOpcode::TSTORE.binary_bit_mask())
                {
                    fe(1)
                } else {
                    fe(0)
                },
                storage_access_idx: fe(next_storage_access_idx.clone()),
            };
            if (p.opcode == OlaOpcode::SSTORE.binary_bit_mask()
                || p.opcode == OlaOpcode::SLOAD.binary_bit_mask())
                && !p.is_ext_line
            {
                *next_storage_access_idx += 1;
            }
            trace.exec.push(step);
        }
    }
}

fn process_mem(trace: &mut Trace, mem: &mut HashMap<u64, Vec<MemExePiece>>) {
    let mut sorted: Vec<(u64, MemExePiece)> = Vec::new();
    let mut env_idxs: Vec<u64> = mem.keys().cloned().collect();
    env_idxs.sort_unstable();
    for env_idx in env_idxs {
        if let Some(mut pieces) = mem.remove(&env_idx) {
            pieces.sort_unstable_by(|a, b| match a.addr.cmp(&b.addr) {
                std::cmp::Ordering::Equal => a.clk.cmp(&b.clk),
                other => other,
            });
            let to_extend: Vec<(u64, MemExePiece)> =
                pieces.into_iter().map(|m| (env_idx, m)).collect();
            sorted.extend(to_extend);
        }
    }
    let mut pre: Option<MemExePiece> = None;
    for (env_idx, m) in &sorted {
        let is_first_line = pre.is_none();
        let p = pre.unwrap_or_default();
        let is_addr_zero = fe_is_zero(m.addr);
        let is_region_stack = is_addr_zero || m.addr < MEM_STACK_REGION.end;
        let is_region_heap =
            !is_addr_zero && m.addr >= MEM_HEAP_REGION.start && m.addr < MEM_HEAP_REGION.end;
        let is_region_rw = is_addr_zero || m.addr < MEM_HEAP_REGION.end;
        let is_region_prophet = !is_addr_zero && m.addr >= MEM_PROPHET_REGION.start;
        let is_addr_change = if !is_first_line {
            p.addr != m.addr
        } else {
            true
        };

        let diff_addr = if !is_first_line {
            if is_region_prophet {
                GoldilocksField::ZERO
            } else {
                fe(m.addr - p.addr)
            }
        } else {
            GoldilocksField::ZERO
        };
        let diff_addr_inv = if diff_addr == GoldilocksField::ZERO || is_region_prophet {
            GoldilocksField::ZERO
        } else {
            GoldilocksField::ONE / diff_addr
        };
        let diff_clk = if is_first_line || is_region_prophet || is_addr_change {
            GoldilocksField::ZERO
        } else {
            fe(m.clk - p.clk)
        };
        let region_rc = if is_region_stack {
            None
        } else if is_region_heap {
            Some(
                GoldilocksField::ZERO
                    - GoldilocksField::from_canonical_u64(2_u64.pow(32) - 1)
                    - fe(m.addr),
            )
        } else {
            Some(GoldilocksField::ZERO - fe(m.addr))
        };
        let rw_addr_unchanged = !is_region_prophet && !is_addr_change;
        let sort_rc = if is_first_line
            || is_region_prophet
            || (is_region_heap && p.addr < MEM_STACK_REGION.end)
        {
            None
        } else if is_addr_change {
            Some(diff_addr)
        } else {
            Some(diff_clk)
        };
        if let Some(rc) = sort_rc.clone() {
            let mut row = u32_to_rc_no_filter(rc.0 as u32);
            row.filter_looked_for_mem_sort = GoldilocksField::ONE;
            trace.builtin_rangecheck.push(row);
        }
        if let Some(rc) = region_rc.clone() {
            let mut row = u32_to_rc_no_filter(rc.0 as u32);
            row.filter_looked_for_mem_region = GoldilocksField::ONE;
            trace.builtin_rangecheck.push(row);
        }

        let row = MemoryTraceCell {
            env_idx: fe(*env_idx),
            addr: fe(m.addr),
            clk: fe(m.clk),
            is_rw: fe_bool(is_region_rw),
            op: fe_option_opcode(m.opcode),
            is_write: fe_bool(m.is_write),
            diff_addr,
            diff_addr_inv,
            diff_clk,
            diff_addr_cond: region_rc.unwrap_or_default(),
            filter_looked_for_main: fe_bool(m.opcode != None),
            rw_addr_unchanged: fe_bool(rw_addr_unchanged),
            region_prophet: fe_bool(is_region_prophet),
            region_heap: fe_bool(is_region_heap),
            value: fe(m.value),
            rc_value: sort_rc.unwrap_or_default(),
        };

        trace.memory.push(row);
        pre = Some(m.clone());
    }
}

fn process_rc(trace: &mut Trace, rc: Vec<RcExePiece>) {
    for r in rc {
        let mut row = u32_to_rc_no_filter(r.value);
        row.filter_looked_for_cpu = GoldilocksField::ONE;
        trace.builtin_rangecheck.push(row);
    }
}

fn process_bitwise(trace: &mut Trace, bitwise: Vec<BitwiseExePiece>) {
    let split = |n: u32| {
        let l0 = (n & 0xff) as u64;
        let l1 = (n >> 8 & 0xff) as u64;
        let l2 = (n >> 16 & 0xff) as u64;
        let l3 = (n >> 24 & 0xff) as u64;
        (l0, l1, l2, l3)
    };
    for b in bitwise {
        let (op0_0, op0_1, op0_2, op0_3) = split(b.op0);
        let (op1_0, op1_1, op1_2, op1_3) = split(b.op1);
        let (res_0, res_1, res_2, res_3) = split(b.res);
        let row = BitwiseCombinedRow {
            opcode: b.opcode.binary_bit_mask(),
            op0: fe(b.op0 as u64),
            op1: fe(b.op1 as u64),
            res: fe(b.res as u64),
            op0_0: fe(op0_0),
            op0_1: fe(op0_1),
            op0_2: fe(op0_2),
            op0_3: fe(op0_3),
            op1_0: fe(op1_0),
            op1_1: fe(op1_1),
            op1_2: fe(op1_2),
            op1_3: fe(op1_3),
            res_0: fe(res_0),
            res_1: fe(res_1),
            res_2: fe(res_2),
            res_3: fe(res_3),
        };
        trace.builtin_bitwise_combined.push(row);
    }
}

fn process_cmp(trace: &mut Trace, bitwise: Vec<CmpExePiece>) {
    for c in bitwise {
        let abs_diff = if c.op0 > c.op1 {
            GoldilocksField::from_canonical_u32(c.op0 - c.op1)
        } else {
            GoldilocksField::from_canonical_u32(c.op1 - c.op0)
        };
        let abs_diff_inv = if abs_diff == GoldilocksField::ZERO {
            GoldilocksField::ZERO
        } else {
            abs_diff.inverse()
        };

        let row = CmpRow {
            op0: fe(c.op0 as u64),
            op1: fe(c.op1 as u64),
            gte: fe_bool(c.is_gte),
            abs_diff,
            abs_diff_inv,
            filter_looking_rc: GoldilocksField::ONE,
        };
        trace.builtin_cmp.push(row);
    }
}

fn process_poseidon(trace: &mut Trace, poseidon: Vec<PoseidonPiece>) {
    for p in poseidon {
        let inputs: Vec<GoldilocksField> = p
            .inputs
            .iter()
            .map(|n| GoldilocksField::from_canonical_u64(*n))
            .collect();
        let (_, poseidon_rows) =
            calculate_arbitrary_poseidon_and_generate_intermediate_trace(inputs.as_slice());
        let poseidon_rows: Vec<PoseidonRow> = poseidon_rows
            .iter()
            .map(|r| {
                let mut row = r.clone();
                row.filter_looked_normal = true;
                row
            })
            .collect();

        let chunk_main = PoseidonChunkRow {
            env_idx: fe(p.env_idx),
            clk: p.clk as u32,
            opcode: fe(OlaOpcode::POSEIDON.binary_bit_mask()),
            dst: fe(p.dst_addr),
            op0: fe(p.src_addr),
            op1: fe(p.len),
            acc_cnt: GoldilocksField::ZERO,
            value: [GoldilocksField::ZERO; 8],
            cap: [GoldilocksField::ZERO; 4],
            hash: [GoldilocksField::ZERO; 12],
            is_ext_line: GoldilocksField::ZERO,
        };
        let chunk_exts: Vec<PoseidonChunkRow> = poseidon_rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut ext = chunk_main.clone();
                ext.op0 = ext.op0 + fe((i * 8) as u64);
                ext.acc_cnt = fe(min((i * 8 + 8) as u64, p.len));
                let mut value: [GoldilocksField; 8] = Default::default();
                value.copy_from_slice(&r.input[0..8]);
                ext.value = value;
                let mut cap: [GoldilocksField; 4] = Default::default();
                cap.copy_from_slice(&r.input[8..12]);
                ext.cap = cap;
                ext.hash = r.output;
                ext.is_ext_line = GoldilocksField::ONE;
                ext
            })
            .collect();
        trace.builtin_poseidon_chunk.push(chunk_main);
        trace.builtin_poseidon_chunk.extend(chunk_exts);
        trace.builtin_poseidon.extend(poseidon_rows);
    }
}

fn process_tape(trace: &mut Trace, tape: Vec<TapeExePiece>) {
    for t in tape {
        let is_init = t.opcode.is_none();
        let row = TapeRow {
            is_init,
            opcode: fe_option_opcode(t.opcode),
            addr: fe(t.addr),
            value: fe(t.value),
            filter_looked: fe_bool(!is_init),
        };
        trace.tape.push(row);
    }
}

fn process_sccall(trace: &mut Trace, sccall: Vec<SCCallPiece>) {
    for s in sccall {
        let clk_caller_ret = s.clk_caller_call + if s.caller_op1_imm { 2 } else { 1 };
        let row = SCCallRow {
            caller_env_idx: fe(s.caller_env_idx),
            addr_storage: fe_4(s.caller_storage_addr),
            addr_code: fe_4(s.caller_code_addr),
            caller_op1_imm: fe_bool(s.caller_op1_imm),
            clk_caller_call: fe(s.clk_caller_call),
            clk_caller_ret: fe(clk_caller_ret),
            regs: fe_10(s.reg_caller),
            callee_env_idx: fe(s.callee_env_idx),
            clk_callee_end: fe(s.clk_callee_end),
        };
        trace.sc_call.push(row);
    }
}

fn add_treekey_poseidon(trace: &mut Trace, storage: Vec<StorageExePiece>) {
    for s in storage {
        let values: Vec<GoldilocksField> = s
            .contract_addr
            .into_iter()
            .chain(s.storage_key.into_iter())
            .map(|n| fe(n))
            .collect();
        // calculate_poseidon_and_generate_intermediate_trace
        let mut inputs = [GoldilocksField::ZERO; 12];
        inputs[0..8].copy_from_slice(&values);
        let mut r = calculate_poseidon_and_generate_intermediate_trace(inputs);
        r.filter_looked_treekey = true;
        trace.builtin_poseidon.push(r);
    }
}

fn u32_to_rc_no_filter(n: u32) -> RangeCheckRow {
    let val = n as u64;
    let limb_lo = val & 0xffff;
    let limb_hi = val >> 16 & 0xffff;
    RangeCheckRow {
        val: fe(val),
        limb_lo: fe(limb_lo),
        limb_hi: fe(limb_hi),
        filter_looked_for_mem_sort: GoldilocksField::ZERO,
        filter_looked_for_mem_region: GoldilocksField::ZERO,
        filter_looked_for_cpu: GoldilocksField::ZERO,
        filter_looked_for_comparison: GoldilocksField::ZERO,
        filter_looked_for_storage: GoldilocksField::ZERO,
    }
}

fn fe_option_opcode(op: Option<OlaOpcode>) -> GoldilocksField {
    if let Some(opcode) = op {
        fe(opcode.binary_bit_mask())
    } else {
        GoldilocksField::ZERO
    }
}

fn fe_is_zero(n: u64) -> bool {
    GoldilocksField::from_canonical_u64(n) == GoldilocksField::ZERO
}

fn fe_non_zero(n: u64) -> bool {
    GoldilocksField::from_canonical_u64(n) != GoldilocksField::ZERO
}

fn fe_option(n: Option<u64>) -> GoldilocksField {
    GoldilocksField::from_canonical_u64(n.unwrap_or_default())
}

fn fe_bool(b: bool) -> GoldilocksField {
    if b {
        fe(1)
    } else {
        fe(0)
    }
}

fn fe(n: u64) -> GoldilocksField {
    GoldilocksField::from_canonical_u64(n)
}

fn fe_4(nums: [u64; 4]) -> [GoldilocksField; 4] {
    [fe(nums[0]), fe(nums[1]), fe(nums[2]), fe(nums[3])]
}
fn fe_10(nums: [u64; 10]) -> [GoldilocksField; 10] {
    [
        fe(nums[0]),
        fe(nums[1]),
        fe(nums[2]),
        fe(nums[3]),
        fe(nums[4]),
        fe(nums[5]),
        fe(nums[6]),
        fe(nums[7]),
        fe(nums[8]),
        fe(nums[9]),
    ]
}
