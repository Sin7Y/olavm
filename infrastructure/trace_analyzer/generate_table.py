# import xlsxwriter module
import json
from enum import Enum

import xlsxwriter
import argparse

REG_NUM = 10
CTX_REG_NUM = 4
class OpcodeValue(Enum):
    SEL_ADD = 2 ** 31
    SEL_MUL = 2 ** 30
    SEL_EQ = 2 ** 29
    SEL_ASSERT = 2 ** 28
    SEL_MOV = 2 ** 27
    SEL_JMP = 2 ** 26
    SEL_CJMP = 2 ** 25
    SEL_CALL = 2 ** 24
    SEL_RET = 2 ** 23
    SEL_MLOAD = 2 ** 22
    SEL_MSTORE = 2 ** 21
    SEL_END = 2 ** 20

    SEL_RANGE_CHECK = 2 ** 19
    SEL_AND = 2 ** 18
    SEL_OR = 2 ** 17
    SEL_XOR = 2 ** 16
    SEL_NOT = 2 ** 15
    SEL_NEQ = 2 ** 14
    SEL_GTE = 2 ** 13
    SEL_POSEIDON = 2 ** 12
    SEL_SLOAD = 2 ** 11
    SEL_SSTORE = 2 ** 10


class JsonMainTraceColumnType(Enum):
    CLK = 'clk'
    PC = 'pc'
    CTX_REGS = "ctx_regs"
    REGS = 'regs'
    INST = 'instruction'
    OP1_IMM = 'op1_imm'
    OPCODE = 'opcode'
    IMM_VALUE = 'immediate_data'
    ASM = 'asm'
    REG_SELECTOR = 'register_selector'

    SEL_ADD = 'sel_add'
    SEL_MUL = 'sel_mul'
    SEL_EQ = 'sel_eq'
    SEL_ASSERT = 'sel_assert'
    SEL_MOV = 'sel_mov'
    SEL_JMP = 'sel_jmp'
    SEL_CJMP = 'sel_cjmp'
    SEL_CALL = 'sel_call'
    SEL_RET = 'sel_ret'
    SEL_MLOAD = 'sel_mload'
    SEL_MSTORE = 'sel_mstore'
    SEL_END = 'sel_end'

    SEL_RANGE_CHECK = 'sel_range_check'
    SEL_AND = 'sel_and'
    SEL_OR = 'sel_or'
    SEL_XOR = 'sel_xor'
    SEL_NOT = 'sel_not'
    SEL_NEQ = 'sel_neq'
    SEL_GTE = 'sel_gte'
    SEL_POSEIDON = 'sel_poseidon'
    SEL_SSTORE = 'sel_sstore'
    SEL_SLOAD = 'sel_sload'

class MainTraceColumnType(Enum):
    CLK = 'clk'
    PC = 'pc'

    CTX0 = "ctx0"
    CTX1 = "ctx1"
    CTX2 = "ctx2"
    CTX3 = "ctx3"

    R0 = 'r0'
    R1 = 'r1'
    R2 = 'r2'
    R3 = 'r3'
    R4 = 'r4'
    R5 = 'r5'
    R6 = 'r6'
    R7 = 'r7'
    R8 = 'r8'
    R9 = 'r9'



    INST = 'inst'
    OP1_IMM = 'op1_imm'
    OPCODE = 'opcode'
    IMM_VALUE = 'imm_val'
    ASM = 'asm'

    OP0 = 'op0'
    OP1 = 'op1'
    DST = 'dst'
    AUX0 = 'aux0'
    AUX1 = 'aux1'

    SEL_OP0_R0 = 'sel_op0_r0'
    SEL_OP0_R1 = 'sel_op0_r1'
    SEL_OP0_R2 = 'sel_op0_r2'
    SEL_OP0_R3 = 'sel_op0_r3'
    SEL_OP0_R4 = 'sel_op0_r4'
    SEL_OP0_R5 = 'sel_op0_r5'
    SEL_OP0_R6 = 'sel_op0_r6'
    SEL_OP0_R7 = 'sel_op0_r7'
    SEL_OP0_R8 = 'sel_op0_r8'
    SEL_OP0_R9 = 'sel_op0_r9'

    SEL_OP1_R0 = 'sel_op1_r0'
    SEL_OP1_R1 = 'sel_op1_r1'
    SEL_OP1_R2 = 'sel_op1_r2'
    SEL_OP1_R3 = 'sel_op1_r3'
    SEL_OP1_R4 = 'sel_op1_r4'
    SEL_OP1_R5 = 'sel_op1_r5'
    SEL_OP1_R6 = 'sel_op1_r6'
    SEL_OP1_R7 = 'sel_op1_r7'
    SEL_OP1_R8 = 'sel_op1_r8'
    SEL_OP1_R9 = 'sel_op1_r9'

    SEL_DST_R0 = 'sel_dst_r0'
    SEL_DST_R1 = 'sel_dst_r1'
    SEL_DST_R2 = 'sel_dst_r2'
    SEL_DST_R3 = 'sel_dst_r3'
    SEL_DST_R4 = 'sel_dst_r4'
    SEL_DST_R5 = 'sel_dst_r5'
    SEL_DST_R6 = 'sel_dst_r6'
    SEL_DST_R7 = 'sel_dst_r7'
    SEL_DST_R8 = 'sel_dst_r8'
    SEL_DST_R9 = 'sel_dst_r9'

    SEL_ADD = 'sel_add'
    SEL_MUL = 'sel_mul'
    SEL_EQ = 'sel_eq'
    SEL_ASSERT = 'sel_assert'
    SEL_MOV = 'sel_mov'
    SEL_JMP = 'sel_jmp'
    SEL_CJMP = 'sel_cjmp'
    SEL_CALL = 'sel_call'
    SEL_RET = 'sel_ret'
    SEL_MLOAD = 'sel_mload'
    SEL_MSTORE = 'sel_mstore'
    SEL_END = 'sel_end'

    SEL_RANGE_CHECK = 'sel_range_check'
    SEL_AND = 'sel_and'
    SEL_OR = 'sel_or'
    SEL_XOR = 'sel_xor'
    SEL_NOT = 'sel_not'
    SEL_NEQ = 'sel_neq'
    SEL_GTE = 'sel_gte'
    SEL_POSEIDON = 'sel_poseidon'
    SEL_SSTORE = 'sel_sstore'
    SEL_SLOAD = 'sel_sload'


class MemoryTraceColumnType(Enum):
    ADDR = 'addr'
    CLK = 'clk'
    IS_RW = 'is_rw'
    OP = 'op'
    IS_WRITE = 'is_write'
    VALUE = 'value'
    DIFF_ADDR = 'diff_addr'
    DIFF_ADDR_INV = 'diff_addr_inv'
    DIFF_CLK = 'diff_clk'
    DIFF_ADDR_COND = 'diff_addr_cond'
    FILTER_LOOKED_FOR_MAIN = 'filter_looked_for_main'
    RW_ADDR_UNCHANGED = 'rw_addr_unchanged'
    REGION_PROPHET = 'region_prophet'
    REGION_HEAP = 'region_heap'
    RC_VALUE = 'rc_value'


class RangeCheckTraceColumnType(Enum):
    VAL = 'val'
    LIMB_LO = 'limb_lo'
    LIMB_HI = 'limb_hi'
    FILTER_LOOKED_FOR_MEM_SORT = 'filter_looked_for_mem_sort'
    FILTER_LOOKED_FOR_CPU = 'filter_looked_for_cpu'
    FILTER_LOOKED_FOR_CMP = 'filter_looked_for_comparison'
    FILTER_LOOKED_FOR_STORE = 'filter_looked_for_storage'
    FILTER_LOOKED_FOR_MEM_REGION = 'filter_looked_for_mem_region'


class BitwiseTraceColumnType(Enum):
    BITWISE_TAG = 'opcode'
    OP0 = 'op0'
    OP1 = 'op1'
    RES = 'res'
    OP0_0 = 'op0_0'
    OP0_1 = 'op0_1'
    OP0_2 = 'op0_2'
    OP0_3 = 'op0_3'
    OP1_0 = 'op1_0'
    OP1_1 = 'op1_1'
    OP1_2 = 'op1_2'
    OP1_3 = 'op1_3'
    RES_0 = 'res_0'
    RES_1 = 'res_1'
    RES_2 = 'res_2'
    RES_3 = 'res_3'


class ComparisonTraceColumnType(Enum):
    OP0 = 'op0'
    OP1 = 'op1'
    CMP_FLAG = 'gte'
    ABS_DIFF = 'abs_diff'
    ABS_DIFF_INV = 'abs_diff_inv'
    FILTER_LOOKING_FOR_RANGE_CHECK = 'filter_looking_rc'

class StorageTraceColumnType(Enum):
    CLK = 'clk'
    DIFF_CLK = 'diff_clk'
    OP = 'opcode'
    ROOT = 'root'
    ADD = 'addr'
    VALUE = 'value'

class StorageHashTraceColumnType(Enum):
    IDX_STORAGE = 'idx_storage'
    LAYER = 'layer'
    LAYER_BIT = 'layer_bit'
    ADDR_ACC = 'addr_acc'
    IS_LAYER64 = 'is_layer64'
    IS_LAYER128 = 'is_layer128'
    IS_LAYER192 = 'is_layer192'
    IS_LAYER256 = 'is_layer256'
    ADDR = 'addr'
    CAPS = 'caps'
    PATHS = 'paths'
    SIBLINGS = 'siblings'
    DELTAS = 'deltas'
    FULL_0_1 = 'full_0_1'
    FULL_0_2 = 'full_0_2'
    FULL_0_3 = 'full_0_3'
    PARTIAL = 'partial'
    FULL_1_0 = 'full_1_0'
    FULL_1_1 = 'full_1_1'
    FULL_1_2 = 'full_1_2'
    FULL_1_3 = 'full_1_3'
    OUTPUT = 'output'

class PoseidonHashTraceColumnType(Enum):
    CLK = "clk"
    OPCODE = "opcode"
    INPUT = "input"
    FULL_0_1 = 'full_0_1'
    FULL_0_2 = 'full_0_2'
    FULL_0_3 = 'full_0_3'
    PARTIAL = 'partial'
    FULL_1_0 = 'full_1_0'
    FULL_1_1 = 'full_1_1'
    FULL_1_2 = 'full_1_2'
    FULL_1_3 = 'full_1_3'
    OUTPUT = 'output'

def generate_columns_of_title(worksheet, trace_column_title):
    col = 0
    title_row = 0
    for data in trace_column_title:
        # print(data.name)
        worksheet.write(title_row, col, data.value)
        col += 1


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=['hex', 'dec'], default=['dec'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default="trace.xlsx")

    args = parser.parse_args()

    trace_input = open(args.input, 'r').read()
    trace_json = json.loads(trace_input)

    # print(args.format)
    # print(args.output)
    workbook = xlsxwriter.Workbook(args.output)

    worksheet = workbook.add_worksheet("MainTrace")

    # print(trace_json["exec"][1]["regs"])

    # MainTrace
    generate_columns_of_title(worksheet, MainTraceColumnType)

    # generate main trace table
    row_index = 1
    for row in trace_json["exec"]:
        col = 0
        for data in JsonMainTraceColumnType:
            if data.value == "regs":
                for i in range(0, REG_NUM):
                    if args.format == 'hex':
                        worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}),DEC2HEX({1},8))'.format(
                            row[data.value][i] // (2 ** 32), row[data.value][i] % (2 ** 32)))
                    else:
                        worksheet.write(row_index, col, row[data.value][i])
                    col += 1
            elif data.value == "ctx_regs":
                for i in range(0, CTX_REG_NUM):
                    if args.format == 'hex':
                        worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}),DEC2HEX({1},8))'.format(
                            row[data.value][i] // (2 ** 32), row[data.value][i] % (2 ** 32)))
                    else:
                        worksheet.write(row_index, col, row[data.value][i])
                    col += 1
            elif data.value == 'register_selector':
                if args.format == 'hex':
                    worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}),DEC2HEX({1},8))'.format(
                        row[data.value]["op0"] // (2 ** 32), row[data.value]["op0"] % (2 ** 32)))
                else:
                    worksheet.write(row_index, col, row[data.value]["op0"])
                col += 1
                if args.format == 'hex':
                    worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}),DEC2HEX({1},8))'.format(
                        row[data.value]["op1"] // (2 ** 32), row[data.value]["op1"] % (2 ** 32)))
                else:
                    worksheet.write(row_index, col, row[data.value]["op1"])
                col += 1
                if args.format == 'hex':
                    worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}),DEC2HEX({1},8))'.format(
                        row[data.value]["dst"] // (2 ** 32), row[data.value]["dst"] % (2 ** 32)))
                else:
                    worksheet.write(row_index, col, row[data.value]["dst"])
                col += 1
                if args.format == 'hex':
                    worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}),DEC2HEX({1},8))'.format(
                        row[data.value]["aux0"] // (2 ** 32), row[data.value]["aux0"] % (2 ** 32)))
                else:
                    worksheet.write(row_index, col, row[data.value]["aux0"])
                col += 1
                if args.format == 'hex':
                    worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}),DEC2HEX({1},8))'.format(
                        row[data.value]["aux1"] // (2 ** 32), row[data.value]["aux1"] % (2 ** 32)))
                else:
                    worksheet.write(row_index, col, row[data.value]["aux1"])
                col += 1
                sel_op0_regs = row[data.value]["op0_reg_sel"]
                for reg in sel_op0_regs:
                    worksheet.write(row_index, col, reg)
                    col += 1
                sel_op1_regs = row[data.value]["op1_reg_sel"]
                for reg in sel_op1_regs:
                    worksheet.write(row_index, col, reg)
                    col += 1
                sel_dst_regs = row[data.value]["dst_reg_sel"]
                for reg in sel_dst_regs:
                    worksheet.write(row_index, col, reg)
                    col += 1
            elif data.value == 'asm':
                # print(trace_json["raw_instructions"]['{0}'.format(row["pc"])])
                worksheet.write(row_index, col, '{0}'.format(trace_json["raw_instructions"]['{0}'.format(row["pc"])]))
                col += 1
            else:
                if data.value == "instruction" or data.value == "opcode" or data.value == "aux0":
                    # print("{0}:{1}:{2}".format(data.value, row[data.value],
                    #                            '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                    #                                row[data.value] // (2 ** 32), row[data.value] % (2 ** 32))))
                    worksheet.write(row_index, col,
                                    '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                        row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
                elif data.value == "sel_add" or data.value == "sel_mul" or data.value == "sel_eq" \
                        or data.value == "sel_assert" or data.value == "sel_mov" or data.value == "sel_jmp" \
                        or data.value == "sel_cjmp" or data.value == "sel_call" or data.value == "sel_ret" \
                        or data.value == "sel_mload" or data.value == "sel_mstore" or data.value == "sel_end" \
                        or data.value == "sel_range_check" or data.value == "sel_and" or data.value == "sel_or" \
                        or data.value == "sel_xor" or data.value == "sel_not" or data.value == "sel_neq" \
                        or data.value == "sel_gte" or data.value == "sel_poseidon" or  data.value == "sel_sload"  \
                        or data.value == "sel_sstore" \
                        :
                    # print("sel:{0}:{1}".format(row["opcode"], OpcodeValue[data.name].value))
                    if row["opcode"] == OpcodeValue[data.name].value:
                        worksheet.write(row_index, col, 1)
                    else:
                        worksheet.write(row_index, col, 0)
                else:
                    worksheet.write(row_index, col, row[data.value])
                col += 1
        row_index += 1

    # Memory Trace
    worksheet = workbook.add_worksheet("MemoryTrace")
    generate_columns_of_title(worksheet, MemoryTraceColumnType)

    # generate memory trace table
    row_index = 1
    for row in trace_json["memory"]:
        col = 0
        for data in MemoryTraceColumnType:
            if (
                    data.value == "addr" or data.value == "op" or data.value == "diff_addr_inv" or data.value == "value" or data.value == "diff_addr_cond") \
                    and args.format == 'hex':

                worksheet.write(row_index, col,
                                '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                    row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1
    # print(MemoryTraceColumnType['IS_WRITE'].value)

    # Builtin Range check Trace
    worksheet = workbook.add_worksheet("RangeChkTrace")
    generate_columns_of_title(worksheet, RangeCheckTraceColumnType)

    # generate range check trace table
    row_index = 1
    for row in trace_json["builtin_rangecheck"]:
        col = 0
        for data in RangeCheckTraceColumnType:
            if args.format == 'hex':
                worksheet.write(row_index, col,
                                '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                    row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1

    # Builtin bitwise Trace
    worksheet = workbook.add_worksheet("BitwiseTrace")
    generate_columns_of_title(worksheet, BitwiseTraceColumnType)

    # generate bitwise trace table
    row_index = 1
    for row in trace_json["builtin_bitwise_combined"]:
        col = 0
        for data in BitwiseTraceColumnType:
            if args.format == 'hex':
                worksheet.write(row_index, col,
                                '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                    row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1

    # Builtin Comparison Trace
    worksheet = workbook.add_worksheet("ComparisonTrace")
    generate_columns_of_title(worksheet, ComparisonTraceColumnType)

    # generate comparison trace table
    row_index = 1
    for row in trace_json["builtin_cmp"]:
        col = 0
        for data in ComparisonTraceColumnType:
            if args.format == 'hex':
                worksheet.write(row_index, col,
                                '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                    row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1

    # Storage Trace
    worksheet = workbook.add_worksheet("StorageTrace")
    generate_columns_of_title(worksheet, StorageTraceColumnType)

    # generate storage trace table
    row_index = 1
    for row in trace_json["builtin_storage"]:
        col = 0
        for data in StorageTraceColumnType:
            if data.value == "root" or data.value == "addr" or  data.value == "value" \
                    or data.value == "clk" or data.value == "diff_clk":
                worksheet.write(row_index, col, '{0}'.format(row[data.value]))
            elif args.format == 'hex':
                worksheet.write(row_index, col,
                                '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                    row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1

    # Storage Hash Trace
    worksheet = workbook.add_worksheet("HashStorageTrace")
    generate_columns_of_title(worksheet, StorageHashTraceColumnType)

    # generate storage hash trace table
    row_index = 1
    for row in trace_json["builtin_storage_hash"]:
        col = 0
        for data in StorageHashTraceColumnType:
            if data.value == "addr" or data.value == "caps" or  data.value == "paths" or \
               data.value == "siblings" or data.value == "deltas" or  data.value == "full_0_1" or\
               data.value == "full_0_2" or data.value == "full_0_3" or  data.value == "partial" or\
               data.value == "full_1_0" or data.value == "full_1_1" or  data.value == "full_1_2" or \
               data.value == "full_1_3" or  data.value == "output" or data.value == "addr_acc":
                worksheet.write(row_index, col, '{0}'.format(row[data.value]))
            elif args.format == 'hex':
                worksheet.write(row_index, col,
                                '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                    row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1

    # Poseidon Hash Trace
    worksheet = workbook.add_worksheet("PoseidonTrace")
    generate_columns_of_title(worksheet, PoseidonHashTraceColumnType)

    # generate poseidon hash trace table
    row_index = 1
    for row in trace_json["builtin_posiedon"]:
        col = 0
        for data in PoseidonHashTraceColumnType:
            if data.value == "input" or  data.value == "full_0_1" or \
                    data.value == "full_0_2" or data.value == "full_0_3" or  data.value == "partial" or \
                    data.value == "full_1_0" or data.value == "full_1_1" or  data.value == "full_1_2" or \
                    data.value == "full_1_3" or  data.value == "output":
                worksheet.write(row_index, col, '{0}'.format(row[data.value]))
            elif data.value == "opcode":
                worksheet.write(row_index, col,
                                    '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                        row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1
    workbook.close()


if __name__ == '__main__':
    main()
