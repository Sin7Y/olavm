# import xlsxwriter module
import json
from enum import Enum

import xlsxwriter

class OpcodeValue(Enum):
    SEL_ADD = 2**34
    SEL_MUL = 2**33
    SEL_EQ = 2**32
    SEL_ASSERT = 2**31
    SEL_MOV = 2**30
    SEL_JMP = 2**29
    SEL_CJMP = 2**28
    SEL_CALL = 2**27
    SEL_RET = 2**26
    SEL_MLOAD = 2**25
    SEL_MSTORE = 2**24
    SEL_END = 2**23

    SEL_RANGE_CHECK = 2**22
    SEL_AND = 2**21
    SEL_OR = 2**20
    SEL_XOR = 2**19
    SEL_NOT = 2**18
    SEL_NEQ = 2**17
    SEL_GTE = 2**16

class JsonMainTraceColumnType(Enum):
    CLK = 'clk'
    PC = 'pc'
    FLAG = 'flag'
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

class MainTraceColumnType(Enum):
    CLK = 'clk'
    PC = 'pc'
    FLAG = 'flag'
    R0 = 'r0'
    R1 = 'r1'
    R2 = 'r2'
    R3 = 'r3'
    R4 = 'r4'
    R5 = 'r5'
    R6 = 'r6'
    R7 = 'r7'
    R8 = 'r8'

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

    SEL_OP1_R0 = 'sel_op1_r0'
    SEL_OP1_R1 = 'sel_op1_r1'
    SEL_OP1_R2 = 'sel_op1_r2'
    SEL_OP1_R3 = 'sel_op1_r3'
    SEL_OP1_R4 = 'sel_op1_r4'
    SEL_OP1_R5 = 'sel_op1_r5'
    SEL_OP1_R6 = 'sel_op1_r6'
    SEL_OP1_R7 = 'sel_op1_r7'
    SEL_OP1_R8 = 'sel_op1_r8'

    SEL_DST_R0 = 'sel_dst_r0'
    SEL_DST_R1 = 'sel_dst_r1'
    SEL_DST_R2 = 'sel_dst_r2'
    SEL_DST_R3 = 'sel_dst_r3'
    SEL_DST_R4 = 'sel_dst_r4'
    SEL_DST_R5 = 'sel_dst_r5'
    SEL_DST_R6 = 'sel_dst_r6'
    SEL_DST_R7 = 'sel_dst_r7'
    SEL_DST_R8 = 'sel_dst_r8'

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
    REGION_POSEIDON = 'region_poseidon'
    REGION_ECDSA = 'region_ecdsa'


def generate_columns_of_title(worksheet, trace_column_title):
    col = 0
    title_row = 0
    for data in trace_column_title:
        # print(data.name)
        worksheet.write(title_row, col, data.value)
        col += 1


def main():
    import sys
    trace_input = open(sys.argv[1], 'r').read()
    trace_json = json.loads(trace_input)

    # trace_output = sys.argv[2]
    # workbook = xlsxwriter.Workbook(trace_output)
    workbook = xlsxwriter.Workbook("trace.xlsx")

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
                for i in range(0, 9):
                    worksheet.write(row_index, col, row[data.value][i])
                    col += 1
            elif data.value == 'register_selector':
                worksheet.write(row_index, col, row[data.value]["op0"])
                col += 1
                worksheet.write(row_index, col, row[data.value]["op1"])
                col += 1
                worksheet.write(row_index, col, row[data.value]["dst"])
                col += 1
                worksheet.write(row_index, col, row[data.value]["aux0"])
                col += 1
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
                print(trace_json["raw_instructions"]['{0}'.format(row["pc"])])
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
                  or data.value == "sel_gte"  \
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
            if data.value == "addr" or data.value == "op" or data.value == "diff_addr_inv" or data.value == "value" or data.value == "diff_addr_cond":
                worksheet.write(row_index, col,
                                '=CONCATENATE("0x",DEC2HEX({0},8),DEC2HEX({1},8))'.format(
                                    row[data.value] // (2 ** 32), row[data.value] % (2 ** 32)))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1
    workbook.close()
    print(MemoryTraceColumnType['IS_WRITE'].value)


if __name__ == '__main__':
    main()
