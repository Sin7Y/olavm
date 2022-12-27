# import xlsxwriter module
import json
from enum import Enum

import xlsxwriter


class JsonMainTraceColumnType(Enum):
    CLK = 'clk'
    PC = 'pc'
    FLAG = 'flag'
    REGS = 'regs'
    INST = 'instruction'
    OP1_IMM = 'op1_imm'
    OPCODE = 'opcode'
    IMM_VALUE = 'immediate_data'
    REG_SELECTOR = 'register_selector'


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

    OP0 = 'op0'
    OP1 = 'op1'
    DST = 'dst'
    AUX0 = 'aux0'

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


def main():
    import sys
    trace_input = open(sys.argv[1], 'r').read()
    trace_json = json.loads(trace_input)

    workbook = xlsxwriter.Workbook('trace.xlsx')

    worksheet = workbook.add_worksheet("MainTrace")

    print(trace_json["exec"][1]["regs"])

    # MainTrace
    col = 0
    title_row = 0
    for data in MainTraceColumnType:
        worksheet.write(title_row, col, data.value)
        col += 1
        print('{:15} = {}'.format(data.name, data.value))

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
            else:
                if data.value == "instruction" or data.value == "opcode":
                    worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}/2^32,8),DEC2HEX(MOD({0},2^32),8))'.format(row[data.value]))
                else:
                    worksheet.write(row_index, col, row[data.value])
                col += 1
        row_index += 1

    # Memory Trace
    worksheet = workbook.add_worksheet("MemoryTrace")
    col = 0
    title_row = 0
    for data in MemoryTraceColumnType:
        worksheet.write(title_row, col, data.value)
        col += 1
        print('{:15} = {}'.format(data.name, data.value))

    row_index = 1

    for row in trace_json["memory"]:
        col = 0
        for data in MemoryTraceColumnType:
            if data.value == "addr" or data.value == "op" or data.value == "diff_addr_inv" or data.value == "value" or data.value == "diff_addr_cond":
                worksheet.write(row_index, col, '=CONCATENATE("0x",DEC2HEX({0}/2^32,8),DEC2HEX(MOD({0},2^32),8))'.format(row[data.value]))
            else:
                worksheet.write(row_index, col, row[data.value])
            col += 1
        row_index += 1
    workbook.close()
    print(MemoryTraceColumnType['IS_WRITE'].value)

if __name__ == '__main__':
    main()
