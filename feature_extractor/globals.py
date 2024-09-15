# coding=utf-8
from idautils import *
from idaapi import *
from idc import *

# 划分指令集 MIPS, x86, ARM 的指令操作码的类型

# MIPS
MIPS_ISA = ['mips', 'mipsb']
MIPS_BRANCH = ['beqz', 'beq', 'bne', 'bgez', 'b',
               'bnez', 'bgtz', 'bltz', 'blez', 'bgt',
               'bge', 'blt', 'ble', 'bgtu', 'bgeu',
               'bltu', 'bleu']  # 所有以 b 开头
MIPS_CALL = []
MIPS_LOGIC = ['and', 'sltu', 'sra', 'xor', 'slt',
              'sllv', 'slti', 'srl', 'sll', 'srav',
              'srlv', 'ori', 'xori', 'andi', 'or',
              'nor']
MIPS_ARITH = ['add', 'sub', 'mul', 'div']  # 所有包含该集合元素的指令，都是算数指令
MIPS_TRANSFER = ['beq', 'bne', 'bgtz', 'bltz', 'bgez',
                 'blez', 'j', 'jal', 'jr', 'jalr',
                 'beqz', 'b', 'bltzal', 'bgezal', 'bnez']  # 所有以 b 和 j 开头的

# X86
X86_ISA = ['metapc']
X86_BRANCH = ['jz', 'jnb', 'jne', 'je', 'jg',
              'jle', 'jl', 'jge', 'ja', 'jae',
              'jb', 'jbe', 'jo', 'jno', 'js',
              'jns']
X86_CALL = ['call']
X86_LOGIC = ['and', 'andn', 'andnpd', 'andpd', 'andps',
             'andnps', 'test', 'xor', 'xorpd', 'pslld',
             'or', 'shl', 'sal', 'shr', 'sar',
             'pand', 'por', 'pxor', 'pandn']
X86_ARITH = ['add', 'adc', 'sub', 'sbb', 'inc',
             'dec',  'mul', 'imul', 'div', 'idiv',
             'fadd', 'fsub', 'fsubr', 'fdiv', 'fdivr']
X86_TRANSFER = ['call', 'j', 'jz', 'jnb', 'jne',
              'jle', 'jl', 'jge', 'ja', 'jae',
              'jb', 'jbe', 'jo', 'jno', 'js',
              'jns', 'je', 'jg']  # 所有以 j 开头

# ARM
ARM_ISA = ['ARM', 'ARMB', 'THUMB', 'ARM64']
ARM_BRANCH = ['B', 'BAL', 'BNE', 'BEQ', 'BPL',
              'BMI', 'BCC', 'BLO', 'BCS', 'BHS',
              'BVC', 'BVS', 'BGT', 'BGE', 'BLT',
              'BLE', 'BHI', 'BLS']
ARM_CALL = ['BL']  # 注意：并非直接的调用指令，必须满足一定的条件
ARM_LOGIC = ['AND', 'OR', 'BIC', 'MVN', 'TEQ',
             'TST', 'LSL', 'LSR']  # 会带条件后缀，包含即可
ARM_ARITH = ['ADD', 'SUB', 'MUL', 'DIV',
             'MLA', 'RSB', 'RSC', 'SBC',
             'ADC']  # 会带前后缀，包含即可
ARM_TRANSFER = ['B', 'BX', 'BL', 'BLX', 'BLE',
                'BEQ', 'BNE', 'BCS', 'BCC', 'BMI',
                'BPL', 'BVS', 'BHI', 'BVC', 'BLS',
                'BGE', 'BLT', 'BGT', 'BAL', 'BLO',
                'BHS']

# 一切未知的 isa，对三种指令集借助集合进行合并处理
BRANCH = list(set(MIPS_BRANCH + X86_BRANCH + ARM_BRANCH))
CALL = list(set(MIPS_CALL + X86_CALL + ARM_CALL))
LOGIC = list(set(MIPS_LOGIC + X86_LOGIC + ARM_LOGIC))
ARITH = list(set(MIPS_ARITH + X86_ARITH + ARM_ARITH))
TRANSFER = list(set(MIPS_TRANSFER + X86_TRANSFER + ARM_TRANSFER))


def is_branch(inst_addr, isa):
    """
    判断当前地址处的指令是否是分支指令

    Args:
        inst_addr: 指令的地址
        isa: ISA
    """
    opcode = GetMnem(inst_addr)  # 获得指令的操作码

    if isa in MIPS_ISA:
        if opcode in MIPS_BRANCH:
            return True
        else:
            return False
    elif isa in X86_ISA:
        if opcode in X86_BRANCH:
            return True
        else:
            return False
    elif isa in ARM_ISA:
        if opcode in ARM_BRANCH:
            return True
        else:
            return False
    else:
        if opcode in BRANCH:
            return True
        else:
            return False


def is_arm_call(inst_addr, funcnames):
    """
    判断指令是否是 ARM 架构下的调用指令
    注意：ARM 调用子程序，必须操作数是 非'.text'的函数名
    """
    opcode = GetMnem(inst_addr)

    if opcode in ARM_TRANSFER:
        print "####### ARM opcode: ", opcode
        opnd = GetOpnd(inst_addr, 0)  # 汇编指令中的操作数
        print "####### opnd: ", opnd
        if opnd in funcnames:
            print "######  ARM  CALL"
            return True
        else:
            print "###### ARM: CALL operand is not extern"
            return False
    else:
        return False


def is_call(inst_addr, isa, funcnames):
    """
    判断该地址处的指令是否是调用子程序的指令
    注意：MIPS 和 ARM 没有直接的调用指令，而是通过某些跳转指令实现

    funcnames (list): 二进制文件中所有的函数名
    """
    opcode = GetMnem(inst_addr)  # 获得指令的操作码

    if isa in MIPS_ISA:
        return False
    # x86 有 call 指令
    elif isa in X86_ISA:
        if opcode in X86_CALL:
            return True
        else:
            return False
    elif isa in ARM_ISA:
        return is_arm_call(inst_addr, funcnames)
    else:
        # 未知的 ISA 类型
        if opcode in X86_CALL:
            return True
        elif opcode in ARM_CALL:
            return is_arm_call(inst_addr, funcnames)
        else:
            return False


def is_logic(inst_addr, isa):
    """
    判断当前地址处的指令是否是逻辑指令
    逻辑指令：与或非系列，逻辑移位
    """
    opcode = GetMnem(inst_addr)  # 获得指令的操作码

    if isa in MIPS_ISA:
        if opcode in MIPS_LOGIC:
            return True
        else:
            return False
    elif isa in X86_ISA:
        if opcode in X86_LOGIC:
            return True
        else:
            return False
    elif isa in ARM_ISA:
        # 包含即可
        for logic in ARM_LOGIC:
            if logic in opcode:
                return True
        return False
    else:
        for logic in LOGIC:
            if logic in opcode:
                return True
        return False


def is_arithmetic(inst_addr, isa):
    """
    判断当前地址处的指令是否是算数指令
    算术指令：加减乘除系列
    """
    opcode = GetMnem(inst_addr)  # 获得指令的操作码

    if isa in MIPS_ISA:
        # MIPS 算术指令会有很多变形，检查有无包含即可
        for arith in MIPS_ARITH:
            if arith in opcode:
                return True
        return False
    elif isa in X86_ISA:
        if opcode in X86_ARITH:
            return True
        else:
            return False
    elif isa in ARM_ISA:
        # 包含即可
        for arith in ARM_ARITH:
            if arith in opcode:
                return True
        return False
    else:
        for arith in ARITH:
            if arith in opcode:
                return True
        return False


def is_transfer(inst_addr, isa):
    """
    判断当前地址处的指令是否是转移指令
    转移指令包括：跳转指令，call
    """
    opcode = GetMnem(inst_addr)  # 获得指令的操作码

    if isa in MIPS_ISA:
        # MIPS 中所有以 b 和 j 开头的都是转移指令
        if opcode[0] == 'b' or opcode[0] == 'j':
            return True
        else:
            return False
    elif isa in X86_ISA:
        # X86 中所有以 'j' 开头的助记符都是跳转指令
        if opcode[0] == 'j' or opcode in X86_CALL:
            return True
        else:
            return False
    elif isa in ARM_ISA:
        if opcode in ARM_TRANSFER:
            return True
        else:
            return False
    else:
        if opcode in TRANSFER:
            return True
        else:
            return False
