# coding=utf-8
# cal (用于 attributing) 统计一个基本块内的属性
# get (用于 discovre) 统计一个函数内的属性
import globals
from idautils import *
from idaapi import *
from idc import *


def calInsts(bl):
    """
    计算基本块中指令的条数

    Args:
        bl (tuple): 基本块的(起始地址, 结束地址)
    """
    start = bl[0]
    end = bl[1]
    ea = start
    num = 0
    while ea < end:
        num += 1
        ea = NextHead(ea)  # 给定地址开始找到下一条指令的地址
    return num


def calCalls(bl, isa, funcnames):
    """
    计算基本块内 calls 指令的数目
    Args:
        bl (tuple): 基本块的(起始地址，结束地址)
    """
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        if globals.is_call(inst_addr, isa, funcnames):
            invoke_num += 1
        inst_addr = NextHead(inst_addr)
    return invoke_num


def calLogicInstructions(bl, isa):
    """
    计算基本块 bl 中逻辑指令的条数
    """
    start = bl[0]
    end = bl[1]
    logicInst_num = 0
    inst_addr = start
    while inst_addr < end:
        if globals.is_logic(inst_addr, isa):
            logicInst_num += 1
        inst_addr = NextHead(inst_addr)
    return logicInst_num


def calArithmeticIns(bl, isa):
    """
    计算基本块 bl 中算数指令的条数
    """
    start = bl[0]
    end = bl[1]
    arithInst_num = 0
    inst_addr = start
    while inst_addr < end:
        if globals.is_arithmetic(inst_addr, isa):
            arithInst_num += 1
        inst_addr = NextHead(inst_addr)
    return arithInst_num


def calTransferIns(bl, isa):
    """
    计算基本块 bl 中转移指令的条数
    """
    start = bl[0]
    end = bl[1]
    transferTnst_num = 0
    inst_addr = start
    while inst_addr < end:
        if globals.is_transfer(inst_addr, isa):
            transferTnst_num += 1
        inst_addr = NextHead(inst_addr)
    return transferTnst_num


def getConst(ea, offset):
    """
    从给定指令地址和操作数偏移量中提取字符串和常数

    Args:
        ea: 指令地址
        offset: 操作数偏移量
    """
    strings = []
    consts = []
    optype1 = GetOpType(ea, offset)  # 获取指定地址处的操作数类型

    # 操作数是立即数
    if optype1 == idaapi.o_imm:
        imm_value = GetOperandValue(ea, offset)  # 获取立即数操作数的数值
        # 判断立即数值范围
        if 0 <= imm_value <= 10:
            consts.append(imm_value)
        else:
            # 判断立即数值是否是有效地址
            if idaapi.isLoaded(imm_value) and idaapi.getseg(imm_value):
                str_value = GetString(imm_value)  # 尝试获取字符串值
                # 判断字符串是否为有效字符串
                if str_value is None:
                    str_value = GetString(imm_value + 0x40000)  # 尝试获取片偏移的字符串值
                    # 在 linux 操作系统下，0x40000 是用户空间程序的默认起始地址，也就是程序的基址
                    if str_value is None:
                        consts.append(imm_value)
                    else:
                        # 检查字符串是否为可打印字符
                        re = all(40 <= ord(c) < 128 for c in str_value)
                        if re:
                            strings.append(str_value)
                        else:
                            consts.append(imm_value)
                else:
                    # 直接获取字符串成功，检查是否可打印
                    re = all(40 <= ord(c) < 128 for c in str_value)
                    if re:
                        strings.append(str_value)
                    else:
                        consts.append(imm_value)
            else:
                consts.append(imm_value)

    return strings, consts


def getBBconsts(bl):
    """
    获取基本块 bl 中的字符串&常数列表
    
    Returns:
        strings (list): 字符串
        consts (list): 常量
    """
    strings = []
    consts = []
    start = bl[0]
    end = bl[1]
    inst_addr = start

    # 遍历基本块中的每条指令
    # 如果操作数中有字符串/常量，添加到对应列表中
    while inst_addr < end:
        opcode = GetMnem(inst_addr)
        # 如果是调用指令，则跳过
        if opcode in globals.CALL:
            inst_addr = NextHead(inst_addr)
            continue
        strings_src, consts_src = getConst(inst_addr, 0)  # 第一个操作数
        strings_dst, consts_dst = getConst(inst_addr, 1)  # 第二个操作数
        strings += strings_src
        strings += strings_dst
        consts += consts_src
        consts += consts_dst
        # 尝试获取第三个操作数的常量/字符串
        try:
            strings_dst, consts_dst = getConst(inst_addr, 2)
            consts += consts_dst
            strings += strings_dst
        except:
            pass
        # 移动到下一条指令的地址
        inst_addr = NextHead(inst_addr)

    return strings, consts


def retrieveExterns(bl, ea_externs):
    """
    从基本块 bl 中检索外部引用

    Args:
        bl: 基本块的 (起始地址, 结束地址)
        ea_externs: 外部 {入口地址: 函数/变量名称}

    Returns:
        list: 基本块中外部引用的名称
    """
    externs = []
    start = bl[0]
    end = bl[1]
    inst_addr = start
    # 遍历处理基本块中的每一条指令
    while inst_addr < end:
        # 获取从当前指令地址发出的所有代码引用
        refs = CodeRefsFrom(inst_addr, 1)
        try:
            # 查询所有在 ea_externs 中的引用地址
            ea = [v for v in refs if v in ea_externs][0]  # 外部引用的地址
            # 将对应外部引用的名称添加到 externs 列表中
            externs.append(ea_externs[ea])
        except:
            pass
        inst_addr = NextHead(inst_addr)
    return externs


def getFuncCalls(func, isa, funcnames):
    """
    获取函数内所有基本块中 calls 指令数目
    Args:
        func: 函数对象
    """
    # 获取函数的所有基本块的 (起始地址, 结束地址)列表
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumcalls = 0
    # 遍历每个基本块
    for bl in blocks:
        callnum = calCalls(bl, isa, funcnames)  # 当前基本块的 calls 数
        sumcalls += callnum
    return sumcalls


def getLogicInsts(func, isa):
    """
    获取函数内所有基本块中的逻辑指令的数目
    """
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumInsts = 0
    for bl in blocks:
        Instnum = calLogicInstructions(bl, isa)
        sumInsts += Instnum
    return sumInsts


def getTransferInsts(func, isa):
    """
    获取函数内所有基本块中的转移指令总数
    """
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumInsts = 0
    for bl in blocks:
        Instnum = calTransferIns(bl, isa)
        sumInsts += Instnum
    return sumInsts


def get_stackVariables(func_addr):
    """
    计算函数的 stack variable（局部变量）的数目
    程序的栈存放局部变量和函数调用的参数、返回地址

    Args:
        func_addr: 函数的地址

    Returns:
        栈变量的数目
    """
    args = []
    # 获取函数帧（存储局部变量、参数、返回地址等信息）
    stack = GetFrame(func_addr)
    if not stack:  # 检查该函数的栈帧是否合法（不空、不假）
        return 0
    firstM = GetFirstMember(stack)
    lastM = GetLastMember(stack)
    i = firstM
    # 遍历每一个栈成员，识别局部变量
    while i <= lastM:
        mName = GetMemberName(stack, i)
        mSize = GetMemberSize(stack, i)
        if mSize:
            i = i + mSize
        else:
            i = i+4
        # 检查成员是否是变量，是则加入 args 列表
        if mName not in args and mName and 'var_' in mName:
            args.append(mName)
    return len(args)


def getLocalVariables(func):
    args_num = get_stackVariables(func.startEA)
    return args_num


def getBasicBlocks(func):
    """
    获取函数内基本块的数目
    """
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    return len(blocks)


def getIncommingCalls(func):
    """
    计算给定函数的调用者的数目
    """
    # 查找指定所有指向函数入口地址的引用
    refs = CodeRefsTo(func.startEA, 0)
    re = len([v for v in refs])  # 可以处理 refs 不是列表的情况
    return re


def getIntrs(func):
    """
    获取函数内所有基本块中的指令总数
    """
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumInsts = 0
    for bl in blocks:
        Instnum = calInsts(bl)
        sumInsts += Instnum
    return sumInsts


def getfunc_consts(func):
    """
    获取函数内所有基本块中的常量和字符串
    """
    strings = []
    consts = []
    # 获取函数内所有基本块的（起始地址，结束地址）
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    # 遍历每个基本块，获取其内的字符串和常量列表
    for bl in blocks:
        strs, conts = getBBconsts(bl)
        strings += strs
        consts += conts
    return strings, consts
