# coding=utf-8
from graph_analysis_ida import *
from raw_graphs import *
import cfg_constructor as cfg


def get_unified_funcname(ea):
    """
    获取入口地址处的函数的统一格式（去除函数名前面的.）的名称
    """
    funcname = GetFunctionName(ea)  # 获取函数的原始名称
    if len(funcname) > 0:
        # 如果函数名称的第一个字符是点 '.' 则去掉
        if funcname[0] == '.':
            funcname = funcname[1:]
    return funcname


def processpltSegs():
    """
    处理二进制文件中包含外部符号的段，
    提取函数的名称和地址的双向映射。

    Returns:
        funcdata (dict): {函数名称: 十六进制入口地址}
        datafunc (dict): {入口地址: 函数名称}
    """
    funcdata = {}  # {函数名: 0x入口地址}
    datafunc = {}  # {入口地址: 函数名}

    # 遍历所有段
    for n in xrange(idaapi.get_segm_qty()):
        seg = idaapi.getnseg(n)  # 获取第 n 个段（segment）
        ea = seg.startEA  # 段的起始地址
        segname = SegName(ea)  # 段的名称

        # 只处理包含外部符号的段：.plt、extern、.MIPS.stubs
        if segname in ['.plt', 'extern', '.MIPS.stubs']:
            start = seg.startEA
            end = seg.endEA
            cur = start

            # 遍历段中的每个头部（head）
            while cur < end:
                name = get_unified_funcname(cur)  # 当前地址处的名称（对象可能是函数/常量）
                funcdata[name] = hex(cur)  # {函数名: 十六进制地址}
                datafunc[cur] = name  # {入口地址: 函数名}
                cur = NextHead(cur)  # 移动到一个条目（函数或变量）的地址
    return funcdata, datafunc


def get_func_cfgs_c(ea):
    """
    获取给定地址处的程序中所有函数的CFGs
    Args:
        ea: 程序的第一个段的地址

    Returns:
        list: 所有函数的 raw graph（CFG）
    """
    binary_name = idc.GetInputFile()  # IDA Pro 当前分析的二进制文件的名字
    raw_cfgs = raw_graphs(binary_name)  # 根据二进制文件名创建 raw_graphs 对象
    # 从段中提取 字典 外部{函数名称: 0x入口地址} {入口地址: 函数名称}
    externs_eas, ea_externs = processpltSegs()

    # 由于需要根据 ISA 类型和 funcnames 列表，新增属性
    isa = get_inf_structure().procName  # 当前分析的二进制文件的 ISA 类型
    funcnames = []  # 二进制文件中所有函数的名称列表

    # 遍历处理起始地址ea开始的所有的函数
    # SegStart(ea) 获取ea所属的段的起始地址
    # Functions() 获取指定地址范围内的所有函数的起始地址
    # 先获取所有函数的 名称 列表（因为计算特征需要 -- is_call()
    for funcea in Functions(SegStart(ea)):
        funcname = get_unified_funcname(funcea)  # 获取函数名（统一格式）
        funcnames.append(funcname)

    for funcea in Functions(SegStart(ea)):
        funcname = get_unified_funcname(funcea)  # 获取函数名（统一格式）
        func = get_func(funcea)  # 获取函数对象

        # 过滤：只保留 segment name = '.text' 的函数
        # 其他的，如 '.text.__x86.get_pc_thunk.ax', 'extern' 都不创建 cfg
        segname = SegName(funcea)
        if segname == '.text':
            # 获取该段中所有函数的 raw graph
            icfg = cfg.getCfg(func, externs_eas, ea_externs, isa, funcnames)  # CFG
            func_f = cfg.get_discoverRe_feature(func, icfg[0], isa, funcnames)  # 特征
            raw_g = raw_graph(funcname, icfg, func_f)  # raw graph
            raw_cfgs.append(raw_g)

    return raw_cfgs
