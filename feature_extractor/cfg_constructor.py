# coding=utf-8
import globals
from graph_analysis_ida import *
from graph_property import *
import networkx as nx
from idautils import *
from idaapi import *
from idc import *


def checkCondition(ea):
    """
    检查给定地址处的指令是否是条件分支指令

    Args:
        ea: 指令地址

    Returns:
        bool: 该指令是条件分支指令，返回True，否则False
    """
    conds = globals.BRANCH
    opcode = GetMnem(ea)  # 得到地址处指令的助记符
    if opcode in conds:
        return True
    return False


def checkCB(bl):
    """
    检查基本块中是否包含条件分支指令

    Args:
        bl (tuple): 基本块的 (起始地址, 结束地址)

    Returns:
        addr: 如果找到条件分支指令，返回其地址；
                没找到，返回基本块的最后一条指令的地址
    """
    start = bl[0]  # 基本块的起始地址
    end = bl[1]  # 基本块的结束地址
    ea = start  # 当前指令地址

    while ea < end:
        if checkCondition(ea):
            return ea
        ea = NextHead(ea)  # 更新到下一个有效指令的地址

    return PrevHead(end)


def obtain_block_sequence(func):
    """
    获取函数func的控制块和基本块序列

    Args:
        func (func_t): 函数对象，有 startEA 和 endEA 属性（函数的起始和结束地址（实际上是下一条指令的地址

    Returns:
        control_blocks: 控制块序列
        x: 基本块序列
    """
    control_blocks = {}  # 控制块的字典
    main_blocks = {}  # main 块的字典
    # FlowChart() 获取所有基本块的迭代器模式
    # blocks 函数的[每一个基本块的(起始地址, 结束地址)]
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]

    # 遍历每一个基本块
    for bl in blocks:
        base = bl[0]  # 基本块的起始地址
        end = PrevHead(bl[1])  # 基本块内最后一条有效指令的起始地址
        control_ea = checkCB(bl)  # 该基本块内第一个分支指令的地址
        control_blocks[control_ea] = bl  # {分支指令地址，所属基本块的(起始地址，结束地址)}
        control_blocks[end] = bl  # {基本块最后一条指令地址，(起始地址，结束地址)}
        if func.startEA <= base <= func.endEA:  # 基本块在函数内
            main_blocks[base] = bl  # {基本块的 起始地址，(起始地址，结束地址)}
        x = sorted(main_blocks)  # x 按地址排序 基本块

    return control_blocks, x


def get_discoverRe_feature(func, icfg, isa, funcnames):
    """
    获取函数 `func` 的各种特征并存储在列表中返回。

    Args:
        func: 函数对象
        icfg: 函数的控制流图

    Returns:
        list: 包含各种特征的列表
    """
    features = []   # 初始化特征列表

    # 提取函数的特征
    # 0 call指令的总数
    FunctionCalls = getFuncCalls(func, isa, funcnames)
    features.append(FunctionCalls)
    # 1 逻辑指令的总数
    LogicInstr = getLogicInsts(func, isa)
    features.append(LogicInstr)
    # 2 转移指令的总数
    Transfer = getTransferInsts(func, isa)
    features.append(Transfer)
    # 3 局部变量
    Locals = getLocalVariables(func)
    features.append(Locals)
    # 4 基本块
    BB = getBasicBlocks(func)
    features.append(BB)
    # 5 CFG 边数
    Edges = len(icfg.edges())
    features.append(Edges)
    # 6 函数内部的调用
    Incoming = getIncommingCalls(func)
    features.append(Incoming)
    # 7 指令总数
    Instrs = getIntrs(func)
    features.append(Instrs)
    # 8 between值
    between = retrieveGP(icfg)
    features.append(between)
    # 9 字符串, 10 数值常量
    strings, consts = getfunc_consts(func)  # 获取字符串、数值常量
    features.append(strings)
    features.append(consts)

    return features


def attributingRe(cfg, externs_eas, ea_externs, isa, funcnames):
    """
    给 CFG 中的节点设置属性

    Args:
        cfg (nx.DiGraph): CFG, 每个节点代表一个基本块
        externs_eas: 外部 {函数/变量名: 0x入口地址}
        ea_externs: 外部 {入口地址: 函数/变量名}
    """
    for node_id in cfg:
        # 获取当前节点
        bl = cfg.node[node_id]['label']
        # 'numIns' 指令的总数
        numIns = calInsts(bl)
        cfg.node[node_id]['numIns'] = numIns
        # 'numCalls' 过程调用指令的数目
        numCalls = calCalls(bl, isa, funcnames)
        cfg.node[node_id]['numCalls'] = numCalls
        # 'numLIs' 逻辑指令的数目
        numLIs = calLogicInstructions(bl, isa)
        cfg.node[node_id]['numLIs'] = numLIs
        # 'numAs' 算数指令的数目
        numAs = calArithmeticIns(bl, isa)
        cfg.node[node_id]['numAs'] = numAs
        # 'numTIs' 转移指令的数目
        numTIs = calTransferIns(bl, isa)
        cfg.node[node_id]['numTIs'] = numTIs
        # 'numNc' 字符串&常量的数目
        strings, consts = getBBconsts(bl)
        cfg.node[node_id]['numNc'] = len(strings) + len(consts)
        # 'consts' 常量列表
        cfg.node[node_id]['consts'] = consts
        # 'strings' 字符串列表
        cfg.node[node_id]['strings'] = strings
        # 'externs' 外部引用的名称列表
        externs = retrieveExterns(bl, ea_externs)
        cfg.node[node_id]['externs'] = externs


def getCfg(func, externs_eas, ea_externs, isa, funcnames):
    """
    生成函数func的CFG，其中 CFG 有特征见 attributingRe()

    Args:
        func (func_t): 函数对象，有 startEA 和 endEA 属性（函数的起始和结束地址
        externs_eas (dict): 外部 {函数/变量名称: 0x入口地址}
        ea_externs (dict): 外部 {入口地址: 函数/变量名}

    Returns:
        cfg: 用 networkx 表示的CFG
        0: 用于防止出现没有返回对象的情况!
    """
    # 获取函数的起始地址、结束地址
    func_start = func.startEA
    func_end = func.endEA
    # 创建一个有向图作为 cfg
    cfg = nx.DiGraph()
    # 获取函数的控制块和基本块序列
    control_blocks, main_blocks = obtain_block_sequence(func)
    visited = {}
    start_node = None

    # 遍历每个控制块，构建 CFG 的节点和边
    for bl in control_blocks:
        start = control_blocks[bl][0]  # 该控制块所属的基本块的起始地址
        end = control_blocks[bl][1]  # 该控制块的结束地址
        src_node = (start, end)

        # 如果 src_node 没被访问的话，加入到 cfg 中
        if src_node not in visited:
            src_id = len(cfg)
            visited[src_node] = src_id
            cfg.add_node(src_id)
            cfg.node[src_id]['label'] = src_node  # 设置节点属性'label'，即基本块的(startEA, endEA)
        else:
            src_id = visited[src_node]

        # 标记起始节点和结束节点
        if start == func_start:
            cfg.node[src_id]['c'] = "start"  # 设置节点属性'c'
            start_node = src_node
        if end == func_end:
            cfg.node[src_id]['c'] = "end"

        # 处理指向该控制块的代码引用，添加边到CFG
        refs = CodeRefsTo(start, 0)  # 查找所有引用 start 地址的代码位置的列表（包括数据引用和控制流引用）
        for ref in refs:
            if ref in control_blocks:
                dst_node = control_blocks[ref]
                if dst_node not in visited:
                    visited[dst_node] = len(cfg)
                dst_id = visited[dst_node]
                cfg.add_edge(dst_id, src_id)
                cfg.node[dst_id]['label'] = dst_node
        refs = CodeRefsTo(start, 1)  # 查询引用 start 地址的控制流类型的代码位置的列表（如跳转指令、函数调用等）
        for ref in refs:
            if ref in control_blocks:
                dst_node = control_blocks[ref]
                if dst_node not in visited:
                    visited[dst_node] = len(cfg)
                dst_id = visited[dst_node]
                cfg.add_edge(dst_id, src_id)
                cfg.node[dst_id]['label'] = dst_node

    attributingRe(cfg, externs_eas, ea_externs, isa, funcnames)
    return cfg, 0
\