# coding=utf-8
import networkx as nx


class raw_graph:
    """
    函数的图形表示
    """
    def __init__(self, funcname, g, func_f):
        """
        Args:
            funcname:
            g (cfg_constructor.getCfg):
            func_f (discovRe.get_discoverRe_feature): 函数的特征
        """
        self.funcname = funcname  # 函数名
        self.old_g = g[0]  # cfg_constructor.getCfg 通过 attributingRe 构建的CFG
        self.g = nx.DiGraph()  # 新构建的ACFG
        self.entry = g[1]
        self.fun_features = func_f  # 函数的特征
        self.attributing()  # 调用 attributing 方法，给 g 节点添加特征

    def __len__(self):
        return len(self.g)

    def attributing(self):
        """
        给图 old_g 和 g 的节点添加属性，构建 ACFG
        """
        # 给 old_g 每个节点添加属性 'offs' - 后代序号的列表
        self.obtainOffsprings(self.old_g)

        # 根据 old_g 中节点属性，构建 g 的节点，并添加属性
        for node in self.old_g:
            fvector = self.retrieveVec(node, self.old_g)
            self.g.add_node(node)
            self.g.node[node]['featurev'] = fvector
            # 添加 succs 属性 - 直接后继节点的序号
            self.g.node[node]['succs'] = self.old_g.successors(node)

        # 给图g添加边
        for edge in self.old_g.edges():
            node1 = edge[0]
            node2 = edge[1]
            self.g.add_edge(node1, node2)

    def obtainOffsprings(self, old_g):
        """
        计算 old_g 中每个节点的后代数目，并添加节点属性 'offs'
        """
        nodes = old_g.nodes()
        for node in nodes:
            offsprings = {}
            self.getOffsprings(old_g, node, offsprings)
            old_g.node[node]['offs'] = list(set(offsprings.keys()))  # 去重

        return old_g

    def getOffsprings(self, g, node, offsprings):
        """
        递归计算图g中node节点的后代数目（所有子节点）
        offsprings (dict): {node的后代节点 : 1}
        """
        sucs = g.successors(node)  # sucs 是 node 的直接后继节点（由 edge 连接）
        # 递归添加后继节点
        for suc in sucs:
            if suc not in offsprings:
                offsprings[suc] = 1
                self.getOffsprings(g, suc, offsprings)

    def retrieveVec(self, id_, g):
        """
        生成图g中节点id_的特征向量
        """
        feature_vec = []
        # 0 'consts' 数值常数的数目
        numc = g.node[id_]['consts']
        feature_vec.append(float(len(numc)))
        # 1 'strings' 字符串的数目
        nums = g.node[id_]['strings']
        feature_vec.append(float(len(nums)))
        # 2 'offs' 后代数目
        offs = g.node[id_]['offs']
        feature_vec.append(float(len(offs)))
        # 3 'numAs' 算术指令的数目
        numAs = g.node[id_]['numAs']
        feature_vec.append(float(numAs))
        # 4 'numCalls' call指令数目
        calls = g.node[id_]['numCalls']
        feature_vec.append(float(calls))
        # 5 'numIns' 指令总数
        numIns = g.node[id_]['numIns']
        feature_vec.append(float(numIns))
        # # 6 'numLIs' 逻辑指令的数目
        # numLIs = g.node[id_]['numLIs']
        # feature_vec.append(numLIs)
        # 6 'numTIs' 转移指令的数目
        numTIs = g.node[id_]['numTIs']
        feature_vec.append(float(numTIs))

        return feature_vec


class raw_graphs:
    """
    二进制文件中的原始控制流图列表

    binary_name (str): 二进制文件的名称
    raw_graph_list (list): 该二进制文件中所有函数的原始CFG
    """
    def __init__(self, binary_name):
        self.binary_name = binary_name
        self.raw_graph_list = []

    def __len__(self):
        return len(self.raw_graph_list)

    def append(self, raw_g):
        self.raw_graph_list.append(raw_g)
