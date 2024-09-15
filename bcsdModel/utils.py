import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_score, recall_score
import tensorflow as tf


def parse_command():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('--device', type=str, default='0',
                        help='visible gpu device')
    parser.add_argument('--fea_dim', type=int, default=7,
                        help='feature dimension')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--embed_depth', type=int, default=3,
                        help='embedding network depth')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='output layer dimension')
    parser.add_argument('--iter_level', type=int, default=5,
                        help='iteration times')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--epoch', type=int, default=10,
                        help='epoch number')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--load_path', type=str, default=None,
                        help='path for model loading, "#LATEST#" for the latest checkpoint')
    parser.add_argument('--save_path', type=str,
                        default='./saved_model/graphnn-model',
                        help='path for model saving')
    parser.add_argument('--log_path', type=str,
                        default='./log/log.txt',
                        help='path for training log')
    parser.add_argument('--roc_path', type=str,
                        default='./roc/roc_data.npz',
                        help='path for TPR & FPR')

    parsed_args = parser.parse_args()  # 解析命令行参数
    parsed_args.dtype = tf.float32
    return parsed_args


def get_file_name(DATA, SF, ISA, OP, CM):
    """
    根据不同参数，生成文件名列表

    Args:
        DATA: 数据集的文件夹名称
        SF(tuple): 软件包（开源库）的名称
        ISA(tuple): 指令集/体系结构的类型
        OP(tuple): 优化选项的类别
        CM: 编译器

    Returns:
        list: .json文件名列表
    """
    FILE_NAME = []
    for sf in SF:
        for isa in ISA:
            for op in OP:
                for cm in CM:
                    # 只有文件存在，才加入列表（因为 clang 只有 x86，缺
                    file_name = DATA + sf + isa + op + cm + ".json"
                    if os.path.exists(file_name):
                        FILE_NAME.append(file_name)

    return FILE_NAME


def get_f_dict(FILE_NAME):
    """
    读取列表中所有文件中的所有行记录中的函数名(fname 属性)，构建字典 {函数名：编号}

    Args:
        FILE_NAME(list): JSON文件名列表

    Returns:
        dict: {函数名: 编号}，
        注意是一个函数名对应一个 id
        （每一个JSON文件都有相同的函数名，只在dict中生成一次）
    """
    name_num = 0
    name_dict = {}

    # 遍历每一个文件
    for file_name in FILE_NAME:
        try:
            with open(file_name) as file:
                for line in file:
                    # 解析每行的 JSON 数据
                    g_info = json.loads(line.strip())
                    g_fname = g_info['fname']
                    # 为该行的函数，添加编号，并将映射存到 name_dict 中
                    if g_fname not in name_dict:
                        name_dict[g_fname] = name_num
                        name_num += 1
        except FileNotFoundError:
            print("No file: {}!".format(file_name))

    return name_dict


class graph(object):
    def __init__(self, node_num=0, label=None, name=None):
        """
        初始化 graph 对象
        （在读取 data 中每条行记录时创建相应的 graph,
        node_num 即 'n_num'，label 即 fname 在 fname_dict 中的编号，name 即 'src'
        """
        self.node_num = node_num  # 节点数目
        self.label = label        # 图所属 class 的编号（= 'fname'的编号），根据 label 在 classes 中索引，可得到同名函数的所有 graph 编号
        self.name = name          # 图所属的源文件名
        self.features = []        # 特征
        self.succs = []           # 后继
        self.preds = []           # 前驱

        # 为每个节点在 features, succs, preds 列表中初始化一个空列表
        if node_num > 0:
            for _ in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])

    def add_node(self, feature=[]):
        """
        向 graph 中添加一个新节点
        """
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])

    def add_edge(self, u, v):
        """
        在节点 u 和 v 之间添加一条有向边（新增前驱、后继关系）
        """
        self.succs[u].append(v)
        self.preds[v].append(u)

    def toString(self):
        """
        生成 graph 的字符串表示
        """
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)  # 添加节点 u 的每个特征
            ret += str(len(self.succs[u]))  # 添加节点 u 的后继数目
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)  # 添加节点 u 的每个后继
            ret += '\n'
        return ret


def read_graph(FILE_NAME, FUNC_NAME_DICT):
    """
    从 FILE_NAME 文件中读取 acfg，构建并返回对应的 graph 和 class 列表

    Args:
        FILE_NAME(list): JSON文件名的列表
        FUNC_NAME_DICT(dict): {函数名: 编号}

    Returns:
        graphs(list): 根据 JSON 行内容（acfg） 创建的 graph 组成的列表，一一对应
        classes(list): 和 fname 对应，同一 fname 在不同 JSON 文件行被读取时，会被认为是同一 class!
                       classes 的第 i 个元素（列表）中存放的是 编号为 i 的 fname 对应的所有 graph 在创建时的次序（编号）
    """
    graphs = []
    classes = []
    # 为每个 FUNC_NAME 在 classes 中初始化一个空列表
    if FUNC_NAME_DICT is not None:
        for _ in range(len(FUNC_NAME_DICT)):
            classes.append([])

    # 遍历每个文件
    for file_name in FILE_NAME:
        try:
            with open(file_name) as file:
                # 逐行读取JSON文件的内容
                for line in file:
                    # 解析每行JSON格式的图信息
                    g_info = json.loads(line.strip())
                    # 获取JSON行的信息
                    g_src = g_info['src']
                    g_n_num = g_info['n_num']
                    g_succs = g_info['succs']
                    g_features = g_info['features']
                    g_fname = g_info['fname']

                    # graph 对象的 label = 'fname'（函数方法名称）的编号
                    label = FUNC_NAME_DICT[g_fname]
                    # classes的第 i 个元素，是编号为 i 的 fname 的所有图
                    # classes 存放对应 label 的 fname 函数的所有 graph 的序号
                    classes[label].append(len(graphs))
                    # 创建当前图的 graph 对象，并添加到图列表中
                    cur_graph = graph(g_n_num, label, g_src)

                    # 设置当前图的节点特征
                    for u in range(g_n_num):
                        cur_graph.features[u] = np.array(g_features[u], dtype=np.float32)
                        # 添加边
                        for v in g_succs[u]:
                            cur_graph.add_edge(u, v)

                    # 将当前图加入图列表
                    graphs.append(cur_graph)

        except FileNotFoundError:
            print("No file: {}!".format(file_name))

    return graphs, classes


def partition_data(Gs, classes, partitions, perm):
    """
    将图数据集和类别按照指定的比例进行划分，返回划分后的结果。

    Args:
        Gs(list): graph 列表
        classes(list): 第 i 个元素（列表） = 编号为 i 的 fname 函数所有 graph 的序号
        partitions(list): 数据集划分比例 [train, validation, test]
        perm: 划分前先进行随机排列的列表，用于打乱数据顺序
    """
    C = len(classes)  # 类别数目
    start = 0.0  # 起始比例
    ret = []  # 返回的结果

    # 遍历每个划分比例
    for part in partitions:
        cur_g = []  # 当前划分的 graph 列表
        cur_c = []  # 当前划分的 class 列表
        end = start + part*C  # 计算结束比例

        # 遍历当前划分的类别范围
        for cls in range(int(start), int(end)):
            prev_class = classes[perm[cls]]  # 获取随机排列后的类别索引列表
            cur_c.append([])  # 添加空列表，用于存储当前类别中的图索引

            # 遍历当前类别中的 graph
            for i in range(len(prev_class)):
                cur_g.append(Gs[prev_class[i]])   # 添加对应的 graph
                cur_g[-1].label = len(cur_c) - 1  # 更新 graph 在 cur_g 中的 label
                cur_c[-1].append(len(cur_g) - 1)  # 更新 cur_c 的列表元素中的 graph 的序号
                # 经上述两部操作，cur_c 和 cur_g 中的 graph 和 class 得到在当前 part 的新编号

        ret.append(cur_g)
        ret.append(cur_c)
        start = end

    return ret


def count_isa(Gs):
    """
    统计数据集中各 ISA 的数目
    """
    x86 = 0
    arm = 0
    mips = 0

    for g in Gs:
        src = g.name

        if src.find('x86') != -1:
            x86 += 1
        elif src.find('arm') != -1:
            arm += 1
        elif src.find('mips') != -1:
            mips += 1

    return x86, arm, mips


def classify_by_n_num(Gs, classes):
    """
    将 Gs 中的 graph 按照 n_num 的大小进行分类
    """
    # node_num < 10 为量级进行分类
    Gs_small = []
    classes_small = []
    Gs_large = []
    classes_large = []

    for cls in range(len(classes)):
        # 由于 class 中存放相同函数名的所有 graph，差别不会特别大，默认按照第一个 graph 分类
        prev_class = classes[cls]
        g_0 = Gs[prev_class[0]]

        if g_0.node_num < 10:
            classes_small.append([])
            for i in range(len(prev_class)):
                Gs_small.append(Gs[prev_class[i]])
                Gs_small[-1].label = len(classes_small) - 1
                classes_small[-1].append(len(Gs_small) - 1)
        else:
            classes_large.append([])
            for i in range(len(prev_class)):
                Gs_large.append(Gs[prev_class[i]])
                Gs_large[-1].label = len(classes_large) - 1
                classes_large[-1].append(len(Gs_large) - 1)

    return Gs_small, classes_small, Gs_large, classes_large


def get_pair(Gs, classes, M, start=-1, output_id=False, load_id=None):
    """
    从 Gs 中生成编号从start开始的 M 个 graph 的正、负数据对。
    若 load_id 非空，则直接加载数据对。

    Args:
        Gs (list): graph 组成的列表
        classes (list): 每个 fname 一个 class 列表，包含 graph 的编号
        M (int): 每个批次中数据对的数目（batch_size
        start (int, optional): 开始位置索引，默认-1
        output_id (bool, optional): 是否输出数据对的索引，默认 False
        load_id (list, optional): 先前加载的数据对索引列表，默认 None

    Returns:
        X1_input: 三维数组，[i, j, :] 存放第i个数据对的左graph的第j个节点的特征向量
        X2_input: 三维数组，[i, j, :] 存放第i个数据对的右graph的第j个节点的特征向量
        node1_mask: 三维数组，[i, j, k] 对于第i个数据对的左graph，如果节点k是节点j的后继，则置1
        node2_mask: [i, j, k] 对于第i个数据对的右graph，如果节点k是节点j的后继，则置1
        y_input: 一维列表，[i] 如果第i个数据对相同，则取1，不同则-1
        pos_id (optional): 正样本的id对列表 [(g0_id, g1_id)]
        neg_id (optional): 负样本的id对列表 [(g0_id, h0_id)]
    """
    if load_id is None:
        # 为编号从 start 开始的 M 个 graph 分别生成 1 个正、负样本数据对
        C = len(classes)  # 所有 class 数目

        # 计算结束位置end，并调整 M 以确保不超出 Gs 范围
        if start + M > len(Gs):
            M = len(Gs) - start
        end = start + M

        pos_ids = []  # 正样本数据对的列表 [(G_0, G_1)]
        neg_ids = []  # 负样本数据对的列表 [(G_0, H_0)]

        # 生成 M 个 正样本数据对+负样本数据对
        for g_id in range(start, end):
            g0 = Gs[g_id]
            cls = g0.label
            tot_g = len(classes[cls])  # g0 所属 class 中的 graph 总数

            # 生成一个正样本数据对（必须有同 class 的才行）
            if tot_g >= 2:
                g1_id = classes[cls][np.random.randint(tot_g)]
                while g_id == g1_id:
                    g1_id = classes[cls][np.random.randint(tot_g)]
                pos_ids.append((g_id, g1_id))

            # 生成一个负样本数据对
            cls2 = np.random.randint(C)
            while len(classes[cls2]) == 0 or cls2 == cls:
                cls2 = np.random.randint(C)
            tot_g2 = len(classes[cls2])
            h_id = classes[cls2][np.random.randint(tot_g2)]
            neg_ids.append((g_id, h_id))
    else:
        # 从先前加载的数据对索引列表中加载数据对
        pos_ids = load_id[0]
        neg_ids = load_id[1]

    # 计算正、负样本数目
    M_pos = len(pos_ids)
    M_neg = len(neg_ids)
    M = M_pos + M_neg  # 新 M: 所有正负样本的 graph pair 数目

    # 计算图对中最大的节点个数
    maxN1 = 0  # 图对中左 graph 最大的节点数 node_num
    maxN2 = 0  # 图对中右 graph 最大的节点数 node_num
    for pair in pos_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
    for pair in neg_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)

    feature_dim = len(Gs[0].features[0])  # 由于所有节点的 feature 都是同样大小的列表，所以这里取 0 位置的就可以

    # 初始化数据数组
    X1_input = np.zeros((M, maxN1, feature_dim), dtype=np.float32)
    X2_input = np.zeros((M, maxN2, feature_dim), dtype=np.float32)
    node1_mask = np.zeros((M, maxN1, maxN1), dtype=np.float32)
    node2_mask = np.zeros((M, maxN2, maxN2), dtype=np.float32)
    y_input = np.zeros(M, dtype=np.float32)

    # 填充正样本数据
    # X1_input 由M个矩阵构成，第i个矩阵和id[i]的左graph对应，一共maxN1行，第j行存储该graph的第j个节点的特征feature
    # X2_input 右
    # node1_mask 由M个maxN1*maxN1矩阵构成，第i个矩阵和id[i]的左graph对应，如果graph中编号(u,v)的两个顶点构成边，则矩阵中(u,v)位置置1
    # node2_mask 右
    for i in range(M_pos):
        y_input[i] = 1  # 相同
        g1 = Gs[pos_ids[i][0]]  # G_0
        g2 = Gs[pos_ids[i][1]]  # G_1
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array(g1.features[u], dtype=np.float32)
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array(g2.features[u], dtype=np.float32)
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1

    # 填充负样本数据
    for i in range(M_pos, M_pos + M_neg):
        y_input[i] = -1  # 不同
        g1 = Gs[neg_ids[i-M_pos][0]]  # G_0
        g2 = Gs[neg_ids[i-M_pos][1]]  # H_0
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array(g1.features[u], dtype=np.float32)
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array((g2.features[u]), dtype=np.float32)
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1

    # 返回数据对及其相关信息
    if output_id:
        return X1_input, X2_input, node1_mask, node2_mask, y_input, pos_ids, neg_ids
    else:
        return X1_input, X2_input, node1_mask, node2_mask, y_input


def generate_epoch_pair(Gs, classes, M, output_id=False, load_id=None):
    """
    将图Gs和对应的classes按照M个一批次，生成一个epoch的数据对 (X1, X2, m1, m2, y)

    Args:
        Gs (list): graph 列表
        classes (list): 类别列表
        M (int): 每个批次（batch） 中的数据对数目
        output_id (bool, optional): 是否输出数据对的索引，默认为 False
        load_id (list, optional): 用于加载先前 id 数据的列表，默认为 None
        （即使运行多次，由于第一次已经固定随机序列，每次一个 graph 得到的 id 都会是固定的

    Returns:
        epoch_data (list): 元素是将 Gs 按 M个graph pair一批次 get_pair 得到的 (X1, X2, m1, m2, y)
        id_data: epoch_data 元素对应的 (pos_id, neg_id)
    """
    epoch_data = []
    id_data = []

    if load_id is None:
        # 将 Gs 按 M 个graph为一批次(batch)进行划分，得到数据对
        st = 0
        while st < len(Gs):
            if output_id:
                # 加载graph对数据和对应的id数据
                X1, X2, m1, m2, y, pos_id, neg_id = get_pair(Gs, classes, M, start=st, output_id=True)
                id_data.append((pos_id, neg_id))
            else:
                X1, X2, m1, m2, y = get_pair(Gs, classes, M, start=st)
            epoch_data.append((X1, X2, m1, m2, y))
            st += M  # 更新起始索引
    else:
        # 从 load_id 加载先前保存的数据
        id_data = load_id
        for id_pair in id_data:
            # 根据加载的id数据获取graph对数据
            X1, X2, m1, m2, y = get_pair(Gs, classes, M, load_id=id_pair)
            epoch_data.append((X1, X2, m1, m2, y))

    if output_id:
        # 输出 id 数据
        return epoch_data, id_data
    else:
        return epoch_data


def train_epoch(model, graphs, classes, batch_size, load_data=None):
    """
    模型训练一个 epoch，并返回该 epoch 的平均 loss

    Args:
        model: 训练的模型
        graphs (list): graph 组成的列表
        classes (list): graphs 的类别列表
        batch_size (int): 批处理大小
        load_data (optional): 预加载的数据，默认为 None

    Returns:
        float: 平均 loss 值
    """
    if load_data is None:
        # 生成一个新的 epoch 数据
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        # 使用提供的加载数据
        epoch_data = load_data

    # 随机打乱数据顺序
    perm = np.random.permutation(len(epoch_data))

    cum_loss = 0.0  # 累积 loss

    # 遍历打乱顺序后的数据
    for i in perm:
        # 获取当前索引处的数据
        cur_data = epoch_data[i]
        X1, X2, mask1, mask2, y = cur_data

        # 按批次（batch）的数据对模型进行训练，计算 loss 并累加
        loss = model.train(X1, X2, mask1, mask2, y)
        cum_loss += loss

    avg_loss = cum_loss / len(epoch_data)  # 平均 loss 值
    return avg_loss


def get_truth_pred_loss_epoch(model, graphs, classes, batch_size, load_data=None):
    """
    计算模型在一个 epoch 的 graphs 上的 truth, pred, loss

    Args:
        model: 模型
        graphs (list): graph 组成的列表
        classes (list): graphs 的 class 列表
        batch_size (int): 批大小--每个批次(batch)处理的 graphs 数目
        load_data: 用于评估的预加载数据

    Returns:
        truth: ground truth 列表
        sim: 模型输出的相似度
        loss: 损失值
    """
    tot_diff = []
    tot_truth = []
    cum_loss = 0.0

    # 生成/加载 epoch 数据
    # 决定是否用 验证集、测试集 评测
    if load_data is None:
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    # 累积模型预测的差异和真实值
    for cur_data in epoch_data:
        X1, X2, m1, m2, y = cur_data
        diff = model.calc_diff(X1, X2, m1, m2)  # -cos 即 论文中 Sim(g, g')
        loss = model.calc_loss(X1, X2, m1, m2, y)
        tot_diff += list(diff)  # + 可以实现两个列表的拼接
        tot_truth += list(y > 0)  # y>0则列表中添加True，否则添加False
        cum_loss += loss

    # 将累积数据转换为 numpy 数组
    diff = np.array(tot_diff, dtype=np.float32)
    truth = np.array(tot_truth, dtype=np.float32)

    # (1-diff)/2 = (1+cos)/2，取值范围 [0, 1]，值越大越相似
    sim = (1 - diff) / 2

    avg_loss = cum_loss / len(epoch_data)

    return truth, sim, avg_loss


def get_auc_epoch(truth, pred):
    """
    计算模型一个 epoch 的 AUC 值
    """
    # 计算评估指标 auc、precision、recall
    # thresholds 阈值数组
    # roc_curve() 通过给定的真实标签 truth 和预测分数 (1-diff)/2，计算不同阈值下的 fpr和tpr，
    # 从而生成 roc 曲线的数据点
    fpr, tpr, thresholds = roc_curve(truth, pred)
    model_auc = auc(fpr, tpr)

    return model_auc, fpr, tpr, thresholds


def get_optimal_threshold(fpr, tpr, thresholds):
    """
    roc 曲线左上方最优，约登指数确定最佳阈值
    """
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_idx, optimal_threshold


def cal_precision_recall(truth, pred, threshold):
    """
    使用阈值进行二分类，计算 precision、recall
    """
    pred_bool = (pred > threshold).astype(int)

    precision = precision_score(truth, pred_bool)
    recall = recall_score(truth, pred_bool)

    return precision, recall
