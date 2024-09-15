import argparse
import os
from utils import *
from graphnnSiamese import graphnn
from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def eval_comp(DATA_FILE_NAME, SOFTWARE, COMPILER):
    """
    评估模型在跨编译器上的性能
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    Dtype = tf.float32
    NODE_FEATURE_DIM = 7
    EMBED_DIM = 64
    EMBED_DEPTH = 3
    OUTPUT_DIM = 64
    ITERATION_LEVEL = 5
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    LOAD_PATH = './saved_model/graphnn-model_best'
    LOG_PATH = './log/log_' + "".join(SOFTWARE) + "".join(COMPILER) + '.txt'

    OPTIMIZATION = ('-O3',)
    ISA = ('x86-linux',)

    # 生成存储 acfg 的文件名列表
    FILE_NAME = get_file_name(DATA_FILE_NAME, SOFTWARE, ISA, OPTIMIZATION, COMPILER)
    # 从 data 文件中读取所有函数的 {函数名: 编号} 字典
    FUNC_NAME_DICT = get_f_dict(FILE_NAME)
    # 从 data 中读取 acfg 并构建对应的 graph 和 class 列表
    Gs, classes = read_graph(FILE_NAME, FUNC_NAME_DICT)

    # 加载 classes 的 随机排列 perm
    perm_file_name = './data/class_perm_comp_' + "".join(SOFTWARE) + "".join(COMPILER) + '.npz'
    if os.path.isfile(perm_file_name):
        perm = np.load(perm_file_name)
        if len(perm) < len(classes):
            perm = np.random.permutation(len(classes))
            np.save(perm_file_name, perm)
    else:
        perm = np.random.permutation(len(classes))
        np.save(perm_file_name, perm)

    # 固定跨编译器的数据集的 graph pairs
    comp_pair_file = './data/comp_' + "".join(SOFTWARE) + "".join(COMPILER) + '.json'
    if os.path.isfile(comp_pair_file):
        with open(comp_pair_file) as file:
            comp_ids = json.load(file)
        comp_epoch = generate_epoch_pair(
            Gs, classes, BATCH_SIZE, load_id=comp_ids)
    else:
        comp_epoch, comp_ids = generate_epoch_pair(
            Gs, classes, BATCH_SIZE, output_id=True)
        with open(comp_pair_file, 'w') as outfile:
            json.dump(comp_ids, outfile)

    # 创建图神经网络模型
    gnn = graphnn(N_x=NODE_FEATURE_DIM, Dtype=Dtype, N_embed=EMBED_DIM,
                  depth_embed=EMBED_DEPTH, N_o=OUTPUT_DIM, ITER_LEVEL=ITERATION_LEVEL,
                  lr=LEARNING_RATE)
    gnn.init(LOAD_PATH, LOG_PATH)

    # 输出数据集等信息
    gnn.say("DATA_FILE_NAME: {}".format(DATA_FILE_NAME))
    gnn.say("SOFTWARE: {}".format(",".join(SOFTWARE)))
    gnn.say("COMPILE: {}".format(",".join(COMPILER)))
    gnn.say("Total: {} graphs, {} functions".format(len(Gs), len(classes)))

    # 计算 recall 和 precision
    truth, pred, loss = get_truth_pred_loss_epoch(gnn, Gs, classes, BATCH_SIZE, load_data=comp_epoch)
    auc, fpr, tpr, roc_thresholds = get_auc_epoch(truth, pred)
    optimal_idx, optimal_threshold = get_optimal_threshold(fpr, tpr, roc_thresholds)
    precision, recall = cal_precision_recall(truth, pred, optimal_threshold)
    gnn.say("loss = {}".format(loss))
    gnn.say("AUC = {}".format(auc))
    gnn.say("precision = {}".format(precision))
    gnn.say("recall = {}".format(recall))


def eval_op(DATA_FILE_NAME, SOFTWARE, OPTIMIZATION):
    """
    评估模型在跨优化级别上的性能
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    Dtype = tf.float32
    NODE_FEATURE_DIM = 7
    EMBED_DIM = 64
    EMBED_DEPTH = 3
    OUTPUT_DIM = 64
    ITERATION_LEVEL = 5
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    LOAD_PATH = './saved_model/graphnn-model_best'
    LOG_PATH = './log/log_' + "".join(SOFTWARE) + "".join(OPTIMIZATION) + '.txt'

    ISA = ('x86-linux',)
    COMPILER = ('-gcc',)

    # 生成存储 acfg 的文件名列表
    FILE_NAME = get_file_name(DATA_FILE_NAME, SOFTWARE, ISA, OPTIMIZATION, COMPILER)
    # 从 data 文件中读取所有函数的 {函数名: 编号} 字典
    FUNC_NAME_DICT = get_f_dict(FILE_NAME)
    # 从 data 中读取 acfg 并构建对应的 graph 和 class 列表
    Gs, classes = read_graph(FILE_NAME, FUNC_NAME_DICT)

    # 加载 classes 的 随机排列 perm
    perm_file_name = './data/class_perm_op_' + "".join(SOFTWARE) + "".join(OPTIMIZATION) + '.npy'
    if os.path.isfile(perm_file_name):
        perm = np.load(perm_file_name)
        if len(perm) < len(classes):
            perm = np.random.permutation(len(classes))
            np.save(perm_file_name, perm)
    else:
        perm = np.random.permutation(len(classes))
        np.save(perm_file_name, perm)

    # 固定跨优化级别的数据集的 graph pairs
    op_pair_file = './data/op_' + "".join(SOFTWARE) + "".join(OPTIMIZATION) + '.json'
    if os.path.isfile(op_pair_file):
        with open(op_pair_file) as file:
            op_ids = json.load(file)
        op_epoch = generate_epoch_pair(
            Gs, classes, BATCH_SIZE, load_id=op_ids)
    else:
        op_epoch, op_ids = generate_epoch_pair(
            Gs, classes, BATCH_SIZE, output_id=True)
        with open(op_pair_file, 'w') as outfile:
            json.dump(op_ids, outfile)

    # 创建图神经网络模型
    gnn = graphnn(N_x=NODE_FEATURE_DIM, Dtype=Dtype, N_embed=EMBED_DIM,
                  depth_embed=EMBED_DEPTH, N_o=OUTPUT_DIM, ITER_LEVEL=ITERATION_LEVEL,
                  lr=LEARNING_RATE)
    gnn.init(LOAD_PATH, LOG_PATH)

    # 输出数据集等信息
    gnn.say("DATA_FILE_NAME: {}".format(DATA_FILE_NAME))
    gnn.say("SOFTWARE: {}".format(",".join(SOFTWARE)))
    gnn.say("OPTIMIZATION: {}".format(",".join(OPTIMIZATION)))
    gnn.say("Total: {} graphs, {} functions".format(len(Gs), len(classes)))

    # 计算 recall 和 precision
    truth, pred, loss = get_truth_pred_loss_epoch(gnn, Gs, classes, BATCH_SIZE, load_data=op_epoch)
    auc, fpr, tpr, roc_thresholds = get_auc_epoch(truth, pred)
    optimal_idx, optimal_threshold = get_optimal_threshold(fpr, tpr, roc_thresholds)
    precision, recall = cal_precision_recall(truth, pred, optimal_threshold)
    gnn.say("loss = {}".format(loss))
    gnn.say("AUC = {}".format(auc))
    gnn.say("precision = {}".format(precision))
    gnn.say("recall = {}".format(recall))


def eval_isa(DATA_FILE_NAME, SOFTWARE, ISA):
    """
    评估模型在跨目标体系结构上的性能
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    Dtype = tf.float32
    NODE_FEATURE_DIM = 7
    EMBED_DIM = 64
    EMBED_DEPTH = 3
    OUTPUT_DIM = 64
    ITERATION_LEVEL = 5
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    LOAD_PATH = './saved_model/graphnn-model_best'
    LOG_PATH = './log/log_' + "".join(SOFTWARE) + "-".join(ISA) + '.txt'

    OPTIMIZATION = ('-O0',)
    COMPILER = ('-gcc',)

    # 生成存储 acfg 的文件名列表
    FILE_NAME = get_file_name(DATA_FILE_NAME, SOFTWARE, ISA, OPTIMIZATION, COMPILER)
    # 从 data 文件中读取所有函数的 {函数名: 编号} 字典
    FUNC_NAME_DICT = get_f_dict(FILE_NAME)
    # 从 data 中读取 acfg 并构建对应的 graph 和 class 列表
    Gs, classes = read_graph(FILE_NAME, FUNC_NAME_DICT)

    # 加载 classes 的 随机排列 perm
    perm_file_name = './data/class_perm_isa_' + "".join(SOFTWARE) + "-".join(ISA) + '.npy'
    if os.path.isfile(perm_file_name):
        perm = np.load(perm_file_name)
        if len(perm) < len(classes):
            perm = np.random.permutation(len(classes))
            np.save(perm_file_name, perm)
    else:
        perm = np.random.permutation(len(classes))
        np.save(perm_file_name, perm)

    # 固定跨体系结构的数据集的 graph pairs
    isa_pair_file = './data/isa_' + "".join(SOFTWARE) + "-".join(ISA) + '.json'
    if os.path.isfile(isa_pair_file):
        with open(isa_pair_file) as file:
            isa_ids = json.load(file)
        isa_epoch = generate_epoch_pair(
            Gs, classes, BATCH_SIZE, load_id=isa_ids)
    else:
        isa_epoch, isa_ids = generate_epoch_pair(
            Gs, classes, BATCH_SIZE, output_id=True)
        with open(isa_pair_file, 'w') as outfile:
            json.dump(isa_ids, outfile)

    # 创建图神经网络模型
    gnn = graphnn(N_x=NODE_FEATURE_DIM, Dtype=Dtype, N_embed=EMBED_DIM,
                  depth_embed=EMBED_DEPTH, N_o=OUTPUT_DIM, ITER_LEVEL=ITERATION_LEVEL,
                  lr=LEARNING_RATE)
    gnn.init(LOAD_PATH, LOG_PATH)

    # 输出数据集等信息
    gnn.say("DATA_FILE_NAME: {}".format(DATA_FILE_NAME))
    gnn.say("SOFTWARE: {}".format(",".join(SOFTWARE)))
    gnn.say("ISA: {}".format(",".join(ISA)))
    gnn.say("Total: {} graphs, {} functions".format(len(Gs), len(classes)))

    # 计算 recall 和 precision
    truth, pred, loss = get_truth_pred_loss_epoch(gnn, Gs, classes, BATCH_SIZE, load_data=isa_epoch)
    auc, fpr, tpr, roc_thresholds = get_auc_epoch(truth, pred)
    optimal_idx, optimal_threshold = get_optimal_threshold(fpr, tpr, roc_thresholds)
    precision, recall = cal_precision_recall(truth, pred, optimal_threshold)
    gnn.say("loss = {}".format(loss))
    gnn.say("AUC = {}".format(auc))
    gnn.say("precision = {}".format(precision))
    gnn.say("recall = {}".format(recall))


if __name__ == '__main__':
    # 评估模型的跨编译器的性能
    eval_comp('./data/val_comp/', ('openssl-1.0.1f-',), ('-gcc', '-clang'))
    eval_comp('./data/val_comp/', ('gmp-6.3.0-',), ('-gcc', '-clang'))
    eval_comp('./data/val_comp/', ('curl-7.88-',), ('-gcc', '-clang'))

    # 评估模型的跨优化级别的性能
    eval_op('data/val_op/', ('openssl-1.0.1f-',), ('-O0', '-O3'))
    eval_op('data/val_op/', ('openssl-1.0.1f-',), ('-O1', '-O2'))
    eval_op('data/val_op/', ('gmp-6.3.0-',), ('-O0', '-O3'))
    eval_op('data/val_op/', ('gmp-6.3.0-',), ('-O1', '-O2'))
    eval_op('data/val_op/', ('curl-7.88-',), ('-O0', '-O3'))
    eval_op('data/val_op/', ('curl-7.88-',), ('-O1', '-O2'))

    # 评估模型的跨体系结构的性能
    eval_isa('data/val_isa/', ('openssl-1.0.1f-',), ('arm-linux', 'x86-linux'))
    eval_isa('data/val_isa/', ('openssl-1.0.1f-',), ('arm-linux', 'mips-linux'))
    eval_isa('data/val_isa/', ('openssl-1.0.1f-',), ('mips-linux', 'x86-linux'))
    eval_isa('data/val_isa/', ('gmp-6.3.0-',), ('arm-linux', 'x86-linux'))
    eval_isa('data/val_isa/', ('gmp-6.3.0-',), ('arm-linux', 'mips-linux'))
    eval_isa('data/val_isa/', ('gmp-6.3.0-',), ('mips-linux', 'x86-linux'))
