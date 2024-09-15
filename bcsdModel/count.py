import argparse
import json
import os
from utils import *
from graphnnSiamese import graphnn
from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)


def train_model(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 设置环境变量
    # 从命令行参数中获取超参数的值
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    LOAD_PATH = None
    SAVE_PATH = None
    LOG_PATH = './log/count.txt'

    DATA_FILE_NAME = './data/acfgSSL_{}/'.format(NODE_FEATURE_DIM)
    SOFTWARE = ('openssl-3.0.13-',)
    OPTIMIZATION = ('-O0', '-O1', '-O2', '-O3')
    ISA = ('arm-linux', 'mips-linux', 'x86-linux')
    COMPILER = ('-gcc', '-clang')

    # 生成存储 acfg 的文件名列表
    FILE_NAME = get_file_name(DATA_FILE_NAME, SOFTWARE, ISA, OPTIMIZATION, COMPILER)
    # 从 data 文件中读取所有函数的 {函数名: 编号} 字典
    FUNC_NAME_DICT = get_f_dict(FILE_NAME)
    # 从 data 中读取 acfg 并构建对应的 graph 和 class 列表
    Gs, classes = read_graph(FILE_NAME, FUNC_NAME_DICT)

    # 加载 classes 的 随机排列 perm
    if os.path.isfile('./data/class_perm.npy'):
        perm = np.load('./data/class_perm.npy')  # 加载保存的数组数据
        if len(perm) < len(classes):
            perm = np.random.permutation(len(classes))
            np.save('./data/class_perm.npy', perm)
    else:
        perm = np.random.permutation(len(classes))
        np.save('./data/class_perm.npy', perm)

    # 划分训练集、验证集（也称开发集）、测试集
    Gs_train, classes_train, Gs_val, classes_val, Gs_test, classes_test \
        = partition_data(Gs, classes, [0.8, 0.1, 0.1], perm)

    x86_train, arm_train, mips_train = count_isa(Gs_train)
    x86_val, arm_val, mips_val = count_isa(Gs_val)
    x86_test, arm_test, mips_test = count_isa(Gs_test)

    # 创建 图神经网络 模型，并初始化
    gnn = graphnn(N_x=NODE_FEATURE_DIM, Dtype=Dtype, N_embed=EMBED_DIM,
                  depth_embed=EMBED_DEPTH, N_o=OUTPUT_DIM, ITER_LEVEL=ITERATION_LEVEL,
                  lr=LEARNING_RATE)
    gnn.init(LOAD_PATH, LOG_PATH)

    # 输出相关信息
    gnn.say("Hyperparameters and related configurations:")
    gnn.say(str(args))
    gnn.say("Total     : {} graphs, {} functions".format(len(Gs), len(classes)))
    gnn.say("Training  : {} graphs, {} functions".format(len(Gs_train), len(classes_train)))
    gnn.say("Validation: {} graphs, {} functions".format(len(Gs_val), len(classes_val)))
    gnn.say("Testing   : {} graphs, {} functions".format(len(Gs_test), len(classes_test)))

    gnn.say("x86  in training: {} graphs".format(x86_train))
    gnn.say("arm  in training: {} graphs".format(arm_train))
    gnn.say("mips in training: {} graphs".format(mips_train))

    gnn.say("x86  in validation: {} graphs".format(x86_val))
    gnn.say("arm  in validation: {} graphs".format(arm_val))
    gnn.say("mips in validation: {} graphs".format(mips_val))

    gnn.say("x86  in testing: {} graphs".format(x86_test))
    gnn.say("arm  in testing: {} graphs".format(arm_test))
    gnn.say("mips in testing: {} graphs".format(mips_test))


if __name__ == '__main__':
    args = parse_command()
    train_model(args)
