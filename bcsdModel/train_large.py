import argparse
import json
import os
from utils import *
from graphnnSiamese import graphnn
from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)


def train_model_with_large_subset(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 设置环境变量
    # 从命令行参数中获取设置值
    Dtype = args.dtype  # dtype 作为隐式命令行参数，在上面直接打印
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path
    ROC_PATH = args.roc_path

    TEST_FREQ = 1
    SAVE_FREQ = 5
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

    # 将验证集中的graph按找node_num大小进行分成大小子图集合
    Gs_small, classes_small, Gs_large, classes_large = classify_by_n_num(Gs_val, classes_val)

    # 固定验证集的 graph pairs
    # 在训练集每次训练一个 epoch 后，都需要在同一个验证集上比较性能
    # valid.json 存放将 Gs_val 按 BATCH_SIZE 个 graph 一批次划分得到的 valid_ids
    if os.path.isfile('./data/valid.json'):
        with open('./data/valid.json') as file:
            valid_ids = json.load(file)
        valid_epoch = generate_epoch_pair(Gs_val, classes_val, BATCH_SIZE, load_id=valid_ids)
    else:
        valid_epoch, valid_ids = generate_epoch_pair(Gs_val, classes_val, BATCH_SIZE, output_id=True)
        with open('./data/valid.json', 'w') as outfile:
            json.dump(valid_ids, outfile)

    # 固定大图集合
    if os.path.isfile('./data/large.json'):
        with open('./data/large.json') as file:
            large_ids = json.load(file)
        large_epoch = generate_epoch_pair(Gs_large, classes_large, BATCH_SIZE, load_id=large_ids)
    else:
        large_epoch, large_ids = generate_epoch_pair(Gs_large, classes_large, BATCH_SIZE, output_id=True)
        with open('./data/large.json', 'w') as outfile:
            json.dump(large_ids, outfile)

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

    # 统计验证集中的大图子集
    gnn.say("node_num < 10  in validation: {} graphs, {} functions".format(len(Gs_small), len(classes_small)))
    gnn.say("node_num >= 10 in validation: {} graphs, {} functions".format(len(Gs_large), len(classes_large)))

    # 计算未训练前的 auc 和 loss
    # 训练集
    truth_train_init, pred_train_init, loss_train_init = get_truth_pred_loss_epoch(gnn, Gs_train, classes_train,
                                                                                   BATCH_SIZE)
    auc_train_init, fpr_train_init, tpr_train_init, thresholds = get_auc_epoch(truth_train_init, pred_train_init)
    gnn.say("Initial training loss   = {}".format(loss_train_init))
    gnn.say("Initial training auc    = {}".format(auc_train_init))

    # 验证集
    truth_val_init, pred_val_init, loss_val_init = get_truth_pred_loss_epoch(gnn, Gs_val, classes_val, BATCH_SIZE,
                                                                             load_data=valid_epoch)
    auc_val_init, fpr_val_init, tpr_val_init, thresholds = get_auc_epoch(truth_val_init, pred_val_init)
    gnn.say("Initial validation loss = {}".format(loss_val_init))
    gnn.say("Initial validation auc  = {}".format(auc_val_init))

    # 大图子集
    truth_large_init, pred_large_init, loss_large_init = get_truth_pred_loss_epoch(gnn, Gs_large, classes_large,
                                                                                   BATCH_SIZE,
                                                                                   load_data=large_epoch)
    auc_large_init, fpr_large_init, tpr_large_init, thresholds = get_auc_epoch(truth_large_init, pred_large_init)
    gnn.say("Initial large-scale subset loss = {}".format(loss_large_init))
    gnn.say("Initial large-scale subset auc  = {}".format(auc_large_init))

    # 训练 模型
    best_auc = 0
    for i in range(1, MAX_EPOCH + 1):
        loss_train = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
        gnn.say("EPOCH {}/{} @ {}".format(i, MAX_EPOCH, datetime.now()))
        gnn.say("training loss   = {}".format(loss_train))

        if i % TEST_FREQ == 0:
            # 训练 TEST_FREQ 个 epoch 后，打印 auc 和 loss
            # 训练集
            truth_train, pred_train, loss_train = get_truth_pred_loss_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
            auc_train, fpr_train, tpr_train, roc_thresholds_train = get_auc_epoch(truth_train, pred_train)
            gnn.say("training auc    = {}".format(auc_train))
            gnn.say("training loss   = {}".format(loss_train))

            # 验证集
            truth_val, pred_val, loss_val = get_truth_pred_loss_epoch(gnn, Gs_val, classes_val, BATCH_SIZE,
                                                                      load_data=valid_epoch)
            auc_val, fpr_val, tpr_val, roc_thresholds_val = get_auc_epoch(truth_val, pred_val)
            gnn.say("validation loss = {}".format(loss_val))
            gnn.say("validation auc  = {}".format(auc_val))

            # 大图子集
            truth_large, pred_large, loss_large = get_truth_pred_loss_epoch(gnn, Gs_large, classes_large, BATCH_SIZE,
                                                                            load_data=large_epoch)
            auc_large, fpr_large, tpr_large, thresholds = get_auc_epoch(truth_large, pred_large)
            gnn.say("large-scale subset loss = {}".format(loss_large))
            gnn.say("large-scale subset auc  = {}".format(auc_large))

            if auc_val > best_auc:
                path = gnn.save(SAVE_PATH + '_best')
                best_auc = auc_val
                gnn.say("Model saved in {}".format(path))

        if i % SAVE_FREQ == 0:
            path = gnn.save(SAVE_PATH, epoch=i)
            gnn.say("Model saved in {}".format(path))


if __name__ == '__main__':
    args = parse_command()
    train_model_with_large_subset(args)
