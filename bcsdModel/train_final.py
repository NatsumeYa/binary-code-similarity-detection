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
    """
    确定好最佳超参数后，训练集 + 验证集 重新模型训练，得到最终模型，在测试集上评估性能
    """
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

    # 划分新训练集（原训练集+验证集）、测试集
    Gs_train_new, classes_train_new, Gs_test, classes_test \
        = partition_data(Gs, classes, [0.9, 0.1], perm)

    # 固定测试集的 pairs
    if os.path.isfile('./data/test.json'):
        with open('./data/test.json') as file:
            test_ids = json.load(file)
        test_epoch = generate_epoch_pair(
            Gs_test, classes_test, BATCH_SIZE, load_id=test_ids)
    else:
        test_epoch, test_ids = generate_epoch_pair(
            Gs_test, classes_test, BATCH_SIZE, output_id=True)
        with open('./data/test.json', 'w') as outfile:
            json.dump(test_ids, outfile)

    # 创建 图神经网络 模型，并初始化
    gnn = graphnn(N_x=NODE_FEATURE_DIM, Dtype=Dtype, N_embed=EMBED_DIM,
                  depth_embed=EMBED_DEPTH, N_o=OUTPUT_DIM, ITER_LEVEL=ITERATION_LEVEL,
                  lr=LEARNING_RATE)
    gnn.init(LOAD_PATH, LOG_PATH)

    # 输出相关信息
    gnn.say("Hyperparameters and related configurations:")
    gnn.say(str(args))
    gnn.say("Total     : {} graphs, {} functions".format(len(Gs), len(classes)))
    gnn.say("Training  : {} graphs, {} functions".format(len(Gs_train_new), len(classes_train_new)))
    gnn.say("Testing   : {} graphs, {} functions".format(len(Gs_test), len(classes_test)))

    # 计算未训练前的 auc 和 loss
    truth_train_init, pred_train_init, loss_train_init = get_truth_pred_loss_epoch(gnn, Gs_train_new, classes_train_new,
                                                                                   BATCH_SIZE)
    auc_train_init, fpr_train_init, tpr_train_init, thresholds = get_auc_epoch(truth_train_init, pred_train_init)
    gnn.say("Initial training loss   = {}".format(loss_train_init))
    gnn.say("Initial training auc    = {}".format(auc_train_init))

    # 测试集
    truth_test_init, pred_test_init, loss_test_init = get_truth_pred_loss_epoch(gnn, Gs_test, classes_test, BATCH_SIZE,
                                                                                load_data=test_epoch)
    auc_test_init, fpr_test_init, tpr_test_init, thresholds = get_auc_epoch(truth_test_init, pred_test_init)
    gnn.say("Initial testing loss = {}".format(loss_test_init))
    gnn.say("Initial testing auc  = {}".format(auc_test_init))

    # 训练 模型
    best_auc = 0
    for i in range(1, MAX_EPOCH + 1):
        train_epoch(gnn, Gs_train_new, classes_train_new, BATCH_SIZE)
        gnn.say("EPOCH {}/{} @ {}".format(i, MAX_EPOCH, datetime.now()))

        if i % TEST_FREQ == 0:
            # 训练 TEST_FREQ 个 epoch 后，打印 auc 和 loss
            # 训练集
            truth_train, pred_train, loss_train = get_truth_pred_loss_epoch(gnn, Gs_train_new, classes_train_new,
                                                                            BATCH_SIZE)
            auc_train, fpr_train, tpr_train, roc_thresholds_train = get_auc_epoch(truth_train, pred_train)
            gnn.say("training auc    = {}".format(auc_train))
            gnn.say("training loss   = {}".format(loss_train))

            # 测试集
            truth_test, pred_test, loss_test = get_truth_pred_loss_epoch(gnn, Gs_test, classes_test, BATCH_SIZE,
                                                                         load_data=test_epoch)
            auc_test, fpr_test, tpr_test, roc_thresholds_test = get_auc_epoch(truth_test, pred_test)
            gnn.say("testing loss = {}".format(loss_test))
            gnn.say("testing auc  = {}".format(auc_test))

            if auc_test > best_auc:
                path = gnn.save(SAVE_PATH + '_best')
                best_auc = auc_test
                gnn.say("Model saved in {}".format(path))

        if i % SAVE_FREQ == 0:
            path = gnn.save(SAVE_PATH, epoch=i)
            gnn.say("Model saved in {}".format(path))


if __name__ == '__main__':
    args = parse_command()
    train_model(args)
