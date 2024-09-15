import re
import matplotlib.pyplot as plt
import numpy as np

# 四种数据集的 tag
TRAINING = 0
VALIDATION = 1
TESTING = 2
LARGE_SCALE_SUBSET = 3

# 设置全局字体为 Times New Roman 和大小为 5 号（12 pt）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12
})


def get_loss(tag, line):
    """
    提取字符串中 type 类型的所有的 loss
    注意：这里不匹配 initial * loss
    Args:
        tag (int): 和数据集 tag 对应
        line (str): 训练的 log 文件的一行字符串

    Returns:
        loss (float): 返回提取到 loss，没提取到返回 -1
    """
    loss = -1

    if tag == TRAINING:
        # 提取 training loss
        training_loss_match = re.match(r'training loss\s+= ([\d.]+)', line)
        if training_loss_match:
            loss = float(training_loss_match.group(1))
    elif tag == VALIDATION:
        # 提取 validation loss
        validation_loss_match = re.match(r'validation loss\s+= ([\d.]+)', line)
        if validation_loss_match:
            loss = float(validation_loss_match.group(1))
    elif tag == TESTING:
        # 提取 testing loss
        testing_loss_match = re.match(r'testing loss\s+= ([\d.]+)', line)
        if testing_loss_match:
            loss = float(testing_loss_match.group(1))
    elif tag == LARGE_SCALE_SUBSET:
        # 提取 large-scale subset loss
        large_scale_subset_loss_match = re.match(r'large-scale subset loss\s+= ([\d.]+)', line)
        if large_scale_subset_loss_match:
            loss = float(large_scale_subset_loss_match.group(1))

    return loss


def get_auc(tag, line):
    """
    提取字符串中 type 类型的 auc
    """
    auc = -1

    if tag == TRAINING:
        # 提取 training auc
        training_auc_match = re.match(r'training auc\s+= ([\d.]+)', line)
        if training_auc_match:
            auc = float(training_auc_match.group(1))
    elif tag == VALIDATION:
        # 提取 validation auc
        validation_auc_match = re.match(r'validation auc\s+= ([\d.]+)', line)
        if validation_auc_match:
            auc = float(validation_auc_match.group(1))
    elif tag == TESTING:
        # 提取 testing auc
        testing_auc_match = re.match(r'testing auc\s+= ([\d.]+)', line)
        if testing_auc_match:
            auc = float(testing_auc_match.group(1))
    elif tag == LARGE_SCALE_SUBSET:
        # 提取 large-scale subset auc
        large_scale_subset_auc_match = re.match(r'large-scale subset auc\s+= ([\d.]+)', line)
        if large_scale_subset_auc_match:
            auc = float(large_scale_subset_auc_match.group(1))

    return auc


def task1(filename):
    """
    选择 epoch，绘制图 4-1(a) loss vs epochs + 4-1(b) auc vs epochs
    Args:
        filename: 保存数据的 log 文件
    """
    # 所有需要提取的数据列表
    training_losses = []
    training_aucs = []
    validation_losses = []
    validation_aucs = []
    large_scale_subset_losses = []
    large_scale_subset_aucs = []

    # 读取文件并解析文件内容
    with open(filename, 'r', encoding='UTF-8') as file:  # 打开文本
        lines = file.readlines()

        for line in lines:
            # 提取 training loss
            if training_losses:  # list 非空
                loss = get_loss(TRAINING, line)
                if loss != -1:
                    training_losses.append(loss)
            else:
                # 提取初始值
                initial_training_loss_match = re.match(r'Initial training loss\s+= ([\d.]+)', line)
                if initial_training_loss_match:
                    training_losses.append(float(initial_training_loss_match.group(1)))

            # 提取 training auc
            if training_aucs:
                auc = get_auc(TRAINING, line)
                if auc != -1:
                    training_aucs.append(auc)
            else:
                initial_training_auc_match = re.match(r'Initial training auc\s+= ([\d.]+)', line)
                if initial_training_auc_match:
                    training_aucs.append(float(initial_training_auc_match.group(1)))

            # 提取 validation loss
            if validation_losses:
                loss = get_loss(VALIDATION, line)
                if loss != -1:
                    validation_losses.append(loss)
            else:
                initial_validation_loss_match = re.match(r'Initial validation loss\s+= ([\d.]+)', line)
                if initial_validation_loss_match:
                    validation_losses.append(float(initial_validation_loss_match.group(1)))

            # 提取 validation auc
            if validation_aucs:
                auc = get_auc(VALIDATION, line)
                if auc != -1:
                    validation_aucs.append(auc)
            else:
                initial_validation_auc_match = re.match(r'Initial validation auc\s+= ([\d.]+)', line)
                if initial_validation_auc_match:
                    validation_aucs.append(float(initial_validation_auc_match.group(1)))

            # 提取 large-scale subset loss
            if large_scale_subset_losses:
                loss = get_loss(LARGE_SCALE_SUBSET, line)
                if loss != -1:
                    large_scale_subset_losses.append(loss)
            else:
                initial_large_scale_subset_loss_match = re.match(r'Initial large-scale subset loss\s+= ([\d.]+)', line)
                if initial_large_scale_subset_loss_match:
                    large_scale_subset_losses.append(float(initial_large_scale_subset_loss_match.group(1)))

            # 提取 large-scale subset auc
            if large_scale_subset_aucs:
                auc = get_auc(LARGE_SCALE_SUBSET, line)
                if auc != -1:
                    large_scale_subset_aucs.append(auc)
            else:
                initial_large_scale_subset_auc_match = re.match(r'Initial large-scale subset auc\s+= ([\d.]+)', line)
                if initial_large_scale_subset_auc_match:
                    large_scale_subset_aucs.append(float(initial_large_scale_subset_auc_match.group(1)))

    epochs = list(range(len(training_losses)))

    # 画图 4-1(a) loss vs epoch
    plt.figure()
    plt.plot(epochs, training_losses, label='Training set', color='blue')
    plt.plot(epochs, validation_losses, label='Validation set', color='green')
    plt.plot(epochs, large_scale_subset_losses, label='Large-scale subset', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(left=0)
    plt.legend(loc='lower right')
    plt.savefig('./hypergraphs/4-1-a_loss-epoch.png')
    plt.show()

    # 画图 4-1(b) auc vs epoch
    plt.figure()
    plt.plot(epochs, training_aucs, label='Training set', color='blue')
    plt.plot(epochs, validation_aucs, label='Validation set', color='green')
    plt.plot(epochs, large_scale_subset_aucs, label='Large-scale subset', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.xlim(left=0)
    plt.legend(loc='lower right')
    plt.savefig('./hypergraphs/4-1-b_auc-epoch.png')
    plt.show()


def task2(fn1, fn2, fn3, fn4, fn5):
    """
    选择 lr [1e-2, 1e-3, 1e-4, 1e-4, 1e-6]
    绘制不同 lr 下 loss-epoch 图4-2
    """
    # 所有需要提取的数据列表
    losses2 = []  # lr=e-2 的 loss
    losses3 = []
    losses4 = []
    losses5 = []
    losses6 = []

    # 从文件 fn1 中提取 lr=e-2 的 losses
    with open(fn1, 'r', encoding='UTF-8') as file:  # 打开文本
        lines = file.readlines()

        for line in lines:
            # 提取 validation loss
            if losses2:
                loss = get_loss(VALIDATION, line)
                if loss != -1:
                    losses2.append(loss)
            else:
                initial_validation_loss_match = re.match(r'Initial validation loss\s+= ([\d.]+)', line)
                if initial_validation_loss_match:
                    losses2.append(float(initial_validation_loss_match.group(1)))

    # 从文件 fn2 中提取 lr=e-3 的 losses
    with open(fn2, 'r', encoding='UTF-8') as file:  # 打开文本
        lines = file.readlines()

        for line in lines:
            # 提取 validation loss
            if losses3:
                loss = get_loss(VALIDATION, line)
                if loss != -1:
                    losses3.append(loss)
            else:
                initial_validation_loss_match = re.match(r'Initial validation loss\s+= ([\d.]+)', line)
                if initial_validation_loss_match:
                    losses3.append(float(initial_validation_loss_match.group(1)))

    # 从文件 fn3 中提取 lr=e-4 的 losses
    with open(fn3, 'r', encoding='UTF-8') as file:  # 打开文本
        lines = file.readlines()

        for line in lines:
            # 提取 validation loss
            if losses4:
                loss = get_loss(VALIDATION, line)
                if loss != -1:
                    losses4.append(loss)
            else:
                initial_validation_loss_match = re.match(r'Initial validation loss\s+= ([\d.]+)', line)
                if initial_validation_loss_match:
                    losses4.append(float(initial_validation_loss_match.group(1)))

    # 从文件 fn4 中提取 lr=e-5 的 losses
    with open(fn4, 'r', encoding='UTF-8') as file:  # 打开文本
        lines = file.readlines()

        for line in lines:
            # 提取 validation loss
            if losses5:
                loss = get_loss(VALIDATION, line)
                if loss != -1:
                    losses5.append(loss)
            else:
                initial_validation_loss_match = re.match(r'Initial validation loss\s+= ([\d.]+)', line)
                if initial_validation_loss_match:
                    losses5.append(float(initial_validation_loss_match.group(1)))

    # 从文件 fn5 中提取 lr=e-6 的 losses
    with open(fn5, 'r', encoding='UTF-8') as file:  # 打开文本
        lines = file.readlines()

        for line in lines:
            # 提取 validation loss
            if losses6:
                loss = get_loss(VALIDATION, line)
                if loss != -1:
                    losses6.append(loss)
            else:
                initial_validation_loss_match = re.match(r'Initial validation loss\s+= ([\d.]+)', line)
                if initial_validation_loss_match:
                    losses6.append(float(initial_validation_loss_match.group(1)))

    epochs = list(range(len(losses2)))

    # 画图：不同 lr 下的 loss-epoch 曲线
    plt.figure()
    plt.plot(epochs, losses2, label='lr = 1e-2', color='blue')
    plt.plot(epochs, losses3, label='lr = 1e-3', color='green')
    plt.plot(epochs, losses4, label='lr = 1e-4', color='red')
    plt.plot(epochs, losses5, label='lr = 1e-5', color='purple')
    plt.plot(epochs, losses6, label='lr = 1e-6', color='yellow')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(left=0)
    plt.legend(loc='lower right')
    plt.savefig('./hypergraphs/4-2_lr-loss-epoch.png')
    plt.show()


def task3(fn1, fn2, fn3, fn4):
    """
    选择 embed depth [1, 2, 3, 4]
    绘制 4-3 不同 embed depth 下的 roc 曲线
    """
    # 读取四组 (fpr, tpr) 数据
    data1 = np.load(fn1)
    fpr1 = data1['fpr']
    tpr1 = data1['tpr']

    data2 = np.load(fn2)
    fpr2 = data2['fpr']
    tpr2 = data2['tpr']

    data3 = np.load(fn3)
    fpr3 = data3['fpr']
    tpr3 = data3['tpr']

    data4 = np.load(fn4)
    fpr4 = data4['fpr']
    tpr4 = data4['tpr']

    # 绘制 4 条 ROC 曲线
    plt.figure()
    plt.plot(fpr1, tpr1, label='n = 1', color='blue')
    plt.plot(fpr2, tpr2, label='n = 2', color='green')
    plt.plot(fpr3, tpr3, label='n = 3', color='red')
    plt.plot(fpr4, tpr4, label='n = 4', color='purple')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.savefig('./hypergraphs/4-3_embed_depth_roc.png')
    plt.show()


def task4(fn1, fn2, fn3, fn4, fn5):
    """
    选择 embed size [16, 32, 64, 128, 256]
    绘制不同 embed size 下的 ROC 曲线4-4
    """
    data1 = np.load(fn1)
    fpr1 = data1['fpr']
    tpr1 = data1['tpr']

    data2 = np.load(fn2)
    fpr2 = data2['fpr']
    tpr2 = data2['tpr']

    data3 = np.load(fn3)
    fpr3 = data3['fpr']
    tpr3 = data3['tpr']

    data4 = np.load(fn4)
    fpr4 = data4['fpr']
    tpr4 = data4['tpr']

    data5 = np.load(fn5)
    fpr5 = data5['fpr']
    tpr5 = data5['tpr']

    plt.figure()
    plt.plot(fpr1, tpr1, label='p = 16', color='blue')
    plt.plot(fpr2, tpr2, label='p = 32', color='green')
    plt.plot(fpr3, tpr3, label='p = 64', color='red')
    plt.plot(fpr4, tpr4, label='p = 128', color='purple')
    plt.plot(fpr5, tpr5, label='p = 256', color='yellow')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.savefig('./hypergraphs/4-4_embed_size_roc.png')
    plt.show()


def task5(fn1, fn2, fn3, fn4, fn5):
    """
    选择 iterations [1, 3, 5, 7, 9]
    绘制 4-5 不同 iteration 下的 ROC 曲线
    """
    data1 = np.load(fn1)
    fpr1 = data1['fpr']
    tpr1 = data1['tpr']

    data2 = np.load(fn2)
    fpr2 = data2['fpr']
    tpr2 = data2['tpr']

    data3 = np.load(fn3)
    fpr3 = data3['fpr']
    tpr3 = data3['tpr']

    data4 = np.load(fn4)
    fpr4 = data4['fpr']
    tpr4 = data4['tpr']

    data5 = np.load(fn5)
    fpr5 = data5['fpr']
    tpr5 = data5['tpr']

    plt.figure()
    plt.plot(fpr1, tpr1, label='T = 1', color='blue')
    plt.plot(fpr2, tpr2, label='T = 3', color='green')
    plt.plot(fpr3, tpr3, label='T = 5', color='red')
    plt.plot(fpr4, tpr4, label='T = 7', color='purple')
    plt.plot(fpr5, tpr5, label='T = 9', color='yellow')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.savefig('./hypergraphs/4-5_iter_level_roc_1e.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    task1('./log/log-5-30-4.txt')
    task2('./log/log_lre-2.txt', './log/log_lre-3.txt', './log/log_lre-4.txt', './log/log_lre-5.txt', './log/log_lre-6.txt')
    task3('./roc/roc_depth1.npz', './roc/roc_depth2.npz', './roc/roc_depth3.npz', './roc/roc_depth4.npz')
    task4('./roc/roc_dim16_1.npz', './roc/roc_dim32_1.npz', './roc/roc_dim64_1.npz', './roc/roc_dim128_1.npz', './roc/roc_dim256_1.npz')
    task5('./roc/roc_iter1_1e.npz', './roc/roc_iter3_1e.npz', './roc/roc_iter5_1e.npz', './roc/roc_iter7_1e.npz', './roc/roc_iter9_1e.npz')
