from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def graph_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    """
    图嵌入函数：将图数据进行嵌入表示
    网络结构：X -- affine(W1) -- ReLU --
    (Message -- affine(W2) -- add(with aff W1) -- ReLU -- )* MessageAll
    -- output

    Args:
        X: 输入节点的特征张量，形状为 [batch_size, node_num, N_x]
        msg_mask: 类似邻接矩阵，存储节点间后继关系，形状为 [batch_size, node_num, node_num]
        # 图神经网络的消息传递，也称邻近聚合
        N_x: 输入特征的维度
        N_embed: embed_dim 嵌入层的维度
        N_o: 输出的维度
        iter_level: 迭代次数
        Wnode: 节点特征到嵌入层的权重矩阵
        Wembed: 嵌入层中每个迭代阶段的权重矩阵列表
        W_output: 嵌入层到输出层的权重矩阵
        b_output: 输出层的偏置向量

    Returns:
        output: 输出张量，形状为 [batch_size, N_o]
    """
    # 对输入节点特征进行线性变换，并重新组织形状为 [batch_size, num_nodes, N_embed] 的节点值张量
    node_val = tf.reshape(tf.matmul(tf.reshape(X, [-1, N_x]), Wnode),
                          [tf.shape(X)[0], -1, N_embed])
    # 对节点值张量应用 ReLU 激活函数
    cur_msg = tf.nn.relu(node_val)  # [batch_size, node_num, N_embed]

    # 进行 iter_level 次消息传递迭代
    for t in range(iter_level):
        # 根据消息掩码传递消息
        Li_t = tf.matmul(msg_mask, cur_msg)  # [batch, node_num, embed_dim]
        # 将传递的消息张量 reshape 并应用复杂函数，由 Wembed 中的多个权重矩阵组成
        cur_info = tf.reshape(Li_t, [-1, N_embed])
        for Wi in Wembed:
            if Wi == Wembed[-1]:
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        # 将复杂函数处理后的结果 reshape 回原始形状
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
        # 将节点张量和邻居值相加
        tot_val_t = node_val + neigh_val_t
        # 应用非线性函数（tanh）
        tot_msg_t = tf.nn.tanh(tot_val_t)
        cur_msg = tot_msg_t  # [batch_size, node_num, embed_dim]

    # 汇总所有节点的消息嵌入，对第二维求和(reduce_sum)
    g_embed = tf.reduce_sum(cur_msg, 1)  # [batch_size, embed_dim]
    # 最终输出，对嵌入结果应用权重矩阵和偏置向量进行线性变换
    output = tf.matmul(g_embed, W_output) + b_output
    return output


class graphnn(object):
    """
    定义图神经网络模型
    """

    def __init__(self, N_x, Dtype, N_embed, depth_embed,
                 N_o, ITER_LEVEL, lr, device='/gpu:0'):
        """
        初始化图神经网络模型。

        Args:
            N_x: acfg中节点特征feature的维度
            Dtype: 数据类型（默认 tf.float32）
            N_embed: 嵌入的大小/维度
            depth_embed: 嵌入的深度
            N_o: 输出层的维度
            ITER_LEVEL: 迭代次数
            lr: 学习率
            device: 计算设备，默认 GPU
        """
        self.sess = None
        self.saver = None
        self.log_file = None
        self.NODE_LABEL_DIM = N_x  # 节点特征/标签的维度

        tf.reset_default_graph()  # 重置 Tensorflow 默认图，避免冲突

        with tf.device(device):
            # 初始化模型参数
            Wnode = tf.Variable(tf.truncated_normal(
                shape=[N_x, N_embed], stddev=0.1, dtype=Dtype))
            Wembed = []
            for i in range(depth_embed):
                Wembed.append(tf.Variable(tf.truncated_normal(
                    shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype)))

            W_output = tf.Variable(tf.truncated_normal(
                shape=[N_embed, N_o], stddev=0.1, dtype=Dtype))
            b_output = tf.Variable(tf.constant(0, shape=[N_o], dtype=Dtype))  # 一维数组

            # 定义输入占位符
            X1 = tf.placeholder(Dtype, [None, None, N_x])  # [B, N_node, N_x]
            msg1_mask = tf.placeholder(Dtype, [None, None, None])  # None 表示该维度的大小可变
            # [B, N_node, N_node]
            self.X1 = X1
            self.msg1_mask = msg1_mask

            # 对graph pair的左图X1进行嵌入
            embed1 = graph_embed(X1, msg1_mask, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output)  # [B, N_x]

            X2 = tf.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.placeholder(Dtype, [None, None, None])
            self.X2 = X2
            self.msg2_mask = msg2_mask

            # 对graph pair的右图X2进行嵌入
            embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output)
            label = tf.placeholder(Dtype, [None, ])
            self.label = label
            self.embed1 = embed1
            self.embed2 = embed2

            # 计算余弦相似度
            # embed1 和 embed2 的 shape 都是[batch_size, N_o]（前面都有 Batch_size 第一维度）
            # 加上一个非常小的数 1e-10，用于防止除零错误，对实际结果几乎无影响
            # 最终 cos = embed1 和 embed2 的余弦相似度，即 Sim(g, g')
            cos = tf.reduce_sum(embed1 * embed2, 1) / tf.sqrt(
                tf.reduce_sum(embed1 ** 2, 1) * tf.reduce_sum(embed2 ** 2, 1) + 1e-10)
            diff = -cos
            self.diff = diff

            # 定义损失函数 loss = mean((label - Sim)^2)
            loss = tf.reduce_mean((diff + label) ** 2)
            self.loss = loss

            # 定义优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            self.optimizer = optimizer

    def init(self, LOAD_PATH, LOG_PATH):
        """
        初始化模型和会话

        Args:
            LOAD_PATH(str): 模型加载路径，如果为 None 则从头开始训练
            LOG_PATH(str): log 文件路径
        """
        config = tf.ConfigProto(allow_soft_placement=True)  # 没有GPU时，可以自适应调整到CPU
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=100)  # 保存 checkpoints 文件
        self.sess = sess
        self.saver = saver
        self.log_file = None

        if LOAD_PATH is not None:
            if LOAD_PATH == "#LATEST#":
                checkpoint_path = tf.train.latest_checkpoint('./')  # 查找最新保存的 checkpoint
            else:
                checkpoint_path = LOAD_PATH

            saver.restore(sess, checkpoint_path)  # 从 checkpoint_path 中恢复模型，将文件中的变量值加载到当前会话sess的相应变量中

            if LOG_PATH is not None:
                self.log_file = open(LOG_PATH, 'a+')  # 追加
            self.say("model loaded from file: {} @ {}".format(checkpoint_path, datetime.now()))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH is not None:
                self.log_file = open(LOG_PATH, 'w')  # 覆写
            self.say("Training start @ {}".format(datetime.now()))

    def say(self, string):
        """
        输出信息并写入 log 文件（如果已设置）
        Args:
            string(str): 要输出的字符串
        """
        print(string)
        if self.log_file is not None:
            self.log_file.write(string + '\n')

    def get_embed(self, X1, mask1):
        """
        获取节点嵌入向量
        这里只获取 embed1，相当于 样本，另一个是与之相同/不同的样本

        Args:
            X1: 输入节点的特征数据
            mask1: 节点消息传递掩码

        Returns:
            节点的嵌入向量
        """
        vec, = self.sess.run([self.embed1], feed_dict={self.X1: X1, self.msg1_mask: mask1})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, y):
        """
        计算模型的 loss 值

        Args:
            X1: 第一个输入数据的特征张量
            X2: 第二个输入数据的特征张量
            mask1: 第一个输入数据的消息掩码张量
            mask2: 第二个输入数据的消息掩码张量
            y: ground truth

        Returns:
            float: loss 值
        """
        cur_loss, = self.sess.run([self.loss], feed_dict={self.X1: X1, self.X2: X2,
                                                          self.msg1_mask: mask1, self.msg2_mask: mask2, self.label: y})
        return cur_loss

    def calc_diff(self, X1, X2, mask1, mask2):
        """
        计算两个输入数据的嵌入表示之间的差异

        Args:
            X1: 第一个输入数据的特征张量
            X2: 第二个输入数据的特征张量
            mask1: 第一个输入数据的消息掩码张量
            mask2: 第二个输入数据的消息掩码张量

        Returns:
            嵌入之间的差异
        """
        diff, = self.sess.run([self.diff], feed_dict={self.X1: X1, self.X2: X2,
                                                      self.msg1_mask: mask1, self.msg2_mask: mask2})
        return diff

    def train(self, X1, X2, mask1, mask2, y):
        """
        训练模型，并返回当前数据的 loss 值
        """
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.X1: X1, self.X2: X2,
                                                                        self.msg1_mask: mask1, self.msg2_mask: mask2,
                                                                        self.label: y})
        return loss

    def save(self, path, epoch=None):
        """
        保存模型到指定路径，并可指定保存的训练轮数

        Args:
            path: 模型保存路径
            epoch (optional): 训练轮数

        Returns:
            checkpoint_path: 保存的模型路径
        """
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
