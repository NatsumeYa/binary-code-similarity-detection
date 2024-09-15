一、项目目录说明
1、data
1) acfgSSL_7 训练数据
2) val_comp 评估跨编译器的数据
3) val_isa 评估跨体系结构的数据
4) val_op 评估跨优化选项的数据
5) .npy 数据集的随机排列
6) .json 将数据集按 batch 划分时用到的 id

2、hypergraphs
调节超参数生成的图

3、log
日志

4、roc
保存模型的 fpr 和 tpr 数据，用于绘制 ROC 曲线

5、saved_model
保存模型的权重

6、script
调整超参数用到的.sh脚本

7、主要代码
1) graphnnSiamese.py 相似度检测模型
2) train.py 只是用训练集训练模型
3) train_large.py 包括大图子集时的模型训练
4) train_final.py 训练最终模型
5) tune_hyperparams.py 绘制选择超参数需要的图
6) eval.py 评估模型在跨编译器、跨优化级别、跨体系结构下的recall和precision
7) utils.py
8) count.py 统计训练集、验证集、测试集中三个体系结构下的graph数 & 原函数的数目
