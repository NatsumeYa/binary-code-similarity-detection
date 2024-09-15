一、项目环境
python 2.7.16(32-bit)
反汇编工具 IDA Pro v6.8
idapython-1.7.2_ida6.8_py2.7_win32

二、项目目录说明
1、feature_extractor
对二进制文件（如 .o）反汇编，并提取特征
1) gen_acfg_Record.py 生成二进制文件的所有函数的 acfg 记录
2) preprocessing_ida.py 使用 IDA Pro 分析二进制文件
3) func.py 生成原始控制流图
4) cfg_constructor.py 构建raw_graph并统计属性
5) raw_graphs.py 原始控制流图类
6) graph_analysis_ida.py 统计函数的属性
7) globals.py 三种架构下指令助记符分类

2、python ida-python
执行 Mark Dictionary as  Sources Root

3、data 数据集
1) binary_codes：二进制文件
2) cfgs：IDA Pro 分析出的原始控制流图（raw_graph）
3) acfgs：提取出的带属性的特征流图记录

三、IDA Pro 指令
使用 ida 命令行分析二进制文件
在 feature_extractor 下打开 Terminal，
用 ida 命令行启动 IDA pro，-S### 表示打开 database 时执行脚本文件
其中 XXX.o 是所分析的二进制代码的路径，选项参数的含义请见 https://blog.csdn.net/qq523176585/article/details/112598362
-A 批量执行（不打开 IDA pro 界面）
```commandline
D:\IDAPro\IDApro6.8\idaq.exe -c -S"..\..\feature_extractor\preprocessing_ida.py --path ..\cfgs" .\data\binary_codes\XXX.o
```
注意：文件的相对存储位置！
