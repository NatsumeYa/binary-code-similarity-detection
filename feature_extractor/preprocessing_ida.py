# coding=utf-8
import argparse

import idc
from idc import *
from idaapi import *
from func import *
import os
import pickle


def parse_command():
    """
    --path “生成的 .ida 文件的地址”
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--path", type=str, help="The directory where to store the generated .ida file")
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    # 获取 --path 参数
    args = parse_command()
    path = idc.ARGV[2]  # 由于使用 ida 命令行运行，由此获取 path 参数

    # 设置 IDA 分析选项
    analysis_flags = idc.GetShortPrm(idc.INF_START_AF)
    analysis_flags &= ~idaapi.AF_IMMOFF  # 将 AF_IMMOFF 标志位从 analysis_flags 中清除
    idc.SetShortPrm(idc.INF_START_AF, analysis_flags)  # 关闭选项'automatically make offset'
    idaapi.autoWait()  # 自动等待分析完成

    # 获取二进制文件中所有函数的 CFG
    cfgs = get_func_cfgs_c(FirstSeg())

    # 将获取的 cfgs 用 pickle 序列化后存储到 指定的文件夹
    binary_name = idc.GetInputFile() + '.ida'
    fullpath = os.path.join(path, binary_name)
    cfg_dir = os.path.dirname(fullpath)
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    with open(fullpath, 'wb') as f:
        pickle.dump(cfgs, f)

    idc.Exit(0)  # 退出 IDA pro
