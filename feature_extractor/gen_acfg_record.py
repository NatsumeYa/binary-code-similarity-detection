# coding=utf-8
import json
import pickle
import os
import subprocess

# 由于需要频繁在py命令行和ida命令行切换，统一使用绝对路径
# IDA Pro 可执行文件的位置，脚本位置
IDA_PRO_PATH = r"D:\IDAPro\IDApro6.8\idaq.exe"
PREPROCESSING_PATH = r"D:\BCSD\feature_extractor\preprocessing_ida.py"

# 存放项目、数据（二进制文件、cfg、acfg）的路径
PROJECT_PATH = r"D:\BCSD\feature_extractor"
# 项目下的相对地址
BINARY_PATH = r"data\binary_codes"
CFG_PATH = r"data\cfgs"
ACFG_PATH = r"data\acfgs"


def load_pickle_file(file_path):
    """
    从指定的 pickle 文件中加载对象
    """
    with open(file_path, 'rb') as f:
        cfgs = pickle.load(f)
    return cfgs


def write_json_file(file_path, json_strings):
    """
    将列表中的 JSON 字符串追加写入文件
    """
    # 先检查文件的目录是否存在
    acfgs_dir = os.path.dirname(file_path)
    if not os.path.exists(acfgs_dir):
        os.makedirs(acfgs_dir)

    with open(file_path, 'a') as f:
        # 将每个JSON字符串写入文件，并在每个字符串后添加换行符
        for str in json_strings:
            f.write(str + '\n')


class acfg_record:
    """
    函数的 acfg 的行记录
    """
    def __init__(self, src, n_num, succs, features, fname):
        self.src = src
        self.n_num = n_num
        self.succs = succs
        self.features = features
        self.fname = fname

    def to_dict(self):
        """
        将 acfg_record 对象转化为字典表示
        """
        return {
            "src": self.src,
            "n_num": self.n_num,
            "succs": self.succs,
            "features": self.features,
            "fname": self.fname
        }

    def to_json(self):
        """
        将 acfg_record 对象转化为 JSON 字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)


def cfgs2acfgs(src, cfgs):
    """
    将 cfgs 转化为 acfgs 的 JSON 字符串列表
    """
    acfg_records = []

    # 处理 cfgs 中每个函数的 cfg (raw_graph)，构造对应的 acfg_record
    for cfg in cfgs.raw_graph_list:
        acfg = cfg.g
        acfg_src = src
        acfg_fname = cfg.funcname
        acfg_n_num = len(acfg)  # 节点数目
        acfg_succs = []
        acfg_features = []

        # 处理 cfg 中的每个节点，构造特征向量
        for i in acfg.node:
            acfg_succs.append(acfg.node[i]['succs'])
            acfg_features.append(acfg.node[i]['featurev'])

        record = acfg_record(acfg_src, acfg_n_num, acfg_succs, acfg_features, acfg_fname)
        acfg_records.append(record.to_json())

    return acfg_records


def gen_acfg_record(dir, binary_name):
    """
    从 cfgs 文件夹（.ida）下的名为 binary_name 的函数的 cfg 表示中提取对应的 acfg 行记录，
    并追加到对应的 json 文件
    处理一个二进制文件
    使用相对地址
    """
    # .ida 文件的地址
    src = os.path.join(dir, binary_name) + '.ida'
    ida_path = os.path.join(CFG_PATH, src)

    # 读取 pickle 文件
    cfgs = load_pickle_file(ida_path)

    # 每个函数的 cfg 转化为 acfg_record 的 JSON 字符串表示
    acfg_records = cfgs2acfgs(src, cfgs)

    # 追加进 JSON 文件
    json_file_path = os.path.join(ACFG_PATH, dir) + '.json'
    write_json_file(json_file_path, acfg_records)


def run_terminal_command(dir, file_path):
    """
    用终端启用 IDA Pro 的命令行
    """
    command = IDA_PRO_PATH + ' -c -A -S"' + PREPROCESSING_PATH + \
              ' --path ' + os.path.join(PROJECT_PATH, CFG_PATH, dir) + \
              '" ' + os.path.join(PROJECT_PATH, file_path)
    print command

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print("Error executing command:", stderr.decode('utf-8'))
    except Exception as e:
        print("Exception occurred:", e)


def batch_gen_acfg_record():
    """
    处理 binary_codes 文件夹下的所有二进制文件
    该函数中都是相对地址
    """
    # 获取.o文件目录下的所有文件夹名称
    dirs = [d for d in os.listdir(BINARY_PATH) if os.path.isdir(os.path.join(BINARY_PATH, d))]

    cnt = 0  # 计数器

    # 遍历处理文件夹中所有的二进制文件，生成 cfg 存储到对应CFG路径
    for dir in dirs:
        dir_path = os.path.join(BINARY_PATH, dir)
        print "########## Processing binary codes in ", dir_path

        for binary_code in os.listdir(dir_path):
            # 只处理以 '.o' 结尾的文件（防止混入ida database
            binary_code_path = os.path.join(dir_path, binary_code)

            if binary_code_path.endswith('.o'):
                cnt += 1
                run_terminal_command(dir, binary_code_path)
                gen_acfg_record(dir, binary_code)

    print "Process {} binary codes in total.".format(cnt)


if __name__ == '__main__':
    batch_gen_acfg_record()
