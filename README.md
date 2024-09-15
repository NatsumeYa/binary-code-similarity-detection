# binary-code-similarity-detection
二进制代码的相似度检测  
  
## 项目说明
1、有两个组件：特征提取器（feature_extractor）、相似度检测模型（bcsdModel），  
分别参考Genius和Gemini两个模型，详细见reference_papers。  
2、feature_extractor的原始输入数据是编译好的.o二进制文件，借助反汇编工具IDA Pro得到其汇编代码，  
统计汇编代码的属性值，附加到控制流图（CFG）上，得到带属性的控制流图（ACFG）。  
3、bcsdModel的原始输入数据是不同{编译器，优化级别，目标体系结构}下编译得到的acfg，  
使用图神经网络计算二进制函数的嵌入，再用孪生神经网络架构连接、计算两个二进制函数的相似度。
  
## 目录结构说明
```
├─bcsdModel          # 相似度检测模型                
│  ├─data              # 数据文件夹  
│  │  ├─acfgSSL_7        # 所有的acfg  
│  │  ├─val_comp         # 跨编译器实验所用的 acfg  
│  │  ├─val_isa          # 跨体系结构实验所用的 acfg  
│  │  └─val_op           # 跨优化级别实验所用的 acfg  
│  ├─hypergraphs       # 超参数调整使用的图  
│  ├─log               # 训练日志  
│  ├─roc               # roc数据  
│  ├─saved_model       # 保存的模型  
│  └─script            # 自动训练脚本  
├─feature_extractor  # 特征提取器  
│  ├─data              # 数据文件夹  
│  │  ├─acfgs            # 在cfg基础上统计属性值得到的 acfg  
│  │  ├─binary_codes     # 二进制.o文件  
│  │  └─cfgs             # 使用IDA Pro提取出的 cfg  
│  └─python            # ida python  
└─reference_papers  # 参考文献  
```
