#! /bin/bash
python train.py --embed_depth 1 --epoch 1 --log_path './log/log_depth01.txt' --roc_path './roc/roc_depth1.npz'
python train.py --embed_depth 2 --epoch 1 --log_path './log/log_depth02.txt' --roc_path './roc/roc_depth2.npz'
python train.py --embed_depth 3 --epoch 1 --log_path './log/log_depth03.txt' --roc_path './roc/roc_depth3.npz'
python train.py --embed_depth 4 --epoch 1 --log_path './log/log_depth04.txt' --roc_path './roc/roc_depth4.npz'
