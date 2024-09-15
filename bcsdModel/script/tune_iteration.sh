#! /bin/bash
python train.py --iter_level 1 --epoch 1 --log_path './log/log_iter1_1e.txt' --save_path './saved_model_iter1_1e/graphnn-model' --roc_path './roc/roc_iter1_1e.npz'
python train.py --iter_level 3 --epoch 1 --log_path './log/log_iter3_1e.txt' --save_path './saved_model_iter3_1e/graphnn-model' --roc_path './roc/roc_iter3_1e.npz'
python train.py --iter_level 5 --epoch 1 --log_path './log/log_iter5_1e.txt' --save_path './saved_model_iter5_1e/graphnn-model' --roc_path './roc/roc_iter5_1e.npz'
python train.py --iter_level 7 --epoch 1 --log_path './log/log_iter7_1e.txt' --save_path './saved_model_iter7_1e/graphnn-model' --roc_path './roc/roc_iter7_1e.npz'
python train.py --iter_level 9 --epoch 1 --log_path './log/log_iter9_1e.txt' --save_path './saved_model_iter9_1e/graphnn-model' --roc_path './roc/roc_iter9_1e.npz'
