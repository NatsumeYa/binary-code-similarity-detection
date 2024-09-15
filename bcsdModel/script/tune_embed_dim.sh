#! /bin/bash
python train.py --embed_dim 16 --epoch 1 --log_path './log/log_dim16_1.txt' --save_path './saved_model_dim16_1/graphnn-model' --roc_path './roc/roc_dim16_1.npz'
python train.py --embed_dim 32 --epoch 1 --log_path './log/log_dim32_1.txt' --save_path './saved_model_dim32_1/graphnn-model' --roc_path './roc/roc_dim32_1.npz'
python train.py --embed_dim 64 --epoch 1 --log_path './log/log_dim64_1.txt' --save_path './saved_model_dim64_1/graphnn-model' --roc_path './roc/roc_dim64_1.npz'
python train.py --embed_dim 128 --epoch 1 --log_path './log/log_dim128_1.txt' --save_path './saved_model_dim128_1/graphnn-model' --roc_path './roc/roc_dim128_1.npz'
python train.py --embed_dim 256 --epoch 1 --log_path './log/log_dim256_1.txt' --save_path './saved_model_dim256_1/graphnn-model' --roc_path './roc/roc_dim256_1.npz'
