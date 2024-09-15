#! /bin/bash
python train.py --lr 0.01 --epoch 20 --log_path './log/log_lre-2.txt' --save_path './saved_model-lre-2/graphnn-model'
python train.py --lr 0.001 --epoch 20 --log_path './log/log_lre-3.txt' --save_path './saved_model-lre-3/graphnn-model'
python train.py --lr 0.0001 --epoch 20 --log_path './log/log_lre-4.txt' --save_path './saved_model-lre-4/graphnn-model'
python train.py --lr 0.00001 --epoch 20 --log_path './log/log_lre-5.txt' --save_path './saved_model-lre-5/graphnn-model'
python train.py --lr 0.000001 --epoch 20 --log_path './log/log_lre-6.txt' --save_path './saved_model-lre-6/graphnn-model'
