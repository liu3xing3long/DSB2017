#!/usr/bin/env bash
#python main.py --epochs 1000 --lr 0.0001 --save-freq 50 --test 0
#python main.py --resume ./results/model_best.ckpt  --test 1


# multiple
#python main_multiple.py --epochs 500 --lr 0.001 --save-freq 50 --test 0
python main_multiple.py --resume ./results/model_best.ckpt  --test 1

## multiple fake
#python main_multiple_fake.py --epochs 300 --lr 0.001 --save-freq 50 --test 0
#python main_multiple_fake.py --resume ./results/model_best.ckpt  --test 1