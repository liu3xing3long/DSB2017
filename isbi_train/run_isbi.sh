#!/usr/bin/env bash
#python main.py --epochs 1000 --lr 0.0001 --save-freq 50 --test 0
python main.py --resume ./results/model_best.ckpt  --test 1