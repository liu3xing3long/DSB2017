#!/usr/bin/env bash

gpus=8

srun -p Med -n1 -w BJ-IDC1-10-10-15-73 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=yizhongtest --kill-on-bad-exit=1 \
python main.py
