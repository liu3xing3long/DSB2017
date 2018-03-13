#!/usr/bin/env bash

gpus=1

srun -p Med -n1 -w BJ-IDC1-10-10-15-74 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=dsb2017test --kill-on-bad-exit=1 \
python main.py
