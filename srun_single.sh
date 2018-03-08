#!/usr/bin/env bash

gpus=1

srun -p Med -n1 -w BJ-IDC1-10-10-15-73 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=multi_task_rating --kill-on-bad-exit=1 \
python main.py
