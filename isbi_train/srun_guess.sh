#!/usr/bin/env bash

gpus=1

cd ./guess

srun -p Med -n1 -w BJ-IDC1-10-10-15-74 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=guessisbi --kill-on-bad-exit=1 \
python guess_groundtruth_analyzed.py
