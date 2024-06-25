#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate mw

module load CUDA/12.0.0

python3 train_parallel.py --benchmark="mw" --env_name=basketball-v2-goal-observable

wait
