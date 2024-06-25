#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate gym

module load CUDA/12.0.0

python3 train_parallel.py --benchmark="gym" --env_name="Hopper-v4"

wait
