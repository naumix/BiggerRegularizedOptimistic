#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate mw2

module load cuDNN/8.9.2.26-CUDA-12.2.0

python3 train_parallel_mt.py --seed=0 --task_type=50

wait
