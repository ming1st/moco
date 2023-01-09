#!/bin/bash

#SBATCH -J MOCO_res18
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o ./results/stdout/%x_stdout_%j.txt
#SBATCH -e ./results/stderr/%x_stderr_%j.txt
#SBATCH --gres=gpu:8

conda activate ming-robust
            
python main_moco.py \
    -a resnet18 \
    # -a resnet50 \
    --lr 0.03 \
    --batch-size 256 \
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
    /mnt/server2_hard0/minji/data/ImageNet/IMAGENET0/ImageNet \
    --mlp --moco-t 0.2 --aug-plus --cos

conda deactivate