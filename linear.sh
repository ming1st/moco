#!/bin/bash

#SBATCH -J MOCO_linear
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o ./results/stdout/%x_stdout_%j.txt
#SBATCH -e ./results/stderr/%x_stderr_%j.txt
#SBATCH --gres=gpu:8

conda activate ming-robust
            
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained /mnt/server2_hard0/minji/moco-main/checkpoints/resnet50/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 16 \
  /mnt/server2_hard0/minji/data/ImageNet/IMAGENET0/ImageNet

conda deactivate