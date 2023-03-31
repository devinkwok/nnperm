#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=1-train-%j.out
#SBATCH --error=1-train-%j.err

# hparams
MODEL=(cifar_vgg_16)
DATASET=(cifar10 svhn)
REPLICATE=($(seq 1 1 2))

source ./open_lth/slurm-setup.sh cifar10 svhn
cd open_lth

parallel --delay=15 --linebuffer --jobs=2  \
    python open_lth.py lottery  \
        --default_hparams={1}  \
        --dataset_name={2}  \
        --replicate={3}  \
        --warmup_steps="1ep"  \
        --rewinding_steps=5ep  \
        --batchnorm_replace="layernorm"  \
        --levels=20  \
  ::: ${MODEL[@]}  \
  ::: ${DATASET[@]}  \
  ::: ${REPLICATE[@]}  \
