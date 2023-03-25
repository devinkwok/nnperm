#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=train-cross-task-%j.out
#SBATCH --error=train-cross-task-%j.err

# hparams
MODEL=(cifar_vgg_16 cifar_resnet_20_64)
DATASET=(cifar10 cifar100class10 pixelpermutedcifar10 svhn eurosat)
REPLICATE=(1 2 3 4)

source ./open_lth/slurm-setup.sh cifar10 cifar100 svhn eurosat
cd open_lth

parallel --delay=15 --linebuffer --jobs=3  \
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
