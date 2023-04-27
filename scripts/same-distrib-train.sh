#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=same-distrib-train-%j.out
#SBATCH --error=same-distrib-train-%j.err

# hparams
MODEL=(cifar_vgg_16_64  \
    cifar_resnet_20_64  \
)
# cifar10
DATASET=(cifar10 cifar100)
REPLICATE=($(seq 1 1 4))
START=(0 25000)
END=(25000 50000)

source ./open_lth/slurm-setup.sh cifar10 cifar100
cd open_lth

parallel --delay=15 --linebuffer --jobs=3  \
    python open_lth.py lottery  \
        --warmup_steps=1ep  \
        --default_hparams={2}  \
        --dataset_name={3}  \
        --replicate={1}  \
        --levels=0  \
        --training_steps=160ep  \
        --batchnorm_replace="layernorm"  \
        --subset_start={4}  \
        --subset_end={5}  \
  ::: ${REPLICATE[@]}  \
  ::: ${MODEL[@]}  \
  ::: ${DATASET[@]}  \
  ::: ${START[@]}  \
  :::+ ${END[@]}  \
