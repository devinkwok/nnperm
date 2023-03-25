#!/bin/bash
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=sparse-align-%j.out
#SBATCH --error=sparse-align-%j.err

source ./open_lth/slurm-setup.sh

CKPT_ROOT=$HOME/scratch/open_lth_data/
# cifar10
# vgg    lottery_45792df32ad68649ffd066ae40be4868  \
# resnet lottery_c1db9e608f0c23077ab39f272306cb35  \
CKPT=(  \
    lottery_45792df32ad68649ffd066ae40be4868  \
)
REP_A=(1 3)
REP_B=(2 4)
KERNEL=(cosine linear)
LEVEL=($(seq 0 1 20))

# align all dense-dense cross-task
parallel --delay=5 --linebuffer --jobs=4  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_{2}/level_{5}/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_{3}/level_{5}/main/model_ep160_it0.pth  \
        --kernel={4}  \
    ::: ${CKPT[@]}  \
    ::: ${REP_A[@]}  \
    :::+ ${REP_B[@]}  \
    ::: ${KERNEL[@]}  \
    ::: ${LEVEL[@]}  \
