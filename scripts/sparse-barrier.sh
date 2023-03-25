#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=sparse-barrier-%j.out
#SBATCH --error=sparse-barrier-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
BARRIER_ROOT=$HOME/scratch/2022-nnperm/sparse-to-sparse/
# cifar10
# vgg    lottery_45792df32ad68649ffd066ae40be4868  \
# resnet lottery_c1db9e608f0c23077ab39f272306cb35  \
CKPT=(  \
    lottery_45792df32ad68649ffd066ae40be4868  \
)
REP_A=(1 3)
REP_B=(2 4)
KERNEL=(cosine)
# KERNEL=(cosine linear)


        # --levels="2,6,10,14,18"  \

parallel --delay=15 --linebuffer --jobs=1  \
    python -m scripts.open_lth_barriers  \
        --repdir_a=$CKPT_ROOT/{1}/replicate_{2}/  \
        --repdir_b=$CKPT_ROOT/{1}/replicate_{3}/  \
        --train_ep_it="ep160_it0" \
        --levels="0"  \
        --save_file=$BARRIER_ROOT/{1}/replicate_{2}-{3}/"barrier-{4}-ep160.pt"  \
        --n_train=10000 \
        --kernel={4} \
    ::: ${CKPT[@]}  \
    ::: ${REP_A[@]}  \
    :::+ ${REP_B[@]}  \
    ::: ${KERNEL[@]}  \
