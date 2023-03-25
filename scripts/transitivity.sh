#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=transitivity-%j.out
#SBATCH --error=transitivity-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
SAVE_ROOT=$HOME/scratch/2022-nnperm/transitivity/
CKPTS=(  \
    lottery_8146fc7e9839615729ee764a8019bdc5  \
    lottery_0683974d98b4b6cb906aa0b80db9e2f5  \
)
EPOCHS=(150 20)

parallel --delay=15 --linebuffer --jobs=1  \
    python -m scripts.transitivity  \
        --ckpt_dir=$CKPT_ROOT/{1}/  \
        --ckpt_pattern="replicate_*/level_0/main/model_ep"{2}"_it0.pth"  \
        --save_file=$SAVE_ROOT/{1}-ep{2}.pt  \
    ::: ${CKPTS[@]}  \
    ::: ${EPOCHS[@]}  \
