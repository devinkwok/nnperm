#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=align-all-%j.out
#SBATCH --error=align-all-%j.err

source ./open_lth/slurm-setup.sh

# CKPT_ROOT=$HOME/
CKPT_ROOT=$HOME/scratch/open_lth_data/
# MLP
# "open_lth_data/train_574e51abc295d8da78175b320504f2ba"  \
# VGG from 4x to 1/8, layernorm sgd with warmup
# "scratch/open_lth_data/lottery_2915b34d8b29a209ffee2288466cf5f6"  \
# "scratch/open_lth_data/lottery_3d9c91d3d4133cfcdcb2006da1507cbb"  \
# "scratch/open_lth_data/lottery_8d561a7b273e4d6b2705ba6d627a69bd"  \
# "scratch/open_lth_data/lottery_a309ac4ab15380928661e70ca8b054a1"  \
# "scratch/open_lth_data/lottery_c855d7c25ffef997a89799dc08931e82"  \
# VGG_16_128 "lottery_c8a10b33ede4caf6c026bbb12c7b8ae9"  \
# ResNet_20_64 "lottery_0683974d98b4b6cb906aa0b80db9e2f5"  \
CKPTS=(  \
    "lottery_8146fc7e9839615729ee764a8019bdc5"  \
)
EPOCHS=(  \
    "level_pretrain/main/model_ep1_it0.pth"  \
    "level_pretrain/main/model_ep3_it0.pth"  \
    "level_pretrain/main/model_ep5_it0.pth"  \
    "level_0/main/model_ep10_it0.pth"  \
    "level_0/main/model_ep20_it0.pth"  \
    "level_0/main/model_ep50_it0.pth"  \
    "level_0/main/model_ep80_it0.pth"  \
    "level_0/main/model_ep110_it0.pth"  \
    "level_0/main/model_ep150_it0.pth"  \
)
KERNELS=(linear)

parallel --delay=15 --linebuffer --jobs=3  \
    python -m scripts.align_all  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_1/{2}  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_2/{2}  \
        --save_file=refactor-outputs/sanity-check/{3}-{1}-replicate_1-2-{2}.pt  \
        --kernel={3}  \
    ::: ${CKPTS[@]}  \
    ::: ${EPOCHS[@]}  \
    ::: ${KERNELS[@]}  \

parallel --delay=15 --linebuffer --jobs=3  \
    python -m scripts.align_all  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_3/{2}  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_4/{2}  \
        --save_file=refactor-outputs/sanity-check/{3}-{1}-replicate_3-4-{2}.pt  \
        --kernel={3}  \
    ::: ${CKPTS[@]}  \
    ::: ${EPOCHS[@]}  \
    ::: ${KERNELS[@]}  \
