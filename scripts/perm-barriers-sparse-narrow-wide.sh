#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=perm-barriers-sparse-narrow-wide-%j.out
#SBATCH --error=perm-barriers-sparse-narrow-wide-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
# VGG from 4x to 1/8, layernorm sgd with warmup
STATS_FILES=(  \
    "lottery_2915b34d8b29a209ffee2288466cf5f6"  \
    "lottery_3d9c91d3d4133cfcdcb2006da1507cbb"  \
    "lottery_c855d7c25ffef997a89799dc08931e82"  \
)
WIDE="lottery_8d561a7b273e4d6b2705ba6d627a69bd"

# --ckpt_a=$CKPT_ROOT/{1}/replicate_1/level_pretrain/main/model_ep{2}_it0.pth  \
# --ckpt_b=$CKPT_ROOT/{1}/replicate_2/level_pretrain/main/model_ep{2}_it0.pth  \
parallel --delay=15 --linebuffer --jobs=3  \
    python -m scripts.perm_barriers  \
        --stats_file=refactor-outputs/fix-embed-$WIDE/{1}_1_2_pretrain_ep160_mse.pt  \
        --parent_ckpt=$CKPT_ROOT/$WIDE/replicate_1/level_pretrain/main/model_ep160_it0.pth  \
        --perm_key_a="perm_a"  \
        --perm_key_b="perm_b"  \
        --ckpt_a_dir=$CKPT_ROOT/{1}/replicate_1/  \
        --ckpt_b_dir=$CKPT_ROOT/{1}/replicate_2/  \
        --levels="5,10,15,20"  \
        --ckpt_filename="/main/model_ep160_it0.pth"  \
        --save_file=refactor-outputs/embed-narrow-wide/$WIDE/sparse/{1}_1_2_ep160_mse.pt  \
        --n_train=1000  \
        --n_test=1000  \
        --barrier_resolution=5  \
    ::: ${STATS_FILES[@]}  \
