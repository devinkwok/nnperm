#!/bin/bash
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=super-align-%j.out
#SBATCH --error=super-align-%j.err

source ./open_lth/slurm-setup.sh

CKPT_ROOT=$HOME/scratch/open_lth_data/
SAVE_ROOT=$HOME/scratch/2022-nnperm/
# cifar_vgg_16_64 lottery_8146fc7e9839615729ee764a8019bdc5
# cifar_vgg_16_128 lottery_c8a10b33ede4caf6c026bbb12c7b8ae9
# cifar_resnet_20_64 lottery_0683974d98b4b6cb906aa0b80db9e2f5
CKPTS=(lottery_0683974d98b4b6cb906aa0b80db9e2f5 lottery_8146fc7e9839615729ee764a8019bdc5)
    # "ep2_it0,ep4_it0,ep6_it0,ep8_it0,ep10_it0,ep30_it0"  \
    # "ep6_it0,ep8_it0,ep10_it0,ep30_it0,ep50_it0,ep70_it0"  \
    # "ep10_it0,ep30_it0,ep50_it0,ep70_it0,ep90_it0,ep110_it0"  \
    # "ep50_it0,ep70_it0,ep90_it0,ep110_it0,ep130_it0,ep150_it0"  \
    # "ep2_it0,ep4_it0,ep6_it0,ep8_it0"  \
    # "ep6_it0,ep8_it0,ep10_it0,ep30_it0"  \
    # "ep10_it0,ep30_it0,ep50_it0,ep70_it0"  \
    # "ep50_it0,ep70_it0,ep90_it0,ep110_it0"  \
    # "ep90_it0,ep110_it0,ep130_it0,ep150_it0"  \
COMBINE=(  \
    "ep4_it0,ep6_it0,ep8_it0,ep10_it0,ep30_it0,ep50_it0"  \
    "ep8_it0,ep10_it0,ep30_it0,ep50_it0,ep70_it0,ep90_it0"  \
    "ep30_it0,ep50_it0,ep70_it0,ep90_it0,ep110_it0,ep130_it0"  \
    "ep4_it0,ep6_it0,ep8_it0,ep10_it0"  \
    "ep8_it0,ep10_it0,ep30_it0,ep50_it0"  \
    "ep30_it0,ep50_it0,ep70_it0,ep90_it0"  \
    "ep70_it0,ep90_it0,ep110_it0,ep130_it0"  \
    "ep2_it0,ep4_it0,ep6_it0"  \
    "ep4_it0,ep6_it0,ep8_it0"  \
    "ep6_it0,ep8_it0,ep10_it0"  \
    "ep8_it0,ep10_it0,ep30_it0"  \
    "ep10_it0,ep30_it0,ep50_it0"  \
    "ep30_it0,ep50_it0,ep70_it0"  \
    "ep50_it0,ep70_it0,ep90_it0"  \
    "ep70_it0,ep90_it0,ep110_it0"  \
    "ep90_it0,ep110_it0,ep130_it0"  \
)
BIAS=("reg" "bias")
ALIGN_ROOT=$SAVE_ROOT/super-align-more
REP_A=(1 3 5 7 9)
REP_B=(2 4 6 8 10)

parallel --delay=15 --linebuffer --jobs=8  \
    python -m scripts.super_align  \
        --repdir_a=$CKPT_ROOT/{1}/replicate_{4}/  \
        --repdir_b=$CKPT_ROOT/{1}/replicate_{5}/  \
        --combine_ep_it={2} \
        --save_file=$ALIGN_ROOT-{3}/{1}"/replicate_"{4}"-"{5}"/perm-"{2}.pt  \
        --align_bias={3}  \
    ::: ${CKPTS[@]}  \
    ::: ${COMBINE[@]}  \
    ::: ${BIAS[@]}  \
    ::: ${REP_A[@]}  \
    :::+ ${REP_B[@]}  \
