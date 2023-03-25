#!/bin/bash
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=layer-stability-align-%j.out
#SBATCH --error=layer-stability-align-%j.err

source ./open_lth/slurm-setup.sh

CKPT_ROOT=$HOME/scratch/open_lth_data/

# cifar_vgg_16_64 lottery_8146fc7e9839615729ee764a8019bdc5
# cifar_vgg_16_128 lottery_c8a10b33ede4caf6c026bbb12c7b8ae9
# cifar_resnet_20_64 lottery_0683974d98b4b6cb906aa0b80db9e2f5

# svhn cifar_vgg_16_64 lottery_e64643612c6ec28ecb086a495725d2e7
# svhn cifar_resnet_20_64 lottery_6222be47ff9a8c65a59c78b954f74d80
# cifar-100 cifar_vgg_16_64 lottery_2d45283937ca3ee3af7f4514beab0645
# cifar-100 cifar_resnet_20_64 lottery_df64d8f1bef80877b76ada73c917af8e
    # lottery_e64643612c6ec28ecb086a495725d2e7  \
    # lottery_6222be47ff9a8c65a59c78b954f74d80  \
CKPTS=(  \
    lottery_2d45283937ca3ee3af7f4514beab0645  \
    lottery_df64d8f1bef80877b76ada73c917af8e  \
)
ALIGN="ep1_it0,ep3_it0,ep5_it0,ep10_it0,ep20_it0,ep50_it0,ep80_it0,ep110_it0"
FINAL="ep150_it0"
SUBSET_TYPES="bottom-up,top-down,leave-out,put-in"
# VGG-16 "1,3,4,7,10"
# ResNet-20 "0,1,2,3,4"
LAYERS=("1,3,4,7,10" "0,1,2,3,4")
SAVE_ROOT=$HOME/scratch/2022-nnperm/layer-align-more-bias
# BIAS=("reg" "bias")
REPLICATE_A=(1 3 5 7 9)
REPLICATE_B=(2 4 6 8 10)

parallel --delay=15 --linebuffer --jobs=4  \
    python -m scripts.layer_stability_align  \
        --repdir_a=$CKPT_ROOT/{1}/replicate_{3}/  \
        --repdir_b=$CKPT_ROOT/{1}/replicate_{4}/  \
        --align_ep_it=$ALIGN \
        --final_ep_it=$FINAL \
        --layer_subset_types=$SUBSET_TYPES  \
        --layer_thresholds={2}  \
        --align_bias="bias"  \
        --save_dir=$SAVE_ROOT/{1}/replicate_{3}-{4}/  \
    ::: ${CKPTS[@]}  \
    :::+ ${LAYERS[@]}  \
    ::: ${REPLICATE_A[@]}  \
    :::+ ${REPLICATE_B[@]}  \
