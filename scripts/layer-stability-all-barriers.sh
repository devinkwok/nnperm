#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=layer-stability-all-barriers-%j.out
#SBATCH --error=layer-stability-all-barriers-%j.err

source ./open_lth/slurm-setup.sh cifar100 svhn  # cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
SAVE_ROOT=$HOME/scratch/2022-nnperm/layer-align-more-bias/
BARRIER_ROOT=$HOME/scratch/2022-nnperm/barrier-endtraintest-bias/
# cifar_vgg_16_64 lottery_8146fc7e9839615729ee764a8019bdc5
# cifar_vgg_16_128 lottery_c8a10b33ede4caf6c026bbb12c7b8ae9
# cifar_resnet_20_64 lottery_0683974d98b4b6cb906aa0b80db9e2f5

# svhn cifar_vgg_16_64 lottery_e64643612c6ec28ecb086a495725d2e7
# svhn cifar_resnet_20_64 lottery_6222be47ff9a8c65a59c78b954f74d80
# cifar-100 cifar_vgg_16_64 lottery_2d45283937ca3ee3af7f4514beab0645
# cifar-100 cifar_resnet_20_64 lottery_df64d8f1bef80877b76ada73c917af8e
CKPTS=(  \
    lottery_e64643612c6ec28ecb086a495725d2e7  \
    lottery_6222be47ff9a8c65a59c78b954f74d80  \
    lottery_2d45283937ca3ee3af7f4514beab0645  \
    lottery_df64d8f1bef80877b76ada73c917af8e  \
)
REPLICATE_A=(1 3 5 7 9)
REPLICATE_B=(2 4 6 8 10)

parallel --delay=15 --linebuffer --jobs=1  \
    source ./scripts/layer-stability-barrier.sh  \
        $CKPT_ROOT/{1}/replicate_{2}/  \
        $CKPT_ROOT/{1}/replicate_{3}/  \
        $SAVE_ROOT/{1}/replicate_{2}-{3}/  \
        $BARRIER_ROOT/{1}/replicate_{2}-{3}/  \
    ::: ${CKPTS[@]}  \
    ::: ${REPLICATE_A[@]}  \
    :::+ ${REPLICATE_B[@]}  \
