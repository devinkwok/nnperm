#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=stats-%j.out
#SBATCH --error=stats-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
BARRIER_ROOT=$HOME/scratch/2022-nnperm/stats/
# cifar10
# vgg-16 pruned         lottery_45792df32ad68649ffd066ae40be4868
# vgg-16 train ckpts    lottery_8146fc7e9839615729ee764a8019bdc5
# resnet-20-64 pruned   lottery_c1db9e608f0c23077ab39f272306cb35
# resnet-20-64 train    lottery_0683974d98b4b6cb906aa0b80db9e2f5

REP_A=(1 3)
REP_B=(2 4)
    # "weight_cosine"  \
    # "activation_cosine_cifar10_10000_first"  \
    # "activation_linear_cifar10_50000"  \
TYPE=(
    "weight_linear"  \
)

CKPT=(  \
    lottery_23d95a4841f5114daaeb195dcd3bce62  \
    lottery_688106d9fc0da7db4cebd981434750ee  \
    lottery_812938fad9d6a452c60de777ad8b9ba2  \
    lottery_e1ee2ce029f3688fdba8d16bcd72101b  \
    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d  \
    lottery_3ed390b5f8f0b92d658244d053e538e7  \
)

# baselines (no perm)
NOTYPE=(noperm)  # dummy variable to avoid renumbering other variables
parallel --delay=5 --jobs=1 -u  \
    python -m scripts.stats  \
        --ckpt_a=$CKPT_ROOT/{2}/replicate_{3}/level_0/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{2}/replicate_{4}/level_0/main/model_ep160_it0.pth  \
        --save_file=$BARRIER_ROOT/{1}"/"{2}"/"{3}"-"{4}"/stats-200test-jac-level_0-main-ep160_it0.pt"  \
        --batch_size=10  \
        --n_test=200  \
  ::: ${NOTYPE[@]}  \
  ::: ${CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \
