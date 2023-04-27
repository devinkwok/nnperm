#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=compare-algorithm-%j.out
#SBATCH --error=compare-algorithm-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
BARRIER_ROOT=$HOME/scratch/2022-nnperm/compare-algorithm/
# cifar10
# vgg-16 pruned         lottery_45792df32ad68649ffd066ae40be4868
# vgg-16 train ckpts    lottery_8146fc7e9839615729ee764a8019bdc5
# resnet-20-64 pruned   lottery_c1db9e608f0c23077ab39f272306cb35
# resnet-20-64 train    lottery_0683974d98b4b6cb906aa0b80db9e2f5

REP_A=(1 3)
REP_B=(2 4)
TYPE=(
    "weight_linear"  \
    "weight_loglinear"  \
    "weight_cosine"  \
    "activation_linear_cifar10_10000"  \
    "activation_loglinear_cifar10_10000"  \
    "activation_cosine_cifar10_10000"  \
    "activation_cosine_cifar10_10000_all"  \
    "activation_cosine_cifar10_10000_first"  \
    "activation_cosine_cifar10_50000"  \
)

# align by sparsity
SPARSE_CKPT=(  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_c1db9e608f0c23077ab39f272306cb35  \
)
parallel --delay=5 --jobs=2  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{2}/replicate_{3}/level_0/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{2}/replicate_{4}/level_0/main/model_ep160_it0.pth  \
        --type={1}  \
  ::: ${TYPE[@]}  \
  ::: ${SPARSE_CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \

# align by epoch
TRAIN_CKPT=(  \
    lottery_8146fc7e9839615729ee764a8019bdc5  \
    lottery_0683974d98b4b6cb906aa0b80db9e2f5  \
)
parallel --delay=5 --jobs=2  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{2}/replicate_{3}/level_0/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{2}/replicate_{4}/level_0/main/model_ep160_it0.pth  \
        --type={1}  \
  ::: ${TYPE[@]}  \
  ::: ${TRAIN_CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \

# barriers by sparsity
LEVEL="0,9,12,15"
parallel --delay=5 --jobs=2  \
    python -m scripts.open_lth_barriers  \
        --repdir_a=$CKPT_ROOT/{2}/replicate_{3}  \
        --repdir_b=$CKPT_ROOT/{2}/replicate_{4}  \
        --perm_b=$CKPT_ROOT/{2}/replicate_{4}"/level_0/main/perm_"{1}"-"{2}"-replicate_"{3}"-level_0-main-ep160_it0.pt"  \
        --train_ep_it="ep160_it0"  \
        --levels=$LEVEL  \
        --save_file=$BARRIER_ROOT/"{1}"/"{2}"/"{3}"-"{4}"/"barrier-level_0-main-ep160_it0.pt"  \
        --barrier_resolution=11  \
        --n_train=10000  \
        --n_test=10000  \
  ::: ${TYPE[@]}  \
  ::: ${SPARSE_CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \

# barriers by epoch
EPOCH="ep10_it0,ep20_it0,ep50_it0,ep160_it0"
parallel --delay=5 --jobs=2  \
    python -m scripts.open_lth_barriers  \
        --repdir_a=$CKPT_ROOT/{2}/replicate_{3}  \
        --repdir_b=$CKPT_ROOT/{2}/replicate_{4}  \
        --perm_b=$CKPT_ROOT/{2}/replicate_{4}"/level_0/main/perm_"{1}"-"{2}"-replicate_"{3}"-level_0-main-ep160_it0.pt"  \
        --train_ep_it=$EPOCH  \
        --levels="0"  \
        --save_file=$BARRIER_ROOT/"{1}"/"{2}"/"{3}"-"{4}"/"barrier-level_0-main-ep160_it0.pt"  \
        --barrier_resolution=11  \
        --n_train=10000  \
        --n_test=10000  \
  ::: ${TYPE[@]}  \
  ::: ${TRAIN_CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \

python -m scripts.collate_barriers  \
    --dir=$BARRIER_ROOT  \
    --recursive  \
