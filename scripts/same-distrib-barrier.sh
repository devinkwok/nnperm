#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=same-distrib-barriers-%j.out
#SBATCH --error=same-distrib-barriers-%j.err

source ./open_lth/slurm-setup.sh cifar10 eurosat cifar100 svhn

CKPT_ROOT=$HOME/scratch/open_lth_data/
BARRIER_ROOT=$HOME/scratch/2022-nnperm/same-distrib
# cifar10 vgg_16_64 25000 lottery_1193ab29883d4d3ece83c27af218a493
# cifar10 vgg_16_64 50000 lottery_d91dbbbc96d08ec0886540c215406d64
# cifar10 resnet_20_64 25000 lottery_2f6dd759ece2b346d33b577fd22e2b7a
# cifar10 resnet_20_64 50000 lottery_e73c18ce09cdf0157bcb113765806900
# cifar100 vgg_16_64 25000 lottery_2d61650f65ebf010b53eca9dff5805f4
# cifar100 vgg_16_64 50000 lottery_ae89e400548813924f17ab77ae0b5d5b
# cifar100 resnet_20_64 25000 lottery_a0310708bf9754e65de41c443bd4a173
# cifar100 resnet_20_64 50000 lottery_50250bdc670182c939207cb320663425
CKPT_A=(  \
    lottery_1193ab29883d4d3ece83c27af218a493  \
    lottery_2f6dd759ece2b346d33b577fd22e2b7a  \
    lottery_2d61650f65ebf010b53eca9dff5805f4  \
    lottery_a0310708bf9754e65de41c443bd4a173  \
)
CKPT_B=(  \
    lottery_d91dbbbc96d08ec0886540c215406d64  \
    lottery_e73c18ce09cdf0157bcb113765806900  \
    lottery_ae89e400548813924f17ab77ae0b5d5b  \
    lottery_50250bdc670182c939207cb320663425  \
)
REPLICATE=(1 2 3 4)
TYPE=("weight_linear")

parallel --delay=5 --jobs=2  \
    python -m scripts.open_lth_barriers  \
        --repdir_a=$CKPT_ROOT/{2}/replicate_{4}  \
        --repdir_b=$CKPT_ROOT/{3}/replicate_{4}  \
        --perm_b=$CKPT_ROOT/{3}/replicate_{4}"/level_0/main/perm_"{1}"-"{2}"-replicate_"{4}"-level_0-main-ep160_it0.pt"  \
        --train_ep_it="ep160_it0"  \
        --levels=0  \
        --save_file=$BARRIER_ROOT/{1}"/"{2}"/"{3}"-"{4}"/barrier-level_0-main-ep160_it0.pt"  \
        --barrier_resolution=25  \
        --n_train=50000  \
        --n_test=10000  \
  ::: ${TYPE[@]}  \
  ::: ${CKPT_A[@]}  \
  :::+ ${CKPT_B[@]}  \
  ::: ${REPLICATE[@]}  \

python -m scripts.collate_barriers  \
    --dir=$BARRIER_ROOT  \
    --recursive  \
