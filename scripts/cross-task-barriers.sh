#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cross-task-barriers-%j.out
#SBATCH --error=cross-task-barriers-%j.err

source ./open_lth/slurm-setup.sh cifar10 eurosat cifar100 svhn

CKPT_ROOT=$HOME/scratch/open_lth_data/
# cifar10               lottery_45792df32ad68649ffd066ae40be4868
# eurosat               lottery_d1b69a2da0973637bfc9a76d73a1f19f
# cifar100class10       lottery_ee3a8edc96da470068a5b524300f3ab8
# svhn                  lottery_2123f3764046b82699d86590c19bc401
# pixelpermutedcifar10  lottery_37ac5d99c2d78c509e44808f2d2ed6f9
    # lottery_45792df32ad68649ffd066ae40be4868  \
CKPT=(  \
    lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
    lottery_ee3a8edc96da470068a5b524300f3ab8  \
    lottery_2123f3764046b82699d86590c19bc401  \
    lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
)
REP_A=(1 3)
REP_B=(2 4)

parallel --delay=15 --linebuffer --jobs=2  \
    python -m scripts.open_lth_barriers  \
        --repdir_a=$CKPT_ROOT/{1}/replicate_{2}  \
        --repdir_b=$CKPT_ROOT/{1}/replicate_{3}  \
        --train_ep_it="ep5_it0,ep160_it0"  \
        --levels="0-20"  \
        --save_file=$HOME/scratch/2022-nnperm/cross-task-barriers/cross-task/{1}_1_2_barriers.pt  \
  ::: ${CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \
