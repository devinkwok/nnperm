#!/bin/bash
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=cross-task-align-%j.out
#SBATCH --error=cross-task-align-%j.err

source ./open_lth/slurm-setup.sh

CKPT_ROOT=$HOME/scratch/open_lth_data/
# cifar10            lottery_45792df32ad68649ffd066ae40be4868
# eurosat               lottery_d1b69a2da0973637bfc9a76d73a1f19f
# cifar100class10       lottery_ee3a8edc96da470068a5b524300f3ab8
# svhn                  lottery_2123f3764046b82699d86590c19bc401
# pixelpermutedcifar10  lottery_37ac5d99c2d78c509e44808f2d2ed6f9
CKPT=(  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
    lottery_ee3a8edc96da470068a5b524300f3ab8  \
    lottery_2123f3764046b82699d86590c19bc401  \
    lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
)
# choose any 2 of above
CKPT_A=(  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
    lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
    lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
    lottery_ee3a8edc96da470068a5b524300f3ab8  \
    lottery_ee3a8edc96da470068a5b524300f3ab8  \
    lottery_2123f3764046b82699d86590c19bc401  \
)
CKPT_B=(  \
    lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
    lottery_ee3a8edc96da470068a5b524300f3ab8  \
    lottery_2123f3764046b82699d86590c19bc401  \
    lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
    lottery_ee3a8edc96da470068a5b524300f3ab8  \
    lottery_2123f3764046b82699d86590c19bc401  \
    lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
    lottery_2123f3764046b82699d86590c19bc401  \
    lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
    lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
)
REPLICATE=(1 2 3 4)
REP_A=(1 3)
REP_B=(2 4)
LEVEL=($(seq 0 1 20))

# align all cross-task
parallel --delay=5 --linebuffer --jobs=8  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_{4}/level_{3}/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{2}/replicate_{4}/level_{3}/main/model_ep160_it0.pth  \
        --kernel=linear  \
  ::: ${CKPT_A[@]}  \
  :::+ ${CKPT_B[@]}  \
  ::: ${LEVEL[@]}  \
  ::: ${REPLICATE[@]}  \

# baseline: align all sparse-sparse within task
parallel --delay=5 --linebuffer --jobs=8  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_{2}/level_{4}/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_{3}/level_{4}/main/model_ep160_it0.pth  \
        --kernel=linear  \
  ::: ${CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \
  ::: ${LEVEL[@]}  \
