#!/bin/bash
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=2-align-%j.out
#SBATCH --error=2-align-%j.err

source ./open_lth/slurm-setup.sh

CKPT_ROOT=$HOME/scratch/open_lth_data/
# cifar10 vgg lottery_45792df32ad68649ffd066ae40be4868
# svhn vgg lottery_2123f3764046b82699d86590c19bc401
CKPT=(  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_2123f3764046b82699d86590c19bc401  \
)
REP_A=(1)
REP_B=(2)
KERNEL=(linear)
LEVEL=($(seq 0 1 20))

# align all dense-dense cross-task
parallel --delay=5 --linebuffer --jobs=1  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_{2}/level_{5}/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_{3}/level_{5}/main/model_ep160_it0.pth  \
        --kernel={4}  \
    ::: ${CKPT[@]}  \
    ::: ${REP_A[@]}  \
    :::+ ${REP_B[@]}  \
    ::: ${KERNEL[@]}  \
    ::: ${LEVEL[@]}  \
