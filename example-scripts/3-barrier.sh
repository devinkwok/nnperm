#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=3-barrier-%j.out
#SBATCH --error=3-barrier-%j.err

source ./open_lth/slurm-setup.sh cifar10 svhn

CKPT_ROOT=$HOME/scratch/open_lth_data/
BARRIER_ROOT=$HOME/scratch/2022-nnperm/example-barrier/
# cifar10 vgg lottery_45792df32ad68649ffd066ae40be4868
# svhn vgg lottery_2123f3764046b82699d86590c19bc401
CKPT=(  \
    lottery_45792df32ad68649ffd066ae40be4868  \
    lottery_2123f3764046b82699d86590c19bc401  \
)
REP_A=(1)
REP_B=(2)
KERNEL=(linear)

# can also use `--levels="2,4,6"` to choose specific levels
parallel --delay=15 --linebuffer --jobs=2  \
    python -m scripts.open_lth_barriers  \
        --repdir_a=$CKPT_ROOT/{1}/replicate_{2}/  \
        --repdir_b=$CKPT_ROOT/{1}/replicate_{3}/  \
        --train_ep_it="ep160_it0" \
        --levels="0-20"  \
        --save_file=$BARRIER_ROOT/{1}/replicate_{2}-{3}/"barrier-{4}-ep160.pt"  \
        --n_train=10000 \
        --n_test=10000 \
        --kernel={4} \
    ::: ${CKPT[@]}  \
    ::: ${REP_A[@]}  \
    :::+ ${REP_B[@]}  \
    ::: ${KERNEL[@]}  \


python -m scripts.collate_barriers  \
    --dir=$BARRIER_ROOT  \
    --save_file=$BARRIER_ROOT/barriers.csv
    --recursive  \
