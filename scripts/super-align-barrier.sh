#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=super-align-barrier-%j.out
#SBATCH --error=super-align-barrier-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
SAVE_ROOT=$HOME/scratch/2022-nnperm/
# cifar_vgg_16_64 lottery_8146fc7e9839615729ee764a8019bdc5
# cifar_vgg_16_128 lottery_c8a10b33ede4caf6c026bbb12c7b8ae9
# cifar_resnet_20_64 lottery_0683974d98b4b6cb906aa0b80db9e2f5
CKPT=lottery_8146fc7e9839615729ee764a8019bdc5
BARRIER="ep1_it0,ep3_it0,ep5_it0,ep10_it0,ep20_it0,ep50_it0,ep80_it0,ep110_it0,ep150_it0"
REP_A="1"
REP_B="2"
RESOLUTION=25

ALIGN_ROOT=$SAVE_ROOT/super-align/$CKPT/
REPLICATE="replicate_"$REP_A"-"$REP_B
cd $ALIGN_ROOT/$REPLICATE/
PERM=($(ls *.pt ))
cd -

parallel --delay=15 --linebuffer --jobs=3  \
    python -m scripts.layer_stability_barrier  \
        --repdir_a=$CKPT_ROOT/$CKPT/replicate_$REP_A/  \
        --repdir_b=$CKPT_ROOT/$CKPT/replicate_$REP_B/  \
        --barrier_ep_it=$BARRIER \
        --perm_b2a=$ALIGN_ROOT/$REPLICATE/{1}  \
        --save_file=$SAVE_ROOT/super-align-barrier/$CKPT/$REPLICATE/"barrier-"{1}  \
        --barrier_resolution=$RESOLUTION  \
    ::: ${PERM[@]}  \
