#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=layer-stability-barrier-%j.out
#SBATCH --error=layer-stability-barrier-%j.err

REPDIR_A=$1
REPDIR_B=$2
PERM_DIR=$3
BARRIER_DIR=$4

BARRIER="ep1_it0,ep3_it0,ep5_it0,ep10_it0,ep20_it0,ep50_it0,ep80_it0,ep110_it0,ep150_it0"
RESOLUTION=25

cd $PERM_DIR
PERM=($(ls perm-*.pt ))
cd -

parallel --delay=15 --linebuffer --jobs=3  \
    python -m scripts.layer_stability_barrier  \
        --repdir_a=$REPDIR_A  \
        --repdir_b=$REPDIR_B  \
        --barrier_ep_it=$BARRIER \
        --perm_b2a=$PERM_DIR/{1}  \
        --save_file=$BARRIER_DIR/"barrier-"{1}  \
        --barrier_resolution=$RESOLUTION  \
        --n_train=10000 \
    ::: ${PERM[@]}  \
