#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log_exp_1-%j.out
#SBATCH --error=log_exp_1-%j.err

module load python/3.7
module load pytorch/1.4

if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

# ResNet train_71bc92a970b64a76d7ab7681764b0021
# MLP
# S-Conv
#    train_9d0811cc67a44e1ec85e702a5e01570f  \
# VGG
CKPTS=( \
   train_574e51abc295d8da78175b320504f2ba  \
   train_7312e802e619673d23c7a02eba8aee52)

LOSS=(L2)
# LOSS=(L1 L2)

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_4.py  \
        --save_dir=outputs/rebasin/exp_4  \
        --precompute_dir=outputs/rebasin/exp_1/  \
        --ckpt_root=../../open_lth_data  \
        --n_replicates=5  \
        --ckpt={1}  \
        --loss={2}  \
        --level=train_last  \
    ::: ${CKPTS[@]}  \
    ::: ${LOSS[@]}  \
