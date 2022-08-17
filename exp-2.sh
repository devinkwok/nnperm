#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log_exp_2-%j.out
#SBATCH --error=log_exp_2-%j.err

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

# ResNet
# MLP
# S-Conv
# VGG
CKPTS=(train_71bc92a970b64a76d7ab7681764b0021  \
   train_574e51abc295d8da78175b320504f2ba  \
   train_9d0811cc67a44e1ec85e702a5e01570f  \
   train_7312e802e619673d23c7a02eba8aee52)

LOSS=(L1 L2)
LAYERS=(-1 1)
NOISE=(0 0.1 0.5)

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_2.py  \
        --n_replicates=5  \
        --ckpt={1}  \
        --loss={2}  \
        --n_layers={3}  \
        --weight_noise={4}  \
    ::: ${CKPTS[@]}  \
    ::: ${LOSS[@]}  \
    ::: ${LAYERS[@]}  \
    ::: ${NOISE[@]}  \

# baselines
parallel --delay=15 --linebuffer --jobs=3  \
    python exp_2.py  \
        --n_replicates=5  \
        --ckpt={1}  \
        --loss=L2  \
        --n_layers=1  \
        --weight_noise={2}  \
        --no_scale  \
    ::: ${CKPTS[@]}  \
    ::: ${NOISE[@]}  \

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_2.py  \
        --n_replicates=5  \
        --ckpt={1}  \
        --loss=L2  \
        --n_layers=1  \
        --weight_noise={2}  \
        --no_permute  \
    ::: ${CKPTS[@]}  \
    ::: ${NOISE[@]}  \

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_2.py  \
        --n_replicates=5  \
        --ckpt={1}  \
        --loss=L2  \
        --n_layers=1  \
        --weight_noise={2}  \
        --no_scale  \
        --no_permute  \
    ::: ${CKPTS[@]}  \
    ::: ${NOISE[@]}  \
