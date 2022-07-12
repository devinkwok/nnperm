#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=exp-%j.out
#SBATCH --error=exp-%j.err

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

CKPTS=(train_574e51abc295d8da78175b320504f2ba train_9d0811cc67a44e1ec85e702a5e01570f)

parallel --delay=15 --linebuffer --jobs=3  \
    python experiment.py  \
        --n_replicates=5  \
        --barrier_resolution=10  \
        --test_points=10000  \
        --ckpt={1}  \
    ::: ${CKPTS[@]}  \
