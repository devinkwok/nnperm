#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log_align-all-%j.out
#SBATCH --error=log_align-all-%j.err

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

# CKPT_ROOT=$HOME/scratch/open_lth_data/
CKPT_ROOT=$HOME/open_lth_data/
#VGG from 4x to 1/8, layernorm sgd with warmup
    # lottery_2915b34d8b29a209ffee2288466cf5f6  \
    # lottery_3d9c91d3d4133cfcdcb2006da1507cbb  \
    # lottery_8d561a7b273e4d6b2705ba6d627a69bd  \
    # lottery_a309ac4ab15380928661e70ca8b054a1  \
    # lottery_c855d7c25ffef997a89799dc08931e82  \
CKPTS=(  \
   train_574e51abc295d8da78175b320504f2ba  \
)
EPOCHS=(1)
# EPOCHS=($(seq 10 50 160))
LOSS=(mse dot)

        # --ckpt_a=$CKPT_ROOT/{1}/replicate_1/level_pretrain/main/model_ep{2}_it0.pth  \
        # --ckpt_b=$CKPT_ROOT/{1}/replicate_2/level_pretrain/main/model_ep{2}_it0.pth  \
parallel --delay=15 --linebuffer --jobs=3  \
    python -m scripts.align_all  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_1/main/checkpoint.pth  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_2/main/checkpoint.pth  \
        --save_file=refactor-outputs/layernorm-base/{1}_1_2_pretrain_{2}_{3}.pt  \
        --loss={3}  \
        --n_train=1000  \
        --n_test=1000  \
    ::: ${CKPTS[@]}  \
    ::: ${EPOCHS[@]}  \
    ::: ${LOSS[@]}  \
