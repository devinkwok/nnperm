#!/bin/bash
#SBATCH --partition=main
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

# 5 training checkpoints
# 5 prune levels
LEVELS=(  \
    pretrain_0 pretrain_40 pretrain_80 pretrain_120 pretrain_160  \
    4_160 8_160 12_160 16_160 20_160  \
)

#VGG from 4x to 1/8, layernorm sgd with warmup
CKPTS=(  \
    lottery_2915b34d8b29a209ffee2288466cf5f6  \
    lottery_3d9c91d3d4133cfcdcb2006da1507cbb  \
    lottery_8d561a7b273e4d6b2705ba6d627a69bd  \
    lottery_a309ac4ab15380928661e70ca8b054a1  \
    lottery_c855d7c25ffef997a89799dc08931e82  \
)

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_1.py  \
        --save_dir=outputs/rebasin/exp_1e  \
        --n_replicates=2  \
        --loss=L2  \
        --bias_loss_weight=1  \
        --level={1}  \
        --ckpt={2}  \
    ::: ${LEVELS[@]}  \
    ::: ${CKPTS[@]}  \