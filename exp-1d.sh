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

# 5 training checkpoints
# 5 prune levels
LEVELS=(  \
    pretrain_0 pretrain_40 pretrain_80 pretrain_120 pretrain_160  \
    4_160 8_160 12_160 16_160 20_160  \
)

#VGG from 4x to 1/8, sgd with warmup
CKPTS=(  \
    lottery_3af318151d8504c14c56efe6d5f35ef5  \
    lottery_147b9bfc57aa4d51d54149dfbc9f7e8f  \
    lottery_415b4bc41ff5261db200ed92021242fb  \
    lottery_4f79fd9595f47d0f8a4f04a1bf801ae5  \
    lottery_7815ff752b40a28901eeda9157edc3a9  \
)

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_1.py  \
        --save_dir=outputs/rebasin-nobn/exp_1d  \
        --n_replicates=2  \
        --loss=L2  \
        --bias_loss_weight=1  \
        --level={1}  \
        --ckpt={2}  \
    ::: ${LEVELS[@]}  \
    ::: ${CKPTS[@]}  \
