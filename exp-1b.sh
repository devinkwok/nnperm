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

#VGG from 4x to 1/8
CKPTS=(  \
    lottery_405df0e1af1fd13b750c0dbb6c92d3a5  \
    lottery_4d7656b80d72437f584722d51aedd0fc  \
    lottery_06e3ceea2dae7621529556ef969cf803  \
    lottery_c4249732c49350ed79fec7f29d9f6c7e  \
    lottery_b62907fe0a5dc7dc6bcfa22dea75fe21  \
    lottery_bd85f4b553eb07d3e751c7d9bd03b3bc  \
)

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_1.py  \
        --save_dir=outputs/rebasin-nobn/exp_1b  \
        --n_replicates=2  \
        --loss=L2  \
        --bias_loss_weight=1  \
        --level={1}  \
        --ckpt={2}  \
    ::: ${LEVELS[@]}  \
    ::: ${CKPTS[@]}  \
