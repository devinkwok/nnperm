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

#VGG from 4x to 1/8, nobn or with adam
CKPTS=(  \
    lottery_453eb807615b16a690fca0a6ca941c57  \
    lottery_c75cc006137d0b9a9be44c8b4f2ef4bb  \
    lottery_444f95315c1330543f9221faf493cb1a  \
    lottery_d8d275d1b6e0566cf0b3df73deacd6ad  \
    lottery_6edee361614815c04d791242cbe981ea  \
    lottery_64de8dea8cb3aecfb0196adeff94f644  \
    lottery_4f1179d73eb303281b56f0795dc9def7  \
    lottery_3372810d978476ae9b22102c1d6c2978  \
    lottery_ba02b680967534148b52de85bd747445  \
    lottery_09d85889f1ebdf6d4794939fdd6a31da  \
    lottery_20816e46eb10c9cac751e7105d7f7148  \
)

parallel --delay=15 --linebuffer --jobs=3  \
    python exp_1.py  \
        --save_dir=outputs/rebasin-nobn/exp_1c  \
        --n_replicates=2  \
        --loss=L2  \
        --bias_loss_weight=1  \
        --level={1}  \
        --ckpt={2}  \
    ::: ${LEVELS[@]}  \
    ::: ${CKPTS[@]}  \
