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

#VGG from 1/8 to 4x
CKPTS=(  \
    lottery_bd85f4b553eb07d3e751c7d9bd03b3bc  \
    lottery_b62907fe0a5dc7dc6bcfa22dea75fe21  \
    lottery_c4249732c49350ed79fec7f29d9f6c7e  \
    lottery_06e3ceea2dae7621529556ef969cf803  \
    lottery_4d7656b80d72437f584722d51aedd0fc  \
    lottery_405df0e1af1fd13b750c0dbb6c92d3a5  \
)

# LOSS=(L1 L2)
# BIASWEIGHT=(0 1 10)
LOSS=(L2)
BIASWEIGHT=(1)

parallel --delay=15 --linebuffer --jobs=1  \
    python exp_1.py  \
        --save_dir=outputs/rebasin/exp_1a  \
        --n_replicates=3  \
        --ckpt={1}  \
        --loss={2}  \
        --bias_loss_weight={3}  \
    ::: ${CKPTS[@]}  \
    ::: ${LOSS[@]}  \
    ::: ${BIASWEIGHT[@]}  \
