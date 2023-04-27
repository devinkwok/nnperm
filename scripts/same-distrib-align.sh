#!/bin/bash
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=same-distrib-align-%j.out
#SBATCH --error=same-distrib-align-%j.err

source ./open_lth/slurm-setup.sh

CKPT_ROOT=$HOME/scratch/open_lth_data/
# cifar10 vgg_16_64 25000 lottery_1193ab29883d4d3ece83c27af218a493
# cifar10 vgg_16_64 50000 lottery_d91dbbbc96d08ec0886540c215406d64
# cifar10 resnet_20_64 25000 lottery_2f6dd759ece2b346d33b577fd22e2b7a
# cifar10 resnet_20_64 50000 lottery_e73c18ce09cdf0157bcb113765806900
# cifar100 vgg_16_64 25000 lottery_2d61650f65ebf010b53eca9dff5805f4
# cifar100 vgg_16_64 50000 lottery_ae89e400548813924f17ab77ae0b5d5b
# cifar100 resnet_20_64 25000 lottery_a0310708bf9754e65de41c443bd4a173
# cifar100 resnet_20_64 50000 lottery_50250bdc670182c939207cb320663425
CKPT=(  \
    lottery_1193ab29883d4d3ece83c27af218a493  \
    lottery_d91dbbbc96d08ec0886540c215406d64  \
    lottery_2f6dd759ece2b346d33b577fd22e2b7a  \
    lottery_e73c18ce09cdf0157bcb113765806900  \
    lottery_2d61650f65ebf010b53eca9dff5805f4  \
    lottery_ae89e400548813924f17ab77ae0b5d5b  \
    lottery_a0310708bf9754e65de41c443bd4a173  \
    lottery_50250bdc670182c939207cb320663425  \
)
CKPT_A=(  \
    lottery_1193ab29883d4d3ece83c27af218a493  \
    lottery_2f6dd759ece2b346d33b577fd22e2b7a  \
    lottery_2d61650f65ebf010b53eca9dff5805f4  \
    lottery_a0310708bf9754e65de41c443bd4a173  \
)
CKPT_B=(  \
    lottery_d91dbbbc96d08ec0886540c215406d64  \
    lottery_e73c18ce09cdf0157bcb113765806900  \
    lottery_ae89e400548813924f17ab77ae0b5d5b  \
    lottery_50250bdc670182c939207cb320663425  \
)
REPLICATE=(1 2 3 4)
REP_A=(1 3)
REP_B=(2 4)
# LEVEL=($(seq 0 1 20))
LEVEL=(0)

# align all cross-task
parallel --delay=5 --linebuffer --jobs=8  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_{4}/level_{3}/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{2}/replicate_{4}/level_{3}/main/model_ep160_it0.pth  \
        --type="weight_linear"  \
  ::: ${CKPT_A[@]}  \
  :::+ ${CKPT_B[@]}  \
  ::: ${LEVEL[@]}  \
  ::: ${REPLICATE[@]}  \

# baseline: align all sparse-sparse within task
parallel --delay=5 --linebuffer --jobs=8  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{1}/replicate_{2}/level_{4}/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_{3}/level_{4}/main/model_ep160_it0.pth  \
        --type="weight_linear"  \
  ::: ${CKPT[@]}  \
  ::: ${REP_A[@]}  \
  :::+ ${REP_B[@]}  \
  ::: ${LEVEL[@]}  \
