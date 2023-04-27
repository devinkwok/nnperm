#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=cross-task-noperm-%j.out
#SBATCH --error=cross-task-noperm-%j.err


# training hparams
CKPT_ROOT=$HOME/scratch/open_lth_data/
MODEL=(  \
    cifar_vgg_16  \
    cifar_vgg_16  \
    cifar_vgg_16  \
    cifar_vgg_16  \
    cifar_resnet_20_64  \
    cifar_resnet_20_64  \
    cifar_resnet_20_64  \
    cifar_resnet_20_64  \
)
# open_lth hash of $DATASET
# cifar10               lottery_45792df32ad68649ffd066ae40be4868
# HASH=lottery_45792df32ad68649ffd066ae40be4868
# non-$HASH datasets
# eurosat               lottery_d1b69a2da0973637bfc9a76d73a1f19f
# cifar100class10       lottery_ee3a8edc96da470068a5b524300f3ab8
# svhn                  lottery_2123f3764046b82699d86590c19bc401
# pixelpermutedcifar10  lottery_37ac5d99c2d78c509e44808f2d2ed6f9
DATASET=(  \
    cifar10  \
    svhn  \
    svhn  \
    cifar10  \
    cifar10  \
    svhn  \
    svhn  \
    cifar10  \
)
CKPT_SOURCE=(  \
    lottery_9480d0d676da3d141b1248da8f13929e  \
    lottery_9480d0d676da3d141b1248da8f13929e  \
    lottery_d449c50ba258e1f0f44b0e5170c340bb  \
    lottery_de9a5d81719b9d698c9c84ffbb7b8a60  \
    lottery_2737a26f5502d8b049d92b2b99c93fbd  \
    lottery_2737a26f5502d8b049d92b2b99c93fbd  \
    lottery_ae22ef8d26bd185914b1a5cfeb67cfa0  \
    lottery_fe947f9d7bf6ee74bb85716da28ddd53  \
)
CKPT_TARGET=(  \
    lottery_d449c50ba258e1f0f44b0e5170c340bb  \
    lottery_de9a5d81719b9d698c9c84ffbb7b8a60  \
    lottery_de9a5d81719b9d698c9c84ffbb7b8a60  \
    lottery_d449c50ba258e1f0f44b0e5170c340bb  \
    lottery_ae22ef8d26bd185914b1a5cfeb67cfa0  \
    lottery_fe947f9d7bf6ee74bb85716da28ddd53  \
    lottery_fe947f9d7bf6ee74bb85716da28ddd53  \
    lottery_ae22ef8d26bd185914b1a5cfeb67cfa0  \
)
REPLICATE=(1 2 3 4)
# LEVELS="1-20"
LEVELS="6,8,10,12,14,16"

# branch hparams

source ./open_lth/slurm-setup.sh cifar10 svhn cifar100
cd open_lth

# transport mask with no permutation
parallel --delay=15 --linebuffer --jobs=2  \
    python open_lth.py lottery_branch transport_mask  \
        --default_hparams={4}  \
        --dataset_name={1}  \
        --replicate={5}  \
        --warmup_steps="1ep"  \
        --rewinding_steps="1ep"  \
        --batchnorm_replace="layernorm"  \
        --pruning_mask_source_file=$CKPT_ROOT/{3}/replicate_{5}/level_1/main/mask.pth  \
        --levels=$LEVELS  \
        --reinit_outputs  \
  ::: ${DATASET[@]}  \
  :::+ ${CKPT_TARGET[@]}  \
  :::+ ${CKPT_SOURCE[@]}  \
  :::+ ${MODEL[@]}  \
  ::: ${REPLICATE[@]}  \
