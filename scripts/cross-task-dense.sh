#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=cross-task-dense-%j.out
#SBATCH --error=cross-task-dense-%j.err


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
# cifar100   cifar_vgg_16        lottery_9480d0d676da3d141b1248da8f13929e
# cifar10    cifar_vgg_16        lottery_d449c50ba258e1f0f44b0e5170c340bb
# svhn       cifar_vgg_16        lottery_de9a5d81719b9d698c9c84ffbb7b8a60
# cifar100   cifar_resnet_20_64  lottery_2737a26f5502d8b049d92b2b99c93fbd
# cifar10    cifar_resnet_20_64  lottery_ae22ef8d26bd185914b1a5cfeb67cfa0
# svhn       cifar_resnet_20_64  lottery_fe947f9d7bf6ee74bb85716da28ddd53
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
TYPE=(  \
    "weight_linear"  \
    "activation_linear_cifar10_50000"  \
)
# LEVELS="1-20"
LEVELS="6,8,10,12,14,16"

# branch hparams

source ./open_lth/slurm-setup.sh cifar10 svhn cifar100
cd open_lth

# use dense-dense perm
parallel --delay=15 --linebuffer --jobs=2  \
    python open_lth.py lottery_branch transport_mask  \
        --default_hparams={4}  \
        --dataset_name={1}  \
        --replicate={5}  \
        --warmup_steps="1ep"  \
        --rewinding_steps="1ep"  \
        --batchnorm_replace="layernorm"  \
        --pruning_mask_source_file=$CKPT_ROOT/{3}/replicate_{5}/level_1/main/mask.pth  \
        --pruning_permutation=$CKPT_ROOT/{3}"/replicate_"{5}"/level_0/main/perm_"{6}"-"{2}"-replicate_"{5}"-level_0-main-ep160_it0.pt"  \
        --pruning_infer_permutation_from="file"  \
        --levels=$LEVELS  \
        --reinit_outputs  \
  ::: ${DATASET[@]}  \
  :::+ ${CKPT_TARGET[@]}  \
  :::+ ${CKPT_SOURCE[@]}  \
  :::+ ${MODEL[@]}  \
  ::: ${REPLICATE[@]}  \
  ::: ${TYPE[@]}  \
