#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=cross-task-align-%j.out
#SBATCH --error=cross-task-align-%j.err

source ./open_lth/slurm-setup.sh svhn cifar10 cifar100

CKPT_ROOT=$HOME/scratch/open_lth_data/
# cifar10            lottery_45792df32ad68649ffd066ae40be4868
# eurosat               lottery_d1b69a2da0973637bfc9a76d73a1f19f
# cifar100class10       lottery_ee3a8edc96da470068a5b524300f3ab8
# svhn                  lottery_2123f3764046b82699d86590c19bc401
# pixelpermutedcifar10  lottery_37ac5d99c2d78c509e44808f2d2ed6f9
# CKPT=(  \
#     lottery_45792df32ad68649ffd066ae40be4868  \
#     lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
#     lottery_ee3a8edc96da470068a5b524300f3ab8  \
#     lottery_2123f3764046b82699d86590c19bc401  \
#     lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
# )
# # choose any 2 of above
# CKPT_SOURCE=(  \
#     lottery_45792df32ad68649ffd066ae40be4868  \
#     lottery_45792df32ad68649ffd066ae40be4868  \
#     lottery_45792df32ad68649ffd066ae40be4868  \
#     lottery_45792df32ad68649ffd066ae40be4868  \
#     lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
#     lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
#     lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
#     lottery_ee3a8edc96da470068a5b524300f3ab8  \
#     lottery_ee3a8edc96da470068a5b524300f3ab8  \
#     lottery_2123f3764046b82699d86590c19bc401  \
# )
# CKPT_TARGET=(  \
#     lottery_d1b69a2da0973637bfc9a76d73a1f19f  \
#     lottery_ee3a8edc96da470068a5b524300f3ab8  \
#     lottery_2123f3764046b82699d86590c19bc401  \
#     lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
#     lottery_ee3a8edc96da470068a5b524300f3ab8  \
#     lottery_2123f3764046b82699d86590c19bc401  \
#     lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
#     lottery_2123f3764046b82699d86590c19bc401  \
#     lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
#     lottery_37ac5d99c2d78c509e44808f2d2ed6f9  \
# )

# rewind ep1 ckpts
# svhn vgg_16               lottery_de9a5d81719b9d698c9c84ffbb7b8a60
# cifar10 vgg_16            lottery_d449c50ba258e1f0f44b0e5170c340bb
# cifar100 vgg_16           lottery_9480d0d676da3d141b1248da8f13929e
# svhn resnet_20_64         lottery_fe947f9d7bf6ee74bb85716da28ddd53
# cifar10 resnet_20_64      lottery_ae22ef8d26bd185914b1a5cfeb67cfa0
# cifar100 resnet_20_64     lottery_2737a26f5502d8b049d92b2b99c93fbd
CKPT=(  \
    lottery_de9a5d81719b9d698c9c84ffbb7b8a60  \
    lottery_d449c50ba258e1f0f44b0e5170c340bb  \
    lottery_9480d0d676da3d141b1248da8f13929e  \
    lottery_fe947f9d7bf6ee74bb85716da28ddd53  \
    lottery_ae22ef8d26bd185914b1a5cfeb67cfa0  \
    lottery_2737a26f5502d8b049d92b2b99c93fbd  \
)
# choose any 2 of above
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

    # "weight_linear"  \
# first 8: activation align using target dataset
# last 8: activation align using source dataset
TYPE=(
    "activation_linear_cifar10_50000"  \
    "activation_linear_svhn_50000"  \
    "activation_linear_svhn_50000"  \
    "activation_linear_cifar10_50000"  \
    "activation_linear_cifar10_50000"  \
    "activation_linear_svhn_50000"  \
    "activation_linear_svhn_50000"  \
    "activation_linear_cifar10_50000"  \
    "activation_linear_cifar100_50000"  \
    "activation_linear_cifar100_50000"  \
    "activation_linear_cifar10_50000"  \
    "activation_linear_svhn_50000"  \
    "activation_linear_cifar100_50000"  \
    "activation_linear_cifar100_50000"  \
    "activation_linear_cifar10_50000"  \
    "activation_linear_svhn_50000"  \
)
REPLICATE=(1 2 3 4)
REP_A=(1 3)
REP_B=(2 4)
# LEVEL=($(seq 0 1 20))
LEVEL=(0)

# align all cross-task
parallel --delay=5 --linebuffer --jobs=2  \
    python -m scripts.open_lth_align  \
        --ckpt_a=$CKPT_ROOT/{2}/replicate_{5}/level_{4}/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{1}/replicate_{5}/level_{4}/main/model_ep160_it0.pth  \
        --type={3}  \
        --exclude="fc"  \
        --overwrite  \
  ::: ${CKPT_SOURCE[@]}  \
  :::+ ${CKPT_TARGET[@]}  \
  :::+ ${TYPE[@]}  \
  ::: ${LEVEL[@]}  \
  ::: ${REPLICATE[@]}  \

# # baseline: align all sparse-sparse within task
# parallel --delay=5 --linebuffer --jobs=2  \
#     python -m scripts.open_lth_align  \
#         --ckpt_a=$CKPT_ROOT/{1}/replicate_{3}/level_{5}/main/model_ep160_it0.pth  \
#         --ckpt_b=$CKPT_ROOT/{1}/replicate_{4}/level_{5}/main/model_ep160_it0.pth  \
#         --type={2}  \
#         --exclude="fc"  \
#         --overwrite  \
#   ::: ${CKPT[@]}  \
#   :::+ ${TYPE[@]}  \
#   ::: ${REP_A[@]}  \
#   :::+ ${REP_B[@]}  \
#   ::: ${LEVEL[@]}  \
