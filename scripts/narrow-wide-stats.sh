#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=stats-%j.out
#SBATCH --error=stats-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
STAT_ROOT=$HOME/scratch/2022-nnperm/narrow-wide-stats/

# vgg_16_8   svhn       lottery_fc3385ddd61e71e3234844f1584212ab
# vgg_16_16  svhn       lottery_0623cbc53a96baa5d56e0c04aa910cf3
# vgg_16_32  svhn       lottery_518ee26e9ef0aa8a2e6c0a5423cf44f2
# vgg_16_64  svhn       lottery_89d67a52bb2e16b7aa93e84bf7c85302
# vgg_16_128 svhn       lottery_69fd4b9f1c515d4ee872a935ef48b0d2
# vgg_16_256 svhn       lottery_ec5595f1318dc58e91af299a37eb6071
# vgg_16_8   cifar10    lottery_688106d9fc0da7db4cebd981434750ee
# vgg_16_16  cifar10    lottery_812938fad9d6a452c60de777ad8b9ba2
# vgg_16_32  cifar10    lottery_e1ee2ce029f3688fdba8d16bcd72101b
# vgg_16_64  cifar10    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d
# vgg_16_128 cifar10    lottery_3ed390b5f8f0b92d658244d053e538e7
# vgg_16_256 cifar10    lottery_23d95a4841f5114daaeb195dcd3bce62
# vgg_16_8   cifar100   lottery_a33c07402a0c44b5ce98c8f42cc7de20
# vgg_16_16  cifar100   lottery_f474c902c60155e9ff25d9d4a710a869
# vgg_16_32  cifar100   lottery_4293247e3b0d0d76098cf66d61521885
# vgg_16_64  cifar100   lottery_5e2581215475976433f5c3aee127f570
# vgg_16_128 cifar100   lottery_5d365e1864a0effd4a92034b20b3cc5f
# vgg_16_256 cifar100   lottery_6020cd42026a268e84a95f95b44a931f

# WIDE_CKPT=(  \
#     lottery_23d95a4841f5114daaeb195dcd3bce62  \
#     lottery_688106d9fc0da7db4cebd981434750ee  \
#     lottery_812938fad9d6a452c60de777ad8b9ba2  \
#     lottery_e1ee2ce029f3688fdba8d16bcd72101b  \
#     lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d  \
#     lottery_3ed390b5f8f0b92d658244d053e538e7  \
# )
# NARROW_CKPT=(  \
#     lottery_688106d9fc0da7db4cebd981434750ee  \
#     lottery_688106d9fc0da7db4cebd981434750ee  \
#     lottery_688106d9fc0da7db4cebd981434750ee  \
#     lottery_688106d9fc0da7db4cebd981434750ee  \
#     lottery_688106d9fc0da7db4cebd981434750ee  \
#     lottery_688106d9fc0da7db4cebd981434750ee  \
# )
WIDE_CKPT=(  \
    lottery_3ed390b5f8f0b92d658244d053e538e7  \
    lottery_23d95a4841f5114daaeb195dcd3bce62  \
    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d  \
)
NARROW_CKPT=(  \
    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d  \
    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d  \
    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d  \
)

REP_WIDE=(1 2)
REP_A=(1 3)
REP_B=(2 4)
    # "weight_linear"  \
    # "activation_linear_cifar10_50000"  \
TYPE=(
    "weight_linear"  \
)
EPOCH="ep160_it0"
LEVELS="0"

# stats by sparsity
LEVEL=(0)
parallel --delay=5 --jobs=1 -u  \
    python -m scripts.stats  \
        --ckpt_a=$CKPT_ROOT/{3}/replicate_{5}/level_0/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{3}/replicate_{6}/level_0/main/model_ep160_it0.pth  \
        --perm_a=$CKPT_ROOT/{3}/replicate_{5}"/level_0/main/perm_"{1}"-"{2}"-replicate_"{4}"-level_0-main-ep160_it0.pt"  \
        --perm_b=$CKPT_ROOT/{3}/replicate_{6}"/level_0/main/perm_"{1}"-"{2}"-replicate_"{4}"-level_0-main-ep160_it0.pt"  \
        --target_size_ckpt=$CKPT_ROOT/{2}/replicate_{4}/level_0/main/model_ep160_it0.pth  \
        --save_file=$STAT_ROOT/{1}/{2}"-"{4}/{3}/{5}"-"{6}/"stats-200test-level_0-main-ep160_it0.pt"  \
        --batch_size=20  \
        --n_test=200  \
    ::: ${TYPE[@]}  \
    ::: ${WIDE_CKPT[@]}  \
    :::+ ${NARROW_CKPT[@]}  \
    ::: ${REP_WIDE[@]}  \
    ::: ${REP_A[@]}  \
    :::+ ${REP_B[@]}  \

# baselines (no perm)
NOTYPE=(noperm)  # dummy variable to avoid renumbering other variables
parallel --delay=5 --jobs=1 -u  \
    python -m scripts.stats  \
        --ckpt_a=$CKPT_ROOT/{3}/replicate_{4}/level_0/main/model_ep160_it0.pth  \
        --ckpt_b=$CKPT_ROOT/{3}/replicate_{5}/level_0/main/model_ep160_it0.pth  \
        --target_size_ckpt=$CKPT_ROOT/{2}/replicate_1/level_0/main/model_ep160_it0.pth  \
        --save_file=$STAT_ROOT/{1}/{2}/{3}/{4}"-"{5}/"stats-200test-level_0-main-ep160_it0.pt"  \
        --batch_size=20  \
        --n_test=200  \
    ::: ${NOTYPE[@]}  \
    ::: ${WIDE_CKPT[@]}  \
    :::+ ${NARROW_CKPT[@]}  \
    ::: ${REP_A[@]}  \
    :::+ ${REP_B[@]}  \
