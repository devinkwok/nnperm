#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=transitivity-width-%j.out
#SBATCH --error=transitivity-width-%j.err

source ./open_lth/slurm-setup.sh cifar10

CKPT_ROOT=$HOME/scratch/open_lth_data/
SAVE_ROOT=$HOME/scratch/2022-nnperm/transitivity-width/
# vgg_16_8   cifar10    lottery_688106d9fc0da7db4cebd981434750ee
# vgg_16_16  cifar10    lottery_812938fad9d6a452c60de777ad8b9ba2
# vgg_16_32  cifar10    lottery_e1ee2ce029f3688fdba8d16bcd72101b
# vgg_16_64  cifar10    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d
# vgg_16_128 cifar10    lottery_3ed390b5f8f0b92d658244d053e538e7
# vgg_16_256 cifar10    lottery_23d95a4841f5114daaeb195dcd3bce62
# ResNet_20_16    cifar10    lottery_57c37a2b8fd5a5c74cb0f565fd63e73c
# ResNet_20_32    cifar10    lottery_9cdece881a2eff8a65aeecba308aabe6
# ResNet_20_64    cifar10    lottery_889c23b4d2b571fabafee827cc8697cd
# ResNet_20_128   cifar10    lottery_7ce57051a6cf02ec112ad4d37ffccf93
# ResNet_20_256   cifar10    lottery_42bf0fd10b6e5a3a3e35e74f86c0fc90
# ResNet_20_512   cifar10    lottery_2fec827265eaaaa263b155b4ef1f43a7
CKPTS=(  \
    lottery_688106d9fc0da7db4cebd981434750ee  \
    lottery_812938fad9d6a452c60de777ad8b9ba2  \
    lottery_e1ee2ce029f3688fdba8d16bcd72101b  \
    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d  \
    lottery_3ed390b5f8f0b92d658244d053e538e7  \
    lottery_23d95a4841f5114daaeb195dcd3bce62  \
    lottery_57c37a2b8fd5a5c74cb0f565fd63e73c  \
    lottery_9cdece881a2eff8a65aeecba308aabe6  \
    lottery_889c23b4d2b571fabafee827cc8697cd  \
    lottery_7ce57051a6cf02ec112ad4d37ffccf93  \
    lottery_42bf0fd10b6e5a3a3e35e74f86c0fc90  \
    lottery_2fec827265eaaaa263b155b4ef1f43a7  \
)
EPOCHS=(160)


TYPE=(  \
    weight_linear  \
)
parallel --delay=15 --linebuffer --jobs=1  \
    python -m scripts.transitivity  \
        --ckpt_dir=$CKPT_ROOT/{1}/  \
        --ckpt_pattern="replicate_*/level_0/main/model_ep"{2}"_it0.pth"  \
        --save_file=$SAVE_ROOT/{3}/{1}-ep{2}.pt  \
        --type={3}  \
        --n_paths=5  \
        --max_path_length=3  \
    ::: ${CKPTS[@]}  \
    ::: ${EPOCHS[@]}  \
    ::: ${TYPE[@]}  \


TYPE=(  \
    activation_linear_cifar10_50000_all  \
    activation_cosine_cifar10_50000_all  \
)
parallel --delay=15 --linebuffer --jobs=1  \
    python -m scripts.transitivity  \
        --ckpt_dir=$CKPT_ROOT/{1}/  \
        --ckpt_pattern="replicate_*/level_0/main/model_ep"{2}"_it0.pth"  \
        --save_file=$SAVE_ROOT/{3}/{1}-ep{2}.pt  \
        --type={3}  \
        --n_paths=5  \
        --batch_size=200  \
        --max_path_length=3  \
        --exclude="conv,layernorm"  \
    ::: ${CKPTS[@]}  \
    ::: ${EPOCHS[@]}  \
    ::: ${TYPE[@]}  \
