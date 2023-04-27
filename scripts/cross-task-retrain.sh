#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=cross-task-retrain-%j.out
#SBATCH --error=cross-task-retrain-%j.err

MODEL=cifar_vgg_16
DATASET=(  \
    cifar10  \
    cifar10  \
    cifar100  \
    cifar100  \
    svhn  \
)
RETRAIN=(  \
    svhn  \
    cifar100  \
    svhn  \
    cifar10  \
    cifar10  \
)
# LEVELS="1-20"
LEVELS="2,4,6,8,10,12,14,16,18,20"
REPLICATE=($(seq 1 1 2))

source ./open_lth/slurm-setup.sh cifar10 svhn cifar100
cd open_lth

parallel --delay=15 --linebuffer --jobs=3  \
    python open_lth.py lottery_branch retrain  \
        --default_hparams=$MODEL  \
        --dataset_name={2}  \
        --replicate={1}  \
        --warmup_steps="1ep"  \
        --rewinding_steps="1ep"  \
        --batchnorm_replace="layernorm"  \
        --levels=$LEVELS  \
        --retrain_d_dataset_name={3}  \
        --retrain_d_batch_size=128  \
        --retrain_t_optimizer_name='sgd'  \
        --retrain_t_momentum=0.9  \
        --retrain_t_milestone_steps='80ep,120ep'  \
        --retrain_t_lr=0.1  \
        --retrain_t_gamma=0.1  \
        --retrain_t_weight_decay=1e-4  \
        --retrain_t_training_steps='160ep'  \
        --reinit_outputs  \
  ::: ${REPLICATE[@]}  \
  ::: ${DATASET[@]}  \
  :::+ ${RETRAIN[@]}  \
