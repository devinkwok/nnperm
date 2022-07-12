#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

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

MODEL=(mnist_lenet_300_100 mnist_sconv_16_16 cifar_vgg_16 cifar_resnet_20)
REPLICATE=($(seq 1 1 5))
cd open_lth

parallel --delay=15 --linebuffer --jobs=3  \
    python open_lth.py train  \
        --default_hparams={1}  \
        --replicate={2}  \
  ::: ${MODEL[@]}  \
  ::: ${REPLICATE[@]}  \
