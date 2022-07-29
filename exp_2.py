"""Experiment 2:
Change in error barrier when a checkpoint is randomly permuted with weight noise.
"""
import argparse
from pathlib import Path
import torch

from nnperm import random_transform
from nnperm_utils import named_loss_fn, align_and_error, multiplicative_weight_noise, \
    get_open_lth_hparams, load_open_lth_state_dict, open_lth_model_and_data

## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=Path)
parser.add_argument('--n_replicates', required=True, type=int)
parser.add_argument('--loss', required=True, type=named_loss_fn)

parser.add_argument('--n_layers', default=-1, type=int)
parser.add_argument('--no_scale', default=True, action="store_false")
parser.add_argument('--no_permute', default=True, action="store_false")
parser.add_argument('--weight_noise', default=0., type=float)
parser.add_argument('--noise_samples', default=1, type=int)

parser.add_argument('--train_data', default=False, action="store_true")
parser.add_argument('--barrier_resolution', default=11, type=int)
parser.add_argument('--n_examples', default=10000, type=int)
parser.add_argument('--max_search', default=-1, type=int)
parser.add_argument('--ckpt_root', default="../../open_lth_data/", type=Path)
parser.add_argument('--device', default="cuda", type=str)
args = parser.parse_args()

# get model and data
ckpt_dir = args.ckpt_root / args.ckpt
hparams = get_open_lth_hparams(ckpt_dir)
print(hparams)
model, dataloader = open_lth_model_and_data(hparams, args.n_examples,
                            train=args.train_data, device=args.device)

# experiment
save_dir = Path(f"outputs/exp_2/{args.n_replicates}_{args.train_data}_{args.loss}_{args.n_layers}_{args.no_scale}_{args.no_permute}_{args.weight_noise}_{args.ckpt}")
save_dir.mkdir(parents=True, exist_ok=True)
for i in range(1, args.n_replicates + 1):
    # load checkpoint
    state_dict_f = load_open_lth_state_dict(model, ckpt_dir, i, device=args.device)
    for j in range(args.noise_samples):
        # randomly permute
        state_dict_g, scale, permutation = random_transform(state_dict_f,
            scale=args.no_scale, permute=args.no_permute, n_layers=args.n_layers)
        # add weight noise
        if args.weight_noise > 0:
            state_dict_g = multiplicative_weight_noise(state_dict_g,
                                args.weight_noise, args.n_layers)
        # error barriers with and without realignment
        values = align_and_error(model, state_dict_f, state_dict_g, dataloader,
                            args.barrier_resolution, args.loss, args.max_search)
        torch.save(values, save_dir / f"errors_{i}_{j}.pt")
