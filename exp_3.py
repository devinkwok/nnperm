"""Experiment 3:
How does error barrier change with more/less layers aligned (from first layer up)?
Requires precomputed permutations
"""
import argparse
from pathlib import Path
import torch

from nnperm_utils import named_loss_fn, get_open_lth_hparams, \
    load_open_lth_state_dict, open_lth_model_and_data, partial_align

## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=Path)
parser.add_argument('--n_replicates', required=True, type=int)
parser.add_argument('--loss', required=True, type=named_loss_fn)
parser.add_argument('--precompute_dir', default="outputs/exp_1", type=Path)
parser.add_argument('--save_dir', default="outputs/exp_3", type=Path)
parser.add_argument('--train_data', default=False, action="store_true")
parser.add_argument('--barrier_resolution', default=11, type=int)
parser.add_argument('--n_examples', default=10000, type=int)
parser.add_argument('--ckpt_root', default="../../scratch/open_lth_data/", type=Path)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--bias_loss_weight', default=0., type=float)
parser.add_argument('--level', default="pretrain_last", type=str)
args = parser.parse_args()

# get model and data
level, epoch = args.level.split("_")
ckpt_dir = args.ckpt_root / args.ckpt
hparams = get_open_lth_hparams(ckpt_dir)
print(hparams)
model, dataloader = open_lth_model_and_data(hparams, args.n_examples,
                            train=args.train_data, device=args.device)

# experiment
subdir = f"{args.n_replicates}_{args.train_data}_{args.loss}_{args.ckpt}"
save_dir = args.save_dir / subdir
save_dir.mkdir(parents=True, exist_ok=True)
for i in range(1, args.n_replicates + 1):
    for j in range(1, i):
        save_file = f"errors_{i}_{j}_train_last.pt"
        # save_file = f"errors_{i}_{j}_{args.level}.pt"
        print(f"Computing partial alignment error barriers for {i}, {j}")
        state_dict_f = load_open_lth_state_dict(model, ckpt_dir, i, level=level, epoch=epoch, device=args.device)
        state_dict_g = load_open_lth_state_dict(model, ckpt_dir, j, level=level, epoch=epoch, device=args.device)
        precomputed = torch.load(args.precompute_dir / subdir / save_file)
        perm_f = precomputed["perm_f"]
        perm_g = precomputed["perm_g"]
        errors = partial_align(model, state_dict_f, state_dict_g, precomputed["perm_f"], precomputed["perm_g"],
            dataloader, args.barrier_resolution, args.loss, args.bias_loss_weight)
        # save everything to new file
        torch.save({**precomputed, **errors}, save_dir / save_file)
