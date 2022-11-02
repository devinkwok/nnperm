"""Experiment 4:
How much do permutations diverge from identity in k-cycles?
1. Load precomputed permutations over $n$ replicates
2. Generate $k$-cycles of permutations that are (in principle) identity transformations relative to a given replicate
    e.g. for replicate 0, 0 -> 1, 1 -> 2, 2 -> 0 should be a 3-cycle.
3. Apply composition of cycle permutation to the replicate, calculate error barrier
4. Save the permutation and error barrier (in order to plot # of places that differ from identity, and where differences are distributed)
"""
import argparse
from itertools import permutations
from pathlib import Path
import torch
from nnperm import compose_permutation, inverse_permutation, permute_state_dict

from nnperm_utils import calculate_errors, named_loss_fn, get_open_lth_hparams, \
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

# 1. Load precomputed permutations over $n$ replicates
state_dicts = []
perms = [[None] * args.n_replicates for _ in range(args.n_replicates)]
for i in range(1, args.n_replicates + 1):
    state_dicts.append(load_open_lth_state_dict(
                        model, ckpt_dir, i, level=level, epoch=epoch, device=args.device))
    for j in range(1, i):
        save_file = f"errors_{i}_{j}_train_last.pt"
        # save_file = f"errors_{i}_{j}_{args.level}.pt"
        precomputed = torch.load(args.precompute_dir / subdir / save_file)
        # f and g are both mapped to h
        # to map from f to g, apply s_f composed with s_g^{-1} to go f -> h -> g
        perms[i - 1][j - 1] = compose_permutation(precomputed["perm_f"], inverse_permutation(precomputed["perm_g"]))
        # mapping from g to f is the inverse
        perms[j - 1][i - 1] = inverse_permutation(perms[i - 1][j - 1])

# 2. Generate $k$-cycles of permutations that are (in principle) identity transformations relative to a given replicate
cycle_perms, errors = {}, {}
for n in range(2, args.n_replicates + 1):  # need at least 2 to form a pair
    for cycle in permutations(range(args.n_replicates), n):
        cycle_perm = None
        # 3. Apply composition of cycle permutation to the replicate, calculate error barrier
        for i, j in zip(cycle, cycle[1:] + cycle[0:1]):
            # cycle ends by going from last element to first
            if cycle_perm is None:
                cycle_perm = perms[i][j]
            else:
                cycle_perm = compose_permutation(cycle_perm, perms[i][j])
        state_dict = state_dicts[cycle[0]]
        errors[cycle] = calculate_errors(model, state_dict, permute_state_dict(state_dict, cycle_perm),
                                        dataloader, n_samples=args.barrier_resolution)
        cycle_perms[cycle] = cycle_perm

# 4. Save the permutation and error barrier (in order to plot # of places that differ from identity, and where differences are distributed)
torch.save({"cycle_perms": cycle_perms, "errors": errors},
            args.save_dir / subdir / f"cycles.pt")
