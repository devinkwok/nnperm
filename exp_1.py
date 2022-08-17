"""Experiment 1:
Change in error barrier on pairs of checkpoints after realigning weights.
"""
import argparse
from pathlib import Path
import torch

from nnperm_utils import named_loss_fn, align_and_error, \
    get_open_lth_hparams, load_open_lth_state_dict, open_lth_model_and_data

## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=Path)
parser.add_argument('--n_replicates', required=True, type=int)
parser.add_argument('--loss', required=True, type=named_loss_fn)

parser.add_argument('--train_data', default=False, action="store_true")
parser.add_argument('--barrier_resolution', default=11, type=int)
parser.add_argument('--n_examples', default=10000, type=int)
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
save_dir = Path(f"outputs/exp_OT_1_norm/{args.n_replicates}_{args.train_data}_{args.loss}_{args.ckpt}")
save_dir.mkdir(parents=True, exist_ok=True)
for i in range(1, args.n_replicates + 1):
    for j in range(1, i):
        print(f"Computing error barriers for {i}, {j}")
        state_dict_f = load_open_lth_state_dict(model, ckpt_dir, i, device=args.device)
        state_dict_g = load_open_lth_state_dict(model, ckpt_dir, j, device=args.device)
        values = align_and_error(model, state_dict_f, state_dict_g, dataloader,
                            args.barrier_resolution, args.loss)
        torch.save(values, save_dir / f"errors_{i}_{j}.pt")
