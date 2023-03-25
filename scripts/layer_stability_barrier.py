# take 2 ckpt dirs, permutation, list of epochs
# compute alignments between ckpts for every epoch
import sys
import argparse
from pathlib import Path
import torch

from nnperm.error import get_barrier_stats
from nnperm.perm import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_open_lth_data, find_open_lth_ckpt, device


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--repdir_a', required=True, type=Path)
parser.add_argument('--repdir_b', required=True, type=Path)
parser.add_argument('--barrier_ep_it', required=True, type=str)
parser.add_argument('--perm_b2a', required=True, type=Path)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--barrier_resolution', default=25, type=int)
parser.add_argument('--n_train', default=None, type=int)
parser.add_argument('--n_test', default=None, type=int)
parser.add_argument('--overwrite', default=False, action="store_true")
args = parser.parse_args()
print(args)

if args.save_file.exists() and not args.overwrite:
    sys.exit(f"File already exists {args.save_file}")

iterations = args.barrier_ep_it.split(",")
# load the permutation and data
perm, perm_spec = PermutationSpec.load_from_file(args.perm_b2a)
# use the first iteration to load the model hparams
(model_hparams, dataset_hparams), model, _ = get_open_lth_ckpt(find_open_lth_ckpt(args.repdir_a, iterations[0]))
train_dataloader, test_dataloader = get_open_lth_data(dataset_hparams, args.n_train, args.n_test)
print(model_hparams.display)
print(dataset_hparams.display)


def barrier(a, b, is_train=False):
    return get_barrier_stats(model, train_dataloader if is_train else test_dataloader,
        a, b, resolution=args.barrier_resolution, reduction="mean", device=device())


# compute barrier for every iteration using permutation
stats_dict = {}
for ep_it in iterations:
    hparams, model, params_a = get_open_lth_ckpt(find_open_lth_ckpt(args.repdir_a, ep_it))
    _, _, params_b = get_open_lth_ckpt(find_open_lth_ckpt(args.repdir_b, ep_it))
    params_b = perm_spec.apply_permutation(params_b, perm)
    stats_dict[f"train-{ep_it}"] = barrier(params_a, params_b, True)
    stats_dict[f"test-{ep_it}"] = barrier(params_a, params_b, False)

args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save(stats_dict, args.save_file)
print(f"Saved to {args.save_file}")
