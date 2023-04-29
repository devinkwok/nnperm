# take 2 checkpoints and precomputed dense-dense and sparse-sparse permutations
# load all saved model epochs (specifically, rewind point and end of training)
# apply no perm, dense-dense perm, or sparse-sparse perms
# compute error barriers for all sparsity levels
import sys
import argparse
from pathlib import Path
import torch

from nnperm.barrier import get_barrier_stats
from nnperm.spec import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_open_lth_data, parse_int_list, device


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--repdir_a', required=True, type=Path)
parser.add_argument('--repdir_b', required=True, type=Path)
parser.add_argument('--perm_a', default=None, type=Path)
parser.add_argument('--perm_b', default=None, type=Path)
parser.add_argument('--target_size_ckpt', default=None, type=Path)  # load this model instead of the ckpts at repdir_a or repdir_b, if doing partial align to a wider network
parser.add_argument('--train_ep_it', required=True, type=str)
parser.add_argument('--levels', required=True, type=str)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--barrier_resolution', default=25, type=int)
parser.add_argument('--n_train', default=None, type=int)
parser.add_argument('--n_test', default=None, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--overwrite', default=False, action="store_true")
args = parser.parse_args()

levels = parse_int_list(args.levels)
train_ep_it = args.train_ep_it.split(",")

# skip if files already exist
if args.save_file.exists() and not args.overwrite:
    sys.exit(f"File already exists {args.save_file}")

# get model and data
(model_hparams, dataset_hparams), model, size_params = get_open_lth_ckpt(args.repdir_a / f"level_{levels[0]}" / "main" / f"model_{train_ep_it[0]}.pth")
if args.target_size_ckpt is not None:
    source_size = size_params
    # TODO temporary hack for layernorm
    _, _, size_params = get_open_lth_ckpt(args.target_size_ckpt)
    # make sure sizes differ by constant ratio
    ratio = None
    for k, v in size_params.items():
        if "layernorm" in k:
            new_ratio = v.shape[0] / source_size[k].shape[0]
            assert ratio is None or ratio == new_ratio
            ratio = new_ratio
    # scale mean/std of layernorm appropriately so they have the correct scale from the source network
    print(f"Scaling layernorm by {ratio} due to added padding")
    (model_hparams, _), model, _ = get_open_lth_ckpt(args.target_size_ckpt, layernorm_scaling=ratio)

train_dataloader, test_dataloader = get_open_lth_data(dataset_hparams, args.n_train, args.n_test)
print(model_hparams.display)
print(dataset_hparams.display)


def load_and_apply_permutation(params, perm_file):
    perm, perm_spec = PermutationSpec.load_from_file(perm_file)
    if args.target_size_ckpt is not None:  # pad params
        target_size = perm_spec.get_sizes(size_params)
        params = perm_spec.apply_padding(params, target_size)
    return perm_spec.apply_permutation(params, perm)


def barrier(a, b, is_train=False):
    return get_barrier_stats(model, train_dataloader if is_train else test_dataloader,
        a, b, resolution=args.barrier_resolution, reduction="mean", device=device())


stats = {}
for level in levels:
    for ckpt_iter in train_ep_it:
        file_a = args.repdir_a / f"level_{level}" / "main" / f"model_{ckpt_iter}.pth"
        file_b = args.repdir_b / f"level_{level}" / "main" / f"model_{ckpt_iter}.pth"
        _, _, params_a = get_open_lth_ckpt(file_a)
        _, _, params_b = get_open_lth_ckpt(file_b)
        if args.perm_a is not None:
            if args.perm_b is None:
                print(f"Transporting A {file_a} to B {file_b} using perm {args.perm_a}")
            else:
                print(f"Barrier between permuted {args.perm_a} A {file_a} and permuted {args.perm_b} B {file_b}.")
            params_a = load_and_apply_permutation(params_a, args.perm_a)
        if args.perm_b is not None:
            if args.perm_a is None:
                print(f"Transporting B {file_b} to A {file_a} using perm {args.perm_b}")
            params_b = load_and_apply_permutation(params_b, args.perm_b)
        else:
            print(f"Baseline barrier for {file_a}, {file_b}")
        stats[f"level_{level}-{ckpt_iter}-train"] = barrier(params_a, params_b, True)
        stats[f"level_{level}-{ckpt_iter}-test"] = barrier(params_a, params_b, False)

args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save(stats, args.save_file)
print(f"Saved to {args.save_file}")
