# Apply precomputed permutation onto sparsity levels, get barriers
import argparse
from pathlib import Path
import torch

from nnperm.barrier import get_barrier_stats
from nnperm.spec import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_open_lth_data, device


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--stats_file', required=True, type=Path)
parser.add_argument('--parent_ckpt', required=True, type=str)
parser.add_argument('--perm_key_a', default=None, type=str)
parser.add_argument('--perm_key_b', default=None, type=str)
parser.add_argument('--ckpt_a_dir', required=True, type=Path)
parser.add_argument('--ckpt_b_dir', required=True, type=Path)
parser.add_argument('--levels', required=True, type=str)
parser.add_argument('--ckpt_filename', required=True, type=Path)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--barrier_resolution', default=25, type=int)
parser.add_argument('--n_train', default=None, type=int)
parser.add_argument('--n_test', default=None, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

# get model and data
stats = torch.load(args.stats_file, map_location=device())
wide_ckpt = stats[args.parent_ckpt] if args.parent_ckpt in stats else args.parent_ckpt

(model_hparams, dataset_hparams), model, wide_params = get_open_lth_ckpt(Path(wide_ckpt))
perm_spec = PermutationSpec.from_sequential_model(wide_params)
target_sizes = perm_spec.get_sizes(wide_params)
train_dataloader, test_dataloader = get_open_lth_data(dataset_hparams, args.n_train, args.n_test)
print(model_hparams.display)
print(dataset_hparams.display)

# get perm
if args.perm_key_a is None:
    perm_a = perm_spec.get_identity_permutation(wide_params)
else:
    perm_a = stats[args.perm_key_a]
if args.perm_key_b is None:
    perm_b = perm_spec.get_identity_permutation(wide_params)
else:
    perm_b = stats[args.perm_key_b]

def barrier(a, b, is_train=False):
    return get_barrier_stats(model, train_dataloader if is_train else test_dataloader,
        a, b, resolution=args.barrier_resolution, reduction="mean", device=device())

# compute barriers for each level
if "-" in args.levels:
    start, end = args.levels.split("-")
    levels = range(start, end + 1)
else:
    levels = [int(x) for x in args.levels.split(",")]

stats = {}
for level in levels:
    ckpt_a = args.ckpt_a_dir / f"level_{level}" / args.ckpt_filename
    ckpt_b = args.ckpt_b_dir / f"level_{level}" / args.ckpt_filename
    _, _, params_a = get_open_lth_ckpt(ckpt_a)
    _, _, params_b = get_open_lth_ckpt(ckpt_b)
    params_a = perm_spec.apply_permutation(perm_spec.apply_padding(params_a, target_sizes), perm_a)
    params_b = perm_spec.apply_permutation(perm_spec.apply_padding(params_b, target_sizes), perm_b)
    print(f"Computing error barrier for {ckpt_a}, {ckpt_b}")
    stats[f"train_level_{level}"] = barrier(params_a, params_b, True)
    stats[f"test_level_{level}"] = barrier(params_a, params_b, False)

args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save(stats, args.save_file)
print(f"Saved to {args.save_file}")
