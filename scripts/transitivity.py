# align all ckpts A, B, C... in ckpt_dir to a ckpt X
# compose permutations A -> X -> B and so forth
# compare error barrier of composition (vs A -> B)
import argparse
from itertools import permutations
from pathlib import Path
import torch
import numpy as np

from nnperm.barrier import get_barrier_stats
from nnperm.spec import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_open_lth_data, device
from nnperm.open_lth_align import open_lth_align


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', required=True, type=Path)
parser.add_argument('--ckpt_pattern', required=True, type=str)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--type', default="weight_linear", type=str)
parser.add_argument('--exclude', default=None, type=str)
parser.add_argument('--barrier_resolution', default=25, type=int)
parser.add_argument('--n_train', default=10000, type=int)
parser.add_argument('--n_test', default=10000, type=int)
parser.add_argument('--n_paths', default=20, type=int)
parser.add_argument('--max_path_length', default=5, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=1000, type=int)
args = parser.parse_args()

ckpts = list(Path(args.ckpt_dir).glob(args.ckpt_pattern))

(model_hparams, dataset_hparams), model, params = get_open_lth_ckpt(ckpts[0])
if "resnet" in model_hparams.model_name:
    perm_spec = PermutationSpec.from_residual_model(params)
else:
    perm_spec = PermutationSpec.from_sequential_model(params)
train_dataloader, test_dataloader = get_open_lth_data(dataset_hparams, args.n_train, args.n_test, batch_size=args.batch_size)

#DEBUG sanity check
for k in perm_spec.group_to_axes.keys():
    assert not k.startswith("P")

# 1. get pairwise permutations: rows=from, cols=to
perms = [[None] * len(ckpts) for _ in range(len(ckpts))]
for a, b in zip(*np.triu_indices(len(ckpts), 1)):
    print(f"Aligning {ckpts[a]}, {ckpts[b]}")
    a2b_file, b2a_file = open_lth_align(
        ckpts[a], ckpts[b],
        type=args.type,
        batch_size=args.batch_size,
        exclude=args.exclude,
        save_inverse=True)
    perms[a][b], ps_ab = perm_spec.load_from_file(a2b_file)
    perms[b][a], ps_ba = perm_spec.load_from_file(b2a_file)
    #DEBUG sanity check
    for k in perms[a][b].keys():
        assert not k.startswith("P"), a2b_file
    for k in perms[b][a].keys():
        assert not k.startswith("P"), b2a_file
    for k in ps_ab.group_to_axes.keys():
        assert not k.startswith("P"), a2b_file
    for k in ps_ba.group_to_axes.keys():
        assert not k.startswith("P"), b2a_file



def barrier(a, b, is_train=False):
    return get_barrier_stats(model, train_dataloader if is_train else test_dataloader,
        a, b, resolution=args.barrier_resolution, reduction="mean", device=device())


# 2. get fixed points, error barriers for $k$-length paths composed of permutations
all_stats = []
for n in range(2, args.max_path_length + 1):  # need at least 2 to form a pair
    paths = list(permutations(range(len(ckpts)), n))
    # randomly choose args.n_paths
    paths = [paths[i] for i in np.random.permutation(len(paths))[:args.n_paths]]
    for path in paths:
        print(f"Barriers for path {path}")
        last_perm = perm_spec.get_identity_permutation(params)
        # 3. Apply composition of path permutation to the replicate
        for i, j in zip(path[:-1], path[1:]):
            last_perm = last_perm.compose(perms[i][j])
        _, _, params_a = get_open_lth_ckpt(ckpts[path[0]])
        _, _, params_b = get_open_lth_ckpt(ckpts[path[-1]])
        permuted_a = perm_spec.apply_permutation(params_a, last_perm)
        cycle_perm = last_perm.compose(perms[path[-1]][path[0]])
        cycle_a = perm_spec.apply_permutation(params_a, cycle_perm)
        stats = {
            "path": path,
            "path_train": barrier(params_b, permuted_a, True),
            "path_test": barrier(params_b, permuted_a, False),
            "cycle_train": barrier(params_a, cycle_a, True),
            "cycle_test": barrier(params_a, cycle_a, False),
            "fixed_points": cycle_perm.fixed_points(),
        }
        all_stats.append(stats)

args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save({"ckpts": ckpts, "stats": all_stats}, args.save_file)
print(f"Saved to {args.save_file}")
