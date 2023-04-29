# compare 2 permutations via their per-layer fixed points
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

from nnperm.spec import PermutationSpec

## Setup
parser = argparse.ArgumentParser()
parser.add_argument(dest='perm_dirs', type=Path, nargs='+')
args = parser.parse_args()

for perm_dir in args.perm_dirs:
    # assume all permutations in dir should be compared together
    perms = {}
    perm_spec = None
    # load everything to memory
    for file in perm_dir.glob("perm*.pt"):
        perm, perm_spec = PermutationSpec.load_from_file(file)
        perms[file] = perm
    # then compare pairwise fixed points
    fixed_points = defaultdict(list)
    fixed_points_summary = defaultdict(list)
    perm_keys = list(perms.keys())
    layers = list(perm_spec.group_to_axes.keys())
    idx_a, idx_b = np.triu_indices(len(perm_keys))
    # get fixed points per layer
    for i, (a, b) in enumerate(zip(idx_a, idx_b)):
        perm_a, perm_b = perms[perm_keys[a]], perms[perm_keys[b]]
        perm_a2b2a = perm_a.compose(perm_b.inverse())
        fp = perm_a2b2a.fixed_points()
        # save each layer as a 2D np array with separate axes for perm a/b triangular index, channels
        for layer in layers:
            fixed_points[layer].append(fp[layer])
            # save summary stat (average over channels to get proportion of fixed points)
            fixed_points_summary[layer].append(np.mean(fp[layer]))
    fixed_points = {k: np.stack(v, axis=0) for k, v in fixed_points.items()}
    fixed_points_summary = {k: np.stack(v, axis=0) for k, v in fixed_points_summary.items()}

    np.savez_compressed(perm_dir / "fixed_points.npz", **fixed_points,
                        perms=[str(x) for x in perm_keys])
    np.savez_compressed(perm_dir / "fixed_points_summary.npz", **fixed_points_summary,
                        perms=[str(x) for x in perm_keys])
