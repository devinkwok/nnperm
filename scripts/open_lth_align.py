# Compute the permutation between 2 open_lth checkpoints
# save permutations in a format that makes sense to open_lth
import argparse
from pathlib import Path

from nnperm.open_lth_align import open_lth_align


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_a', required=True, type=Path)
parser.add_argument('--ckpt_b', required=True, type=Path)
parser.add_argument('--type', default="weight_linear", type=str)  # this also names the permutation
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=400, type=int)
parser.add_argument('--exclude', default=None, type=str)
parser.add_argument('--overwrite', default=False, action="store_true")
parser.add_argument('--save_inverse', default=False, action="store_true")  # save a2b using inverse perm
parser.add_argument('--target_size_ckpt_a', default=None, type=Path)  # use these if ckpt_a needs to be padded
parser.add_argument('--target_size_ckpt_b', default=None, type=Path)
parser.add_argument('--prune_type', default='sparse_global', type=str)
parser.add_argument('--prune_randomize', default='identity', type=str)
parser.add_argument('--prune_fraction', default=0., type=float)
args = parser.parse_args()


open_lth_align(
    args.ckpt_a,
    args.ckpt_b,
    type=args.type,
    seed=args.seed,
    batch_size=args.batch_size,
    exclude=args.exclude,
    overwrite=args.overwrite,
    save_inverse=args.save_inverse,
    target_size_ckpt_a=args.target_size_ckpt_a,
    target_size_ckpt_b=args.target_size_ckpt_b,
    prune_type=args.prune_type,
    prune_randomize=args.prune_randomize,
    prune_fraction=args.prune_fraction
)
