# take 2 checkpoints and precomputed dense-dense and sparse-sparse permutations
# load all saved model epochs (specifically, rewind point and end of training)
# apply no perm, dense-dense perm, or sparse-sparse perms
# compute error barriers for all sparsity levels
import argparse
from pathlib import Path
import torch

from nnperm.error import get_barrier_stats
from nnperm.perm import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_open_lth_data, parse_int_list, device


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--repdir_a', required=True, type=Path)
parser.add_argument('--repdir_b', required=True, type=Path)
parser.add_argument('--train_ep_it', required=True, type=str)
parser.add_argument('--levels', required=True, type=str)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--barrier_resolution', default=25, type=int)
parser.add_argument('--n_train', default=None, type=int)
parser.add_argument('--n_test', default=None, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--kernel', default="linear", type=str)
args = parser.parse_args()


levels = parse_int_list(args.levels)
train_ep_it = args.train_ep_it.split(",")

# get model and data
(model_hparams, dataset_hparams), model, params = get_open_lth_ckpt(args.repdir_a / f"level_{levels[0]}" / "main" / f"model_{train_ep_it[0]}.pth")
perm_spec = PermutationSpec.from_sequential_model(params)
train_dataloader, test_dataloader = get_open_lth_data(dataset_hparams, args.n_train, args.n_test)
print(model_hparams.display)
print(dataset_hparams.display)


def barrier(a, b, is_train=False):
    return get_barrier_stats(model, train_dataloader if is_train else test_dataloader,
        a, b, resolution=args.barrier_resolution, reduction="mean", device=device())

def all_barriers(perms, perm_type):
    stats = {}
    for perm_file, level in zip(perms, levels):
        for ckpt_iter in train_ep_it:
            file_a = args.repdir_a / f"level_{level}" / "main" / f"model_{ckpt_iter}.pth"
            file_b = args.repdir_b / f"level_{level}" / "main" / f"model_{ckpt_iter}.pth"
            print(f"Computing error barrier for {file_a}, {file_b}")
            _, _, params_a = get_open_lth_ckpt(file_a)
            _, _, params_b = get_open_lth_ckpt(file_b)
            if perm_file is not None:
                perm, perm_spec = PermutationSpec.load_from_file(perm_file)
                params_b = perm_spec.apply_permutation(params_b, perm)
            stats[f"train-{perm_type}-level_{level}-{ckpt_iter}"] = barrier(params_a, params_b, True)
            stats[f"test-{perm_type}-level_{level}-{ckpt_iter}"] = barrier(params_a, params_b, False)
    return stats


source_root = args.repdir_b.parent
target_root = args.repdir_a.parent
dense_perm = source_root / args.repdir_b.stem / "level_0" / "main" / f"perm-{args.kernel}-to-{target_root.stem}-{args.repdir_a.stem}-level_0-main.pt"
sparse_perms = [source_root / args.repdir_b.stem / f"level_{i}" / "main" / f"perm-{args.kernel}-to-{target_root.stem}-{args.repdir_a.stem}-level_{i}-main.pt" for i in levels]

stats_dict = {
    **all_barriers([None] * len(levels), "noperm"),
    **all_barriers(sparse_perms, "sparse"),
    **all_barriers([dense_perm] * len(levels), "dense"),
}

args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save(stats_dict, args.save_file)
print(f"Saved to {args.save_file}")
