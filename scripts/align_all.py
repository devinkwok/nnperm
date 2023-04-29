import argparse
from pathlib import Path
import torch

from nnperm.align import WeightAlignment
from nnperm.barrier import get_barrier_stats
from nnperm.spec import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_open_lth_data, device


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_a', required=True, type=Path)
parser.add_argument('--ckpt_b', required=True, type=Path)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--kernel', required=True, type=str)
parser.add_argument('--barrier_resolution', default=11, type=int)
parser.add_argument('--n_train', default=10000, type=int)
parser.add_argument('--n_test', default=10000, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--save_all_similarities', default=False, action="store_true")
args = parser.parse_args()

# get model and data
(model_hparams, dataset_hparams), model, params_a = get_open_lth_ckpt(args.ckpt_a)
_, _, params_b = get_open_lth_ckpt(args.ckpt_b)
train_dataloader, test_dataloader = get_open_lth_data(dataset_hparams, args.n_train, args.n_test)
perm_spec = PermutationSpec.from_sequential_model(params_a)
print(model_hparams.display)
print(dataset_hparams.display)


def align(x, y):
    align_obj = WeightAlignment(
                perm_spec,
                kernel=args.kernel,
                init_perm=None,
                max_iter=100,
                seed=args.seed,
                order="random",
                verbose=False)
    p, similarities = align_obj.fit(x, y)
    aligned, _ = align_obj.transform()  # this returns np arrays
    stats = align_obj.summarize_last_similarities()
    return p, similarities, aligned, stats

perm, align_loss, aligned_params, align_stats = align(params_a, params_b)
randperm = perm_spec.get_random_permutation(params_b)
randperm_params = perm_spec.apply_permutation(params_b, randperm)
known_perm, known_align_loss, known_aligned_params, known_align_stats = align(params_b, randperm_params)


if args.save_all_similarities:
    sim_stats = {
        "align_loss": align_loss,
        "known_align_loss": known_align_loss,
    }
else:
    sim_stats = {
        "align_stats": align_stats,
        "known_align_stats": known_align_stats,
    }


def barrier(a, b, is_train=False):
    return get_barrier_stats(model, train_dataloader if is_train else test_dataloader,
        a, b, resolution=args.barrier_resolution, reduction="mean", device=device())


stats = {
    "args": vars(args),
    "hparams": hparams,
    "randperm": dict(randperm),
    "perm": dict(perm),
    "train_control": barrier(params_a, params_b, True),
    "test_control": barrier(params_a, params_b, False),
    "train_randperm": barrier(params_a, randperm_params, True),
    "test_randperm": barrier(params_a, randperm_params, False),
    "train_aligned": barrier(params_a, aligned_params, True),
    "test_aligned": barrier(params_a, aligned_params, False),
    "known_perm": known_perm,
    "train_known_control": barrier(params_b, randperm_params, True),
    "test_known_control": barrier(params_b, randperm_params, False),
    "train_known_aligned": barrier(params_b, known_aligned_params, True),
    "test_known_aligned": barrier(params_b, known_aligned_params, False),
}
args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save({**stats, **sim_stats}, args.save_file)
print(f"Saved to {args.save_file}")
