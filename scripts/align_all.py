import argparse
from pathlib import Path
import torch

from nnperm.align import WeightAlignment
from nnperm.error import get_barrier_stats
from nnperm.perm import PermutationSpec
from nnperm.utils import load_open_lth_model, load_data


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
parser.add_argument('--device', default="cuda", type=str)
args = parser.parse_args()

# get model and data
hparams, model, params_a = load_open_lth_model(args.ckpt_a, args.device)
_, _, params_b = load_open_lth_model(args.ckpt_b, args.device)
train_dataloader = load_data(hparams, args.n_train, True)
test_dataloader = load_data(hparams, args.n_test, False)
perm_spec = PermutationSpec.from_sequential_model(params_a)
target_sizes = None
print(hparams)

align_obj = WeightAlignment(
            perm_spec,
            kernel=args.kernel,
            init_perm=None,
            max_iter=100,
            seed=42,
            order="random",
            verbose=False)
perm, align_loss = align_obj.fit(params_a, params_b)
aligned_params = align_obj.transform()  # this returns np arrays

randperm = align_obj.perm_spec.get_random_permutation(params_b)
randperm_params = align_obj.perm_spec.apply_permutation(params_b, randperm)
known_perm, known_align_loss = align_obj.fit(params_b, randperm_params)
known_aligned_params = align_obj.transform()

def barrier(a, b, is_train=False):
    return get_barrier_stats(model, train_dataloader if is_train else test_dataloader,
        a, b, # mask_a, mask_b,
        resolution=args.barrier_resolution, reduction="mean", device=args.device)

stats = {
    "args": vars(args),
    "hparams": hparams,
    "randperm": randperm,
    "perm": perm,
    "align_loss": align_loss,
    "train_control": barrier(params_a, params_b, True),
    "test_control": barrier(params_a, params_b, False),
    "train_randperm": barrier(params_a, randperm_params, True),
    "test_randperm": barrier(params_a, randperm_params, False),
    "train_aligned": barrier(params_a, aligned_params, True),
    "test_aligned": barrier(params_a, aligned_params, False),
    "known_perm": known_perm,
    "known_align_loss": known_align_loss,
    "train_known_control": barrier(params_b, randperm_params, True),
    "test_known_control": barrier(params_b, randperm_params, False),
    "train_known_aligned": barrier(params_b, known_aligned_params, True),
    "test_known_aligned": barrier(params_b, known_aligned_params, False),
}
args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save(stats, args.save_file)
print(f"Saved to {args.save_file}")
