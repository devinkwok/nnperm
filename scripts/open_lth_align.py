# Compute the permutation between 2 open_lth checkpoints
# save permutations in a format that makes sense to open_lth
import sys
import argparse
from pathlib import Path

from nnperm.align import WeightAlignment, ActivationAlignment
from nnperm.perm import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_dataloader, get_device


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_a', required=True, type=Path)
parser.add_argument('--ckpt_b', required=True, type=Path)
parser.add_argument('--type', default="", type=str)  # this also names the permutation
parser.add_argument('--kernel', required=True, type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--n_train', default=0, type=int)
parser.add_argument('--n_test', default=0, type=int)
parser.add_argument('--batch_size', default=2500, type=int)
parser.add_argument('--exclude', default=None, type=str)
parser.add_argument('--overwrite', default=False, action="store_true")
args = parser.parse_args()

def name_from_path(open_lth_path):
    branch = open_lth_path.parent
    level = branch.parent
    replicate = level.parent
    experiment = replicate.parent
    return "-".join([args.type, args.kernel, experiment.stem, replicate.stem, level.stem, branch.stem])
b2a_save_file = args.ckpt_b.parent / f"perm{name_from_path(args.ckpt_a)}.pt"
a2b_save_file = args.ckpt_a.parent / f"perm{name_from_path(args.ckpt_b)}.pt"
# skip if files already exist
if b2a_save_file.exists() and a2b_save_file.exists() and not args.overwrite:
    sys.exit(f"Files already exist {b2a_save_file}, {a2b_save_file}")

# get model and data
(model_hparams, dataset_hparams), model, params_a = get_open_lth_ckpt(args.ckpt_a)
_, _, params_b = get_open_lth_ckpt(args.ckpt_b)
perm_spec = PermutationSpec.from_sequential_model(params_a)
print(model_hparams.display)
print(dataset_hparams.display)

if "activation" in args.type:
    assert args.n_train == 0 or args.n_test == 0, "Cannot align with both train and test examples."
    if args.n_train != 0:
        dataloader = get_dataloader(dataset_hparams, args.n_train, train=True, batch_size=args.batch_size)
    else:
        dataloader = get_dataloader(dataset_hparams, args.n_test, train=False, batch_size=args.batch_size)
    align_obj = ActivationAlignment(
        perm_spec,
        model,
        dataloader=dataloader,
        exclude=args.exclude.split(",") if args.exclude is not None else None,
        kernel=args.kernel,
        verbose=False,
        device=get_device(),
    )
else:
    align_obj = WeightAlignment(
                perm_spec,
                kernel=args.kernel,
                init_perm=None,
                max_iter=100,
                seed=args.seed,
                order="random",
                verbose=False)

perm, align_loss = align_obj.fit(params_a, params_b)
aligned_params, _ = align_obj.transform()  # this returns np arrays

# save a copy of each permutation to ckpt_a and ckpt_b locations
perm_spec.save_to_file(perm, b2a_save_file)
perm_spec.save_to_file(perm.inverse(), a2b_save_file)
print(f"Saved to {b2a_save_file} and {a2b_save_file}.")
