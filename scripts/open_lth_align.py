# Compute the permutation between 2 open_lth checkpoints
# save permutations in a format that makes sense to open_lth
import sys
import argparse
from pathlib import Path

from nnperm.align import WeightAlignment, ActivationAlignment, PartialActivationAlignment, PartialWeightAlignment
from nnperm.perm import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_dataloader, get_device


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
args = parser.parse_args()

def name_from_path(open_lth_path):
    epoch = open_lth_path.stem.split("model_")[1]
    branch = open_lth_path.parent
    level = branch.parent
    replicate = level.parent
    experiment = replicate.parent
    path_str = "-".join([experiment.stem, replicate.stem, level.stem, branch.stem, epoch])
    return f"perm_{args.type}-{path_str}.pt"

b2a_save_file = args.ckpt_b.parent / name_from_path(args.ckpt_a)
a2b_save_file = args.ckpt_a.parent / name_from_path(args.ckpt_b)
# skip if files already exist
if b2a_save_file.exists() and not args.overwrite:
    if not args.save_inverse:
        sys.exit(f"File already exists {b2a_save_file}")
    elif a2b_save_file.exists():
        sys.exit(f"Files already exist {b2a_save_file}, {a2b_save_file}")

exclude_layers = args.exclude.split(",") if args.exclude is not None else None
# get model and data
(model_hparams_a, dataset_hparams_a), model_a, params_a = get_open_lth_ckpt(args.ckpt_a)
(model_hparams_b, dataset_hparams_b), model_b, params_b = get_open_lth_ckpt(args.ckpt_b)
if "resnet" in model_hparams_a.model_name:
    print("Aligning residual model")
    perm_spec = PermutationSpec.from_residual_model(params_a)
else:
    print("Aligning sequential model")
    perm_spec = PermutationSpec.from_sequential_model(params_a)

perm_spec = perm_spec.subset_perm(exclude_axes=exclude_layers)
print(model_hparams_a.display)
print(dataset_hparams_a.display)

# TODO temporary hack for layernorm
def get_target_size_model(model, target_size_ckpt):
    source_size = model.state_dict()
    if target_size_ckpt is not None:
        _, _, size_params = get_open_lth_ckpt(target_size_ckpt)
        # make sure sizes differ by constant ratio
        ratio = None
        for k, v in size_params.items():
            if "layernorm" in k:
                new_ratio = v.shape[0] / source_size[k].shape[0]
                assert ratio is None or ratio == new_ratio
                ratio = new_ratio
        # scale mean/std of layernorm appropriately so they have the correct scale from the source network
        print(f"Scaling layernorm by {ratio} due to added padding")
        _, model, _ = get_open_lth_ckpt(target_size_ckpt, layernorm_scaling=ratio)
    return model


type, kernel, *other_args = args.type.split("_")
AlignClass = ActivationAlignment if type == "activation" else WeightAlignment
if any(x != y for x, y in zip(perm_spec.get_sizes(params_a).values(), perm_spec.get_sizes(params_b).values())):
    AlignClass = PartialActivationAlignment if type == "activation" else PartialWeightAlignment
    # get the correct size of model to embed into
    model_a = get_target_size_model(model_a, args.target_size_ckpt_a)
    model_b = get_target_size_model(model_b, args.target_size_ckpt_b)
    print(f"Sizes differ, using {AlignClass.__name__}")  # if sizes differ, use partial alignment
if type == "activation":
    dataset_name = other_args[0]
    n_train = int(other_args[1])
    intermediate_type = "last" if len(other_args) < 3 else other_args[2]
    if dataset_name == dataset_hparams_a.dataset_name:
        dataset_hparams = dataset_hparams_a
    elif dataset_name == dataset_hparams_b.dataset_name:
        dataset_hparams = dataset_hparams_b
    else:
        raise ValueError(f"Dataset must be either from A: {dataset_hparams_a.dataset_name}, or B: {dataset_hparams_b.dataset_name}")
    dataloader = get_dataloader(dataset_hparams, n_train, train=True, batch_size=args.batch_size)
    align_obj = AlignClass(
        perm_spec,
        model_a=model_a,
        model_b=model_b,
        dataloader=dataloader,
        intermediate_type=intermediate_type,
        exclude=exclude_layers,
        kernel=kernel,
        verbose=False,
        device=get_device(),
    )
else:
    align_obj = AlignClass(
                perm_spec,
                kernel=kernel,
                init_perm=None,
                max_iter=100,
                seed=args.seed,
                order="random",
                verbose=False)

perm, align_loss = align_obj.fit(params_a, params_b)

# save a copy of each permutation to ckpt_a and ckpt_b locations
perm_spec.save_to_file(perm, b2a_save_file)
if args.save_inverse:
    perm_spec.save_to_file(perm.inverse(), a2b_save_file)
    print(f"Saved to {b2a_save_file} and {a2b_save_file}.")
else:
    print(f"Saved to {b2a_save_file}.")
