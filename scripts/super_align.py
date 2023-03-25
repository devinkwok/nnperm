# super-alignments: Compute the permutation between two sets of open_lth checkpoints
# Combine each set of checkpoints using super-alignment technique (stack tensors)
import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd
import torch

from nnperm.align import WeightAlignment
from nnperm.perm import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, find_open_lth_ckpt


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--repdir_a', required=True, type=Path)
parser.add_argument('--repdir_b', required=True, type=Path)
parser.add_argument('--combine_ep_it', required=True, type=str)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--kernel', default="linear", type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--align_bias', default="bias", type=str)
args = parser.parse_args()


# get model and data
def combine_checkpoints(ckpt_dir, checkpoints):
    combined = defaultdict(list)
    for ep_it in checkpoints:
        file = find_open_lth_ckpt(ckpt_dir, ep_it)
        hparams, _, params = get_open_lth_ckpt(file)
        for k, v in params.items():
            combined[k].append(v)
    # use one of the ckpts to setup perm_spec, because tensor shapes are used to infer perm_spec
    if "resnet" in hparams["Model"]["model_name"]:
        print("Aligning residual model")
        perm_spec = PermutationSpec.from_residual_model(params)
    else:
        print("Aligning sequential model")
        perm_spec = PermutationSpec.from_sequential_model(params)
    stacked_params = {k: torch.stack(v, dim=-1) for k, v in combined.items()}
    return perm_spec, stacked_params


# super align
combine_ep_it = args.combine_ep_it.split(",")
perm_spec, params_a = combine_checkpoints(args.repdir_a, combine_ep_it)
_, params_b = combine_checkpoints(args.repdir_b, combine_ep_it)
align_obj = WeightAlignment(
            perm_spec,
            kernel=args.kernel,
            init_perm=None,
            max_iter=100,
            seed=args.seed,
            order="random",
            align_bias=(args.align_bias != "bias"),
            verbose=False)
perm, align_loss = align_obj.fit(params_a, params_b)
stats = align_obj.summarize_last_similarities(**{
    "save_file": args.save_file,
    "combine_ep_it": args.combine_ep_it,
})

# save perm and stats
args.save_file.parent.mkdir(parents=True, exist_ok=True)
perm_spec.save_to_file(perm, args.save_file)

df = pd.DataFrame(stats)
csv_file = Path(str(args.save_file) + ".csv")
df.to_csv(csv_file)

print(f"Saved to {args.save_file} and {csv_file}.")
