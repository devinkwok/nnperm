# take 2 ckpt dirs, list of times T, list of layers K
# aligns frankenstein networks which combine early and end of training weights
# to check whether layer depth affects alignment quality over training
import argparse
from pathlib import Path
import pandas as pd

from nnperm.align import WeightAlignment
from nnperm.perm import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, find_open_lth_ckpt


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--repdir_a', required=True, type=Path)
parser.add_argument('--repdir_b', required=True, type=Path)
parser.add_argument('--align_ep_it', required=True, type=str)
parser.add_argument('--final_ep_it', required=True, type=str)
parser.add_argument('--layer_thresholds', required=True, type=str)
parser.add_argument('--layer_subset_types', required=True, type=str)
parser.add_argument('--save_dir', required=True, type=Path)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--kernel', default="linear", type=str)
parser.add_argument('--align_bias', default="bias", type=str)
args = parser.parse_args()


# get model and data, weights at end of training
(model_hparams, dataset_hparams), model, final_a = get_open_lth_ckpt(find_open_lth_ckpt(args.repdir_a, args.final_ep_it))
_, _, final_b = get_open_lth_ckpt(find_open_lth_ckpt(args.repdir_b, args.final_ep_it))
if "resnet" in model_hparams.model_name:
    print("Aligning residual model")
    perm_spec = PermutationSpec.from_residual_model(final_a)
else:
    print("Aligning sequential model")
    perm_spec = PermutationSpec.from_sequential_model(final_a)
print(model_hparams.display)
print(dataset_hparams.display)

# check that thresholds are in layer names
param_keys = list(final_a.keys())
iterations = args.align_ep_it.split(",")    # weights at time T
thresholds = args.layer_thresholds.split(",")  # threshold K for subset
subset_types = args.layer_subset_types.split(",")
for t in thresholds:
    if not any(t in k for k in param_keys):
        raise ValueError(f"Threshold {t} does not match any parameter name in {param_keys}")

# frankenstein network: combine subset of weights S at time T with ~S from end of training
# align 2 frankensteins, get alignment similarity stats
# assume state dicts are in order of depth
def frankenstein_network(params_to_insert, params, insert_keys, invert=False):
    if invert:  # swap order of params and params_to_insert to get inverse
        return frankenstein_network(params, params_to_insert, insert_keys, invert=False)
    output = {k: v for k, v in params.items()}
    for k in insert_keys:
        output[k] = params_to_insert[k]
    return output


def replace_network_subset(params_to_insert, params, subset_type, threshold_keyword):
    if subset_type == "bottom-up":          # replace up to threshold_keyword (exclusive)
        keys = param_keys[:min(i for i, k in enumerate(param_keys) if threshold_keyword in k)]
        return frankenstein_network(params_to_insert, params, keys)
    if subset_type == "top-down":           # replace all after threshold_keyword (inclusive)
        keys = param_keys[:min(i for i, k in enumerate(param_keys) if threshold_keyword in k)]
        return frankenstein_network(params_to_insert, params, keys, invert=True)
    if subset_type == "put-in":             # replace only threshold_keyword
        keys = [k for k in param_keys if threshold_keyword in k]
        return frankenstein_network(params_to_insert, params, keys)
    if subset_type == "leave-out":          # replace all but threshold_keyword
        keys = [k for k in param_keys if threshold_keyword in k]
        return frankenstein_network(params_to_insert, params, keys, invert=True)


def perm_filename(ep_it, type, threshold):
    return args.save_dir / f"perm-{args.kernel}-{ep_it}-{type}-{threshold}.pt"


def align_and_save(params_a, params_b, ep_it, type, threshold):
    align_obj = WeightAlignment(
                perm_spec,
                kernel=args.kernel,
                init_perm=None,
                max_iter=100,
                seed=args.seed,
                order="random",
                align_bias=(args.align_bias != "bias"),
                verbose=False)
    perm, _ = align_obj.fit(params_a, params_b)
    sim_stats = align_obj.summarize_last_similarities(**{
                "ep_it": ep_it,
                "type": type,
                "threshold": threshold,
    })
    # keep perms in separate files for cross-compatibility with barrier scripts
    perm_spec.save_to_file(perm, perm_filename(ep_it, type, threshold))
    return sim_stats


args.save_dir.mkdir(parents=True, exist_ok=True)

# control: no alignment at end of training
identity_perm = perm_spec.get_identity_permutation(final_a)
perm_spec.save_to_file(identity_perm, perm_filename(args.final_ep_it, "identity", "all"))

# gold standard: use alignment at end of training
stats = []
stats += align_and_save(final_a, final_b, args.final_ep_it, "final", "all")

for ep_it in iterations:
    ckpt_a = find_open_lth_ckpt(args.repdir_a, ep_it)
    ckpt_b = find_open_lth_ckpt(args.repdir_b, ep_it)
    print(f"Aligning {ckpt_a} and {ckpt_b}")
    _, _, params_a = get_open_lth_ckpt(ckpt_a)
    _, _, params_b = get_open_lth_ckpt(ckpt_b)
    # baseline: all aligned at time T
    stats += align_and_save(params_a, params_b, ep_it, "ckpt", "none")
    # get one of the following subsets: bottom-up(top-down), single(leave-one-out)
    for subset_type in subset_types:
        for threshold_keyword in thresholds:
            # positive: frankenstein each S for each T (bottom-up and single)
            # negative: frankenstein each ~S for all T (top-down and leave-one-out)
            # which of positive or negative is run depends on threshold_keyword
            frankenstein_a = replace_network_subset(params_a, final_a, subset_type, threshold_keyword)
            frankenstein_b = replace_network_subset(params_b, final_b, subset_type, threshold_keyword)
            stats += align_and_save(frankenstein_a, frankenstein_b, ep_it, subset_type, threshold_keyword)

# save stats together in one csv file
csv_file = args.save_dir / f"align_stats-{args.kernel}.csv"
df = pd.DataFrame(stats)
df.to_csv(csv_file)
print(f"Stats saved to {csv_file}")
