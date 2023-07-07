"""
For one pair of networks A and B, (optionally permute and) compute per-layer:
    - Number of channels
    - Similarity stats
    - Sparsity masks of each ckpt
        - total number and masked
        - size of intersection and union
    - Weight magnitudes
        - at endpoints
        - at interpolated alpha=0.5
    - L2 norm between A and B
    - Per-channel correlation coefficients on:
        - weights
        - activations
    - Per-example max error barrier
    - Per-example activation correlation coefficients
"""
import argparse
import sys
import time
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from functorch import make_functional_with_buffers, jacrev
from scipy.stats import pearsonr

from nnperm.eval import evaluate_intermediates
from nnperm.barrier import interpolate_dict
from nnperm.spec import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_open_lth_data, get_device, to_numpy, to_torch_device


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_a', required=True, type=Path)
parser.add_argument('--ckpt_b', required=True, type=Path)
parser.add_argument('--perm_a', default=None, type=Path)
parser.add_argument('--perm_b', default=None, type=Path)
parser.add_argument('--n_test', default=10000, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--save_file', required=True, type=Path)
parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument('--overwrite', default=False, action="store_true")
args = parser.parse_args()

# skip if files already exist
if args.save_file.exists() and not args.overwrite:
    sys.exit(f"File already exists {args.save_file}")


def reshape_keeping_first_dim(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    keep_dims = 1 if len(x.shape) == 1 else x.shape[0]  # if only 1 dim, summarize over it
    # leave 1st dim intact, summarize all remaining dims
    return x.reshape(keep_dims, -1)


class OnlineStats:
    def update(self):
        raise NotImplementedError
    def compute(self):
        raise NotImplementedError
    def one_shot(self, *args):
        self.update(*args)
        return self.compute()

class OnlineAvg(OnlineStats):
    def __init__(self):
        self.n = []
        self.sum_x = []
        self.sum_x_sq = []
        self.mean = []  # sanity check
        self.std = []  # sanity check
        self.min = []
        self.max = []
    def update(self, x):
        with torch.no_grad():
            x = reshape_keeping_first_dim(x)
            self.n.append(x.shape[1])
            self.sum_x.append(torch.sum(x, dim=1).detach().cpu().numpy())
            self.sum_x_sq.append(torch.sum(x**2, dim=1).detach().cpu().numpy())
            self.mean.append(torch.mean(x, dim=1).detach().cpu().numpy())
            self.std.append(torch.std(x, dim=1).detach().cpu().numpy())
            self.min.append(torch.min(x, dim=1).values.detach().cpu().numpy())
            self.max.append(torch.max(x, dim=1).values.detach().cpu().numpy())
    def compute(self):
        n = np.sum(self.n)
        sum_x = np.sum(np.stack(self.sum_x, axis=0), axis=0)
        sum_x_sq = np.sum(np.stack(self.sum_x_sq, axis=0), axis=0)
        online_mean = sum_x / n
        sum_centered_sq = np.sqrt(sum_x_sq - sum_x**2 / n)
        online_std = sum_centered_sq / np.sqrt(n - 1)
        if len(self.n) == 1:  # return the more numerically stable version if not in online mode
            online_mean = self.mean[0]
            online_std = self.std[0]
        else:
            est_std = np.average(self.std, axis=0, weights=(np.array(self.n) - 1))
            est_mean = np.average(self.mean, axis=0, weights=self.n)
            # if np.any(np.abs(online_mean - est_mean) > 1e-2):
            #     print(f"WARNING mean disagrees with online mean: {np.max(np.abs(online_mean - est_mean))}")
            # if np.any(np.abs(online_std - est_std) > 1e-2):
            #     print(f"WARNING std disagrees with online std: {np.max(np.abs(online_std - est_std))}")
        return {
            "n": n.item(),
            "mean": online_mean,
            "std": online_std,
            "min": np.min(np.stack(self.min, axis=0), axis=0),
            "max": np.max(np.stack(self.max, axis=0), axis=0),
            "sum": sum_x,
            "sqsum": sum_x_sq,
        }


class OnlinePearsonR(OnlineStats):
    def __init__(self):
        self.x = OnlineAvg()
        self.y = OnlineAvg()
        self.sum_xy = []
        self.r = []  # track pearsonr as a sanity check versus online computation
    def update(self, x, y):
        with torch.no_grad():
            self.x.update(x)
            self.y.update(y)
            assert x.shape == y.shape
            x = reshape_keeping_first_dim(x)
            y = reshape_keeping_first_dim(y)
            self.sum_xy.append(torch.sum(x * y, dim=1).detach().cpu().numpy())
            self.r.append(np.array([pearsonr(i.flatten().detach().cpu().numpy(), j.flatten().detach().cpu().numpy())[0] for i, j in zip(x, y)]))
    def compute(self):
        n = np.sum(self.x.n)
        x = self.x.compute()
        y = self.y.compute()
        sum_xy = np.sum(np.stack(self.sum_xy, axis=0), axis=0)
        online_r = (n * sum_xy - x["sum"] * y["sum"]) / np.sqrt(n * x["sqsum"] - x["sum"]**2) / np.sqrt(n * y["sqsum"] - y["sum"]**2)
        if len(self.r) == 1:  # return the more numerically stable version if not in online mode
            online_r = self.r[0]
        else:
            est_r = np.average(self.r, axis=0, weights=self.x.n)
            # if np.any(np.abs(online_r - est_r) > 1e-2):
            #     print(f"WARNING pearsonr disagrees with online r: {np.max(np.abs(online_r - est_r))}")
        return {
            "pearsonr": online_r
        }


class TripleStats(OnlineStats):
    def __init__(self) -> None:
        self.corr_ab = OnlinePearsonR()
        self.corr_ai = OnlinePearsonR()
        self.corr_bi = OnlinePearsonR()
        self.err_ab = OnlineAvg()
        self.err_ai = OnlineAvg()
        self.err_bi = OnlineAvg()
    def update(self, a, b, i):
        self.corr_ab.update(a, b)
        self.corr_ai.update(a, i)
        self.corr_bi.update(b, i)
        self.err_ab.update(a - b)
        self.err_ai.update(a - i)
        self.err_bi.update(b - i)
    def compute(self):
        return {
            **{"a_" + k: v for k, v in self.corr_ab.x.compute().items()},
            **{"b_" + k: v for k, v in self.corr_ab.y.compute().items()},
            **{"i_" + k: v for k, v in self.corr_ai.y.compute().items()},
            **{"ab_" + k: v for k, v in self.corr_ab.compute().items()},
            **{"ai_" + k: v for k, v in self.corr_ai.compute().items()},
            **{"bi_" + k: v for k, v in self.corr_bi.compute().items()},
            **{"ab_" + k: v for k, v in self.err_ab.compute().items()},
            **{"ai_" + k: v for k, v in self.err_ai.compute().items()},
            **{"bi_" + k: v for k, v in self.err_bi.compute().items()},
        }


def assert_not_equal(iterable_a, iterable_b, iterable_i):
    for a, b, i in zip(iterable_a, iterable_b, iterable_i):
        if isinstance(a, np.ndarray):
            assert np.any(a != i) and np.any(b != i) and np.any(a != b)
        else:
            assert torch.any(a != i) and torch.any(b != i) and torch.any(a != b)


"""
SETUP
"""
if not args.debug:
    (model_hparams, dataset_hparams), model, params_a = get_open_lth_ckpt(args.ckpt_a)
    _, _, params_b = get_open_lth_ckpt(args.ckpt_b, layernorm_scaling=1)
    params_a = to_numpy(params_a)
    params_b = to_numpy(params_b)
    params_for_perm_spec = params_a
    # assume perm_spec and perm size are identical for A and B
    perm_a, perm_b, perm_spec = None, None, None
    if args.perm_a is not None:
        perm_a, perm_spec = PermutationSpec.load_from_file(args.perm_a)
    if args.perm_b is not None:
        perm_b, perm_spec = PermutationSpec.load_from_file(args.perm_b)
    # create new perm_spec when both perm_a and perm_b are None
    if perm_spec is None:  
        if "resnet" in model_hparams.model_name:
            print("Residual model")
            perm_spec = PermutationSpec.from_residual_model(params_for_perm_spec)
        else:
            print("Sequential model")
            perm_spec = PermutationSpec.from_sequential_model(params_for_perm_spec)
    # data for activations
    _, test_dataloader = get_open_lth_data(dataset_hparams, 1, args.n_test, batch_size=args.batch_size)
    print(model_hparams.display)
    print(dataset_hparams.display)
# DEBUG #
else:
    def get_device():
        return "cpu"
    model = torch.nn.Sequential(torch.nn.Linear(20, 15), torch.nn.LayerNorm(15), torch.nn.ReLU(), torch.nn.Linear(15, 10))
    params_a = model.state_dict()
    params_b = {k: v + torch.randn_like(v) for k, v in params_a.items()}
    examples = torch.randn(100, 20)
    labels = torch.randint(0, 10, size=[100])
    test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(examples, labels), batch_size=50)
    perm_spec = PermutationSpec.from_sequential_model(params_a)
    perm_b = perm_spec.get_random_permutation(params_b)
    perm_a = None
    torch.save({
        "params_a": params_a,
        "params_b": params_b,
        "examples": examples,
        "labels": labels,
        "perm": dict(perm_b),
        }, args.save_file.parent / (args.save_file.stem + "-params.pt"))
    params_a = to_numpy(params_a)
    params_b = to_numpy(params_b)

# pad and permute params
def assert_perm_size_equal(target_size, perm):
    if perm is None:
        return
    perm_sizes = perm.sizes()
    for k in target_size.keys():
        assert target_size[k] == perm_sizes[k], (target_size[k], perm_sizes[k])

target_size = perm_spec.get_sizes(params_for_perm_spec)
assert_perm_size_equal(target_size, perm_a)
assert_perm_size_equal(target_size, perm_b)
params_a = perm_spec.apply_padding(params_a, target_size)
params_b = perm_spec.apply_padding(params_b, target_size)
if perm_a is not None:
    params_a = perm_spec.apply_permutation(params_a, perm_a)
if perm_b is not None:
    params_b = perm_spec.apply_permutation(params_b, perm_b)
params_int = interpolate_dict(params_a, params_b, 0.5)
assert_not_equal(params_a.values(), params_b.values(), params_int.values())

# for channel-wise stats, get layers that share the same outputs to a given perm
# this allows comparison between weights and activations
layers_to_perm = {}
for k, layers in perm_spec.group_to_axes.items():
    for layer_name, dim, is_input in layers:
        if not is_input:
            # remove parameter type (e.g. "".weight", ".bias") from layer name
            if layer_name in layers_to_perm:
                print(f"Warning: overwriting layer {layer_name} perm from {layers_to_perm[layer_name]} to {k}")
            layers_to_perm[layer_name] = k

def combine_params_by_perm(params):
    output = defaultdict(list)
    for k, v in params.items():
        if k in layers_to_perm:
            # flatten all except channels (output dim)
            output[layers_to_perm[k]].append(v.reshape(v.shape[0], -1))
        else:
            print(f"Warning, layer {k} not assigned to any perm")
    return {k: np.concatenate(v, axis=1) for k, v in output.items()}

ch_params_a = combine_params_by_perm(params_a)
ch_params_b = combine_params_by_perm(params_b)
ch_params_int = combine_params_by_perm(params_int)

assert_not_equal(ch_params_a.values(), ch_params_b.values(), ch_params_int.values())


"""
STATS
"""
class ExecTime:
    def __init__(self, message) -> None:
        self.message = message
    def __enter__(self):
        print(self.message, end="...\t")
        self.start = time.time()
    def __exit__(self, *args):
        print(f"{time.time() - self.start:0.4f} seconds")


# - Number of channels
# size of each layer
with ExecTime("\tsizes"):
    stats = {
        "layer_weight_all_size": {k: v.size for k, v in params_a.items()},
        "channel_weight_all_size": perm_spec.get_sizes(params_a),
    }


# - Weight magnitudes, correlations, MSE
#     - at endpoints
#     - at interpolated alpha=0.5
with ExecTime("\tweight magnitude"):
    stats = {**stats,
        "layer_weight": {k: TripleStats().one_shot(params_a[k].flatten(), params_b[k].flatten(), params_int[k].flatten()) for k in params_a.keys()},
        "channel_weight": {k: TripleStats().one_shot(ch_params_a[k], ch_params_b[k], ch_params_int[k]) for k in ch_params_a.keys()},
    }


# Activations

def acc_and_loss(output, label):
    acc = torch.argmax(output, dim=-1) == label
    loss = loss_fn(output, label)
    return acc.detach().cpu().numpy(), loss.detach().cpu().numpy()

def barriers(out_a, out_b, out_i, label):
    acc_a, loss_a = acc_and_loss(out_a, label)
    acc_b, loss_b = acc_and_loss(out_b, label)
    acc_int, loss_int = acc_and_loss(out_int, label)
    return {
        "a_acc": acc_a,
        "b_acc": acc_b,
        "i_acc": acc_int,
        "barrier_acc": acc_int - (acc_a + acc_b) / 2,
        "a_loss": loss_a,
        "b_loss": loss_b,
        "i_loss": loss_int,
        "barrier_loss": loss_int - (loss_a + loss_b) / 2,
    }


loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
model_a = deepcopy(model)
model_b = deepcopy(model)
model_int = deepcopy(model)
model_a.load_state_dict(to_torch_device(params_a, get_device()))
model_b.load_state_dict(to_torch_device(params_b, get_device()))
model_int.load_state_dict(to_torch_device(params_int, get_device()))

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = deepcopy(model)
        self.loss_fn = loss_fn
    def forward(self, x, y):
        z = self.model(x)
        return self.loss_fn(z, y)

fmodel, fparams_a, buffers = make_functional_with_buffers(ModelWithLoss(model_a, loss_fn))
_, fparams_b, buffers = make_functional_with_buffers(ModelWithLoss(model_b, loss_fn))
_, fparams_i, buffers = make_functional_with_buffers(ModelWithLoss(model_int, loss_fn))
assert_not_equal(fparams_a, fparams_b, fparams_i)

eval_a = evaluate_intermediates(model_a, test_dataloader, get_device())
eval_b = evaluate_intermediates(model_b, test_dataloader, get_device())
eval_int = evaluate_intermediates(model_int, test_dataloader, get_device())

# need to batch activations, compute correlation online
stats["layer_activation"] = defaultdict(TripleStats)
stats["channel_activation"] = defaultdict(TripleStats)
stats["example_activation"] = defaultdict(list)
stats["layer_jacobian"] = defaultdict(TripleStats)
stats["channel_jacobian"] = defaultdict(TripleStats)
stats["example_jacobian"] = defaultdict(list)
stats["example_barrier"] = {"output": []}

for j, ((input_a, hidden_a, out_a, label_a), (input_b, hidden_b, out_b, label_b),  \
        (input_int, hidden_int, out_int, label_int)) in enumerate(zip(eval_a, eval_b, eval_int)):
    print(f"\tbatch {j} ({len(label_a)})...")
    n_examples = len(label_a)
    assert torch.all(label_a == label_b) and torch.all(label_a == label_int)
    assert_not_equal(out_a, out_b, out_int)

# activations
    with ExecTime("\t\tactivations"):
        for k in hidden_a.keys():
            a = hidden_a[k]
            b = hidden_b[k]
            i = hidden_int[k]
            stats["layer_activation"][k].update(a.flatten(), b.flatten(), i.flatten())
# - Per-example activation correlation coefficients
            # if layernorm, need to reshape examples back in
            if not "layernorm" in k:
                assert n_examples == a.shape[0] and a.shape[0] == b.shape[0] and a.shape[0] == i.shape[0]
            stats["example_activation"][k].append(TripleStats().one_shot(a.reshape(n_examples, -1), b.reshape(n_examples, -1), i.reshape(n_examples, -1)))
# per-channel mse, activation correlation between A and B, A/B and int
            # swap output dims to position 0 so that we iterate over channels
            # sum over all except output channels
            a_ch = torch.moveaxis(a, 1, 0)
            b_ch = torch.moveaxis(b, 1, 0)
            i_ch = torch.moveaxis(i, 1, 0)
            stats["channel_activation"][k].update(a_ch, b_ch, i_ch)

# - Per-example midpoint error barrier
    with ExecTime("\t\tbarriers"):
        stats["example_barrier"]["output"].append(barriers(out_a, out_b, out_int, label_a))

# jacobians
    with ExecTime("\t\tjacobians"):
        jac_a = jacrev(fmodel, argnums=0)(fparams_a, buffers, input_a, label_a)
        jac_b = jacrev(fmodel, argnums=0)(fparams_b, buffers, input_a, label_a)
        jac_i = jacrev(fmodel, argnums=0)(fparams_i, buffers, input_a, label_a)
        assert_not_equal(jac_a, jac_b, jac_i)
        for (k, v), j_a, j_b, j_i in zip(params_a.items(), jac_a, jac_b, jac_i):
            target_shape = tuple([len(input_a)] + list(v.shape))
            assert j_a.shape == target_shape, (k, j_a.shape, target_shape)
            assert j_b.shape == target_shape, (k, j_b.shape, target_shape)
            assert j_i.shape == target_shape, (k, j_i.shape, target_shape)
            stats["layer_jacobian"][k].update(j_a.flatten(), j_b.flatten(), j_i.flatten())
            stats["example_jacobian"][k].append(TripleStats().one_shot(j_a, j_b, j_i))
            j_a_ch = torch.moveaxis(j_a, 1, 0)
            j_b_ch = torch.moveaxis(j_b, 1, 0)
            j_i_ch = torch.moveaxis(j_i, 1, 0)
            stats["channel_jacobian"][k].update(j_a_ch, j_b_ch, j_i_ch)

    if (j + 1) * args.batch_size >= args.n_test:
        break  #TODO temporary hack for early stopping: n_test doesn't work with open_lth.api


def flatten_stats_dict(stats_dict):
    # case 1: dict[prefix + stat_name, dict[layer_name, np.ndarray]]
    # case 2: dict[prefix, dict[layer_name, dict[stat_name, np.ndarray|numeric]]],
    # case 3: dict[prefix, dict[layer_name, OnlineStats]],
    # case 4: dict[prefix, dict[layer_name, list[dict[stat_name, np.ndarray]]]],
    # turn 4, 3, 2 into 1
    output_dict = defaultdict(dict)
    for prefix, layer_dict in stats_dict.items():
        for layer_name, item in layer_dict.items():
            # turn 4 into 2
            if isinstance(item, list):
                per_layer_output = defaultdict(list)
                for stats_dict in item:
                    for k, v in stats_dict.items():
                        if isinstance(v, int):  # save number of averaged items as single int
                            if k in per_layer_output:  # check number is same for all batches
                                assert per_layer_output[k] == v
                            per_layer_output[k] = v
                        else:
                            per_layer_output[k].append(v)
                layer_dict[layer_name] = {k: np.concatenate(v) if isinstance(v, list) else v for k, v in per_layer_output.items()}
            # turn 3 into 2
            elif isinstance(item, OnlineStats):
                layer_dict[layer_name] = item.compute()
        # turn 2 into 1
        for layer_name, item in layer_dict.items():
            if isinstance(item, dict):
                for stat_name, stat in item.items():
                    if isinstance(stat, np.ndarray) and stat.size == 1:
                        stat = stat.item()
                    elif isinstance(stat, list) and len(stat) == 1:
                        stat = stat[0]
                    output_dict[f"{prefix}_{stat_name}"][layer_name] = stat
            else:
                output_dict[prefix][layer_name] = item
    return output_dict


# - Sparsity masks of each ckpt
def mask_stats(a, b):
    mask_a = (reshape_keeping_first_dim(a) != 0)
    mask_b = (reshape_keeping_first_dim(b) != 0)
    return {
        "a_mask": np.count_nonzero(mask_a, axis=1),
        "b_mask": np.count_nonzero(mask_b, axis=1),
        "ab_intersection": np.count_nonzero(np.logical_and(mask_a, mask_b), axis=1),
        "ab_union": np.count_nonzero(np.logical_or(mask_a, mask_b), axis=1),
    }

with ExecTime("\tmasks"):
    mask_stats = {
        "layer_weight": {k: mask_stats(v.flatten(), params_b[k].flatten()) for k, v in params_a.items()},
        "channel_weight": {k: mask_stats(v, ch_params_b[k]) for k, v in ch_params_a.items()},
    }


with ExecTime(f"Combining stats"):
    stats = {
        **flatten_stats_dict(stats),
        **flatten_stats_dict(mask_stats),
    }

args.save_file.parent.mkdir(parents=True, exist_ok=True)
torch.save(stats, args.save_file)
print(f"Saved to {args.save_file}")
