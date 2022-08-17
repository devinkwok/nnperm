import collections
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torchvision

import sys
sys.path.append("open_lth")
from open_lth.models import registry
from open_lth.foundations.hparams import ModelHparams

from nnperm import ComputationGraph, _is_identity, _is_shortcut, canonical_normalization, get_normalizing_permutation, permute_state_dict


# class SaveIntermediateHook:
#     """This is used to get intermediate values in forward() pass.
#     """

#     def __init__(self, named_modules, target_device='cpu'):
#         self.module_names = collections.OrderedDict()
#         for name, module in named_modules:
#             self.module_names[module] = name
#             module.register_forward_hook(self)
#         self.device = target_device
#         self.reset()

#     def reset(self):
#         self.values = []

#     def __call__(self, module, args, return_val):
#         layer_name = self.module_names[module]
#         args = [x.detach().clone().to(device=self.device) for x in args]
#         return_val = return_val.detach().clone().to(device=self.device)
#         self.values.append((layer_name, *args, return_val))

#     def get_module_names(self):
#         return [x for x in self.module_names.values()]

#     def get_saved_names(self):
#         return [x[0] for x in self.values]

#     def get_inputs(self):
#         return [x[1:-1] for x in self.values]

#     def get_outputs(self):
#         return [x[-1] for x in self.values]

#     def is_identity(self, x, y):
#         return len(x.flatten()) == len(y.flatten()) and torch.all(x.flatten() == y.flatten())

#     def is_relu_output(self, x):
#         return torch.all(x >= 0.).item()

#     def get_intermediates(self):
#         intermediates = OrderedDict()
#         for v in self.values:
#             for i, x in enumerate(v[1:]):
#                 is_return_value = (i == len(v) - 2)
#                 key = v[0] + (".out" if is_return_value else ".in")
#                 is_unique = True
#                 for n, y in intermediates.items():
#                     if self.is_identity(x, y):
#                         print(f"{key} and {n} are equal, omitting {key}")
#                         is_unique = False
#                         break
#                 if is_unique:
#                     assert key not in intermediates
#                     intermediates[key] = x
#         return intermediates

#     def get_graph(self, input_tensor):
#         outputs = {'input': input_tensor}
#         for v in self.values:
#             assert v[0] not in outputs
#             outputs[v[0]] = v[-1]
#         graph_nodes = {}
#         for v in self.values:
#             name = v[0]
#             inputs = []
#             for x in v[1:-1]:
#                 for n, y in outputs.items():
#                     if self.is_identity(x, y):
#                         inputs.append(n)
#             assert len(inputs) == len(v[1:-1])
#             assert name not in graph_nodes
#             graph_nodes[name] = inputs
#         return graph_nodes


# def computation_graph(model, n_test_points=10):
#     hook = SaveIntermediateHook(model.named_modules())
#     first_layer = next(iter(model.state_dict().values()))
#     shape = [n_test_points] + list(first_layer.shape[1:])
#     test_data = torch.randn(shape)
#     model(test_data)
#     print(hook.get_graph(test_data))
#     #TODO need to handle case where parent module has the same input/output as a set of child layers, we don't want that


def infer_open_lth_computation_graph(input_size,
        state_dict,
        input_name='input',
        skip_names=['skip', 'shortcut'],
        first_skip_idx=2,
        ignore_names=['running_mean', 'running_var', 'num_batches_tracked'],
):
    """Uses names from state_dict to assume how the computation graph is structured.

    Assumes:
    * the first shortcut points to the output of the first layer
    * subsequent shortcuts point to output of previous shortcut
    * shortcuts apply an optional linear transform, then are added to the output of the previous (block) layer

    Args:
        state_dict (dict): from torch.nn.Module.state_dict()
    """
    on_skip_path = False
    skip_key = input_name
    prev_key = input_name
    graph_dict = {input_name: {'shape': input_size}}
    for i, (k, v) in enumerate(state_dict.items()):
        if i == first_skip_idx:
            skip_key = k
        if any(n in k for n in ignore_names):
            continue  # do not include in graph
        elif any(n in k for n in skip_names):
            # assume skip layer connects to previous skip (or input)
            if on_skip_path:
                graph_dict[k] = {'parents': [skip_key]}
            else:
                graph_dict[k] = {'parents': [prev_key, skip_key]}
                on_skip_path = True
            skip_key = k
        else:  # assume layer is sequentially applied
            if on_skip_path:
                graph_dict[k] = {'parents': [prev_key, skip_key]}
                on_skip_path = False
            else:
                graph_dict[k] = {'parents': [prev_key]}
            prev_key = k
    graph_dict['output'] = {'parents': [prev_key]}
    return ComputationGraph.from_dict(state_dict, graph_dict)


# error barrier
def error_barrier_from_losses(errors, reduction='none'):
    n_samples = errors.shape[0]
    alphas = error_barrier_linspace_sample(n_samples)
    error_barriers = [e - (a * errors[0] + (1 - a) * errors[-1]) for e, a in zip(errors, alphas)]
    error_barriers = np.stack(error_barriers, axis=0)
    if reduction == 'mean':
        error_barriers = np.mean(error_barriers, axis=1)
    if reduction == 'sum':
        error_barriers = np.sum(error_barriers, axis=1)
    return error_barriers

def evaluate_per_sample(model, dataloader, state_dict=None, loss_fn=None, device="cuda", in_place=False):
    if state_dict is not None:
        if not in_place:
            model = deepcopy(model)
            state_dict = deepcopy(state_dict)
        model.load_state_dict(state_dict)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_examples, batch_labels in dataloader:
            batch_examples = batch_examples.to(device=device)
            y = model(batch_examples)
            if loss_fn is not None:
                y = loss_fn(y, batch_labels.to(device=device))
            outputs.append(y.cpu().detach().numpy())
    return np.concatenate(outputs)

def calculate_errors(model, model_state_dict1, model_state_dict2, dataloader, n_samples=10):
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    errors = []
    for alpha in tqdm(error_barrier_linspace_sample(n_samples)[1:-1]):
        avg_weight = collections.OrderedDict()

        for k in model_state_dict1.keys():
            avg_weight[k] = alpha*model_state_dict1[k].clone() + (1-alpha)*model_state_dict2[k].clone()

        errors.append(evaluate_per_sample(model, dataloader, state_dict=avg_weight, loss_fn=ce_loss))
    error1 = evaluate_per_sample(model, dataloader, state_dict=model_state_dict1, loss_fn=ce_loss)
    error2 = evaluate_per_sample(model, dataloader, state_dict=model_state_dict2, loss_fn=ce_loss)
    errors = np.stack([error1] + errors + [error2], axis=0)
    return errors

def error_barrier_linspace_sample(n_samples):
    return np.linspace(0., 1., n_samples)


# open_lth
def get_open_lth_hparams(save_dir):
    with open(save_dir / f"replicate_1/main/hparams.log", 'r') as f:
        hparam_lines = f.readlines()
    hparams = {}
    for line in hparam_lines:
        line = line.strip()
        if line.endswith(" Hyperparameters"):
            header = line[:-len(" Hyperparameters")]
            hparams[header] = {}
        elif line.startswith("* "):
            k, v = line[len("* "):].split(" => ")
            hparams[header][k] = v
        else:
            raise ValueError(line)
    return hparams

def add_skip_weights_to_open_lth_resnet(model):
    """
    Special logic for ResNet
    * assumes skip connections are called "shortcut" and are empty if input and output shape equal
    * adds identity matrix to empty skip connections (identity matrix) to allow permutation of all skip layers
    """
    model = deepcopy(model)
    for block in model.blocks:
        if len(block.shortcut) == 0:
            c = block.conv1.in_channels
            block.shortcut = nn.Conv2d(c, c, 1, bias=False)
            permutation_weights = torch.eye(c).reshape(c, c, 1, 1).to(device=block.conv1.weight.device)
            block.shortcut.weight.data = nn.parameter.Parameter(permutation_weights)
    return model

def load_open_lth_model(hparams, device):
    model = registry.get(ModelHparams(
        hparams["Model"]["model_name"],
        hparams["Model"]["model_init"],
        hparams["Model"]["batchnorm_init"])).to(device=device)
    if "resnet" in hparams["Model"]["model_name"]:
        model = add_skip_weights_to_open_lth_resnet(model)
    return model

def load_open_lth_data(hparams, n_examples, train, device, data_root):
    if hparams["Dataset"]["dataset_name"] == "cifar10":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.CIFAR10(root=data_root / "cifar10",
                    train=train, download=False, transform=transforms)
    elif hparams["Dataset"]["dataset_name"] == "mnist":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = torchvision.datasets.MNIST(root=data_root / "mnist",
                    train=train, download=False, transform=transforms)
    else:
        raise ValueError(f"Unsupported dataset {hparams['Dataset']['dataset_name']}")
    dataset = torch.utils.data.Subset(dataset, np.arange(n_examples))
    return torch.utils.data.DataLoader(dataset, batch_size=n_examples, shuffle=False)

def open_lth_model_and_data(hparams, n_examples, train=False, device="cuda",
        data_root=Path("../../open_lth_datasets/"),
    ):
    model = load_open_lth_model(hparams, device)
    dataloader = load_open_lth_data(hparams, n_examples, train, device, data_root)
    return model, dataloader

def load_open_lth_state_dict(model, path, replicate, device="cuda"):
    ckpt = torch.load(path / f"replicate_{replicate}/main/checkpoint.pth",
                    map_location=torch.device(device))
    model = deepcopy(model)
    # use model to fill in missing shortcut weights
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    state_dict = model.state_dict()
    assert len(unexpected_keys) == 0
    for k in missing_keys:
        assert _is_shortcut(k, state_dict[k]) and _is_identity(k, state_dict[k])
    return deepcopy(state_dict)


# experiments
def named_loss_fn(loss_fn_name: str) -> nn.Module:
    if loss_fn_name == "L1":
        return nn.L1Loss()
    elif loss_fn_name == "L2":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unrecognized loss function {loss_fn_name}")

def multiplicative_weight_noise(state_dict, std, n_layers=-1,
        include_keywords=[], exclude_keywords=[],
    ):
    state_dict = deepcopy(state_dict)
    for k, v in state_dict.items():
        if n_layers == 0:  # ignore if n_layers < 0
            break  # stop when n_layers of weight noise added
        if not include_keywords or any(x in k for x in include_keywords):
            if not exclude_keywords or not any(x in k for x in exclude_keywords):
                noise = torch.empty_like(v).normal_(mean=1., std=std)
                state_dict[k] = v * noise
                n_layers -= 1
    return state_dict

def align_and_error(model, state_dict_f, state_dict_g, dataloader, n_samples, loss_fn, max_search):

    def get_errors(state_dict_f, state_dict_g):
        return calculate_errors(model, state_dict_f, state_dict_g,
                        dataloader, n_samples=n_samples)

    normalized_f, scale_f = canonical_normalization(state_dict_f)
    normalized_g, scale_g = canonical_normalization(state_dict_g)
    s_f, s_g, loss = get_normalizing_permutation(state_dict_f, state_dict_g,
        loss_fn=loss_fn, max_search=max_search, cache=True)
    permuted_f = permute_state_dict(state_dict_f, s_f)
    permuted_g = permute_state_dict(state_dict_g, s_g)
    np_s_f, np_s_g, np_loss = get_normalizing_permutation(normalized_f, normalized_g,
        loss_fn=loss_fn, max_search=max_search, cache=True)
    norm_and_perm_f = permute_state_dict(normalized_f, np_s_f)
    norm_and_perm_g = permute_state_dict(normalized_g, np_s_g)

    ## sanity check
    # random_f, _, _ = random_transform(state_dict_f, scale=False)
    # values_f = evaluate_per_sample(model, dataloader, state_dict=state_dict_f)
    # values_g = evaluate_per_sample(model, dataloader, state_dict=state_dict_g)
    # norm_f = evaluate_per_sample(model, dataloader, state_dict=normalized_f)
    # perm_f = evaluate_per_sample(model, dataloader, state_dict=permuted_f)
    # rand_f = evaluate_per_sample(model, dataloader, state_dict=random_f)
    # print(np.mean((values_f - values_g)**2))
    # print(np.mean((values_f - norm_f)**2))
    # print(np.mean((values_f - perm_f)**2))
    # print(np.mean((values_f - rand_f)**2))

    return {
            "scale_f": scale_f,
            "scale_g": scale_g,
            "perm_f": s_f,
            "perm_g": s_g,
            "perm_loss": loss,
            "norm_and_perm_f": np_s_f,
            "norm_and_perm_g": np_s_g,
            "norm_and_perm_loss": np_loss,
            "original_barriers": get_errors(state_dict_f, state_dict_g),
            "normalized_barriers": get_errors(normalized_f, normalized_g),
            "permuted_barriers": get_errors(permuted_f, permuted_g),
            "norm_and_perm_barriers": get_errors(norm_and_perm_f, norm_and_perm_g),
        }
