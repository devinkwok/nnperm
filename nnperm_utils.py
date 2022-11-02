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

from nnperm import _is_identity, _is_shortcut, canonical_normalization, get_normalizing_permutation, normalize_batchnorm, permute_state_dict

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
    try:
        with open(save_dir / f"replicate_1/level_0/main/hparams.log", 'r') as f:
            hparam_lines = f.readlines()
    except FileNotFoundError:
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
    model = registry.get(ModelHparams.create_from_dict(hparams["Model"])).to(device=device)
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

def load_open_lth_state_dict(model, path, replicate, level="pretrain", epoch="last", device="cuda"):
    ckpt_name = "checkpoint" if epoch == "last" else f"model_ep{epoch}_it0"
    level_name = "" if level == "train" else f"level_{level}/"
    ckpt = torch.load(path / f"replicate_{replicate}/{level_name}main/{ckpt_name}.pth",
                    map_location=torch.device(device))
    model = deepcopy(model)
    if epoch == "last":
        ckpt = ckpt["model_state_dict"]
    # use model to fill in missing shortcut weights
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
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

def align_and_error(model, state_dict_f, state_dict_g, dataloader, n_samples, loss_fn, bias_loss_weight=0.):

    def get_errors(prefix, normalized_f, normalized_g, permute=False):
        outputs = {}
        if permute:
            s_f, s_g, loss = get_normalizing_permutation(
                normalized_f, normalized_g, loss_fn=loss_fn, bias_loss_weight=bias_loss_weight)
            permuted_f = permute_state_dict(state_dict_f, s_f)
            permuted_g = permute_state_dict(state_dict_g, s_g)
            outputs[f"{prefix}_f"] = s_f
            outputs[f"{prefix}_g"] = s_g
            outputs[f"{prefix}_loss"] =loss
        else:
            permuted_f = state_dict_f
            permuted_g = state_dict_g
        outputs[f"{prefix}_barriers"] = calculate_errors(model, permuted_f, permuted_g, dataloader, n_samples=n_samples)
        return outputs

    unnormalized_f = normalize_batchnorm(state_dict_f)
    unnormalized_g = normalize_batchnorm(state_dict_g)
    resetbn_f = normalize_batchnorm(state_dict_f, reset_batchnorm=True)
    resetbn_g = normalize_batchnorm(state_dict_g, reset_batchnorm=True)
    normalized_f, scale_f = canonical_normalization(state_dict_f)
    normalized_g, scale_g = canonical_normalization(state_dict_g)
    normalized_resetbn_f, scale_resetbn_f = canonical_normalization(state_dict_f, reset_batchnorm=True)
    normalized_resetbn_g, scale_resetbn_g = canonical_normalization(state_dict_g, reset_batchnorm=True)
    return {
        "scale_f": scale_f,
        "scale_g": scale_g,
        "scale_resetbn_f": scale_resetbn_f,
        "scale_resetbn_g": scale_resetbn_g,
        **get_errors("original", unnormalized_f, unnormalized_g, permute=False),
        **get_errors("permuted", unnormalized_f, unnormalized_g, permute=True),
        **get_errors("permuted_resetbn", resetbn_f, resetbn_g, permute=True),
        **get_errors("normalized", normalized_f, normalized_g, permute=False),
        **get_errors("norm_and_perm", normalized_f, normalized_g, permute=True),
        **get_errors("norm_and_perm_resetbn", normalized_resetbn_f, normalized_resetbn_g, permute=True),
    }
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


def partial_align(model, state_dict_f, state_dict_g, perm_f, perm_g, dataloader, n_samples, loss_fn, bias_loss_weight=0.):
    """
    Partially permute network, from first layer only to all but last 2 layers.
    """

    def get_errors(state_dict_f, state_dict_g):
        return calculate_errors(model, state_dict_f, state_dict_g,
                        dataloader, n_samples=n_samples)

    errors = {}
    # go backwards from 2nd last layer to first layer,
    # so we can reuse the same perm array setting 1 layer to None at a time
    perm_f = list(perm_f)
    perm_g = list(perm_g)
    for i in range(len(perm_f) - 2, 0, -1):
        perm_f[i] = None
        perm_g[i] = None
        # set all perms to None
        permuted_f = permute_state_dict(state_dict_f, perm_f)
        permuted_g = permute_state_dict(state_dict_g, perm_g)
        errors[f"partial_barriers_{i}"] = get_errors(permuted_f, permuted_g)
    return errors
