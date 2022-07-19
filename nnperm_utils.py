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

from nnperm import canonical_normalization, geometric_realignment, permute_state_dict

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

def evaluate_per_sample(model, dataloader, state_dict=None, loss_fn=None, device="cuda"):
    if state_dict is not None:
        model = deepcopy(model)
        model.load_state_dict(deepcopy(state_dict))
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

def open_lth_model_and_data(hparams, n_examples, train=False, device="cuda",
        data_root=Path("../../open_lth_datasets/"),
    ):
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=n_examples, shuffle=False)
    model = registry.get(ModelHparams(
        hparams["Model"]["model_name"],
        hparams["Model"]["model_init"],
        hparams["Model"]["batchnorm_init"])).to(device=device)
    return model, dataloader

def load_open_lth_state_dict(path, replicate, device="cuda"):
    ckpt = torch.load(path / f"replicate_{replicate}/main/checkpoint.pth",
                    map_location=torch.device(device))
    return deepcopy(ckpt["model_state_dict"])


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

def align_and_error(model, state_dict_f, state_dict_g, dataloader, n_samples, loss_fn):

    def get_errors(state_dict_f, state_dict_g):
        return calculate_errors(model, state_dict_f, state_dict_g,
                        dataloader, n_samples=n_samples)

    normalized_f, scale_f = canonical_normalization(state_dict_f)
    normalized_g, scale_g = canonical_normalization(state_dict_g)
    s_f, s_g, loss = geometric_realignment(state_dict_f, state_dict_g, loss_fn=loss_fn)
    permuted_f = permute_state_dict(state_dict_f, s_f)
    permuted_g = permute_state_dict(state_dict_g, s_g)
    np_s_f, np_s_g, np_loss = geometric_realignment(normalized_f, normalized_g, loss_fn=loss_fn)
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
