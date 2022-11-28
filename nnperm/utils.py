from copy import deepcopy
from pathlib import Path
from typing import Dict
import torch
import numpy as np
import torchvision


import sys
sys.path.append("open_lth")
from open_lth.foundations.hparams import load_hparams_from_file, ckpt_hparam_path, ModelHparams
from open_lth.models import registry
from open_lth.platforms.platform import get_platform


def load_data(hparams, n_examples, train, data_root=None):
    if data_root is None:
        data_root = Path(get_platform().dataset_root)
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
    dataset = torch.utils.data.Subset(dataset, torch.arange(n_examples))
    return torch.utils.data.DataLoader(dataset, batch_size=n_examples, shuffle=False)


def load_open_lth_model(ckpt, device):
    hparams = load_hparams_from_file(ckpt_hparam_path(ckpt))
    model = registry.get(ModelHparams.create_from_dict(hparams["Model"])).to(device=device)
    params = torch.load(ckpt)
    if "model_state_dict" in params:
        params = params["model_state_dict"]
    return hparams, model, params


def to_torch_device(state_dict: Dict[str, np.ndarray], device="cuda"):
    return {k: torch.tensor(v, device=device) if type(v) is not torch.Tensor  \
            else v.detach().clone() for k, v in state_dict.items()}


def to_numpy(state_dict: Dict[str, np.ndarray]):
    return {k: v.detach().cpu().numpy() if type(v) is torch.Tensor  \
            else v for k, v in state_dict.items()}


def multiplicative_weight_noise(state_dict, std, n_layers=-1,
        include_keywords=[], exclude_keywords=[],
    ):
    state_dict = deepcopy(state_dict)
    for k, v in state_dict.items():
        if n_layers == 0:  # ignore if n_layers < 0
            break  # stop when n_layers of weight noise added
        if not include_keywords or any(x in k for x in include_keywords):
            if not exclude_keywords or not any(x in k for x in exclude_keywords):
                noise = 1 + np.random.randn(*v.shape) * std
                state_dict[k] = v * noise
                n_layers -= 1
    return state_dict
