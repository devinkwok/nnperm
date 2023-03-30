from copy import deepcopy
from typing import Dict, List
import torch
import numpy as np


import sys
sys.path.append("open_lth")
from open_lth.api import get_ckpt, get_dataset_hparams, get_dataloader, find_ckpt_by_it, get_device


def device():
    return get_device()


def get_open_lth_ckpt(ckpt_path):
    return get_ckpt(ckpt_path)


def get_open_lth_data(dataset_hparams, n_train, n_test, batch_size=5000):
    train_dataloader = get_dataloader(dataset_hparams, n_train, train=True, batch_size=batch_size)
    test_dataloader = get_dataloader(dataset_hparams, n_test, train=False, batch_size=batch_size)
    return train_dataloader, test_dataloader


def find_open_lth_ckpt(replicate_dir, ep_it):
    return find_ckpt_by_it(replicate_dir, ep_it)


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


def to_torch_device(state_dict: Dict[str, np.ndarray], device="cuda"):
    return {k: torch.tensor(v, device=device) if not isinstance(v, torch.Tensor)  \
            else v.detach().clone().to(device=device) for k, v in state_dict.items()}


def to_numpy(state_dict: Dict[str, np.ndarray]):
    return {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor)  \
            else v for k, v in state_dict.items()}


def keys_match(a: dict, b: dict):
    for k in set(list(a.keys()) + list(b.keys())):
        if not (k in a and k in b):
            return False
    return True


def is_valid_key(key: str, include_keywords: List[str] = None, exclude_keywords: List[str] = None):
    if include_keywords is not None:
        if not any(k in key for k in include_keywords):
            return False
    if exclude_keywords is not None:
        if any(k in key for k in exclude_keywords):
            return False
    return True


def parse_int_list(arg):  # parse str containing range or list of ints
    if "-" in arg:
        start, end = arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in arg.split(",")]
