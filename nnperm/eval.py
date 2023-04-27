from copy import deepcopy
from collections import OrderedDict, defaultdict
from typing import Iterable, List, Tuple
import numpy as np
import torch
import torch.nn as nn


def is_identity(x: torch.tensor, y: torch.tensor):
    if len(x.flatten()) != len(y.flatten()):
        return False
    if len(x.shape) > 2:  # open_lth layernorm moves output dim to end
        x_1 = torch.moveaxis(x, 1, -1)
        if torch.all(x_1.flatten() == y.flatten()):
            return True
    return torch.all(x.flatten() == y.flatten())


def match_key(key: str, include: List[str] = None, exclude: List[str] = None):
    if include is not None:
        if not any(k in key for k in include):
            return False
    if exclude is not None:
        if any(k in key for k in exclude):
            return False
    return True


class SaveIntermediateHook:
    """This is used to get intermediate values in forward() pass.
    """
    def __init__(self, named_modules: Iterable[Tuple[str, nn.Module]], include: List[str]=None, exclude: List[str]=None, device='cpu', verbose=False):
        self.named_modules = list(named_modules)
        self.device = device
        self.include = include
        self.exclude = exclude
        self.verbose = verbose
        self.intermediates = OrderedDict()

    def __enter__(self):
        self.module_names = OrderedDict()
        self.handles = []
        for name, module in self.named_modules:
            self.module_names[module] = name
            self.handles.append(module.register_forward_hook(self))
        return self.intermediates

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for handle in self.handles:
            handle.remove()
        self.intermediates = OrderedDict()

    def __call__(self, module, args, return_val):
        layer_name = self.module_names[module]
        for arg in args:
            self._add_if_missing(layer_name + ".in", arg)
        self._add_if_missing(layer_name + ".out", return_val)

    def _add_if_missing(self, key, value):
        # copy to prevent value from changing in later operations
        if match_key(key, self.include, self.exclude):
            value = value.detach().clone().to(device=self.device)
            for k, v in self.intermediates.items():
                if is_identity(v, value):
                    if self.verbose: print(f"{key} and {k} are equal, omitting {key}")
                    return
            assert key not in self.intermediates, key
            self.intermediates[key] = value


def evaluate_intermediates(
        model,
        dataloader,
        device="cuda",
        named_modules: Iterable[Tuple[str, nn.Module]]=None,
        include: List[str]=None,
        exclude: List[str]=None,
        verbose=False,
):
    if named_modules is None:
        named_modules = list(model.named_modules())
    if verbose: print(model, "MODULES", *[k for k, v in named_modules], sep="\n")
    model.to(device=device)
    model.eval()
    intermediates = SaveIntermediateHook(
        named_modules, include=include, exclude=exclude, device=device)
    with torch.no_grad():
        for i, (batch_examples, labels) in enumerate(dataloader):
            with intermediates as hidden:
                if verbose: print(f"...batch {i}")
                batch_examples = batch_examples.to(device=device)
                labels = labels.to(device=device)
                output = model(batch_examples)
                yield batch_examples, hidden, output, labels


def evaluate_model(model, dataloader, state_dict=None, device="cuda", return_accuracy=False, loss_fn=None):
    if state_dict is not None:
        model = deepcopy(model)
        model.load_state_dict(state_dict)
    eval_iterator = evaluate_intermediates(
        model, dataloader, device, named_modules=[])
    all_outputs, all_acc, all_loss = [], [], []
    for _, _, outputs, labels in eval_iterator:
        acc = torch.argmax(outputs, dim=-1) == labels
        all_outputs.append(outputs.detach().cpu().numpy())
        all_acc.append(acc.detach().cpu().numpy())
        if loss_fn is not None:
            loss = loss_fn(outputs, labels)
            all_loss.append(loss.detach().cpu().numpy())
    all_loss = None if loss_fn is None else np.concatenate(all_loss)
    if return_accuracy or loss_fn is not None:
        return all_outputs, np.concatenate(all_acc), all_loss
    return all_outputs
