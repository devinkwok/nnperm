from collections import defaultdict
from copy import deepcopy
from typing import Dict, Union
from tqdm import tqdm
import torch.nn as nn
import numpy as np

from nnperm.utils import to_torch_device
from nnperm.eval import evaluate_model


def linear_interpolate(start, end, alpha):
    return (1 - 1. * alpha) * start + alpha * end  # multiply by 1. to turn bool into float


def _wrap_in_default_dict(value):
    if isinstance(value, dict):
        return value
    return defaultdict(lambda: value)


def interpolate_dict(
        start: Union[np.ndarray, Dict[str, np.ndarray]],
        end: Union[np.ndarray, Dict[str, np.ndarray]],
        alpha: Union[float, Dict[str, np.ndarray]],
):
    start = _wrap_in_default_dict(start)
    end = _wrap_in_default_dict(end)
    alpha = _wrap_in_default_dict(alpha)
    return {k: linear_interpolate(start[k], end[k], alpha[k])  \
            for k in start.keys()}


class EnsembleModel(nn.Module):
    def __init__(self, model, params_a, params_b, alpha: float=0.5):
        super().__init__()
        self.alpha = alpha
        self.model_a = self._init_model(model, params_a)
        self.model_b = self._init_model(model, params_b)

    def _init_model(self, model, params):
        params = deepcopy(params)
        model = deepcopy(model)
        model.load_state_dict(params)
        return model

    def forward(self, x):  # averages at logits
        return linear_interpolate(self.model_a(x), self.model_b(x), self.alpha)

    @classmethod
    def models_for_interpolation(cls, model, params_a, params_b, align_mask):
        """Weight split for interpolation:
        alpha, model_a, model_b
        0, A_shared|A_unique, A_shared|B_unique
        1, B_shared|A_unique, B_shared|B_unique

        In particular, if align_mask = 0, all params are unique, and interpolating gives the same ensemble model:
            model_a = params_a|params_b
            model_b = params_a|params_b
        If align_mask = 1, all params are shared, and interpolating the ensemble model is equivalent to interpolating the original params:
            model_a = params_a|params_a
            model_b = params_b|params_b
        """
        b_with_shared_a = interpolate_dict(params_b, params_a, align_mask)
        a_with_shared_b = interpolate_dict(params_a, params_b, align_mask)
        model_a = cls(model, params_a, b_with_shared_a)
        model_b = cls(model, a_with_shared_b, params_b)
        return model_a, model_b


def reduce(values, reduction, axis):
    if reduction == 'mean':
        return np.mean(values, axis=axis)
    if reduction == 'sum':
        return np.sum(values, axis=axis)
    return values


def barrier(values, interpolation, reduction='none'):
    barriers = [x - linear_interpolate(values[0], values[-1], a)  \
                      for x, a in zip(values, interpolation)]
    barriers = np.stack(barriers, axis=0)
    return reduce(barriers, reduction=reduction, axis=1)


def get_barrier_stats(
        model,
        dataloader,
        params_a,
        params_b,
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        resolution=11,
        reduction="mean",
        device="cuda",
):
    params_a = to_torch_device(params_a, device=device)
    params_b = to_torch_device(params_b, device=device)
    interpolation = np.linspace(0., 1., resolution)
    eval_loss, acc = [], []
    for alpha in tqdm(interpolation):
        combined = interpolate_dict(params_a, params_b, alpha)
        _, accuracy, loss = evaluate_model(model, dataloader, state_dict=combined, device=device, loss_fn=loss_fn, return_accuracy=True)
        eval_loss.append(loss)
        acc.append(accuracy)
    eval_loss = np.stack(eval_loss, axis=0)
    acc = np.stack(acc, axis=0)
    return {
        "interpolation": interpolation,
        "eval_loss": reduce(eval_loss, reduction, axis=1),
        "acc": reduce(acc, reduction, axis=1),
        "loss_barrier": barrier(eval_loss, interpolation, reduction=reduction),
        "acc_barrier": barrier(acc, interpolation, reduction=reduction)
    }
