from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from nnperm.utils import to_torch_device


class EnsembleModel(nn.Module):
    def __init__(self, model, params_a, params_b, out_weight_a, out_weight_b):
        super().__init__()
        self.model_a = deepcopy(model).load_state_dict(deepcopy(params_a))
        self.model_b = deepcopy(model).load_state_dict(deepcopy(params_b))
        self.out_weight_a = out_weight_a
        self.out_weight_b = out_weight_b

    def forward(self, x):
        y_a = self.model_a(x)
        y_b = self.model_b(x)
        return self.out_weight_a * y_a + self.out_weight_b * y_b

# weight split for interpolation:
# alpha, model_1, model_2
# 0, A_shared|A_unique, A_shared|B_unique
# 1, B_shared|A_unique, B_shared|B_unique


def interpolate_params(alpha, params_a, params_b):
    return {k: alpha * params_a[k] + (1 - alpha) * params_b[k]  \
            for k in params_a.keys()}


def evaluate(model,
        dataloader,
        state_dict=None,
        loss_fn=lambda x, l: x,  # default is identity
        device="cuda",
        return_accuracy=False,
):
    if state_dict is not None:
        model = deepcopy(model)
        state_dict = to_torch_device(state_dict, device)
        model.load_state_dict(state_dict)
    model.eval()
    loss, acc = [], []
    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device=device)
            labels = labels.to(device=device)
            y = model(x)
            if return_accuracy:
                accuracy = torch.argmax(y, dim=-1) == labels
                acc.append(accuracy.cpu().detach().numpy())
            loss.append(loss_fn(y, labels).cpu().detach().numpy())
    if return_accuracy:
        return np.concatenate(loss), np.concatenate(acc)
    else:
        return np.concatenate(loss)


def reduce(values, reduction, axis):
    if reduction == 'mean':
        return np.mean(values, axis=axis)
    if reduction == 'sum':
        return np.sum(values, axis=axis)
    return values


def barrier(values, interpolation, reduction='none'):
    barriers = [x - ((1 - a) * values[0] + a * values[-1]) \
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
    params_a = to_torch_device(params_a)
    params_b = to_torch_device(params_b)
    interpolation = np.linspace(0., 1., resolution)
    eval_loss, acc = [], []
    for alpha in tqdm(interpolation):
        combined = interpolate_params(alpha, params_a, params_b)
        l, a = evaluate(model, dataloader, state_dict=combined,
                        loss_fn=loss_fn, device=device, return_accuracy=True)
        eval_loss.append(l)
        acc.append(a)
    eval_loss = np.stack(eval_loss, axis=0)
    acc = np.stack(acc, axis=0)
    return {
        "interpolation": interpolation,
        "eval_loss": reduce(eval_loss, reduction, axis=1),
        "acc": reduce(acc, reduction, axis=1),
        "loss_barrier": barrier(eval_loss, interpolation, reduction=reduction),
        "acc_barrier": barrier(acc, interpolation, reduction=reduction)
    }
