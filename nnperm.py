"""
Weight normalization and permutation finding algorithm for MLPs and ConvNets.

Assumptions made by `nnperm`:
* BatchNorm precedes non-linearity
* for ResNets, Conv layer has no bias, but is followed by BatchNorm with bias
* if last Conv layer has X output channels, final linear layer takes X inputs
    - this means pooling/stride should reduce all image dims
* network is not going to be trained further (BatchNorm running mean/var aren't used)

Procedure:
* roll all batch norms into previous conv so that batch norms are all (1, 0)
* normalize each conv layer, taking norm over non-output dims
* apply permutation finding algorithm
* permute to canonical form
"""

from copy import deepcopy
from itertools import product
from typing import Iterable
from tqdm import tqdm
import torch
import numpy as np
from torch import nn


"""
Weight normalization
"""
def _is_weight_param(key, value) -> bool:
    return "weight" in key and value.dim() > 1

def _w_and_b(state_dict: dict, include_batchnorm=False) -> Iterable:
    w_k, w = None, None
    for k, v in state_dict.items():
        if "weight" in k:
            w_k, w = k, v
        if "bias" in k:
            assert w is not None, k
            assert w_k[:-len(".weight")] == k[:-len(".bias")]
            assert w.shape[0] == len(v), (k, w.shape, v.shape)
            if include_batchnorm or _is_weight_param(w_k, w):
                yield (w_k, w), (k, v)
            w = None

def _broadcast(source: np.ndarray, target: np.ndarray, dim: int) -> np.ndarray:
    if type(source) == torch.Tensor or type(source) == np.ndarray:
        new_shape = [1] * len(target.shape)
        new_shape[dim] = -1
        source = source.reshape(*new_shape)
    return source

def inverse_scale(scale: list) -> list:
    """Gives inverse of scaling factors.

    Args:
        scale (list): Scaling terms per layer. Each term
            is either a number (integer or float) or a ndarray-like
            with length equal to the output dimension of the layer.

    Returns:
        list: s^{-1} for scaling factors s.
    """
    return [None if s is None else 1 / s for s in scale]

def get_normalizing_scale(state_dict: dict) -> list:
    """Gets scaling terms that normalize weights to a norm of 1.

    Args:
        state_dict (dict): pytorch weights

    Returns:
        list: scaling terms per layer. Each term
            is either a number (integer or float) or a ndarray-like
            with length equal to the output dimension of the layer.
    """
    layers = list(_w_and_b(state_dict))
    scale = [1]  # temporary value
    for (w_k, w), (b_k, b) in layers:
        w = w * _broadcast(scale[-1], w, 1)
        with torch.no_grad():
            scale.append(torch.norm(w.view(w.shape[0], -1), dim=1))
    scale[-1] = None  # do not scale outputs
    return inverse_scale(scale[1:])  # remove temporary value

def scale_state_dict(state_dict: dict, scale: list, in_place=False) -> dict:
    """Scales weights of neural network.
    Should not change output of neural network.

    Args:
        state_dict (dict): pytorch weights
        scale (list): Scaling terms per layer. Each term
            is either a number (integer or float) or a ndarray-like
            with length equal to the output dimension of the layer.

    Returns:
        dict: pytorch weights, scaled
    """
    if not in_place:
        state_dict = deepcopy(state_dict)
    layers = list(_w_and_b(state_dict))
    assert len(scale) == len(layers), (len(scale), len(layers))
    s_prev = None
    for ((w_k, w), (b_k, b)), s in zip(layers, scale):
        if s_prev is not None:  # undo previous layer scale
            w = w / _broadcast(s_prev, w, 1)
        if s is not None:  # apply current scale
            w = w * _broadcast(s, w, 0)
            state_dict[b_k] = b * s
        state_dict[w_k] = w
        s_prev = s
    return state_dict

def normalize_batchnorm(state_dict: dict, in_place=False) -> dict:
    """Combines batchnorm with previous linear or convolution layer.
    Assumes there is no non-linearity between the previous layer and batchnorm.
    Sets batchnorm weights to 1 and biases to 0.
    Should not change output of neural network.

    Args:
        state_dict (dict): pytorch weights

    Returns:
        dict: pytorch weights, with batchnorm set to (1, 0)
    """
    if not in_place:
        new_state_dict = deepcopy(state_dict)
    layers = list(_w_and_b(state_dict, include_batchnorm=True))
    for i, ((w_k, w), (b_k, b)) in enumerate(layers):
        if w.dim() == 1:  # batchnorm
            # multiply with previous linear layer
            (prev_w_k, prev_w), (prev_b_k, prev_b) = layers[i-1]
            assert prev_w.dim() > 1  # not another batchnorm
            assert prev_w.shape[0] == len(w)
            new_state_dict[prev_w_k] = prev_w * _broadcast(w, prev_w, 0)
            new_state_dict[prev_b_k] = prev_b + b
            new_state_dict[w_k] = torch.ones_like(w)
            new_state_dict[b_k] = torch.zeros_like(b)
    return new_state_dict

def canonical_normalization(state_dict: dict) -> dict:
    """Normalizes weights in each layer of neural network.
    Should not change output of neural network.

    Args:
        state_dict (dict): pytorch weights

    Returns:
        dict: pytorch weights with normalized weights
    """
    state_dict = normalize_batchnorm(state_dict)
    scale = get_normalizing_scale(state_dict)
    return scale_state_dict(state_dict, scale, in_place=True), scale

"""
Weight permutation
"""
def _is_shortcut(key, value) -> bool:
    return "shortcut" in key

def _permute_layer(weight: np.ndarray, permutation: list, on_output=True) -> np.ndarray:
    if permutation is None:
        return weight
    if on_output:
        assert weight.shape[0] == len(permutation), (weight.shape, len(permutation))
        return weight[permutation, ...]
    assert weight.shape[1] == len(permutation), (weight.shape, len(permutation))
    return weight[:, permutation, ...]

def permute_state_dict(state_dict: dict, permutation: list) -> dict:
    """Permutes weights.
    Should not change output of neural network.

    Args:
        state_dict (dict): pytorch weights
        permutation (list): list of permutations per layer, each
            permutation is either None (no permutation) or a list
            of integers with length equal to the layer's output dimension.

    Returns:
        dict: pytorch weights, permuted
    """
    new_state_dict = deepcopy(state_dict)
    layers = list(_w_and_b(state_dict))
    assert len(permutation) == len(layers)
    s_prev = None
    for ((w_k, w), (b_k, b)), s in zip(layers, permutation):
        w = _permute_layer(w, s_prev, on_output=False)
        new_state_dict[w_k] = _permute_layer(w, s, on_output=True)
        new_state_dict[b_k] = _permute_layer(b, s)
        s_prev = s
    return new_state_dict

def inverse_permutation(permutation: list) -> list:
    """Gives inverse of permutation.

    Args:
        permutation (list): Permutations per layer. Each
            permutation is either None (no permutation) or a list
            of integers with length equal to the layer's output dimension.

    Returns:
        list: s^{-1} for each permutation s.
    """
    with torch.no_grad():
        return [s if s is None else torch.argsort(s) for s in permutation]

def compose_permutation(permutation_f: list, permutation_g: list) -> list:
    """Applies permutation g to f.

    Args:
        permutation_f (list): First permutation, list of permutations per layer.
            Each permutation is either None (no permutation) or a list
            of integers with length equal to the layer's output dimension.
        permutation_g (list): Second permutation (applied after f).

    Returns:
        list: f \circ g, or equivalently, g(f(\cdot)).
    """
    permutations = []
    for s, t in zip(permutation_f, permutation_g):
        if s is None:
            permutations.append(t)
        elif t is None:
            permutations.append(s)
        else:
            permutations.append(s[t])
    return permutations

def geometric_realignment(normalized_state_dict_f: dict, normalized_state_dict_g: dict,
        loss_fn=nn.MSELoss(), max_search=100, cache_permutations=True
    ) -> list:
    """Finds permutation of weights that minimizes difference between two neural nets.
    Uses Tom's algorithm (see writeup).

    Args:
        normalized_state_dict_f (dict): pytorch weights for model f, should run canonical_normalization first
        normalized_state_dict_g (dict): pytorch weights for model g, should run canonical_normalization first
        loss_fn (dict, optional): method to compute difference between weights.
            Defaults to nn.MSELoss().

    Returns:
        tuple: Permutations and loss per layer (s_f, s_g, loss),
            where s_f is the canonical permutation for model f,
            s_g is the canonical permutation for model g,
            and loss is all compared differences between the weights of the two models.
    """
    layers_f = list(_w_and_b(normalized_state_dict_f))[:-1]
    layers_g = list(_w_and_b(normalized_state_dict_g))[:-1]
    assert len(layers_f) == len(layers_g)
    s = [(None, None, float('inf'))]  # temporary
    for ((w_k, w_f), _), ((_, w_g), _) in zip(layers_f, layers_g):
        assert w_f.shape == w_g.shape
        w_f = _permute_layer(w_f, s[-1][0], on_output=False)
        w_g = _permute_layer(w_g, s[-1][1], on_output=False)
        with torch.no_grad():
            # limit number of positions to search for speed
            limited_weights = w_f.reshape(w_f.shape[0], -1)[:, 0:min(max_search, w_f.shape[1])]
            argsort_f = torch.argsort(limited_weights, axis=0).transpose(1, 0)
            argsort_g = torch.argsort(w_g.reshape(w_g.shape[0], -1), axis=0).transpose(1, 0)
            # cache permutations so they don't have to be generated every iteration
            permuted_w_f = torch.stack([_permute_layer(w_f, i) for i in argsort_f], axis=0)
            if cache_permutations:
                permuted_w_g = torch.stack([_permute_layer(w_g, i) for i in argsort_g], axis=0)
        print(f"Aligning {w_k} ({len(argsort_g)} non-output positions {len(argsort_g)**2} max pairs)...")
        best_f, best_g, best_loss = s[0]
        losses = []
        # transposed so that we iterate over non-output dims
        for i, j in tqdm(product(range(len(argsort_f)), range(len(argsort_g))), total=len(argsort_f) * len(argsort_g)):
            if cache_permutations:
                permuted_g = permuted_w_g[j]
            else:
                permuted_g = _permute_layer(w_g, argsort_g[j])
            losses.append(loss_fn(permuted_w_f[i], permuted_g))
            if losses[-1] < best_loss:
                best_f, best_g, best_loss = argsort_f[i], argsort_g[j], losses[-1]
        s.append((best_f, best_g, torch.tensor(losses)))
    permutations = s[1:] + [(None, None, 0.)] # remove temporary, add identity to last layer
    s_f, s_g, loss = list(zip(*permutations))
    return s_f, s_g, loss

def canonical_permutation(normalized_state_dict_f: dict, normalized_state_dict_g: dict, loss_fn=nn.MSELoss(), normalize=True, permute=True) -> tuple:
    """Permutes two networks to match weights as closely as possible.

    Args:
        normalized_state_dict_f (dict): pytorch weights for model f, should run canonical_normalization first
        normalized_state_dict_g (dict): pytorch weights for model g, should run canonical_normalization first
        loss_fn (dict, optional): method to compute difference between weights.
            Defaults to nn.MSELoss().

    Returns:
        tuple: permuted state dicts for f and g models.
    """
    s_f, s_g, _ = geometric_realignment(normalized_state_dict_f, normalized_state_dict_g)
    permuted_state_dict_f = permute_state_dict(normalized_state_dict_f, s_f)
    permuted_state_dict_g = permute_state_dict(normalized_state_dict_g, s_g)
    return permuted_state_dict_f, permuted_state_dict_g

"""
Random transforms
"""
def torch_chisq(x):
    return torch.randn(x)**2

def random_scale(state_dict: dict, n_layers=-1,
        scale_distribution=torch_chisq,
    ) -> list:
    scale = [scale_distribution(v.shape[0]).to(device=v.device) \
        for k, v in state_dict.items() if _is_weight_param(k, v)]
    scale[-1] = None
    if n_layers > 0:
        for i in range(n_layers, len(scale)):
            scale[i] = None
    return scale

def random_permutation(state_dict: dict, n_layers=-1) -> list:
    permutation = [torch.randperm(v.shape[0]).to(device=v.device) \
        for k, v in state_dict.items() if _is_weight_param(k, v)]
    permutation[-1] = None
    if n_layers > 0:
        for i in range(n_layers, len(permutation)):
            permutation[i] = None
    return permutation

def random_transform(state_dict: dict, scale=True, permute=True, n_layers=-1,
        scale_distribution=torch_chisq,
    ) -> dict:
    if permute:
        permutation = random_permutation(state_dict, n_layers=n_layers)
        state_dict = permute_state_dict(state_dict, permutation)
    if scale:
        scale = random_scale(state_dict, n_layers=n_layers,
                    scale_distribution=scale_distribution)
        state_dict = scale_state_dict(state_dict, scale)
    return state_dict, permutation, scale
