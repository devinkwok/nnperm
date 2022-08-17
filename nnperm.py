"""
Weight normalization and permutation finding algorithm for MLPs and ConvNets.

Assumptions made by `nnperm`:
* BatchNorm precedes non-linearity
* for ResNets:
    - Conv layer has no bias, but is followed by BatchNorm with bias
    - shortcut connections always have weights (if not transformed, set as identity matrix without bias)
    - the first shortcut points to the output of the first layer, subsequent shortcuts point to output of previous shortcut
    - shortcuts apply an optional linear transform, then are added to the output of the previous (block) layer
* only 1 level of shortcut connections allowed, each shortcut layer takes from output of previous shortcut layer (or input)
* if last Conv layer has X output channels, final linear layer takes X inputs
    - this means pooling/stride should reduce all image dims
* network is not going to be trained further (BatchNorm running mean/var aren't used)
* if using cache=True in get_normalizing_permutation, loss function is applied elementwise (e.g. MSE or MAE loss)

Procedure:
* roll all batch norms into previous conv so that batch norms are all (1, 0)
* normalize each conv layer, taking norm over non-output dims
* apply permutation finding algorithm (note, only evaluates weight similarity, not bias)
* permute to canonical form
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import json
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
from collections import deque
from tqdm import tqdm
import torch
import numpy as np
from torch import nn


@dataclass
class Param:
    state_dict: Dict[str, torch.Tensor]
    size: torch.Size
    name: str
    dim_in: Optional[int] = None
    dim_out: Optional[int] = None

    def is_scalable(self):
        return self.name not in self.state_dict

    def is_permutable(self):
        return self.dim_out is not None

    @property
    def value(self):
        return self.state_dict[self.name]

    def is_batchnorm(self) -> bool:
        return "weight" in self.name and self.value.dim() == 1

    def is_weight_param(self) -> bool:
        return "weight" in self.name and len(self.size) > 1

    def is_identity(self) -> bool:
        with torch.no_grad():
            v = self.value.squeeze()
            return len(v.shape) == 2 and v.shape[0] == v.shape[1] and \
                torch.all(v == torch.eye(v.shape[0], device=v.device)).item()


class Node:
    def __init__(self, param: Param, parents: Set[Node] = None, children: Set[Node] = None) -> None:
        self.param = param
        self.parents = set() if parents is None else parents
        self.children = set() if children is None else children
        for parent in self.parents:
            parent.children.add(self)
        for child in self.children:
            child.parents.add(self)

    @property
    def name(self):
        return self.param.name

    def __repr__(self) -> str:
        return self.name + '\t[' + ','.join([p.name for p in self.parents]) + ']\t[' + ','.join([c.name for c in self.children]) + ']'


class ComputationGraph:

    def __init__(self, nodes: Dict[str, Node]) -> None:
        self.nodes = nodes
        self.input = None
        self.output = None
        for v in nodes.values():
            if len(v.parents) == 0:
                assert self.input is None
                self.input = v
            if len(v.children) == 0:
                assert self.output is None, nodes
                self.output = v

    @staticmethod
    def from_dict(state_dict, graph_dict):
        nodes = {}
        for key, v in graph_dict.items():
            assert key not in nodes
            parents = {nodes[k] for k in v.get('parents', [])}
            if key in state_dict:
                size = state_dict[key].shape
                dim_out = 0
                if len(size) > 1:
                    dim_in = 1
                else:
                    dim_in = 0
            else:  # special layers (input, output)
                size = v.get('shape', None)
                dim_out = None
                dim_in = None
            param = Param(state_dict, size, key, dim_in=dim_in, dim_out=dim_out)
            nodes[key] = Node(param, parents=parents)
        return ComputationGraph(nodes)

    @staticmethod
    def from_file(state_dict, file):
        with open(file, 'r') as f:
            graph_dict = json.load(f)
        return ComputationGraph.from_dict(state_dict, graph_dict)

    def to_dict(self):
        graph_dict = {}
        def append_to_graph(node):
            graph_dict[node.name] = [child.name for child in node.children]
        self.recursive_apply(append_to_graph, self.input)
        return graph_dict

    def to_file(self, file):
        with open(file, 'w') as f:
            json.dump(self.to_dict(), f)

    def recursive_apply(self, apply_fn: Callable, curr_node: Node):
        apply_fn(curr_node)
        for child in curr_node.children:
            self.recursive_apply(apply_fn, child)

    def __repr__(self):
        strings = []
        for node in self.nodes.items():
            strings.append(str(node))
        return '\n\t'.join(f'{i} {x}' for i, x in enumerate(strings))


# a group of weights that share a transform
@dataclass
class PermutationSet:
    params: List[Tuple[bool, int, Param]]  # use in loss, dim, param


"""
Helper functions
"""

def _w_and_b(state_dict: dict, include_batchnorm=False) -> Iterable:
    prev_w = deque()
    for k, v in state_dict.items():
        if "weight" in k:
            if len(prev_w) > 1:
                yield prev_w.popleft(), (None, None)
            prev_w.append((k, v))
        if "bias" in k:
            assert len(prev_w) > 0
            # if weight layer is missing bias, yield weight and batchnorm bias, then yield batchnorm weight alone
            for w_k, w in prev_w:
                if include_batchnorm or _is_weight_param(w_k, w):
                    if v is not None and w.shape[0] == len(v):
                        yield (w_k, w), (k, v)
                        k, v = None, None  # only yield bias once
                    else:
                        yield (w_k, w), (None, None)
            prev_w.clear()

def _generate_transform(gen_fn, state_dict, n_layers=-1, transform_shortcut="none") -> list:
    transform = []
    shortcut_idx = 0
    for k, v in state_dict.items():
        if _is_weight_param(k, v):
            if _is_skip(k, v):
                # 1. do not transform previous block layer or shorcut
                if transform_shortcut == "none":
                    if _is_identity(k, v):  # if shortcut is identity, pass through shortcut transform
                        # transform[-1] = transform[shortcut_idx]  # cancel previous layer transform, apply shortcut transform instead to match output of shortcut layer
                        # transform.append(transform[shortcut_idx])
                        pass
                    else:  # if shortcut has weights, reset shortcut transform
                        # transform[-1] = None
                        shortcut_idx = len(transform)
                        # transform.append(None)
                    transform[-1] = None
                    transform.append(None)
                # 2. transform if shortcut has weights
                elif transform_shortcut == "shortcut":
                    if _is_identity(k, v):  # if shortcut is identity, pass through previous shortcut transform
                        # transform[-1] = transform[shortcut_idx]
                        # transform.append(shortcut_transform)
                        transform[-1] = None
                        transform.append(None)
                    else:
                        shortcut_idx = len(transform)
                        transform.append(transform[-1])  # if shortcut has weights, copy previous transform and reset shortcut transform
                # 3. transform all shortcuts
                elif transform_shortcut == "block":
                    shortcut_idx = len(transform)
                    transform.append(transform[-1])  # copy previous transform
                else:
                    raise ValueError(f"transform_shortcut must be one of 'none', 'shortcut', or 'block': {transform_shortcut}")
            else:
                with torch.no_grad():
                    transform.append(gen_fn(v).to(device=v.device))
    transform[-1] = None
    if n_layers > 0:
        for i in range(n_layers, len(transform)):
            transform[i] = None
    return transform

def _transform_state_dict(apply_fn, state_dict, transform, in_place=False):
    if not in_place:
        state_dict = deepcopy(state_dict)
    layers = list(_w_and_b(state_dict))
    assert len(transform) == len(layers), (len(transform), len(layers))
    shortcut_prev = transform[0]  #TODO this is a temporary hack to account for first conv layer in ResNet
    s_prev = None
    with torch.no_grad():
        for i in range(len(layers)):
            (w_k, w), (b_k, b) = layers[i]
            s = transform[i]
            if i < len(layers) - 1:
                if s is None and _is_skip(*layers[i + 1][0]) and \
                        _is_identity(*layers[i + 1][0], *layers[i + 1][1]):
                    s = shortcut_prev
            # 1. shortcut is identity, apply shortcut_prev to input of next layer
            if _is_identity(w_k, w, b_k, b):
                s_prev = shortcut_prev
            else:
                # 2. shortcut has weights, apply shortcut_prev to input of shortcut, s to input of next layer
                if _is_skip(w_k, w):
                    s_prev = shortcut_prev  # apply previous shortcut transform to input
                    shortcut_prev = s  # save current transform for next shortcut layer
                if s_prev is not None:  # undo previous layer transform
                    w = apply_fn(w, s_prev, False)
                if s is not None:  # apply current transform
                    w = apply_fn(w, s, True)
                    state_dict[b_k] = apply_fn(b, s, True)
                state_dict[w_k] = w
                s_prev = s
    return state_dict

def _align_state_dict(align_fn, *state_dicts, align_shortcut="none"):
    layers = [list(_w_and_b(d)) for d in state_dicts]
    first_model = layers[0]  # use this model to test layer types
    if len(layers) > 1:
        for model in layers[1:]:
            assert len(first_model) == len(model)
            for ((key_f, w_f), _), ((key_g, w_g), _) in zip(first_model, model):
                assert key_f == key_g
                assert w_f.shape == w_g.shape
    layers = list(zip(*layers))  # iterate per layer over all models
    s = [None]  # temporary
    shortcut_idx = 1
    with torch.no_grad():
        for i in range(len(layers) - 1):
            # 1. current layer is shortcut, copy previous permutation
            if _is_skip(*first_model[i][0]):
                s.append(s[-1])
                if not _is_identity(*first_model[i][0], *first_model[i][1]):
                    shortcut_idx = i  # if shortcut has weights, move shortcut transform to here
            elif i < len(layers) - 1 and _is_skip(*first_model[i + 1][0]):
                # 2. next layer is shortcut, do not permute weights
                if align_shortcut == "none":
                    s.append(align_fn(*layers[i], s_prev=s[-1], do_align=False))
                # 3. next layer is shortcut, use shortcut to permute weights if not identity
                elif align_shortcut == "shortcut":
                    if _is_identity(*first_model[i + 1][0], *first_model[i + 1][1]):
                        s.append(align_fn(*layers[i], s_prev=s[-1], do_align=False))
                    else:
                        s.append(align_fn(*layers[i + 1], s_prev=s[shortcut_idx]))
                # 4. permute using block weights
                elif align_shortcut == "block":
                    s.append(align_fn(*layers[i], s_prev=s[-1]))
                else:
                    raise ValueError(f"align_shortcut must be one of 'none', 'shortcut', or 'block': {align_shortcut}")
            else:  # if previous layer is shortcut, need to apply shortcut transform
                if i > 0 and _is_skip(*first_model[i - 1][0]):
                    s.append(align_fn(*layers[i], s_prev=s[shortcut_idx]))
                else:
                    s.append(align_fn(*layers[i], s_prev=s[-1]))
                    
    # remove temporary, add identity to last layer
    transform = s[1:] + [align_fn(*layers[-1], s_prev=s[-1], do_align=False)]
    return transform


"""
Weight normalization
"""
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

def get_normalizing_scale(state_dict: dict, align_shortcut="none") -> list:
    """Gets scaling terms that normalize weights to a norm of 1.

    Args:
        state_dict (dict): pytorch weights

    Returns:
        list: scaling terms per layer. Each term
            is either a number (integer or float) or a ndarray-like
            with length equal to the output dimension of the layer.
    """
    def _norm_layer_fn(layer, s_prev=None, do_align=True):
        if not do_align:
            return None
        (_, w), _ = layer
        if s_prev is not None:
            w = w / _broadcast(s_prev, w, 1)
        return 1 / torch.norm(w.view(w.shape[0], -1), dim=1)

    return _align_state_dict(_norm_layer_fn, state_dict, align_shortcut=align_shortcut)

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
    return _transform_state_dict(lambda w, s, out: \
        w * _broadcast(s, w, 0) if out else w / _broadcast(s, w, 1),
        state_dict, scale, in_place=in_place)

def normalize_batchnorm(graph: ComputationGraph, in_place=False) -> dict:
    """Combines batchnorm with previous linear or convolution layer.
    Assumes there is no non-linearity between the previous layer and batchnorm.
    Sets batchnorm weights to 1 and biases to 0.
    Should not change output of neural network.

    Args:
        state_dict (dict): pytorch weights

    Returns:
        dict: pytorch weights, with batchnorm set to (1, 0)
    """
    nodes = [graph.output]
    while len(nodes) > 0:
        curr_node = nodes.pop()
        if curr_node.param.is_batchnorm():
            batchnorm = curr_node
            assert len(curr_node.parents) == 1
            curr_node.parents[0]
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
            new_state_dict[w_k] = torch.ones_like(w)
            if b is not None:
                new_state_dict[prev_b_k] = prev_b + b
                new_state_dict[b_k] = torch.zeros_like(b)
    return new_state_dict

def canonical_normalization(state_dict: dict, align_shortcut="none") -> dict:
    """Normalizes weights in each layer of neural network.
    Should not change output of neural network.

    Args:
        state_dict (dict): pytorch weights

    Returns:
        dict: pytorch weights with normalized weights
    """
    state_dict = normalize_batchnorm(state_dict)
    scale = get_normalizing_scale(state_dict, align_shortcut=align_shortcut)
    return scale_state_dict(state_dict, scale, in_place=True), scale


"""
Weight permutation
"""
def _permute_layer(weight: np.ndarray, permutation: list, on_output=True) -> np.ndarray:
    if permutation is None:
        return weight
    if on_output:
        assert weight.shape[0] == len(permutation), (weight.shape, len(permutation))
        return weight[permutation, ...]
    assert weight.shape[1] == len(permutation), (weight.shape, len(permutation))
    return weight[:, permutation, ...]

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

def permute_state_dict(state_dict: dict, permutation: list, in_place=False) -> dict:
    """Permutes weights.
    Should not change output of neural network.
    Shortcuts should have the same permutation as the previous block layer
    (if not permuted, set to None for both layers).

    Args:
        state_dict (dict): pytorch weights
        permutation (list): list of permutations per layer, each
            permutation is either None (no permutation) or a list
            of integers with length equal to the layer's output dimension.

    Returns:
        dict: pytorch weights, permuted
    """
    return _transform_state_dict(_permute_layer, state_dict, permutation, in_place=in_place)

def _geometric_realignment(layers_f, layers_g, s_prev=None, do_align=True,
        loss_fn=nn.MSELoss(), max_search=-1, keep_loss="summary", cache=True
    ):
    (w_k, w_f), _ = layers_f
    (_, w_g), _ = layers_g
    if s_prev is not None:
        w_f = _permute_layer(w_f, s_prev[0], on_output=False)
        w_g = _permute_layer(w_g, s_prev[1], on_output=False)
    if not do_align:
        return None, None, loss_fn(w_f, w_g)
    w_f = w_f.reshape(w_f.shape[0], -1)
    w_g = w_g.reshape(w_g.shape[0], -1)
    print(f"Aligning {w_k} ({w_f.shape[1]} non-output positions {w_f.shape[1]**2} max pairs)...")
    if not (keep_loss == "all" or keep_loss == "summary" or keep_loss == "single"):
        raise ValueError(f"Invalid keep_loss {keep_loss}")
    with torch.no_grad():
        truncated_f = w_f[:, 0:min(max_search, w_f.shape[1])]
        argsort_f = torch.argsort(truncated_f, axis=0).T
        argsort_g = torch.argsort(w_g, axis=0).T  # transpose to iterate over non-output dims
        if cache:  # cache losses so they don't have to be generated every iteration
            losses_per_pair = torch.empty(len(w_f), len(w_g))
            for i, j in tqdm(product(range(len(w_f)), range(len(w_g))
                            ), total=len(w_f)*len(w_g)):
                losses_per_pair[i, j] = loss_fn(w_f[i], w_g[j])
        # go through all pairs of positions, saving permutation with lowest loss
        best_f, best_g, best_loss = None, None, float('inf')
        loss_stats = []
        for i in tqdm(range(len(argsort_f))):
            if not cache:  # (not cached) generate each f permutation once only
                permuted_f = _permute_layer(w_f, argsort_f[i])
            if keep_loss != "single":
                losses = torch.empty(len(argsort_g))
            for j in range(len(argsort_g)):
                if cache:  # (cached) look up pairwise losses using permutations
                    loss = torch.mean(losses_per_pair[argsort_f[i], argsort_g[j]])
                else:  # (not cached) permute g and calculate loss
                    permuted_g = _permute_layer(w_g, argsort_g[j])
                    loss = loss_fn(permuted_f, permuted_g)
                if loss < best_loss:
                    best_f, best_g, best_loss = argsort_f[i], argsort_g[j], loss
                if keep_loss != "single":
                    losses[j] = loss
            if keep_loss == "all":
                loss_stats.append(losses)
            elif keep_loss == "summary":
                loss_stats.append(torch.tensor([torch.min(losses), *torch.std_mean(losses), torch.max(losses)]))
        if keep_loss == "single":
            loss_stats = [best_loss]
    return best_f, best_g, torch.stack(loss_stats)

def get_normalizing_permutation(normalized_state_dict_f: dict, normalized_state_dict_g: dict,
        loss_fn=nn.MSELoss(), max_search=-1, cache=True, align_shortcut="none", keep_loss="summary",
    ) -> list:
    """Finds permutation of weights that minimizes difference between two neural nets.
    Uses Tom's algorithm (see writeup).

    Realignment options for shortcut connections:
    1. "none": do not align layers before shortcut connection
    2. "shortcut": align layers using shortcut weights (if present), block weights before shortcut connection copy this permutation
    3. "block": align layers using block weights, shortcut weights copy this permutation

    Args:
        normalized_state_dict_f (dict): pytorch weights for model f, should run canonical_normalization first
        normalized_state_dict_g (dict): pytorch weights for model g, should run canonical_normalization first
        loss_fn (callable, optional): method to compute difference between weights.
            Defaults to nn.MSELoss().
        max_search (int): Number of weights to search. Actual number of pairs searched is
            min(max_search, total_weights) * total_weights. Defaults to -1 (search all pairs).
        cache (bool): If True, use memory to store losses per output pair (faster),
            otherwise computes permutations and losses on the fly. Defaults to True.
        align_shortcut (str): Controls how block weights in layer before shortcut connections are aligned.
            If "none", do not align such weights. If "shortcut", use shortcut layer weights if present
            (e.g. downsampling layer) for alignment, and do not align if weights are missing or identity.
            If "block", align using weights of the block layer, and apply to shortcut layer weights.
            Defaults to "none".
        keep_loss (str): If "all", return all pairwise losses. If "summary",
            return min, mean, std, max for each search position in f.
            If "none" return min, mean, std, max over entire search.

    Returns:
        tuple: Permutations and loss per layer (s_f, s_g, loss),
            where s_f is the canonical permutation for model f,
            s_g is the canonical permutation for model g,
            and loss is all compared differences between the weights of the two models.
    """
    def _wrapper(layer_f, layer_g, s_prev=None, do_align=True):
        return _geometric_realignment(layer_f, layer_g, s_prev=s_prev, do_align=do_align,
            loss_fn=loss_fn, max_search=max_search, keep_loss=keep_loss, cache=cache)

    permutations = _align_state_dict(_wrapper, normalized_state_dict_f,
            normalized_state_dict_g, align_shortcut=align_shortcut)
    return list(zip(*permutations))

def canonical_permutation(normalized_state_dict_f: dict, normalized_state_dict_g: dict,
        loss_fn=nn.MSELoss(), normalize=True, permute=True, align_shortcut="none",
    ) -> tuple:
    """Permutes two networks to match weights as closely as possible.

    Args:
        normalized_state_dict_f (dict): pytorch weights for model f, should run canonical_normalization first
        normalized_state_dict_g (dict): pytorch weights for model g, should run canonical_normalization first
        loss_fn (dict, optional): method to compute difference between weights.
            Defaults to nn.MSELoss().

    Returns:
        tuple: permuted state dicts for f and g models.
    """
    s_f, s_g, _ = get_normalizing_permutation(normalized_state_dict_f,
            normalized_state_dict_g, align_shortcut=align_shortcut)
    permuted_state_dict_f = permute_state_dict(normalized_state_dict_f, s_f)
    permuted_state_dict_g = permute_state_dict(normalized_state_dict_g, s_g)
    return permuted_state_dict_f, permuted_state_dict_g

"""
Random transforms
"""
def torch_chisq(x):
    return torch.randn(x)**2

def random_scale(state_dict: dict, n_layers=-1,
        scale_distribution=torch_chisq, scale_shortcut="none",
    ) -> list:
    return _generate_transform(lambda x: scale_distribution(x.shape[0]),
        state_dict, n_layers=n_layers, transform_shortcut=scale_shortcut)

def random_permutation(state_dict: dict, n_layers=-1, permute_shortcut="none") -> list:
    return _generate_transform(lambda x: torch.randperm(x.shape[0]),
        state_dict, n_layers=n_layers, transform_shortcut=permute_shortcut)

def random_transform(state_dict: dict, scale=True, permute=True, n_layers=-1,
        scale_distribution=torch_chisq, transform_shortcut="none",
    ) -> dict:
    if permute:
        permutation = random_permutation(state_dict, n_layers=n_layers,
                                    transform_shortcut=transform_shortcut)
        state_dict = permute_state_dict(state_dict, permutation)
    if scale:
        scale = random_scale(state_dict, n_layers=n_layers,
                    scale_distribution=scale_distribution)
        state_dict = scale_state_dict(state_dict, scale)
    return state_dict, permutation, scale
