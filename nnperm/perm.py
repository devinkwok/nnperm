from copy import deepcopy
from re import match
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from nnperm.utils import is_valid_key, to_numpy


def perm_from_matrix(perm_matrix: np.ndarray) ->  np.ndarray:
    n = perm_matrix.shape[0]
    assert len(perm_matrix.shape) == 2 and n == perm_matrix.shape[1]
    x, y = np.nonzero(perm_matrix)
    assert len(x) == n and np.all(x == np.arange(n))
    return y


def perm_to_matrix(perm: np.ndarray) -> np.ndarray:
    return np.eye(len(perm))[perm]


def perm_inverse(perm: np.ndarray) -> np.ndarray:
    return np.argsort(perm)


def perm_compose(perm_f: np.ndarray, perm_g: np.ndarray) -> np.ndarray:
    """Apply f, then g as $g(f(x))$ or $g \circ f$.
    """
    return perm_f[perm_g]


class Permutations(dict):

    @staticmethod
    def from_matrices(perm_matrices: Dict[str, np.ndarray]):
        return Permutations({n: perm_from_matrix(x) for n, x in perm_matrices.items()})

    def to_matrices(self) -> Dict[str, np.ndarray]:
        return Permutations({n: perm_to_matrix(i) for n, i in self.items()})

    def sizes(self):
        return {k: len(v) for k, v in self.items()}

    def fixed_points(self):
        return {k: v == np.arange(len(v)) for k, v in self.items()}

    def inverse(self):
        """Gives inverse of permutation.

        Args:
            permutation (list): Permutations per layer. Each
                permutation is either None (no permutation) or a list
                of integers with length equal to the layer's output dimension.

        Returns:
            list: s^{-1} for each permutation s.
        """
        return Permutations({n: perm_inverse(i) for n, i in self.items()})


    def compose(self, perm_to_apply: Dict[str, np.ndarray]):
        """Applies permutation g to f as $g \circ f$, where f is self.

        Args:
            perm_to_apply (list): permutation g, list of permutations per layer.
                Each permutation is either None (no permutation) or a list
                of integers with length equal to the layer's output dimension.

        Returns:
            list: f \circ g, or equivalently, f(g(\cdot)).
        """
        output = {}
        for name in set(list(self.keys()) + list(perm_to_apply.keys())):
            if name in self and name in perm_to_apply:
                output[name] = perm_compose(self[name], perm_to_apply[name])
            elif name in self:
                output[name] = self[name]
            elif name in perm_to_apply:
                output[name] = perm_to_apply[name]
        return Permutations(output)


class PermutationSpec:
    """
        axes_to_perm: str (name of layer): Tuple[ (for each dim in layer shape) Union[None (dim not permuted), str (name of permutation assigned to dim)]]
        perm_to_axes: str (names of distinct permutations): List[Tuple[str (name of layer with perm), int (dim with this perm)]]
    """
    def __init__(self,
            axes_to_perm: Dict[str, Tuple[Union[None, str]]],
            perm_to_axes: Dict[str, List[Tuple[str, int]]] = None,
    ):
        self.axes_to_perm = axes_to_perm
        if perm_to_axes is None:
            perm_to_axes = self._get_perm_to_axes(axes_to_perm)
        self.perm_to_axes = perm_to_axes

    def _get_perm_to_axes(self, axes_to_perm):
        perm_to_axes = defaultdict(list)
        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))
        return dict(perm_to_axes)

    def subset_perm(self, include_perms: List = None, exclude_perms: List = None):
        axes_to_perm = self.axes_to_perm
        perm_to_axes = {}
        for k, v in self.perm_to_axes.items():
            if is_valid_key(k, include_perms, exclude_perms):
                perm_to_axes[k] = v
        return PermutationSpec(axes_to_perm, perm_to_axes)

    def get_perm_mask(self):
        # anything that's in perm is set to 1, remainder set to 0.
        perm_mask = {}
        perm_keys = list(self.perm_to_axes.keys())
        for wk, axis_perms in self.axes_to_perm.items():
            perm_mask[wk] = 0.
            for perm in axis_perms:  # if param is permuted by anything in perm, set mask to 1s
                if perm is not None and perm in perm_keys:
                    perm_mask[wk] = 1.
        return perm_mask

    @staticmethod
    def _sequential_model_spec(state_dict: Dict[str, nn.Module], input_dim=1, output_dim=0):
        # assume output is always 1st dim, input is always 2nd dim
        spec = {}
        perm = "Pseq_0-in"
        for i, (k, v) in enumerate(state_dict.items()):
            dim_tuple = [None] * len(v.shape)
            # add input perm if exists, unless first layer
            if len(v.shape) > 1 and i > 0:  #TODO change to detect "bias" in key instead of using shape?
                dim_tuple[input_dim] = perm
                perm = f"Pseq_{i}-{k}"  # new perm name for output
            # add output perm if exists
            if len(v.shape) > 0:
                dim_tuple[output_dim] = perm
            spec[k] = dim_tuple
        # remove last perm (on outputs)
        for k, v in spec.items():
            spec[k] = [None if perm == x else x for x in v]
        return spec

    @staticmethod
    def from_sequential_model(state_dict: Dict[str, nn.Module]):
        spec = PermutationSpec._sequential_model_spec(state_dict)
        spec = {k: tuple(v) for k, v in spec.items()}
        return PermutationSpec(spec)

    @staticmethod
    def from_residual_model(state_dict: Dict[str, nn.Module], block_key="blocks\.(\\d+)\.", shortcut_key=".+\.(\d+)\.shortcut\.", input_dim=1, output_dim=0):
        """
        Assume: block_key, shortcut_key are regex that return an identifier as the first regex group.
        Note shortcut_key overrides block_key.
        Regular block has block_key params with same identifier.
        Regular block takes input from previous skip connection and outputs to same skip connection.
        Shortcut block has block_key params followed by shortcut_key params.
        Shortcut block takes input from previous skip connection and outputs to new skip connection.
        """
        spec = PermutationSpec._sequential_model_spec(state_dict, input_dim, output_dim)
        skip_perm = None
        last_shortcut = None
        last_block = None
        perm2perm = {}
        # go through sequential spec and map perm->skip perms
        for i, k in enumerate(state_dict.keys()):
            shortcut = match(shortcut_key, k)
            # entering shortcut, takes precedence over block
            if shortcut is not None:
                if last_shortcut is None or shortcut.group(1) != last_shortcut.group(1):
                    last_shortcut = shortcut
                    # detach from sequential perms, and connect directly to previous skip
                    next_skip = spec[k][input_dim]
                    spec[k][input_dim] = skip_perm
                    # map output of block (sequential input to shortcut) to next skip
                    skip_perm = f"Pskip_{i}-{shortcut.group(1)}"
                    perm2perm[spec[k][output_dim]] = skip_perm
                    perm2perm[next_skip] = skip_perm
            else:
                block = match(block_key, k)
                if block is not None:
                    # entering new block from sequential, let skip_perm be input
                    if last_block is None:
                        skip_perm = f"Pskip_{i}-{block.group(1)}"
                        perm2perm[spec[k][input_dim]] = skip_perm
                    # entering new block from another block, map input perm to skip_perm
                    elif block.group(1) != last_block.group(1):
                        perm2perm[spec[k][input_dim]] = skip_perm
                else:  # regular sequential layer
                    if last_block is not None:   # leaving block, map input perm to skip_perm
                        perm2perm[spec[k][input_dim]] = skip_perm
                last_block = block
        for k, v in spec.items():  # replace every name from perm2perm so that key->value 
            spec[k] = tuple([perm2perm[p] if p in perm2perm else p for p in v])
        return PermutationSpec(spec)

    def _generate_transform(self, gen_fn: callable, state_dict: Dict[str, np.ndarray]):
        """
        Args:
            gen_fn (callable): a function which produces an object based on the following
                layers: List[Tuple[np.ndarray (params), int (dim)]]
            state_dict (Dict[str, np.ndarray]): parameters of model

        Returns:
            Dict[str, obj]: mapping from name of permutation to transform object
        """
        output = {}
        for p, axes in self.perm_to_axes.items():
            output[p] = gen_fn([(state_dict[layer_name], dim) for layer_name, dim in axes])
        return output

    @staticmethod
    def layer_size(params: List[Tuple[np.ndarray, int]]):
        size = params[0][0].shape[params[0][1]]
        for layer, dim in params:
            assert layer.shape[dim] == size, (layer.shape, dim, size)
        return size

    def get_sizes(self, state_dict: Dict[str, np.ndarray]):
        return self._generate_transform(self.layer_size, state_dict)

    def get_random_permutation(self, state_dict: Dict[str, np.ndarray], random_state=None):
        random_state = np.random.RandomState(42) if random_state is None else random_state
        rand_perm_fn = lambda p: random_state.permutation(self.layer_size(p))
        return Permutations(self._generate_transform(rand_perm_fn, state_dict))

    def get_identity_permutation(self, state_dict: Dict[str, np.ndarray]):
        identity_perm_fn = lambda p: np.arange(self.layer_size(p))
        return Permutations(self._generate_transform(identity_perm_fn, state_dict))

    def _transform_state_dict(self, apply_fn: callable, state_dict: Dict[str, np.ndarray], transform_dict: Dict[str, object]) -> Dict[str, np.ndarray]:
        """
        Args:
            apply_fn (callable): a function which applies transform to parameter at dim, with signature
                transform: object, parameter: np.ndarray, dim: int
            state_dict (Dict[str, np.ndarray]): parameters of model
            transform_dict (Dict[str, object]): name of permutation to transformation object

        Returns:
            Dict[str, np.ndarray]: copy of state dict with transformation applied
        """
        output = deepcopy(to_numpy(state_dict))
        for p, transform in transform_dict.items():
            for layer_name, dim in self.perm_to_axes[p]:
                output[layer_name] = apply_fn(transform, output[layer_name], dim)
        return output

    @staticmethod
    def permute_layer(permutation: List[int], param: np.ndarray, dim) -> np.ndarray:
        if permutation is None:
            return param
        assert param.shape[dim] == len(permutation), (param.shape, len(permutation))
        return np.take(param, permutation, axis=dim)

    def apply_permutation(self, state_dict: Dict[str, np.ndarray], permutations: Dict[str, List[int]]):
        return self._transform_state_dict(self.permute_layer, state_dict, permutations)

    def apply_rand_perm(self, state_dict: Dict[str, np.ndarray]):
        return self.apply_permutation(state_dict, self.get_random_permutation(state_dict))

    @staticmethod
    def append_zeros(n: int, param: np.ndarray, dim) -> np.ndarray:
        pad_dims = [(0, 0)] * len(param.shape)
        assert n - param.shape[dim] >= 0
        pad_dims[dim] = (0, n - param.shape[dim])
        return np.pad(param, pad_dims, mode="constant")

    def apply_padding(self, state_dict: Dict[str, np.ndarray], target_size: Dict[str, int]):
        return self._transform_state_dict(self.append_zeros, state_dict, target_size)

    def save_to_file(self, permutations: Dict[str, List[int]], file: str):
        torch.save({
            "permutations": dict(permutations),  # save Permutations object as dict
            "axes_to_perm": self.axes_to_perm,
            "perm_to_axes": self.perm_to_axes,
        }, file)

    @staticmethod
    def load_from_file(file: str):
        data = torch.load(file)
        perm = Permutations(data["permutations"])
        perm_spec = PermutationSpec(data["axes_to_perm"], data["perm_to_axes"])
        return perm, perm_spec
