from copy import deepcopy
import numpy as np
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from nnperm.utils import to_numpy


def perm_to_list(perm_matrix: np.ndarray) ->  np.ndarray:
    n = perm_matrix.shape[0]
    assert len(perm_matrix.shape) == 2 and n == perm_matrix.shape[1]
    x, y = np.where(perm_matrix == 1)
    assert len(x) == n and np.all(x == np.arange(n))
    return y


def perm_to_matrix(perm: np.ndarray) -> np.ndarray:
    return np.eye(len(perm))[perm]


def perm_inverse(perm: np.ndarray) -> np.ndarray:
    return np.argsort(perm)


def perm_compose(perm_f: np.ndarray, perm_g: np.ndarray) -> np.ndarray:
    return perm_f[perm_g]


class Permutations(dict):

    @staticmethod
    def from_matrices(perm_matrices: Dict[str, np.ndarray]):
        perms = {n: perm_to_list(x) for n, x in perm_matrices.items()}
        return Permutations(perms)

    def to_matrices(self) -> Dict[str, np.ndarray]:
        return {n: perm_to_matrix(i) for n, i in self.items()}

    def inverse(self):
        """Gives inverse of permutation.

        Args:
            permutation (list): Permutations per layer. Each
                permutation is either None (no permutation) or a list
                of integers with length equal to the layer's output dimension.

        Returns:
            list: s^{-1} for each permutation s.
        """
        return {n: perm_inverse(i) for n, i in self.items()}


    def compose(self, perm_to_apply: Dict[str, np.ndarray]):
        """Applies permutation g to f.

        Args:
            permutation_f (list): First permutation, list of permutations per layer.
                Each permutation is either None (no permutation) or a list
                of integers with length equal to the layer's output dimension.
            permutation_g (list): Second permutation (applied after f).

        Returns:
            list: f \circ g, or equivalently, g(f(\cdot)).
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
    def __init__(self, axes_to_perm: Dict[str, Tuple[str, Union[None, str]]]):
        self.axes_to_perm = axes_to_perm
        self.perm_to_axes = self._get_perm_to_axes(axes_to_perm)

    def _get_perm_to_axes(self, axes_to_perm):
        perm_to_axes = defaultdict(list)
        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))
        return dict(perm_to_axes)

    @staticmethod
    def from_sequential_model(state_dict: Dict[str, nn.Module]):
        spec = {}
        perm = "Pseq_0-in"
        for i, (k, v) in enumerate(state_dict.items()):
            # assume output is always 1st dim, input is always 2nd dim
            dim_tuple = [None] * len(v.shape)
            # add input perm if exists, unless first layer
            if len(v.shape) > 1 and i > 0:
                dim_tuple[1] = perm
                perm = f"Pseq_{i}-{k}"  # new perm name for output
            # add output perm if exists
            if len(v.shape) > 0:
                dim_tuple[0] = perm
            spec[k] = dim_tuple
        # remove last perm (on outputs)
        for k, v in spec.items():
            spec[k] = tuple([None if perm == x else x for x in v])
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
            assert layer.shape[dim] == size, layer.shape
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
