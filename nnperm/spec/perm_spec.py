import numpy as np
import torch
from typing import Dict, List

from nnperm.perm import Permutations
from nnperm.spec.model_spec import ModelSpec


class PermutationSpec(ModelSpec):
    """
        axes_to_group: str (name of layer): Tuple[ (for each dim in layer shape) Union[None (dim not permuted), Tuple[str (name of perm assigned to dim), bool (if input)]]]
        group_to_axes: str (names of distinct perms): List[Tuple[str (name of layer with perm), int (dim with this perm), bool (if input)]]
    """
    def get_random_permutation(self, state_dict: Dict[str, np.ndarray], random_state=None):
        random_state = np.random.RandomState(42) if random_state is None else random_state
        rand_perm_fn = lambda p: random_state.permutation(self.layer_size(p))
        return Permutations(self._generate_transform(rand_perm_fn, state_dict))

    def get_identity_permutation(self, state_dict: Dict[str, np.ndarray]):
        identity_perm_fn = lambda p: np.arange(self.layer_size(p))
        return Permutations(self._generate_transform(identity_perm_fn, state_dict))

    @staticmethod
    def permute_layer(permutation: List[int], param: np.ndarray, dim, is_input=False) -> np.ndarray:
        if permutation is None:
            return param
        assert param.shape[dim] == len(permutation), (param.shape, len(permutation))
        return np.take(param, permutation, axis=dim)

    def apply_permutation(self, state_dict: Dict[str, np.ndarray], permutations: Dict[str, List[int]]):
        return self._transform_state_dict(self.permute_layer, state_dict, permutations)

    def apply_rand_perm(self, state_dict: Dict[str, np.ndarray]):
        return self.apply_permutation(state_dict, self.get_random_permutation(state_dict))

    @staticmethod
    def append_zeros(n: int, param: np.ndarray, dim, is_input=False) -> np.ndarray:
        pad_dims = [(0, 0)] * len(param.shape)
        assert n - param.shape[dim] >= 0
        pad_dims[dim] = (0, n - param.shape[dim])
        return np.pad(param, pad_dims, mode="constant")

    def apply_padding(self, state_dict: Dict[str, np.ndarray], target_size: Dict[str, int]):
        return self._transform_state_dict(self.append_zeros, state_dict, target_size)

    def save_to_file(self, permutations: Dict[str, List[int]], file: str):
        torch.save({
            "permutations": dict(permutations),  # save Permutations object as dict
            "axes_to_group": self.axes_to_group,
            "group_to_axes": self.group_to_axes,
        }, file)

    @staticmethod
    def load_from_file(file: str):
        data = torch.load(file)
        perm = Permutations(data["permutations"])
        if "axes_to_perm" in data:  #TODO deprecated: load old spec when is_input is missing
            INPUT_DIM = 1  # assume input_dim=1 and output_dim=0
            axes_to_group, group_to_axes = {}, {}
            for k, axes in data["axes_to_perm"].items():
                axes_to_group[k] = [v if v is None else (v, i == INPUT_DIM) for i, v in enumerate(axes)]
            for k, (name, dim) in data["perm_to_axes"].items():
                group_to_axes[k] = (name, dim, dim == INPUT_DIM)
        else:
            axes_to_group = data["axes_to_group"]
            group_to_axes = data["group_to_axes"]
        perm_spec = PermutationSpec(axes_to_group, group_to_axes)
        return perm, perm_spec
