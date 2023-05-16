from collections import defaultdict
from copy import deepcopy
from re import match
import numpy as np
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from nnperm.utils import is_valid_key, to_numpy


def sequential_model_spec(state_dict: Dict[str, nn.Module], input_dim=1, output_dim=0):
    # assume output is always 1st dim, input is always 2nd dim
    spec = {}
    group = f"seq_0-in"
    for i, (k, v) in enumerate(state_dict.items()):
        dim_tuple = [None] * len(v.shape)
        # add input group if exists, unless first layer
        if len(v.shape) > 1 and i > 0:
            dim_tuple[input_dim] = (group, True)
            group = f"seq_{i}-{k}"  # new group name for output
        # add output group if exists
        if len(v.shape) > 0:
            dim_tuple[output_dim] = (group, False)
        spec[k] = dim_tuple
    # remove last group (on outputs)
    for k, v in spec.items():
        spec[k] = [None if x == (group, False) else x for x in v]
    return spec


def residual_model_spec(state_dict: Dict[str, nn.Module], block_key="blocks\.(\\d+)\.", shortcut_key=".+\.(\d+)\.shortcut\.", input_dim=1, output_dim=0):
    """
    Assume: block_key, shortcut_key are regex that return an identifier as the first regex group.
    Note shortcut_key overrides block_key.
    Regular block has block_key params with same identifier.
    Regular block takes input from previous skip connection and outputs to same skip connection.
    Shortcut block has block_key params followed by shortcut_key params.
    Shortcut block takes input from previous skip connection and outputs to new skip connection.
    """
    spec = sequential_model_spec(state_dict, input_dim, output_dim)
    skip_group = None
    last_shortcut = None
    last_block = None
    group2group = {}
    # go through sequential spec and map group->skip group
    for i, k in enumerate(state_dict.keys()):
        shortcut = match(shortcut_key, k)
        # entering shortcut, takes precedence over block
        if shortcut is not None:
            if last_shortcut is None or shortcut.group(1) != last_shortcut.group(1):
                last_shortcut = shortcut
                # detach from sequential groups, and connect directly to previous skip
                next_skip = spec[k][input_dim][0]
                spec[k][input_dim] = (skip_group, True)
                # map output of block (sequential input to shortcut) to next skip
                skip_group = f"skip_{i}-{shortcut.group(1)}"
                group2group[spec[k][output_dim][0]] = skip_group
                group2group[next_skip] = skip_group
        else:
            block = match(block_key, k)
            if block is not None:
                # entering new block from sequential, let skip_group be input
                if last_block is None:
                    skip_group = f"skip_{i}-{block.group(1)}"
                    group2group[spec[k][input_dim][0]] = skip_group
            else:  # regular sequential layer
                if last_block is not None:   # leaving block, map input group to skip_group
                    group2group[spec[k][input_dim][0]] = skip_group
            last_block = block
    # replace every name from group2group so that key->value
    for k, v in spec.items():
        axes = []
        for i, x in enumerate(v):
            if x is not None and x[0] in group2group:
                axes.append((group2group[x[0]], x[1]))
            else:
                axes.append(x)
        spec[k] = tuple(axes)
    return spec


class ModelSpec:
    """
        axes_to_group: str (name of layer): Tuple[ (for each dim in layer shape) Union[None (dim not grouped), Tuple[str (name of group assigned to dim), bool (if input)]]]
        group_to_axes: str (names of distinct groups): List[Tuple[str (name of layer with group), int (dim with this group), bool (if input)]]
    """
    def __init__(self,
            axes_to_group: Dict[str, Tuple[Union[None, str]]],
            group_to_axes: Dict[str, List[Tuple[str, int]]] = None,
    ):
        self.axes_to_group = axes_to_group
        if group_to_axes is None:
            group_to_axes = self._get_group_to_axes(axes_to_group)
        self.group_to_axes = group_to_axes

    def _get_group_to_axes(self, axes_to_group):
        group_to_axes = defaultdict(list)
        for wk, axis_groups in axes_to_group.items():
            for axis, group in enumerate(axis_groups):
                if group is not None:
                    name, is_input = group
                    group_to_axes[name].append((wk, axis, is_input))
        return dict(group_to_axes)

    @classmethod
    def from_sequential_model(cls, state_dict: Dict[str, nn.Module], input_dim=1, output_dim=0):
        spec = sequential_model_spec(state_dict, input_dim=input_dim, output_dim=output_dim)
        # make lists into tuples
        spec = {k: tuple(v) for k, v in spec.items()}
        return cls(spec)

    @classmethod
    def from_residual_model(cls, state_dict: Dict[str, nn.Module], block_key="blocks\.(\\d+)\.", shortcut_key=".+\.(\d+)\.shortcut\.", input_dim=1, output_dim=0):
        """
        Assume: block_key, shortcut_key are regex that return an identifier as the first regex group.
        Note shortcut_key overrides block_key.
        Regular block has block_key params with same identifier.
        Regular block takes input from previous skip connection and outputs to same skip connection.
        Shortcut block has block_key params followed by shortcut_key params.
        Shortcut block takes input from previous skip connection and outputs to new skip connection.
        """
        spec = residual_model_spec(state_dict, block_key=block_key, shortcut_key=shortcut_key, input_dim=input_dim, output_dim=output_dim)
        # make lists into tuples
        spec = {k: tuple(v) for k, v in spec.items()}
        return cls(spec)

    def subset(self, include_groups: List = None, exclude_groups: List = None, include_axes: List = None, exclude_axes: List = None):
        axes_to_group = {}
        remove_axes = set()
        for k, v in self.axes_to_group.items():
            if is_valid_key(k, include_axes, exclude_axes):
                layers = []
                for dim in v:  # if not None, has form (param_name, is_input)
                    if dim is not None and is_valid_key(dim[0], include_groups, exclude_groups):
                        layers.append(dim)
                    else:
                        layers.append(None)
                axes_to_group[k] = tuple(layers)
            else:
                remove_axes.add(k)
        group_to_axes = {}
        for k, v in self.group_to_axes.items():
            if is_valid_key(k, include_groups, exclude_groups):
                group_to_axes[k] = [axis for axis in v if axis[0] not in remove_axes]  # axis has form (param_name, dim, is_input)
        return self.__class__(axes_to_group, group_to_axes)

    def get_mask(self):
        # anything that's in a group is set to 1, remainder set to 0.
        mask = {}
        keys = list(self.group_to_axes.keys())
        for wk, axis_groups in self.axes_to_group.items():
            mask[wk] = 0.
            for group in axis_groups:  # if param has any axis in group, set mask to 1s
                if group is not None and group[0] in keys:  # if not None, has form (param_name, is_input)
                    mask[wk] = 1.
        return mask

    def _generate_transform(self, gen_fn: callable, state_dict: Dict[str, np.ndarray]):
        """
        Args:
            gen_fn (callable): a function which produces an object based on the following
                layers: List[Tuple[np.ndarray (params), int (dim)]]
            state_dict (Dict[str, np.ndarray]): parameters of model

        Returns:
            Dict[str, obj]: mapping from name of group to transform object
        """
        output = {}
        for p, axes in self.group_to_axes.items():
            output[p] = gen_fn([(state_dict[layer_name], dim, is_input) for layer_name, dim, is_input in axes])
        return output

    def _transform_state_dict(self, apply_fn: callable, state_dict: Dict[str, np.ndarray], transform_dict: Dict[str, object]) -> Dict[str, np.ndarray]:
        """
        Args:
            apply_fn (callable): a function which applies transform to parameter at dim, with signature
                transform: object, parameter: np.ndarray, dim: int
            state_dict (Dict[str, np.ndarray]): parameters of model
            transform_dict (Dict[str, object]): name of group to transformation object

        Returns:
            Dict[str, np.ndarray]: copy of state dict with transformation applied
        """
        output = deepcopy(to_numpy(state_dict))
        for p, transform in transform_dict.items():
            for layer_name, dim, is_input in self.group_to_axes[p]:
                output[layer_name] = apply_fn(transform, output[layer_name], dim, is_input=is_input, layer_name=layer_name)
        return output

    @staticmethod
    def layer_size(params: List[Tuple[np.ndarray, int, bool]]):
        size = params[0][0].shape[params[0][1]]
        for layer, dim, is_input in params:
            assert layer.shape[dim] == size, (layer.shape, dim, size)
        return size

    def get_sizes(self, state_dict: Dict[str, np.ndarray]):
        return self._generate_transform(self.layer_size, state_dict)
