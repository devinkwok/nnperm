from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch.nn as nn

from nnperm.spec.model_spec import ModelSpec
from nnperm.utils import is_valid_key


def multiply_along_axis(array, scale, axis):
    assert len(scale) == array.shape[axis]
    broadcast_shape = np.ones_like(array.shape)
    broadcast_shape[axis] = -1
    return array * scale.reshape(*broadcast_shape)


def rollback_normalization(linear_w, linear_b, norm_w, norm_b, var_est, mu_est, output_dim=0):
    """Combines normalization affine transform and stats into single weight and bias.
    Specifically, if w and b are the weights and biases of a linear transform,
    and mu, sd, gamma, beta are the parameters of a normalization layer:

    ((w * x + b) - mu) / sd * gamma + beta
    = (w * gamma / sd) * x + (b - mu) * gamma / sd + beta

    Only use as preprocessing for alignment.
    This will NOT give an equivalent function for layernorm
    due to dependence of mean and std on parameters.

    Args:
        linear_w (np.ndarray): linear transform weights
        linear_b (np.ndarray): linear transform biases
        norm_w (np.ndarray): layernorm weights (gamma)
        norm_b (np.ndarray): layernorm biases (beta)
        var_est (np.ndarray): layernorm variance (estimated from data)
        mu_est (np.ndarray): layernorm mean (estimated from data)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            same order as args, but only linear_w and linear_b are not 1 or 0 respectively.
    """
    def assert_correct_shapes(*params):
        for param in params:
            assert len(param.shape) == 1 and len(param) == linear_w.shape[output_dim]

    assert_correct_shapes(linear_b, norm_w, norm_b, var_est, mu_est)
    scale = norm_w / np.sqrt(var_est)
    linear_w = multiply_along_axis(linear_w, scale, axis=output_dim)
    linear_b = (linear_b - mu_est) * scale + norm_b
    return linear_w, linear_b, np.ones_like(norm_w), np.zeros_like(norm_b), np.ones_like(var_est), np.zeros_like(mu_est)


def _find_param(layers, include, exclude):
    keys = []
    for k, _, is_input in layers:
        if not is_input and is_valid_key(k, include_keywords=include, exclude_keywords=exclude, require_all=True):
            keys.append(k)
    assert len(keys) == 1, (keys, include, exclude)
    return keys[0]


class Scales(dict):

    @staticmethod
    def from_matrices(scale_matrices: Dict[str, np.ndarray]):
        return Scales({k: np.diag(v) for k, v in scale_matrices.items()})

    def to_matrices(self) -> Dict[str, np.ndarray]:
        return {k: np.diag(v) for k, v in self.items()}

    def sizes(self):
        return {k: len(v) for k, v in self.items()}

    def fixed_points(self):
        return {k: v == 1. for k, v in self.items()}

    def inverse(self):
        return Scales({k: 1. / v for k, v in self.items()})

    def compose(self, scale_to_apply: Dict[str, np.ndarray]):
        output = {}
        for name in set(list(self.keys()) + list(scale_to_apply.keys())):
            if name in self and name in scale_to_apply:
                output[name] = self[name] * scale_to_apply[name]
            elif name in self:
                output[name] = self[name]
            elif name in scale_to_apply:
                output[name] = scale_to_apply[name]
        return Scales(output)


class ScaleSpec(ModelSpec):
    """
        axes_to_group: str (name of layer): Tuple[ (for each dim in layer shape) Union[None (dim not scaled), Tuple[str (name of scale assigned to dim), bool (if input and scaling is inverted)]]]
        group_to_axes: str (names of distinct scales): List[Tuple[str (name of layer with scale), int (dim with this scale), bool (if input and scaling is inverted)]]
    """
    def apply_rollback_layernorm(self, state_dict: Dict[str, nn.Module], layernorm_key="layernorm"):
        """WARNING: this will not automatically give an equivalent function.
        To keep the function unchanged, the computed mean and std of LayerNorm layers
        needs to be forced to 0 and 1 respectively.
        """
        pass #TODO

    def apply_rollback_batchnorm(self, state_dict: Dict[str, nn.Module], batchnorm_key="bn"):
        output = {}

        def update_param(key, value):
            assert key not in output
            output[key] = value

        # find exactly one linear and one batchnorm layer per group
        for group, axes in self.group_to_axes.items():
            w_k = _find_param(axes, ["weight"], [batchnorm_key])
            b_k = _find_param(axes, ["bias"], [batchnorm_key])
            gamma_k = _find_param(axes, [batchnorm_key, "weight"], None)
            beta_k = _find_param(axes, [batchnorm_key, "bias"], None)
            var_k = _find_param(axes, [batchnorm_key, "running_var"], None)
            mu_k = _find_param(axes, [batchnorm_key, "running_mean"], None)
            w, b, gamma, beta, var, mu = rollback_normalization(
                state_dict[w_k], state_dict[b_k],
                state_dict[gamma_k], state_dict[beta_k],
                state_dict[var_k], state_dict[mu_k])
            update_param(w_k, w)
            update_param(b_k, b)
            update_param(gamma_k, gamma)
            update_param(beta_k, beta)
            update_param(var_k, var)
            update_param(mu_k, mu)

        # copy over remaining params
        for k, v in state_dict.items():
            if k not in output:
                output[k] = v
        return output

    @staticmethod
    def _channel_norm(params, normalize=False):
        #FIXME need to account for input AND output scaling
        flat_params = []
        for layer, dim, is_input in params:
            if not is_input:
                flat_params.append(np.moveaxis(layer, dim, 0).reshape(layer.shape[dim], -1))
        flat_params = np.concatenate(flat_params, axis=-1)
        norm = np.linalg.norm(flat_params, axis=-1)
        if normalize:
            norm = norm / np.sqrt(flat_params.shape[-1])
        return norm

    def get_norm(self, state_dict: Dict[str, nn.Module], normalize=False):
        #FIXME need to account for input AND output scaling, go from input to output
        return Scales(self._generate_transform(
            lambda x: self._channel_norm(x, normalize=normalize), state_dict))

    def get_avg_norm(self, *state_dicts: List[Dict[str, nn.Module]]):
        all_norms = defaultdict(list)
        for state_dict in state_dicts:
            for k, v in self.get_norm(state_dict).items():
                all_norms[k].append(v)
        return Scales({k: np.mean(v) for k, v in all_norms.items()})

    @staticmethod
    def scale_layer(scale: List[float], param: np.ndarray, dim, is_input=False, layer_name=None) -> np.ndarray:
        if scale is None:
            return param
        if "running_var" in layer_name:
            scale = scale**2
        return multiply_along_axis(param, 1. / scale if is_input else scale, axis=dim)

    def apply_scale(self, state_dict: Dict[str, nn.Module], scales: Dict[str, List[float]]):
        return self._transform_state_dict(self.scale_layer, state_dict, scales)

    def get_random_scale(self, state_dict: Dict[str, nn.Module], random_state=None, scale_min=0.9, scale_max=1.1):
        random_state = np.random.RandomState(42) if random_state is None else random_state
        rand_scale_fn = lambda p: random_state.uniform(scale_min, scale_max, self.layer_size(p))
        return Scales(self._generate_transform(rand_scale_fn, state_dict))

    def get_identity_scale(self, state_dict: Dict[str, np.ndarray]):
        identity_scale_fn = lambda p: np.ones(self.layer_size(p))
        return Scales(self._generate_transform(identity_scale_fn, state_dict))

    def apply_rand_scale(self, state_dict: Dict[str, nn.Module]):
        return self.apply_scale(state_dict, self.get_random_scale(state_dict))
