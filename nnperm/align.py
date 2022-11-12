from copy import deepcopy
from typing import Dict
from tqdm import tqdm
from torch import Tensor
import numpy as np
from scipy.optimize import linear_sum_assignment

from nnperm.perm import PermutationSpec


def keys_match(a: dict, b: dict):
    for k in set(list(a.keys()) + list(b.keys())):
        if not (k in a and k in b):
            return False
    return True


def product_loss(a: np.ndarray, b: np.ndarray):
    # identical vectors maximize dot product
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    dot_similarity = np.outer(norm_a, norm_b) - (a @ b.T)
    return np.clip(dot_similarity, 0, 1e10)


def mse_loss(a: np.ndarray, b: np.ndarray):
    a = a.reshape(a.shape[0], a.shape[1], 1)
    b = b.T.reshape(1, b.shape[1], b.shape[0])
    return np.sum((a - b)**2, axis=1)


def bootstrap_loss(a: np.ndarray, b: np.ndarray,
        loss_fn: callable,
        n_samples: int,
        random_state: np.random.RandomState,
):
    # randomly sample from a and b
    sample_a = random_state.randint(a.shape[1], size=[a.shape[0], n_samples])
    sample_b = random_state.randint(a.shape[1], size=[a.shape[0], n_samples])
    return loss_fn(np.take_along_axis(a, sample_a, axis=1),
                   np.take_along_axis(b, sample_b, axis=1))


class WeightAlignment:

    def __init__(self,
            perm_spec: PermutationSpec,
            loss: str="mse",
            bootstrap: int=0,
            target_sizes: Dict[str, int] = None,
            init_perm: Dict[str, int] = None,
            max_iter: int=100,
            cost_ramp_slope: float=1.,
            cost_margin: float=-0.1,
            seed: int=42,
            order: str="random",
            verbose: bool=False,
    ):
        self.perm_spec = perm_spec
        self.loss = loss
        self.bootstrap = bootstrap
        self.target_sizes = target_sizes
        self.perms_ = init_perm
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.cost_ramp_slope = cost_ramp_slope
        self.order = order
        self.cost_margin = cost_margin

        if self.loss == "mse":
            loss_fn = mse_loss
        elif self.loss == "dot":
            loss_fn = product_loss
        else:
            raise ValueError("Alignment loss should be one of 'mse', 'dot'.")

        if self.bootstrap > 0:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = lambda a, b: bootstrap_loss(
                a, b, loss_fn, self.bootstrap, self.random_state)

    def _pad_params(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        sizes_a = self.perm_spec.get_sizes(params_a)
        sizes_b = self.perm_spec.get_sizes(params_b)
        assert keys_match(sizes_a, sizes_b)
        assert np.all([sizes_a[k] == sizes_b[k] for k in sizes_a.keys()])
        if self.target_sizes is None:
            self.target_sizes = sizes_a
        else:
            assert keys_match(sizes_a, self.target_sizes)
            assert np.all([max(sizes_a[k], sizes_b[k]) <= self.target_sizes[k] for k in sizes_a.keys()])
            assert np.all([self.target_sizes[k] <= sizes_a[k] + sizes_b[k] for k in sizes_a.keys()])
            params_a = self.perm_spec.apply_padding(params_a, self.target_sizes)
            params_b = self.perm_spec.apply_padding(params_b, self.target_sizes)
        return params_a, params_b, sizes_a, sizes_b

    def _alignment_order(self, n):
        if self.order == "random":  # randomly choose order in which to solve permutations
            return self.random_state.permutation(n)
        elif self.order == "forward":
            return np.arange(n)
        elif self.order == "backward":
            return np.flip(np.arange(n))
        else:
            raise ValueError("Alignment order should be 'random', 'forward', or 'backward'.")

    @staticmethod
    def _crop_cost_matrix(cost, n_a, n_b):
        cropped_cost = np.zeros_like(cost)
        cropped_cost[:n_a, :n_b] = cost[:n_a, :n_b]
        return cropped_cost

    def _masked_linear_sum_assignment(self, align_cost, n_a, n_b):
        if n_a < align_cost.shape[0] or n_b < align_cost.shape[0]:
            cost = self._crop_cost_matrix(align_cost, n_a, n_b)
            cost_ramp = np.linspace(
                    np.min(cost) + self.cost_margin,
                    self.cost_ramp_slope * np.max(cost) - self.cost_margin,
                    align_cost.shape[0] - max(n_a, n_b)) / 2  # divide by 2 because cost is added twice when aligning zero pairs
            cost[:n_a, max(n_a, n_b):] = cost_ramp.reshape(1, -1)
            cost[max(n_a, n_b):, :n_b] = cost_ramp.reshape(-1, 1)
        else:
            cost = align_cost
        s_row, s_col = linear_sum_assignment(cost, maximize=False)
        assert np.all(s_row == np.arange(align_cost.shape[0]))
        loss = np.sum(self._crop_cost_matrix(align_cost, n_a, n_b)[s_row, s_col])
        return s_col, loss

    def fit(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        """Find a permutation of `params_b` to make them match `params_a`."""
        if type(next(iter(params_a.values()))) is Tensor:
            params_a = self.perm_spec.torch_to_numpy(params_a)
            params_b = self.perm_spec.torch_to_numpy(params_b)
        self.params_ = deepcopy(params_b)
        params_a, self.params_, sizes_a, sizes_b = self._pad_params(params_a, self.params_)
        if self.perms_ is None:
            self.perms_ = self.perm_spec.get_identity_permutation(params_a)
        perm_names = list(self.perms_.keys())
        self.losses_ = {}
        # find best permutation based on random order of solving them
        for i in tqdm(range(self.max_iter)):
            early_stop = True
            for p_ix in self._alignment_order(len(perm_names)):
                p = perm_names[p_ix]
                n = self.target_sizes[p]
                align_cost = np.zeros((n, n))
                # add losses for every param that has same perm
                for layer_name, axis in self.perm_spec.perm_to_axes[p]:
                    w_a = params_a[layer_name]
                    w_b = self.get_permuted_param(layer_name, self.params_, except_axis=axis)
                    w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
                    align_cost += self.loss_fn(w_a, w_b)
                new_perm, loss = self._masked_linear_sum_assignment(align_cost, sizes_a[p], sizes_b[p])
                loss_diff = self.losses_[p] - loss if p in self.losses_ else loss
                if self.verbose:
                    print(f"{i}/{p}: {loss_diff}")
                if loss_diff > -1e-12:
                    # apply new permutation if loss is improved
                    early_stop = False
                self.losses_[p] = loss
                self.perms_[p] = new_perm
            if early_stop:
                break
        return self.perms_, self.losses_

    def transform(self, params: Dict[str, np.ndarray] = None):
        if self.perms_ is None:
            raise ValueError("Permutation not generated yet.")
        if params is not None:
            params = self.perm_spec.apply_padding(params, self.target_sizes)
        else:
            params = self.params_
        return self.perm_spec.apply_permutation(params, self.perms_)

    def fit_transform(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        self.fit(params_a, params_b)
        return self.transform()

    def get_permuted_param(self, k, params, except_axis=None):
        """Get parameter `k` from `params`, with the permutations applied."""
        w = params[k]
        for axis, p in enumerate(self.perm_spec.axes_to_perm[k]):
            # Skip the axis we're trying to permute.
            if axis == except_axis:
                continue
            # None indicates that there is no permutation relevant to that axis.
            if p is not None:
                w = PermutationSpec.permute_layer(self.perms_[p], w, axis)
        return w
