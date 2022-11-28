from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Union
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

from nnperm.perm import PermutationSpec
from nnperm.utils import keys_match, to_numpy
from nnperm.kernel import get_kernel_from_name


class WeightAlignment:

    def __init__(self,
            perm_spec: PermutationSpec,
            kernel: Union[str, callable]="linear",
            init_perm: Dict[str, int] = None,
            max_iter: int=100,
            seed: int=None,
            order: Union[str, List[List[int]]]="random",
            verbose: bool=False,
    ):
        self.perm_spec = perm_spec
        self.kernel = kernel
        self.init_perm = init_perm
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.order = order
        if type(self.kernel) is str:
            self.kernel_fn = get_kernel_from_name(self.kernel, self.seed)
        else:
            self.kernel_fn = self.kernel
        self.perms_ = self.init_perm

    def _alignment_order(self):
        perm_names = list(self.perm_spec.perm_to_axes.keys())
        if self.order == "random":  # randomly choose order in which to solve permutations
            return [[perm_names[i] for i in self.random_state.permutation(
                        len(perm_names))] for _ in range(self.max_iter)]
        elif self.order == "forward":
            return [perm_names] * self.max_iter
        elif self.order == "backward":
            return [list(reversed(perm_names))] * self.max_iter
        else:
            for x in self.order:
                assert len(x) == len(perm_names), self.order
            return self.order

    def _linear_sum_assignment(self, gram_matrix, perm_name):
        s_row, s_col = linear_sum_assignment(gram_matrix, maximize=True)
        assert np.all(s_row == np.arange(gram_matrix.shape[0]))
        return s_col, gram_matrix

    def _prep_params(self, params_a, params_b, require_same_size=True):
        params_a = to_numpy(params_a)
        params_b = deepcopy(to_numpy(params_b))
        self.params_ = params_b
        sizes_a = self.perm_spec.get_sizes(params_a)
        sizes_b = self.perm_spec.get_sizes(params_b)
        assert keys_match(sizes_a, sizes_b)
        if require_same_size:
            assert np.all([sizes_a[k] == sizes_b[k] for k in sizes_a.keys()])
            return params_a, params_b, sizes_a
        else:
            return params_a, params_b, sizes_a, sizes_b

    def fit(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        """Find a permutation of `params_b` to make them match `params_a`."""
        params_a, params_b, sizes = self._prep_params(params_a, params_b)
        if self.perms_ is None:
            self.perms_ = self.perm_spec.get_identity_permutation(params_a)
        self.similarity_ = defaultdict(list)
        # find best permutation based on random order of solving them
        for perm_names in tqdm(self._alignment_order()[:self.max_iter]):
            early_stop = True
            for i, p in enumerate(perm_names):
                n = sizes[p]
                gram_matrix = np.zeros((n, n))
                # add similarities for every param that has same perm
                for layer_name, axis in self.perm_spec.perm_to_axes[p]:
                    w_a = params_a[layer_name]
                    w_b = self.get_permuted_param(layer_name, params_b, except_axis=axis)
                    w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
                    gram_matrix += self.kernel_fn(w_a, w_b)
                new_perm, similarity = self._linear_sum_assignment(gram_matrix, p)
                sim_diff = np.sum(similarity) - np.sum(self.similarity_[p][-1]) if len(self.similarity_[p]) > 0 else np.sum(similarity) 
                if self.verbose:
                    print(f"{i}/{p}: {sim_diff}")
                if sim_diff > 1e-12:  # apply new permutation if similarity improves
                    early_stop = False
                self.perms_[p] = new_perm
                self.similarity_[p].append(similarity)
            if early_stop:
                break
        return self.perms_, self.similarity_

    def transform(self, params: Dict[str, np.ndarray] = None):
        if self.perms_ is None:
            raise ValueError("Permutation not generated yet.")
        if params is not None:
            params = self.perm_spec.apply_padding(params, self.target_sizes)
        else:
            params = self.params_
        return self.perm_spec.apply_permutation(to_numpy(params), self.perms_)

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


class PartialWeightAlignment(WeightAlignment):

    def __init__(self,
        perm_spec: PermutationSpec,
        target_sizes: Dict[str, int],
        kernel: Union[str, callable]="linear",
        init_perm: Dict[str, int] = None,
        max_iter: int=100,
        seed: int=None,
        order: Union[str, List[List[int]]]="random",
        verbose: bool=False,
        cost_ramp_slope: float = 1,
        cost_margin: float = -0.1,
    ):
        super().__init__(perm_spec, kernel, init_perm, max_iter, seed, order, verbose)
        self.target_sizes = target_sizes
        self.cost_ramp_slope = cost_ramp_slope
        self.cost_margin = cost_margin

    def _prep_params(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        params_a, params_b, sizes_a, sizes_b = super()._prep_params(params_a, params_b, require_same_size=False)
        assert keys_match(sizes_a, self.target_sizes)
        assert np.all([max(sizes_a[k], sizes_b[k]) <= self.target_sizes[k] for k in sizes_a.keys()])
        assert np.all([self.target_sizes[k] <= sizes_a[k] + sizes_b[k] for k in sizes_a.keys()])
        params_a = self.perm_spec.apply_padding(params_a, self.target_sizes)
        params_b = self.perm_spec.apply_padding(params_b, self.target_sizes)
        return params_a, params_b, self.target_sizes

    def _linear_sum_assignment(self, gram_matrix, n_a, n_b):
        cropped_gram = np.zeros_like(gram_matrix)
        cropped_gram[:n_a, :n_b] = gram_matrix[:n_a, :n_b]
        # TODO change to similarity ramp
        cost_ramp = np.linspace(
                np.min(cropped_gram) + self.cost_margin,
                self.cost_ramp_slope * np.max(cropped_gram) - self.cost_margin,
                gram_matrix.shape[0] - max(n_a, n_b)) / 2  # divide by 2 because cost is added twice when aligning zero pairs
        cropped_gram[:n_a, max(n_a, n_b):] = cost_ramp.reshape(1, -1)
        cropped_gram[max(n_a, n_b):, :n_b] = cost_ramp.reshape(-1, 1)
        return super()._linear_sum_assignment(cropped_gram, n_a, n_b)

