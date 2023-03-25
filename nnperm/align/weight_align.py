from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Union
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

from nnperm.perm import PermutationSpec, perm_compose, perm_inverse
from nnperm.utils import keys_match, to_numpy
from nnperm.kernel import get_kernel_from_name


class WeightAlignment:
    #TODO ensure that calling fit resets all computed params_ in object
    def __init__(self,
            perm_spec: PermutationSpec,
            kernel: Union[str, callable]="linear",
            init_perm: Dict[str, int] = None,
            max_iter: int=100,
            seed: int=None,
            order: Union[str, List[List[int]]]="random",
            verbose: bool=False,
            align_bias: bool=True,
    ):
        self.perm_spec = perm_spec
        self.kernel = kernel
        self.init_perm = init_perm
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.order = order
        self.align_bias = align_bias
        if isinstance(self.kernel, str):
            self.kernel_fn = get_kernel_from_name(self.kernel, self.seed)
        else:
            self.kernel_fn = self.kernel

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
        gram_matrix = gram_matrix[:, s_col]
        return s_col, gram_matrix

    def _prep_params(self, params_a, params_b, require_same_size=True):
        params_a = to_numpy(params_a)
        params_b = deepcopy(to_numpy(params_b))
        self.params_ = params_b
        self.sizes_a_ = self.perm_spec.get_sizes(params_a)
        self.sizes_b_ = self.perm_spec.get_sizes(params_b)
        assert keys_match(self.sizes_a_, self.sizes_b_)
        if require_same_size:
            assert np.all([self.sizes_a_[k] == self.sizes_b_[k] for k in self.sizes_a_.keys()])
            return params_a, params_b, self.sizes_a_
        else:
            return params_a, params_b, self.sizes_a_, self.sizes_b_

    def _get_permuted_param(self, k, params, except_axis=None):
        """Get parameter `k` from `params`, with the permutations applied."""
        w = params[k]
        for axis, p in enumerate(self.perm_spec.axes_to_perm[k]):
            # Skip the axis we're trying to permute.
            if axis == except_axis:
                continue
            # None indicates that there is no permutation relevant to that axis.
            if p is not None and p in self.perms_:
                w = PermutationSpec.permute_layer(self.perms_[p], w, axis)
        return w

    def fit(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        """Find a permutation of `params_b` to make them match `params_a`."""
        params_a, params_b, sizes = self._prep_params(params_a, params_b)
        if self.init_perm is None:
            self.perms_ = self.perm_spec.get_identity_permutation(params_a)
        else:
            self.perms_ = deepcopy(self.init_perm)
        self.similarity_ = defaultdict(list)
        # find best permutation based on random order of solving them
        for perm_names in tqdm(self._alignment_order()[:self.max_iter]):
            early_stop = True
            for i, p in enumerate(perm_names):
                n = sizes[p]
                gram_matrix = np.zeros((n, n))
                # add similarities for every param that has same perm
                for layer_name, axis in self.perm_spec.perm_to_axes[p]:
                    if "bias" in layer_name and not self.align_bias:
                        continue  #TODO temp hack to stop NaNs
                    w_a = params_a[layer_name]
                    # w_b = self.params_[layer_name]
                    w_b = self._get_permuted_param(layer_name, params_b, except_axis=axis)
                    w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
                    output = self.kernel_fn(w_a, w_b)
                    if np.any(np.isnan(output)):
                        print(f"NaNs found after kernel operation: {layer_name} {params_a[layer_name].shape}, {axis}")
                        import pdb
                        pdb.set_trace()
                    gram_matrix += output
                new_perm, similarity = self._linear_sum_assignment(gram_matrix, p)
                sim_diff = np.trace(similarity) - np.trace(self.similarity_[p][-1]) if len(self.similarity_[p]) > 0 else np.trace(similarity) 
                if self.verbose:
                    print(f"{i}/{p}: {sim_diff}")
                if sim_diff > 1e-12:
                    early_stop = False
                self.perms_[p] = new_perm
                # self.perms_[p] = perm_compose(new_perm, self.perms_[p])
                # apply new permutation
                # for layer_name, axis in self.perm_spec.perm_to_axes[p]:
                #     self.params_[layer_name] = PermutationSpec.permute_layer(new_perm, self.params_[layer_name], axis)
                self.similarity_[p].append(similarity)
            if early_stop:
                break
        return self.perms_, self.similarity_

    def _get_align_mask(self):  # this will change for partial weight permutation
        return self.perm_spec.get_perm_mask()

    def transform(self, params: Dict[str, np.ndarray] = None):
        if self.perms_ is None:
            raise ValueError("Permutation not generated yet.")
        if params is None:
            params = self.params_
        permuted_params = self.perm_spec.apply_permutation(to_numpy(params), self.perms_)
        self.align_mask_ = self._get_align_mask()
        return permuted_params, self.align_mask_

    def fit_transform(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        self.fit(params_a, params_b)
        return self.transform()

    def summarize_last_similarities(self, **other_info):
        """Summarize align_loss stats per layer: quartiles, mean, std, num_elements
        Returns list of dicts suitable for dataframe
        """
        # get similarity at last align iteration only
        last_sim = {k: v[-1] for k, v in self.similarity_.items()}
        # get stats
        sim_stats = []
        for k, v in last_sim.items():
            min, low_quartile, median, high_quartile, max = np.quantile(v, [0., 0.25, 0.5, 0.75, 1.])
            sim_stats.append({
                **other_info,
                "layer": k,
                "size": len(v),
                "mean": np.mean(v),
                "std": np.std(v),
                "min": min,
                "low_quartile": low_quartile,
                "median": median,
                "high_quartile": high_quartile,
                "max": max,
                "id_similarity": np.mean(v[np.arange(len(v)), np.arange(len(v))]),
                "perm_similarity": np.mean(v[np.arange(len(v)), self.perms_[k]]),
            })
        return sim_stats
