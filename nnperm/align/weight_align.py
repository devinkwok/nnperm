from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Union
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

from nnperm.perm import PermutationSpec, perm_compose
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
            align_bias: bool=True,
            epsilon: float=1e-12,
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
        self.epsilon = epsilon
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

    def _init_fit(self, params_a, params_b):
        # check parameter sizes
        sizes_a = self.perm_spec.get_sizes(params_a)
        sizes_b = self.perm_spec.get_sizes(params_b)
        assert keys_match(sizes_a, sizes_b)
        assert np.all([sizes_a[k] == sizes_b[k] for k in sizes_a.keys()])
        # initialize variables
        self.similarity_ = defaultdict(list)
        self.params_a_ = to_numpy(params_a)
        self.params_b_ = to_numpy(params_b)
        self.params_ = deepcopy(self.params_b_)  # permuted params
        if self.init_perm is None:
            self.perms_ = self.perm_spec.get_identity_permutation(self.params_a_)
        else:
            self.perms_ = deepcopy(self.init_perm)

    def _get_gram_matrix(self, perm_key):
        n = len(self.perms_[perm_key])
        gram_matrix = np.zeros((n, n))
        # add similarities for every param that has same perm
        for layer_name, axis in self.perm_spec.perm_to_axes[perm_key]:
            if "bias" in layer_name and not self.align_bias:
                continue
            w_a = self.params_a_[layer_name]
            w_b = self.params_[layer_name]
            w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
            w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
            output = self.kernel_fn(w_a, w_b)
            gram_matrix += output
        return gram_matrix

    def _update_permutations(self, perm_key, new_p, similarity):
        self.params_ = self.perm_spec.apply_permutation(self.params_, {perm_key: new_p})
        self.perms_[perm_key] = perm_compose(self.perms_[perm_key], new_p)
        # permute similarity matrix so that max is on diagonal
        similarity = PermutationSpec.permute_layer(new_p, similarity, 1)
        return similarity

    def fit(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        # depends on _init_fit, _get_gram_matrix, _update_permutations
        self._init_fit(params_a, params_b)
        last_sim = 0
        for order in tqdm(self._alignment_order()[:self.max_iter]):
            total_sim = 0
            for perm_key in order:
                similarity = self._get_gram_matrix(perm_key)
                _, new_p = linear_sum_assignment(similarity, maximize=True)
                similarity = self._update_permutations(perm_key, new_p, similarity)
                # track total and per layer similarity
                self.similarity_[perm_key].append(similarity)
                score = np.trace(similarity)
                total_sim += score
            if total_sim - last_sim <  self.epsilon:
                break  # stop if similarity is not increasing
            last_sim = total_sim
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
