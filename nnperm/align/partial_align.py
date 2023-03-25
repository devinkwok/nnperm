from typing import Dict, List, Union
import numpy as np

from nnperm.align import WeightAlignment
from nnperm.perm import PermutationSpec
from nnperm.utils import keys_match

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
        self.params_ = params_b
        return params_a, params_b, self.target_sizes

    def _linear_sum_assignment(self, gram_matrix, perm_name):
        n_a = self.sizes_a_[perm_name]
        n_b = self.sizes_b_[perm_name]
        cropped_gram = np.full_like(gram_matrix, np.max(gram_matrix) + 1)
        cropped_gram[:n_a, :n_b] = gram_matrix[:n_a, :n_b]
        return super()._linear_sum_assignment(cropped_gram, perm_name)

    # def _split_perms(self, perms, sizes_a, sizes_b):
    #     mask = {}
    #     output_perms = {}
    #     for k, perm in perms.items():
    #         size_b = sizes_b[k]
    #         n_aligned = sizes_a[k] + size_b - self.target_sizes[k]
    #         mask[k] = (perm >= n_aligned)[:n_aligned]  # ignore padding
    #         print(self.target_sizes[k], n_aligned, perm, mask[k])
    #         # split permutation by aligned and unaligned idx
    #         unaligned_a = np.where(mask[k])[0]
    #         print(unaligned_a)
    #         unaligned_b = perm[n_aligned:]
    #         print(unaligned_b)
    #         # map unaligned_b to unaligned_a idx
    #         perm[unaligned_a] = unaligned_b[:len(unaligned_a)]
    #         perm[n_aligned:] = unaligned_b[len(unaligned_a):]
    #         output_perms[k] = perm
    #     return output_perms, mask

    def fit(self, params_a: Dict[str, np.ndarray], params_b: Dict[str, np.ndarray]):
        super().fit(params_a, params_b)
        # self.perms_, self.mask_ = self._split_perms(self.perms_, self.sizes_a_, self.sizes_b_)
        return self.perms_, self.similarity_
