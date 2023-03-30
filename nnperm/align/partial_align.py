from typing import Dict, List, Union
import numpy as np

from nnperm.align import WeightAlignment
from nnperm.perm import PermutationSpec

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
    ):
        super().__init__(perm_spec, kernel, init_perm, max_iter, seed, order, verbose)
        self.target_sizes = target_sizes

    def _init_fit(self, params_a, params_b):
        # save original sizes to crop gram matrix
        self.sizes_a_ = self.perm_spec.get_sizes(params_a)
        self.sizes_b_ = self.perm_spec.get_sizes(params_b)
        # check sizes
        for k in self.sizes_a_.keys():
            assert max(self.sizes_a_[k], self.sizes_b_[k]) <= self.target_sizes[k]
            assert self.target_sizes[k] <= self.sizes_a_[k] + self.sizes_b_[k]
        # create a mask for gram matrix, True means aligned, False means unaligned
        self.similarity_mask_ = {}
        for perm_key in self.sizes_a_.keys():
            self.similarity_mask_[perm_key] = np.zeros(
                (self.target_sizes[perm_key], self.target_sizes[perm_key]), dtype=bool)
            self.similarity_mask_[perm_key][:self.sizes_a_[perm_key], :self.sizes_b_[perm_key]] = True
        # pad both models to same size as target
        params_a = self.perm_spec.apply_padding(params_a, self.target_sizes)
        params_b = self.perm_spec.apply_padding(params_b, self.target_sizes)
        super()._init_fit(params_a, params_b)

    def _get_gram_matrix(self, p):
        gram_matrix = super()._get_gram_matrix(p)
        # set unaligned channels to max similarity so they are aligned preferentially
        cropped_gram = np.full_like(gram_matrix, np.max(gram_matrix) + 1)
        cropped_gram = gram_matrix + cropped_gram * np.logical_not(self.similarity_mask_)
        return cropped_gram

    def _update_permutations(self, perm_key, new_p, similarity):
        similarity = super()._update_permutations(perm_key, new_p, similarity)
        # permute mask (note: need to do this first as similarity is already permuted)
        self.similarity_mask_[perm_key] = PermutationSpec.permute_layer(
            new_p, self.similarity_mask_[perm_key], 1)
        # use mask to set similarity of unaligned elements to 0
        similarity = similarity * self.similarity_mask_[perm_key]
        return similarity
