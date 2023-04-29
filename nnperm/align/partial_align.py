from typing import Dict, List, Union
import numpy as np
import torch

from nnperm.align.weight_align import WeightAlignment
from nnperm.align.activation_align import ActivationAlignment
from nnperm.spec import PermutationSpec


def _partial_init_fit(params_a, params_b, perm_spec, target_sizes):
    # save original sizes to crop gram matrix
    sizes_a = perm_spec.get_sizes(params_a)
    sizes_b = perm_spec.get_sizes(params_b)
    # if no target_size set, get whichever size is larger in a and b
    if target_sizes is None:
        target_sizes = {k: max(sizes_a[k], sizes_b[k]) for k in sizes_a.keys()}
    # check sizes
    for k in sizes_a.keys():
        assert max(sizes_a[k], sizes_b[k]) <= target_sizes[k]
        assert target_sizes[k] <= sizes_a[k] + sizes_b[k]
    # create a mask for gram matrix, True means aligned, False means unaligned
    similarity_mask = {}
    for perm_key in sizes_a.keys():
        similarity_mask[perm_key] = np.zeros(
            (target_sizes[perm_key], target_sizes[perm_key]), dtype=bool)
        similarity_mask[perm_key][:sizes_a[perm_key], :sizes_b[perm_key]] = True
    # pad both models to same size as target
    params_a = perm_spec.apply_padding(params_a, target_sizes)
    params_b = perm_spec.apply_padding(params_b, target_sizes)
    return params_a, params_b, sizes_a, sizes_b, similarity_mask


def _crop_gram_matrix(gram_matrix, similarity_mask):
    # set unaligned channels to max similarity so they are aligned preferentially
    cropped_gram = np.full_like(gram_matrix, np.max(gram_matrix) + 1)
    cropped_gram = gram_matrix + cropped_gram * np.logical_not(similarity_mask)
    return cropped_gram


def _update_similarity_mask(similarity_mask, perm_key, new_p, similarity):
    # permute mask (note: need to do this first as similarity is already permuted)
    similarity_mask[perm_key] = PermutationSpec.permute_layer(
        new_p, similarity_mask[perm_key], 1)
    # use mask to set similarity of unaligned elements to 0
    return similarity * similarity_mask[perm_key], similarity_mask


class PartialWeightAlignment(WeightAlignment):

    def __init__(self,
        perm_spec: PermutationSpec,
        kernel: Union[str, callable]="linear",
        init_perm: Dict[str, int] = None,
        max_iter: int=100,
        seed: int=None,
        order: Union[str, List[List[int]]]="random",
        verbose: bool=False,
        target_sizes: Dict[str, int] = None,
    ):
        super().__init__(perm_spec, kernel, init_perm, max_iter, seed, order, verbose)
        self.target_sizes = target_sizes

    def _init_fit(self, params_a, params_b):
        params_a, params_b, self.sizes_a_, self.sizes_b_, self.similarity_mask_ = _partial_init_fit(params_a, params_b, self.perm_spec, self.target_sizes)
        super()._init_fit(params_a, params_b)

    def _get_gram_matrix(self, p):
        gram_matrix = super()._get_gram_matrix(p)
        return _crop_gram_matrix(gram_matrix, self.similarity_mask_)

    def _update_permutations(self, perm_key, new_p, similarity):
        similarity = super()._update_permutations(perm_key, new_p, similarity)
        similarity, self.similarity_mask_ = _update_similarity_mask(self.similarity_mask_, perm_key, new_p, similarity)
        return similarity


class PartialActivationAlignment(ActivationAlignment):
    def __init__(self,
        perm_spec: PermutationSpec,
        dataloader: torch.utils.data.DataLoader,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module = None,
        exclude: List[str] = None,
        intermediate_type: str="last",
        kernel: Union[str, callable] = "linear",
        verbose: bool = False,
        device: str = "cuda",
        target_sizes: Dict[str, int] = None,
    ):
        super().__init__(perm_spec=perm_spec, dataloader=dataloader, model_a=model_a, model_b=model_b, exclude=exclude, intermediate_type=intermediate_type, kernel=kernel, verbose=verbose, device=device)
        self.target_sizes = target_sizes

    def _init_fit(self, params_a, params_b):
        params_a, params_b, self.sizes_a_, self.sizes_b_, self.similarity_mask_ = _partial_init_fit(params_a, params_b, self.perm_spec, self.target_sizes)
        # either model_a or model_b should have same size as params_a and params_b
        super()._init_fit(params_a, params_b)

    def _get_gram_matrix(self, p):
        gram_matrix = super()._get_gram_matrix(p)
        return _crop_gram_matrix(gram_matrix, self.similarity_mask_)

    def _update_permutations(self, perm_key, new_p, similarity):
        similarity = super()._update_permutations(perm_key, new_p, similarity)
        similarity, self.similarity_mask_ = _update_similarity_mask(self.similarity_mask_, perm_key, new_p, similarity)
        return similarity
