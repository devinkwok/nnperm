from typing import List, Union
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch

from nnperm.align.weight_align import WeightAlignment
from nnperm.spec import PermutationSpec
from nnperm.eval import evaluate_intermediates
from nnperm.utils import to_torch_device


class ActivationAlignment(WeightAlignment):
    def __init__(self,
            perm_spec: PermutationSpec,
            dataloader: torch.utils.data.DataLoader,
            model_a: torch.nn.Module,
            model_b: torch.nn.Module = None,
            exclude: List[str]=None,
            intermediate_type: str="last",  # first, all, last
            kernel: Union[str, callable]="linear",
            verbose: bool=False,
            device: str="cuda",
    ):
        super().__init__(
            perm_spec=perm_spec, kernel=kernel, init_perm=None, max_iter=1, verbose=verbose, seed=None, order="forward", align_bias=True)
        self.model_a = model_a
        self.model_b = model_a if model_b is None else model_b
        self.dataloader = dataloader
        self.exclude = exclude
        self.intermediate_type = intermediate_type
        self.device = device

    def _get_activations(self, model, params, layers_to_perm):
        # for each permutation, get all intermediate outputs that output to this permutation
        model = deepcopy(model)
        model.load_state_dict(to_torch_device(params, device=self.device))
        hidden_batches = evaluate_intermediates(model, self.dataloader, self.device, exclude=self.exclude, verbose=self.verbose)
        for _, intermediates, _, _ in hidden_batches:
            perm_to_hidden = defaultdict(list)
            for k, v in intermediates.items():
                if k.endswith(".out"):  # ignore inputs to layers
                    perm = layers_to_perm.get(k[:-len(".out")], None)
                    if perm is not None:
                        perm_to_hidden[perm].append(v)
            # format the activations
            for k, v in perm_to_hidden.items():
                # transpose so dim 0 is output, dim 1 is number of examples
                v = [np.moveaxis(x.detach().cpu().numpy(), 0, -1) for x in v]
                v = [x.reshape(x.shape[0], -1) for x in v]
                if self.intermediate_type == "first":
                    perm_to_hidden[k] = v[0]
                elif self.intermediate_type == "all":
                    perm_to_hidden[k] = np.concatenate(v, axis=-1)
                else:
                    perm_to_hidden[k] = v[-1]
            yield perm_to_hidden

    def _init_fit(self, params_a, params_b):
        super()._init_fit(params_a, params_b)
        self.gram_matrix_ = {}
        # get map of layer names to perm names
        layers_to_perm = {}
        for perm, axes in self.perm_spec.group_to_axes.items():
            for layer_name, dim, is_input in axes:
                if not is_input:
                    # remove parameter type (e.g. "".weight", ".bias") from layer name
                    layer_name = layer_name[:-(len(layer_name.split(".")[-1]) + 1)]
                    layers_to_perm[layer_name] = perm
        # get gram matrix in batches to control memory usage
        for hidden_a, hidden_b in zip(self._get_activations(self.model_a, params_a, layers_to_perm),
                                        self._get_activations(self.model_b, params_b, layers_to_perm)):
            for k in hidden_a.keys():
                if k in self.gram_matrix_:
                    self.gram_matrix_[k] += self.kernel_fn(hidden_a[k], hidden_b[k])
                else:
                    self.gram_matrix_[k] = self.kernel_fn(hidden_a[k], hidden_b[k])

    def _get_gram_matrix(self, perm_key):
        return self.gram_matrix_[perm_key]
