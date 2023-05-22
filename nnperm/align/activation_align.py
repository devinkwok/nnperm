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
            append_keys: str = ["relu"],  # use this to include outputs without params to the last permutation group, typically is a non-linear activation function
    ):
        super().__init__(
            perm_spec=perm_spec, kernel=kernel, init_perm=None, max_iter=1, verbose=verbose, seed=None, order="forward", align_bias=True)
        self.model_a = model_a
        self.model_b = model_a if model_b is None else model_b
        self.dataloader = dataloader
        self.exclude = exclude
        self.intermediate_type = intermediate_type
        self.device = device
        self.append_keys = append_keys

    def _get_activation_keys(self, model, layers_to_perm):
        x, y = next(iter(self.dataloader))
        tmp_batch = [(x[:1], y[:1])]
        # for each permutation, get all intermediate outputs that output to this permutation
        # NOTE: do not exclude any layers as that may mess up the perm assignment to layers without params
        hidden_batches = evaluate_intermediates(model, tmp_batch, self.device, exclude=None, verbose=self.verbose)
        _, intermediates, _, _ = next(hidden_batches)
        perm_to_layers = defaultdict(list)
        last_perm = None
        for k in intermediates.keys():
            if k.endswith(".out"):  # ignore inputs to layers
                # include outputs of layers without parameters in the last perm group
                if any(x in k for x in self.append_keys):
                    perm_to_layers[last_perm].append(k)
                else:
                    perm = layers_to_perm.get(k[:-len(".out")], None)
                    if perm is not None:
                        perm_to_layers[perm].append(k)
                        last_perm = perm
        # choose the appropriate layers to include
        for k, v in perm_to_layers.items():
            v = [x for x in v if is_valid_key(x, exclude_keywords=self.exclude)]
            if self.intermediate_type == "first":
                perm_to_layers[k] = v[:1]
            elif self.intermediate_type == "all":
                perm_to_layers[k] = v
            else:
                perm_to_layers[k] = v[-1:]
        return perm_to_layers

    def _get_activations(self, model, params, perm_to_layers):
        # for each permutation, get all intermediate outputs that output to this permutation
        model = deepcopy(model)
        model.load_state_dict(to_torch_device(params, device=self.device))
        include_layers = list(set([k for v in perm_to_layers.values() for k in v]))
        hidden_batches = evaluate_intermediates(model, self.dataloader, self.device, exclude=self.exclude, include=include_layers, verbose=self.verbose)
        perm_to_hidden = {}
        for _, intermediates, _, _ in hidden_batches:
            # format the activations
            for k, layer_names in perm_to_layers.items():
                v = [intermediates[x] for x in layer_names]
                # transpose so dim 0 is output, dim 1 is number of examples
                v = [torch.moveaxis(x, 0, -1) for x in v]
                v = [x.reshape(x.shape[0], -1) for x in v]
                v = [x.detach().cpu().numpy() for x in v]
                if len(v) == 1:
                    perm_to_hidden[k] = v[0]
                else:
                    perm_to_hidden[k] = np.concatenate(v, axis=-1)
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
        perm_to_layers = self._get_activation_keys(self.model_a, layers_to_perm)
        for hidden_a, hidden_b in zip(self._get_activations(self.model_a, params_a, perm_to_layers),
                                        self._get_activations(self.model_b, params_b, perm_to_layers)):
            for k in hidden_a.keys():
                if k in self.gram_matrix_:
                    self.gram_matrix_[k] += self.kernel_fn(hidden_a[k], hidden_b[k])
                else:
                    self.gram_matrix_[k] = self.kernel_fn(hidden_a[k], hidden_b[k])

    def _get_gram_matrix(self, perm_key):
        return self.gram_matrix_[perm_key]
