from collections import OrderedDict
import torch
import numpy as np
import jax
from rebasin.weight_matching import PermutationSpec, weight_matching


def sequential_permutation_spec(state_dict: dict) -> PermutationSpec:
    spec = OrderedDict()
    perm = "Pseq_0-in"
    for i, (k, v) in enumerate(state_dict.items()):
        # assume output is always 1st dim, input is always 2nd dim
        dim_tuple = [None] * len(v.shape)
        # add input perm if exists, unless first layer
        if len(v.shape) > 1 and i > 0:
            dim_tuple[1] = perm
            perm = f"Pseq_{i}-{k}"  # new perm name for output
        # add output perm if exists
        if len(v.shape) > 0:
            dim_tuple[0] = perm
        spec[k] = dim_tuple
    # remove last perm (on outputs)
    for k, v in spec.items():
        spec[k] = tuple([None if perm == x else x for x in v])
    return PermutationSpec.from_axes_to_perm(spec)


def torch_to_jax(state_dict):
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def torch_weight_matching(state_dict_a, state_dict_b):
    state_dict_a = torch_to_jax(state_dict_a)
    state_dict_b = torch_to_jax(state_dict_b)
    spec = sequential_permutation_spec(state_dict_a)
    perms = weight_matching(jax.random.PRNGKey(4), spec, state_dict_a, state_dict_b)
    s_1, s_2, diffs = [], [], []
    for p in perms.values():
        s_1.append(torch.arange(len(p)))
        s_2.append(torch.tensor(np.array(p), dtype=torch.long))
        diffs.append(torch.zeros(len(p)))  # temporary hack
    s_1 += [None] # empty last layer
    s_2 += [None] # empty last layer
    return s_1, s_2, diffs
