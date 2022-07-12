"""
Old code for comparison, only works with MLPs.
"""
import collections
import torch
import numpy as np


def canonical_renormalization(model_state_dict):
    renormalized_model_state_dict = collections.OrderedDict()
    list_of_keys = list(model_state_dict.keys())
    model_state_dict_keys = list_of_keys[:-2]
    fc_weight_key, fc_bias_key = list_of_keys[-2:]
    c_old = 1
    for k in model_state_dict_keys:
        if "weight" in k:
            weight_curr = c_old*model_state_dict[k]
            c_curr = torch.norm(weight_curr, dim=1)
            renormalized_model_state_dict[k] = weight_curr/c_curr.reshape(-1, 1)
            c_old = c_curr
        elif "bias" in k:
            renormalized_model_state_dict[k] = model_state_dict[k]/c_old

    renormalized_model_state_dict[fc_weight_key] = c_old*model_state_dict[fc_weight_key]
    renormalized_model_state_dict[fc_bias_key] = model_state_dict[fc_bias_key]
    return renormalized_model_state_dict

L1Loss = torch.nn.L1Loss()
MSELoss = torch.nn.MSELoss()
def loss(x, y, p):
    if p == 1:
        return L1Loss(x, y)
    elif p == 2:
        return MSELoss(x, y)

def find_permutations(model_state_dict_g, model_state_dict_f, p=2):
    s = []
    model_state_dict_keys = list(model_state_dict_g.keys())[:-2]
    for k in model_state_dict_keys:
        if "weight" in k:
            layer_g = model_state_dict_g[k]
            layer_f = model_state_dict_f[k]
            if len(s) > 0:
                prev_i_g, prev_i_f, _ = s[-1]
                layer_g = layer_g[:, prev_i_g, ...]
                layer_f = layer_f[:, prev_i_f, ...]

            argsort_g = np.argsort(layer_g, axis=0)
            argsort_f = np.argsort(layer_f, axis=0)
            
            c, r = layer_g.shape
            
            best = (None, None, float('inf'))
            max_best = 0
            for i_g in range(r):
                permuted_layer_g = layer_g[argsort_g[:, i_g], :]
                for i_f in range(r):
                    permuted_layer_f = layer_f[argsort_f[:, i_f], :]
                    curr_loss = loss(permuted_layer_g, permuted_layer_f, p)
                    max_best = max(max_best, curr_loss)
                    if best[2] > curr_loss:
                        best = (argsort_g[:, i_g], argsort_f[:, i_f], curr_loss)
            s.append(best)
    found_permutations = s + [(None, None, float('inf'))]
    s_1, s_2, diffs = list(zip(*found_permutations))
    return s_1, s_2, diffs

def permutate_state_dict_mlp(model_state_dict, s):

    permuted_model_state_dict = collections.OrderedDict()

    list_of_keys = list(model_state_dict.keys())
    model_state_dict_keys = list_of_keys[:-2]
    fc_weight_key, fc_bias_key = list_of_keys[-2:]

    s_curr = 0
    for i, k in enumerate(model_state_dict_keys):
        if "weight" in k:
            if i==0:
                s_curr = s[0]
                permuted_model_state_dict[k] = model_state_dict[k][s_curr, :]
            else:
                permuted_model_state_dict[k] = model_state_dict[k][:, s_curr]
                s_curr = s[i//2]
                permuted_model_state_dict[k] = permuted_model_state_dict[k][s_curr, :]
        else:
            permuted_model_state_dict[k] = model_state_dict[k][s_curr]

    permuted_model_state_dict[fc_weight_key] = model_state_dict[fc_weight_key][:, s_curr]
    permuted_model_state_dict[fc_bias_key] = model_state_dict[fc_bias_key]

    return permuted_model_state_dict
