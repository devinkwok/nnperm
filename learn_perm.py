from copy import deepcopy
from typing import List
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from nnperm import random_transform
from rebasin.torch_utils import sequential_permutation_spec
from rebasin.weight_matching import PermutationSpec


def sinkhorn_operator(
        x: torch.tensor,
        iters=3,
        epsilon=1e-8,
        test_convergence=False,
    ):
    x = torch.exp(x)  # sinkhorn theorem requires all positive entries
    ones = torch.ones_like(x)
    for i in range(iters):  # repeat a few times to get something close to a perm matrix
        old_x = x
        x = x / (x @ ones)  # rows
        x = x / (x.T @ ones).T  # columns
        if test_convergence:
            difference = torch.linalg.norm(x - old_x)
            if difference < epsilon:
                print(f"Sinkhorn operator converged within {epsilon} after {i} iterations.")
                return x
    if test_convergence:
        print(f"Sinkhorn operator reached max iterations {iters} with successive Frobenius norm difference {difference}.")
    return x


class Permutation(nn.Module):

    def __init__(self, n: int, iters: int = 3, device=None, dtype=None) -> None:
        super().__init__()
        self.n_permute = n
        self.sinkhorn_iters = iters
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.unnormalized_perm = nn.Parameter(torch.empty((n, n), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.unnormalized_perm)

    def forward(self):
        perm_matrix = sinkhorn_operator(self.unnormalized_perm, self.sinkhorn_iters)
        if not self.training:  # turn into actual permutation during test time
            _, perm_matrix = linear_sum_assignment(perm_matrix)  # first output is always arange
        return perm_matrix


class PermutableParameter:

    def __init__(self,
            perm_dim: int,
            paramA: torch.tensor,
            paramB: torch.tensor,
    ):
        self.perm_dim = perm_dim
        assert paramA.shape == paramB.shape
        self.shape = paramA.shape
        # permute the params so that the perm dim is first
        dim_permutation = [i for i in range(len(self.shape))]
        dim_permutation[0] = perm_dim
        dim_permutation[perm_dim] = 0
        self.dim_permutation = dim_permutation
        self.halfparamA = self._transform_param(paramA)
        self.halfparamB = self._transform_param(paramB)

    @staticmethod
    def from_state_dict(prefix: str, param_names: List[str], perm_dim: int, state_dictA: dict, state_dictB: dict):
        params = []
        for name in param_names:
            key = f"{prefix}.{name}"
            param = PermutableParameter(perm_dim, state_dictA[key], state_dictB[key])
            params.append(param)
        return params

    def _transform_param(self, param):
        halfparam = torch.permute(param / 2, self.dim_permutation)
        halfparam = halfparam.reshape(self.shape[self.perm_dim], -1)
        halfparam.requires_grad = False
        return halfparam

    def forward(self, permutation):
        param = self.halfparamA + permutation @ self.halfparamB
        # undo the permutation to bring perm dim back to its original place
        param = torch.permute(param, self.dim_permutation)
        return param.reshape(self.shape)


class PermutableModule(nn.Module):
    pass  # abstract class


class PermutableLinear(PermutableModule):

    def __init__(self,
            source_name: str,
            state_dictA: dict,
            state_dictB: dict,
    ):
        super().__init__()
        self.weight, self.bias = PermutableParameter.from_state_dict(
            source_name, ["weight", "bias"], 0, state_dictA, state_dictB)

    def forward(self, x, permutation):
        w = self.weight.forward(permutation)
        b = self.bias.forward(permutation)
        return x @ w.T + b


class PermutatableSequential(nn.Module):

    def __init__(self,
            source_model: nn.Sequential,
            state_dictA: dict,
            state_dictB: dict,
            perm_spec: PermutationSpec
    ):
        super().__init__()
        self.perm_spec = perm_spec
        self.is_permutable = set()
        # copy modules
        for name, module in source_model._modules.items():
            if type(module) is nn.Linear:
                self.add_module(name, PermutableLinear(name, state_dictA, state_dictB))
                self.is_permutable.add(name)
            else:
                self.add_module(name, module)
        # create permutations
        self.permutations = {}
        for perm_name, perm_list in perm_spec.perm_to_axes.items():
            layer_name, dim = perm_list[0] # get dims to permute
            n_permute_A = state_dictA[layer_name].shape[dim]
            n_permute_B = state_dictB[layer_name].shape[dim]
            assert n_permute_A == n_permute_B, f"perm dim at {layer_name} doesn't match between models A and B"
            self.permutations[perm_name] = Permutation(n_permute_A)

    def forward(self, x):
        # generate permutation matrices first
        perm_matrices = {n: p.forward() for n, p in self.permutations.items()}
        for name, module in self._modules.items():
            if name in self.is_permutable:
                perm_name = self.perm_spec.axes_to_perm[name]
                x = module.forward(x, perm_matrices[perm_name])
            else:
                x = module.forward(x)
        return x

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Softmax(3),
    )
    state_dictA = deepcopy(model.state_dict())
    state_dictB, rand_perm, _ = random_transform(state_dictA, scale=False, permute=True)
    perm_spec = sequential_permutation_spec(state_dictA)
    perm_model = PermutatableSequential(model, state_dictA, state_dictB, perm_spec)
    print(perm_model)

    x = torch.randn(4, 10)
    y = perm_model(x)
    print(y.shape)

