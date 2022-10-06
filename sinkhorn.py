from itertools import product
from typing import List, Tuple
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def _sinkhorn_normalize(matrix: np.ndarray, row=False):
    ones = np.ones_like(matrix)
    if row:
        return matrix / (matrix @ ones)
    else:
        return matrix / (matrix.T @ ones).T

def _sinkhorn_operator(matrix: np.ndarray, epsilon=1e-8, max_iterations=1000):
    matrix = np.exp(matrix)  # sinkhorn theorem requires all positive entries
    for i in range(max_iterations):
        new_matrix = _sinkhorn_normalize(_sinkhorn_normalize(matrix, row=True))
        difference = np.linalg.norm(new_matrix - matrix)
        matrix = new_matrix
        if difference < epsilon:
            print(f"Sinkhorn operator converged within {epsilon} after {i} iterations.")
            return matrix
    print(f"Sinkhorn operator reached max iterations {max_iterations} with successive Frobenius norm difference {difference}.")
    return matrix

def _pairwise_loss(source, target, vector_loss_fn=nn.MSELoss()):
    with torch.no_grad():
        source = torch.tensor(source, dtype=torch.float64)
        target = torch.tensor(target, dtype=torch.float64)
        losses_per_pair = torch.empty(len(source), len(target))
        for i, j in tqdm(product(range(len(source)), range(len(target))
                        ), total=len(source)*len(target)):
            losses_per_pair[i, j] = vector_loss_fn(source[i], target[j])
    return losses_per_pair.numpy()

def find_permutation(source: np.ndarray, target: np.ndarray, max_tries=10, tau=1e-1, vector_loss_fn=nn.MSELoss()) -> Tuple[List, float, float]:
    """Uses a result from optimal transport to find a permutation that minimizes
    the loss between two matrices.
    Specifically entropic regularization + sinkhorn's theorem. See the reference:
        Mena, G., Belanger, D., Munoz, G., & Snoek, J. (2017).
        Sinkhorn networks: Using optimal transport techniques to learn permutations.
        In NIPS Workshop in Optimal Transport and Machine Learning (Vol. 3).

    Args:
        source (np.ndarray): matrix to compare against
        target (np.ndarray): matrix to permute
        tau (float, optional): Initial value of hyperparameter for soft permutation.
            Defaults to 1e-1.
        vector_loss_fn (Callable, optional): Function to compare source and target.
            Defaults to torch.nn.MSELoss().

    Returns:
        Tuple[List, float, float]: permutation idx that minimize loss of target[idx] to source,
            margin of soft permutations (smallest value in diagonal), loss achieved by permutation
    """
    # theorem 1: S(X / \tau) = argmax_P trace(P^T X) + \tau h(P)
    # where S is the sinkhorn operator, P is the permutation, \tau is a hparam, h is entropy
    # we want to find P that minimizes pairwise loss
    # which is equivalent to finding argmax_P (P^T X) where X = -loss
    loss = _pairwise_loss(source, target, vector_loss_fn=vector_loss_fn)
    permutation_1, permutation_2 = linear_sum_assignment(loss)
    best_loss = np.sum(loss[permutation_1, permutation_2])
    return permutation_1, permutation_2, best_loss

if __name__ == "__main__":
    # test known permutation
    n = 10
    source = np.arange(n**2, dtype=float)
    perm = np.random.permutation(n)
    source = source.reshape(n, n)
    target = source[perm]
    inverse_perm, margin, loss = find_permutation(source, target)
    assert np.all(perm[inverse_perm] == np.arange(n))
    print("Permutation recovered!", source, target, perm, inverse_perm, perm[inverse_perm], margin, loss, sep="\n")
    # test unknown permutation with noise
    source = np.random.randn(n, n)
    target = np.random.randn(n, n)
    print("Random matrices", source, target, sep="\n")
    for i in range(-4, 2):
        inverse_perm, margin, loss = find_permutation(source, target, tau=10**i)
        print(10**i, inverse_perm, margin, loss)
