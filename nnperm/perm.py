import numpy as np
from typing import Dict


def perm_from_matrix(perm_matrix: np.ndarray) ->  np.ndarray:
    n = perm_matrix.shape[0]
    assert len(perm_matrix.shape) == 2 and n == perm_matrix.shape[1]
    x, y = np.nonzero(perm_matrix)
    assert len(x) == n and np.all(x == np.arange(n))
    return y


def perm_to_matrix(perm: np.ndarray) -> np.ndarray:
    return np.eye(len(perm))[perm]


def perm_inverse(perm: np.ndarray) -> np.ndarray:
    return np.argsort(perm)


def perm_compose(perm_f: np.ndarray, perm_g: np.ndarray) -> np.ndarray:
    """Apply f, then g as $g(f(x))$ or $g \circ f$.
    """
    return perm_f[perm_g]


class Permutations(dict):

    @staticmethod
    def from_matrices(perm_matrices: Dict[str, np.ndarray]):
        return Permutations({n: perm_from_matrix(x) for n, x in perm_matrices.items()})

    def to_matrices(self) -> Dict[str, np.ndarray]:
        return Permutations({n: perm_to_matrix(i) for n, i in self.items()})

    def sizes(self):
        return {k: len(v) for k, v in self.items()}

    def fixed_points(self):
        return {k: v == np.arange(len(v)) for k, v in self.items()}

    def inverse(self):
        """Gives inverse of permutation.

        Args:
            permutation (list): Permutations per layer. Each
                permutation is either None (no permutation) or a list
                of integers with length equal to the layer's output dimension.

        Returns:
            list: s^{-1} for each permutation s.
        """
        return Permutations({n: perm_inverse(i) for n, i in self.items()})


    def compose(self, perm_to_apply: Dict[str, np.ndarray]):
        """Applies permutation g to f as $g \circ f$, where f is self.

        Args:
            perm_to_apply (list): permutation g, list of permutations per layer.
                Each permutation is either None (no permutation) or a list
                of integers with length equal to the layer's output dimension.

        Returns:
            list: f \circ g, or equivalently, f(g(\cdot)).
        """
        output = {}
        for name in set(list(self.keys()) + list(perm_to_apply.keys())):
            if name in self and name in perm_to_apply:
                output[name] = perm_compose(self[name], perm_to_apply[name])
            elif name in self:
                output[name] = self[name]
            elif name in perm_to_apply:
                output[name] = perm_to_apply[name]
        return Permutations(output)
