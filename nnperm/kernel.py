import torch
import numpy as np

import sys
sys.path.append("./repsim/")
from repsim.kernels import Linear, SquaredExponential
from repsim.util import pdist2


def _apply_repsim_kernel(x, y, kernel_fn):
    return kernel_fn(torch.tensor(x).to(
        dtype=torch.float), torch.tensor(y).to(dtype=torch.float)).numpy()


def linear_kernel(x, y):
    return _apply_repsim_kernel(x, y, Linear())


def mse_kernel(x, y):
    return _apply_repsim_kernel(x, y, lambda a, b: -pdist2(a, b))


def sqexp_kernel(x, y):
    try:
        return _apply_repsim_kernel(x, y, SquaredExponential(length_scale="auto"))
    except RuntimeWarning:
        return _apply_repsim_kernel(x, y, SquaredExponential(length_scale=1.))


def bootstrap_kernel(a: np.ndarray, b: np.ndarray,
        kernel: callable,
        n_samples: int,
        random_state: np.random.RandomState,
):  # randomly sample from a and b
    sample_a = random_state.randint(a.shape[1], size=[a.shape[0], n_samples])
    sample_b = random_state.randint(a.shape[1], size=[a.shape[0], n_samples])
    return kernel(np.take_along_axis(a, sample_a, axis=1),
                   np.take_along_axis(b, sample_b, axis=1))


def get_kernel_from_name(name: str, seed=None):
    if "linear" in name:  # hack: change to use torch tensors instead of np.ndarray
        kernel_fn = linear_kernel
    elif "mse" in name:
        kernel_fn = mse_kernel
    elif "sqexp" in name:
        kernel_fn = sqexp_kernel
    else:
        raise ValueError(f"Unrecognized name for kernel: {name}")

    if "bootstrap" in name:
        n_bootstrap = [int(x) for x in name.split("_") if x.isdigit()][0]
        kernel_fn = lambda x, y: bootstrap_kernel(
            x, y, kernel_fn, n_bootstrap, np.random.RandomState(seed))
    return kernel_fn
