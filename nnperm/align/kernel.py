import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def linear_kernel(x, y):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    return x @ y.T


def cosine_kernel(x, y):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    return cosine_similarity(x, y)


def loglinear_kernel(x, y):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    distance = 2 * np.log(np.maximum(euclidean_distances(x, y), 1e-16))
    sanity_check = np.sum(np.log(np.maximum((np.linalg.norm(x - y, axis=1))**2, 1e-16)))
    assert abs(np.trace(distance) - sanity_check) < 1e-8, (np.trace(distance), sanity_check)
    return -distance


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
    if "linear" in name:
        kernel_fn = linear_kernel
    elif "cosine" in name:
        kernel_fn = cosine_kernel
    elif "loglinear" in name:
        kernel_fn = loglinear_kernel
    else:
        raise ValueError(f"Unrecognized name for kernel: {name}")

    if "bootstrap" in name:
        n_bootstrap = [int(x) for x in name.split("_") if x.isdigit()][0]
        bootstrap_fn = lambda x, y: bootstrap_kernel(
            x, y, kernel_fn, n_bootstrap, np.random.RandomState(seed))
        return bootstrap_fn

    return kernel_fn
