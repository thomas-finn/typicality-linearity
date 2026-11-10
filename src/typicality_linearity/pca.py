"""Principal component analysis with NumPy."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PcaResult:
    projections: np.ndarray  # (n_samples, n_components)
    components: np.ndarray  # (n_components, n_features)
    explained_variance_ratio: np.ndarray  # (n_components,)
    mean: np.ndarray  # (n_features,)


def fit_pca(x, n_components):
    """Fit PCA to x with shape (n_samples, n_features).

    Each component has a fixed sign: its largest absolute loading is positive.
    This makes the plots the same between runs.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    centred = x - mean
    _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)

    components = vt[:n_components]
    largest = np.abs(components).argmax(axis=1)
    signs = np.sign(components[np.arange(len(components)), largest])
    components = components * signs[:, None]

    variance = singular_values**2
    ratio = variance[:n_components] / variance.sum()
    projections = centred @ components.T
    return PcaResult(projections, components, ratio, mean)


def explained_variance_ratio_all(x):
    """Return the explained variance ratio of all components of x.

    x has shape (n_samples, n_features). There are min(n_samples, n_features)
    values; at most n_samples - 1 of them are not zero.
    """
    x = np.asarray(x, dtype=np.float64)
    centred = x - x.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    variance = singular_values**2
    return variance / variance.sum()


def n_components_for_variance(ratio, threshold):
    """Return the smallest number of components whose ratios sum to threshold.

    ratio: explained variance ratios, largest first. A small tolerance stops
    float rounding from adding one component.
    """
    cumulative = np.cumsum(ratio)
    return int(np.searchsorted(cumulative, threshold - 1e-12) + 1)
