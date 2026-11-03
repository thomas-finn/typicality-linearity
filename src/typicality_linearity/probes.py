"""Linear probes for typicality, folds and correlation metrics.

Two probe types:
- "diff_of_means": the direction is the mean activation of the most typical
  third of the training items minus the mean of the least typical third.
  The score of an activation x is x . direction.
- "ridge": ridge regression of typicality on standardised activations. Alpha is
  chosen from a grid by inner cross-validation on the training items only.

The target y is typicality, where a higher value is more typical.
"""

from dataclasses import dataclass

import numpy as np

PROBE_TYPES = ("diff_of_means", "ridge")


@dataclass
class LinearProbe:
    weights: np.ndarray  # (d_model,)
    centre: np.ndarray  # (d_model,), subtracted before the weights
    scale: np.ndarray  # (d_model,), divides after the centre
    offset: float  # added to the score

    def predict(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.centre) / self.scale
        return z @ self.weights + self.offset


def activation_direction(probe):
    """Return the probe direction in activation space.

    score = ((x - centre) / scale) . weights = x . (weights / scale) + constant,
    so the direction is weights / scale. For difference-of-means, scale is 1.
    """
    return probe.weights / probe.scale


def cosine_similarity(a, b):
    """Return the cosine similarity, or NaN if one vector is zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("nan")
    return float(a @ b / denominator)


# Metrics


def average_ranks(values):
    """Return ranks 1..n. Tied values get the mean of their ranks."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def pearson(a, b):
    """Return the Pearson correlation, or NaN if one input is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    if denominator == 0:
        return float("nan")
    return float((a * b).sum() / denominator)


def spearman(a, b):
    return pearson(average_ranks(a), average_ranks(b))


METRICS = {"spearman": spearman, "pearson": pearson}


# Folds


def stratified_folds(values, n_folds, rng):
    """Return a fold id for each value.

    Sort the values, cut them into blocks of n_folds, and give each item in a
    block a different random fold. Each fold then has a similar range of values.
    """
    values = np.asarray(values)
    folds = np.empty(len(values), dtype=np.int64)
    order = np.argsort(values, kind="stable")
    for start in range(0, len(values), n_folds):
        block = order[start : start + n_folds]
        folds[block] = rng.permutation(n_folds)[: len(block)]
    return folds


def make_folds(categories, values, n_folds, seed):
    """Return a fold id for each row. Folds are made in each category separately."""
    if n_folds < 2:
        raise ValueError("n_folds must be 2 or more")
    categories = np.asarray(categories)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(categories), dtype=np.int64)
    for category in sorted(set(categories.tolist())):
        rows = np.flatnonzero(categories == category)
        if len(rows) < n_folds:
            raise ValueError(f"Category {category} has fewer rows than folds")
        folds[rows] = stratified_folds(np.asarray(values)[rows], n_folds, rng)
    return folds


def random_folds(n, n_folds, rng):
    """Return a fold id for each of n rows, with nearly equal fold sizes."""
    return rng.permutation(np.arange(n) % n_folds)


# Difference of means


def fit_diff_of_means(x, y):
    x = np.asarray(x, dtype=np.float64)
    n_third = len(y) // 3
    if n_third == 0:
        raise ValueError("Difference of means needs 3 or more training items")
    order = np.argsort(y, kind="stable")
    least_typical = order[:n_third]
    most_typical = order[-n_third:]
    direction = x[most_typical].mean(axis=0) - x[least_typical].mean(axis=0)
    d_model = x.shape[1]
    return LinearProbe(direction, np.zeros(d_model), np.ones(d_model), 0.0)


# Ridge


def standardise_parameters(x, standardise):
    centre = x.mean(axis=0)
    if not standardise:
        return centre, np.ones(x.shape[1])
    scale = x.std(axis=0)
    # A feature that does not change in the training items gets scale 1.
    scale[scale < 1e-12] = 1.0
    return centre, scale


def ridge_weights_for_alphas(z, y_centred, alphas):
    """Return the ridge weights for each alpha, shape (n_alphas, d_model).

    z must be centred. Uses the SVD z = U S V^T:
    w(alpha) = V diag(s / (s^2 + alpha)) U^T y.
    """
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    uty = u.T @ y_centred
    weights = []
    for alpha in alphas:
        weights.append(vt.T @ (s / (s * s + alpha) * uty))
    return np.array(weights)


def choose_ridge_alpha(x, y, alphas, n_inner_folds, standardise, rng):
    """Return the alpha with the smallest inner cross-validation squared error.

    If two alphas have the same error, the larger alpha is chosen.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    folds = random_folds(len(y), n_inner_folds, rng)
    errors = np.zeros(len(alphas))
    for fold in range(n_inner_folds):
        test = folds == fold
        train = ~test
        centre, scale = standardise_parameters(x[train], standardise)
        y_mean = y[train].mean()
        z_train = (x[train] - centre) / scale
        z_test = (x[test] - centre) / scale
        weights = ridge_weights_for_alphas(z_train, y[train] - y_mean, alphas)
        predictions = z_test @ weights.T + y_mean  # (n_test, n_alphas)
        errors += ((predictions - y[test][:, None]) ** 2).sum(axis=0)
    best = np.flatnonzero(errors == errors.min())
    order = np.argsort(alphas)
    for index in order[::-1]:
        if index in best:
            return float(alphas[index])
    raise AssertionError("No alpha found")


def fit_ridge(x, y, alpha, standardise):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    centre, scale = standardise_parameters(x, standardise)
    y_mean = float(y.mean())
    z = (x - centre) / scale
    weights = ridge_weights_for_alphas(z, y - y_mean, [alpha])[0]
    return LinearProbe(weights, centre, scale, y_mean)


# Common interface


def fit_probe(probe_type, x, y, settings, rng):
    """Return (LinearProbe, chosen alpha or None).

    settings: {"ridge_alphas": [...], "ridge_inner_folds": int,
               "ridge_standardise": bool}
    """
    if probe_type == "diff_of_means":
        return fit_diff_of_means(x, y), None
    if probe_type == "ridge":
        alphas = np.asarray(settings["ridge_alphas"], dtype=np.float64)
        alpha = choose_ridge_alpha(
            x,
            y,
            alphas,
            settings["ridge_inner_folds"],
            settings["ridge_standardise"],
            rng,
        )
        return fit_ridge(x, y, alpha, settings["ridge_standardise"]), alpha
    raise ValueError(f"Unknown probe type {probe_type!r}. Use one of {PROBE_TYPES}")
