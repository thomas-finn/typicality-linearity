"""Linear probes for typicality, folds and correlation metrics.

Two probe types:
- "diff_of_means": the direction is the mean activation of the most typical
  1/n_groups of the training items minus the mean of the least typical
  1/n_groups (default: thirds). The means can be weighted (see
  fit_diff_of_means). The score of an activation x is x . direction.
- "ridge": ridge regression of typicality on standardised activations. Alpha is
  chosen from a grid by inner cross-validation on the training items only, or
  is fixed (setting "ridge_alpha").

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


def extreme_groups(y, n_groups):
    """Return (least typical indices, most typical indices).

    Each group has len(y) // n_groups items. Ties are put in order by position.
    """
    y = np.asarray(y, dtype=np.float64)
    n_group = len(y) // n_groups
    if n_groups < 2 or n_group == 0:
        raise ValueError(f"Need n_groups >= 2 and {n_groups} or more items")
    order = np.argsort(y, kind="stable")
    return order[:n_group], order[-n_group:]


def wilcoxon_auc(scores, y, n_groups):
    """Return the Wilcoxon rank-sum (Mann-Whitney) AUC of the scores.

    The two groups are the most and least typical len(y) // n_groups items
    (see extreme_groups). The AUC is the probability that a most typical item
    has a higher score than a least typical item; ties count 0.5. 0.5 is
    chance and 1 is a perfect separation. NaN if a score is not finite.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        return float("nan")
    least, most = extreme_groups(y, n_groups)
    ranks = average_ranks(np.concatenate([scores[most], scores[least]]))
    n_most = len(most)
    n_least = len(least)
    u = ranks[:n_most].sum() - n_most * (n_most + 1) / 2
    return float(u / (n_most * n_least))


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


WEIGHT_GROUPS = ("both", "top")


def group_weights(y, weight_constant):
    """Return exp(weight_constant * y), normalised to sum to 1.

    The max is subtracted before exp, so a large value does not overflow.
    """
    exponent = weight_constant * np.asarray(y, dtype=np.float64)
    weights = np.exp(exponent - exponent.max())
    return weights / weights.sum()


def fit_diff_of_means(x, y, n_groups=3, weight_constant=0.0, weight_groups="both"):
    """Return a difference-of-means probe.

    The top group is the n // n_groups most typical items and the bottom group
    the n // n_groups least typical items. Ties are put in order by position.

    weight_constant c > 0 weights the means. y is typicality (higher is more
    typical), so with y = -rating_zscore the weights are exp(-c * rating_zscore).
    - top group: weight exp(c * y), so the most typical items count more
    - bottom group: weight exp(-c * y) when weight_groups is "both", so the
      least typical items count more. With "top", the bottom mean is not weighted.
    c = 0 gives the unweighted difference of means.
    """
    if weight_groups not in WEIGHT_GROUPS:
        raise ValueError(f"weight_groups must be one of {WEIGHT_GROUPS}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    least_typical, most_typical = extreme_groups(y, n_groups)
    top_weights = group_weights(y[most_typical], weight_constant)
    if weight_groups == "both":
        bottom_weights = group_weights(-y[least_typical], weight_constant)
    else:
        bottom_weights = group_weights(y[least_typical], 0.0)
    direction = top_weights @ x[most_typical] - bottom_weights @ x[least_typical]
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
    If settings has "ridge_alpha" (not None), ridge uses this alpha and does no
    inner cross-validation. "ridge_alphas" and "ridge_inner_folds" are then not
    used.
    Optional diff_of_means settings (default: thirds, no weighting):
    "dom_n_groups", "dom_weight_constant", "dom_weight_groups".
    """
    if probe_type == "diff_of_means":
        probe = fit_diff_of_means(
            x,
            y,
            settings.get("dom_n_groups", 3),
            settings.get("dom_weight_constant", 0.0),
            settings.get("dom_weight_groups", "both"),
        )
        return probe, None
    if probe_type == "ridge":
        fixed = settings.get("ridge_alpha")
        if fixed is not None:
            alpha = float(fixed)
            return fit_ridge(x, y, alpha, settings["ridge_standardise"]), alpha
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
