"""Difference-of-means probes for typicality, folds and metrics.

The probe direction is the mean activation of the most typical 1/n_groups of
the training items minus the mean of the least typical 1/n_groups. The score
of an activation x is x . direction.

The target y is typicality, where a higher value is more typical.
"""

import numpy as np

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


# Difference of means


def fit_diff_of_means(x, y, n_groups=3):
    """Return the difference-of-means direction, shape (d_model,).

    The top group is the n // n_groups most typical items and the bottom group
    the n // n_groups least typical items. Ties are put in order by position.
    """
    x = np.asarray(x, dtype=np.float64)
    least_typical, most_typical = extreme_groups(y, n_groups)
    return x[most_typical].mean(axis=0) - x[least_typical].mean(axis=0)
