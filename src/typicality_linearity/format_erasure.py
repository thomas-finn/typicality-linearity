"""Remove prompt-format information from activations.

Three methods. All use a label for each row: the prompt format (group) of
the row. No typicality values are used.

- Centering: subtract the mean of each group from its rows.
- LEACE (Belrose et al. 2023, "LEACE: Perfect linear concept erasure in
  closed form"): the smallest affine change (in the whitened metric) after
  which no linear classifier can do better than a constant at predicting the
  group. After LEACE, all group means are equal. With k groups, at most k - 1
  directions are changed.
- INLP (Ravfogel et al. 2020, "Null it out"): fit a linear classifier for the
  group, project the activations onto the null space of its weights, and do
  this again. Each step removes up to k directions (orthogonal projection).

An eraser is stored as x' = x - L R (x - centre), with L (d, r) and R (r, d).
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class LinearEraser:
    centre: np.ndarray  # (d,)
    left: np.ndarray  # (d, r)
    right: np.ndarray  # (r, d)
    history: list = field(default_factory=list)  # INLP: one dict per step

    @property
    def rank(self):
        return self.left.shape[1]

    def apply(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x - ((x - self.centre) @ self.right.T) @ self.left.T

    def raw_direction(self, direction):
        """Return v such that apply(x) . direction = x . v + constant."""
        direction = np.asarray(direction, dtype=np.float64)
        return direction - self.right.T @ (self.left.T @ direction)


def identity_eraser(d):
    return LinearEraser(np.zeros(d), np.zeros((d, 0)), np.zeros((0, d)))


def center_by_group(x, groups):
    """Return x with the mean of each group subtracted from its rows."""
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray(groups)
    result = np.empty_like(x)
    for group in np.unique(groups):
        rows = groups == group
        result[rows] = x[rows] - x[rows].mean(axis=0)
    return result


def one_hot(groups):
    groups = np.asarray(groups)
    names = np.unique(groups)
    return (groups[:, None] == names[None, :]).astype(np.float64)


def between_group_variance_fraction(x, groups):
    """Return the fraction of the total variance that the group means explain."""
    x = np.asarray(x, dtype=np.float64)
    total = ((x - x.mean(axis=0)) ** 2).sum()
    if total == 0:
        return 0.0
    group_means = x - center_by_group(x, groups)
    between = ((group_means - x.mean(axis=0)) ** 2).sum()
    return float(between / total)


def fit_leace(x, groups, tolerance=1e-8, rank_tolerance=1e-6):
    """Return the LEACE eraser for the group labels."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    z = one_hot(groups)
    centre = x.mean(axis=0)
    x_centred = x - centre
    z_centred = z - z.mean(axis=0)
    covariance = x_centred.T @ x_centred / n
    cross = x_centred.T @ z_centred / n  # (d, k)

    values, vectors = np.linalg.eigh(covariance)
    keep = values > tolerance * values.max()
    vectors = vectors[:, keep]
    root = np.sqrt(values[keep])
    whiten = (vectors / root) @ vectors.T
    unwhiten = (vectors * root) @ vectors.T

    # In the whitened space the singular values are at most 0.5, so an
    # absolute tolerance removes float noise (for example after centering).
    u, s, _ = np.linalg.svd(whiten @ cross, full_matrices=False)
    u = u[:, s > rank_tolerance]
    if u.shape[1] == 0:
        return identity_eraser(d)
    return LinearEraser(centre, unwhiten @ u, u.T @ whiten)


def split_rows(n, dev_fraction, rng):
    """Return (train rows, dev rows) for a random split."""
    order = rng.permutation(n)
    n_dev = round(n * dev_fraction)
    return np.sort(order[n_dev:]), np.sort(order[:n_dev])


def fit_group_classifier(x, groups, train, c_value, max_iter, seed):
    """Fit a logistic regression on standardised x[train].

    Return (weights in the space of x with shape (k, d), classifier, centre,
    scale). Standardising helps the solver; the weights are then divided by
    the scale.
    """
    centre = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x[train] - centre) / scale
    classifier = LogisticRegression(C=c_value, max_iter=max_iter, random_state=seed)
    classifier.fit(z, np.asarray(groups)[train])
    weights = classifier.coef_ / scale
    return weights, classifier, centre, scale


def group_accuracy(x, groups, dev_fraction, c_value, max_iter, seed):
    """Return the dev accuracy of a linear classifier for the groups."""
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    train, dev = split_rows(len(x), dev_fraction, rng)
    _, classifier, centre, scale = fit_group_classifier(
        x, groups, train, c_value, max_iter, seed
    )
    return float(classifier.score((x[dev] - centre) / scale, groups[dev]))


def orthonormal_new_rows(weights, basis, tolerance=1e-10):
    """Return an orthonormal basis (d, r) of the rows of weights, minus basis."""
    vectors = weights.T  # (d, k)
    if basis.shape[1] > 0:
        vectors = vectors - basis @ (basis.T @ vectors)
    u, s, _ = np.linalg.svd(vectors, full_matrices=False)
    if s.size == 0 or s.max() == 0:
        return np.zeros((weights.shape[1], 0))
    return u[:, s > tolerance * s.max()]


def fit_inlp(
    x,
    groups,
    n_iterations,
    dev_fraction=0.3,
    c_value=1.0,
    max_iter=1000,
    stop_accuracy=None,
    seed=0,
):
    """Return the INLP eraser for the group labels.

    Each step fits a classifier on the train split of the projected x, records
    its dev accuracy, and adds the rows of its weights to the removed space.
    The steps stop after n_iterations, or before a step whose dev accuracy is
    below stop_accuracy (that classifier is then not removed).
    """
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray(groups)
    d = x.shape[1]
    rng = np.random.default_rng(seed)
    train, dev = split_rows(len(x), dev_fraction, rng)
    basis = np.zeros((d, 0))
    history = []
    for step in range(n_iterations):
        projected = x - (x @ basis) @ basis.T
        weights, classifier, centre, scale = fit_group_classifier(
            projected, groups, train, c_value, max_iter, seed
        )
        accuracy = float(
            classifier.score((projected[dev] - centre) / scale, groups[dev])
        )
        stop = stop_accuracy is not None and accuracy < stop_accuracy
        new_rows = np.zeros((d, 0)) if stop else orthonormal_new_rows(weights, basis)
        basis = np.concatenate([basis, new_rows], axis=1)
        history.append(
            {
                "step": step,
                "dev_accuracy": accuracy,
                "removed": int(new_rows.shape[1]),
                "rank": int(basis.shape[1]),
            }
        )
        if stop or new_rows.shape[1] == 0:
            break
    return LinearEraser(np.zeros(d), basis, basis.T.copy(), history)
