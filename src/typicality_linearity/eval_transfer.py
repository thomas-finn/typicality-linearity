"""Evaluation of probes on prompts that are never used for training.

The probes are trained on the training prompts, with or without a baseline.
They are evaluated on the raw activations of the evaluation prompts: no
baseline is applied, because the aim is a typicality direction in the residual
stream itself. The score is linear in the activations, so a probe trained with
the mean contrast baseline can be applied to raw activations.

For the category of the item, the fold probe that did not see the item is used.
The folds are the same for all prompts, so this probe did not see the item in
any prompt.
"""

import numpy as np

from typicality_linearity.probe_transfer import cv_correlation, out_of_fold_predictions
from typicality_linearity.probes import METRICS


def eval_transfer(
    full,
    by_fold,
    x_eval_by_prompt,
    y,
    categories,
    folds,
    category_names,
    train_prompts,
    metric,
    aggregate,
):
    """Return an array (n_train_prompts, n_eval_prompts, n_categories, n_categories).

    [q, e, i, j]: the probe of category i and training prompt q, evaluated on
    prompt e and category j. If i == j, the fold probes are used.
    full[category][prompt]: LinearProbe; by_fold[category][prompt]: list.
    """
    y = np.asarray(y, dtype=np.float64)
    categories = np.asarray(categories)
    eval_prompts = list(x_eval_by_prompt)
    rows_of = {c: np.flatnonzero(categories == c) for c in category_names}
    n_categories = len(category_names)
    result = np.empty(
        (len(train_prompts), len(eval_prompts), n_categories, n_categories)
    )
    for q, train_prompt in enumerate(train_prompts):
        for e, eval_prompt in enumerate(eval_prompts):
            x = np.asarray(x_eval_by_prompt[eval_prompt], dtype=np.float64)
            for i, train_category in enumerate(category_names):
                for j, eval_category in enumerate(category_names):
                    rows = rows_of[eval_category]
                    if i == j:
                        predictions = out_of_fold_predictions(
                            by_fold[train_category][train_prompt],
                            x[rows],
                            folds[rows],
                        )
                        value = cv_correlation(
                            predictions, y[rows], folds[rows], metric, aggregate
                        )
                    else:
                        probe = full[train_category][train_prompt]
                        value = METRICS[metric](probe.predict(x[rows]), y[rows])
                    result[q, e, i, j] = value
    return result


def eval_summary(matrix):
    """Return the mean of the diagonal (CV) and off-diagonal cells."""
    n_categories = matrix.shape[-1]
    diagonal = np.eye(n_categories, dtype=bool)
    return {
        "eval_within": float(np.nanmean(matrix[:, :, diagonal])),
        "eval_between_categories": float(np.nanmean(matrix[:, :, ~diagonal])),
    }


def permute_within(y, categories, rng):
    """Return y with the values shuffled inside each category."""
    y = np.asarray(y)
    categories = np.asarray(categories)
    shuffled = y.copy()
    for category in np.unique(categories):
        rows = np.flatnonzero(categories == category)
        shuffled[rows] = y[rng.permutation(rows)]
    return shuffled


def null_summary(observed, null_values):
    """Return the summary of a permutation null for one statistic.

    p is one-sided: the fraction of null values that are at least the observed
    value, with the observed value counted once.
    """
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[~np.isnan(null_values)]
    n = len(null_values)
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null_values)),
        "null_sd": float(np.std(null_values)),
        "null_q95": float(np.quantile(null_values, 0.95)),
        "null_max": float(np.max(null_values)),
        "p_value": float((1 + np.sum(null_values >= observed)) / (1 + n)),
        "n_permutations": n,
    }
