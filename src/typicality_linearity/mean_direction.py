"""Mean direction over a subset of categories and all prompt formats.

For a subset S of categories, the mean direction is the mean of the unit
directions u(c, q) of the probes of all (c, q), for c in S and all training
prompts q. It is one direction. It is evaluated on the activations of each
prompt p. The score is linear, so

    score(i, p, S) = mean over c in S of P[p][i, c]
    P[p][i, c] = x(i, p) . m_c,   m_c = mean over q of u(c, q)

P is computed once, and the scores of any subset come from P. m_c is not
normalised: it is the literal mean of the unit directions, so a category whose
directions agree between prompts has a longer m_c and counts a little more.

No item is scored with a direction that was trained on it:
- c is the category of item i: u(c, q) is the fold probe of (c, q) that did not
  see i. The folds are the same for all prompts, so this probe did not see item
  i in any prompt.
- c is a different category: u(c, q) is the full probe of (c, q)

The score of a subset is evaluated in each prompt and category separately.

Ranking of subsets ("which subset transfers best"): for subset S and a
held-out category c (c not in S), the score is compared with the mean score of
all subsets of the same size that do not contain c. This removes the effect of
the evaluation category: some categories (weapon, clothing) are easy for all
directions. The advantage of S is the mean of these differences over the
held-out categories. Subsets with fewer than MIN_HELD_OUT held-out categories
are not ranked.

The advantage compares subsets of the same size only. To compare sizes, use
held_out_by_size: the mean held-out score of all subsets of each size.

The prompts in the projections can also be evaluation prompts, which are never
used to train the probes. The directions are averaged over the training prompts
in `full` only.
"""

import numpy as np

from typicality_linearity.probes import activation_direction, average_ranks

MIN_HELD_OUT = 3


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def mean_unit_direction(probes):
    """Return the mean of the unit directions of a list of probes."""
    return np.mean([unit(activation_direction(probe)) for probe in probes], axis=0)


def item_projections(x_by_prompt, categories, folds, full, by_fold):
    """Return {prompt: array (n_items, n_categories)} of P[p][i, c].

    full[category][prompt]: LinearProbe; by_fold[category][prompt]: list.
    The directions are averaged over all prompts in full. The columns are in
    sorted category order.
    """
    categories = np.asarray(categories)
    category_names = sorted(full)
    n_folds = int(np.max(folds)) + 1
    full_mean = {}
    fold_mean = {}
    for category in category_names:
        by_prompt = full[category]
        full_mean[category] = mean_unit_direction(list(by_prompt.values()))
        fold_mean[category] = [
            mean_unit_direction([by_fold[category][q][fold] for q in by_prompt])
            for fold in range(n_folds)
        ]
    projections = {}
    for prompt, x in x_by_prompt.items():
        x = np.asarray(x, dtype=np.float64)
        values = np.empty((len(x), len(category_names)))
        for k, category in enumerate(category_names):
            values[:, k] = x @ full_mean[category]
            own = categories == category
            for fold in range(n_folds):
                rows = own & (folds == fold)
                values[rows, k] = x[rows] @ fold_mean[category][fold]
        projections[prompt] = values
    return projections


def all_subsets(n_categories):
    """Return a bool array (n_categories, n_subsets) of all subsets that are
    not empty and not the full set."""
    masks = []
    for code in range(1, 2**n_categories - 1):
        masks.append([(code >> k) & 1 == 1 for k in range(n_categories)])
    return np.array(masks).T


def column_correlations(scores, y, metric):
    """Return the correlation of y with each column of scores, shape (n_cols,).

    For Spearman, ties in the scores are ignored (the scores are continuous).
    """
    if metric == "spearman":
        scores = np.argsort(np.argsort(scores, axis=0), axis=0).astype(np.float64)
        y = average_ranks(y)
    elif metric != "pearson":
        raise ValueError(f"Unknown metric {metric!r}")
    scores = scores - scores.mean(axis=0)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.sqrt((scores * scores).sum(axis=0) * (y * y).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (y @ scores) / denominator, np.nan)


def subset_scores(projections, y, categories, category_names, subsets, metric):
    """Return an array (n_prompts, n_categories, n_subsets): the correlation of
    the score of each subset with y, in each evaluation category."""
    categories = np.asarray(categories)
    weights = subsets / subsets.sum(axis=0)
    prompts = list(projections)
    result = np.empty((len(prompts), len(category_names), subsets.shape[1]))
    for p, prompt in enumerate(prompts):
        for k, category in enumerate(category_names):
            rows = categories == category
            scores = projections[prompt][rows] @ weights
            result[p, k] = column_correlations(scores, y[rows], metric)
    return result


def rank_subsets(scores, subsets, top=10, eval_scores=None):
    """Return a list of dicts for the best subsets, best first.

    scores: (n_prompts, n_categories, n_subsets). The values are first averaged
    over the prompts.
    eval_scores: optional scores on other prompts, same shape except the first
    axis. They are not used for the ranking. Each record then also has
    "eval_held_out_mean".
    """
    mean_scores = np.nanmean(scores, axis=0)  # (n_categories, n_subsets)
    held_out = ~subsets
    sizes = subsets.sum(axis=0)
    n_categories = subsets.shape[0]

    # Reference: mean score of the subsets of the same size without c.
    reference = np.full((n_categories, n_categories + 1), np.nan)
    for size in np.unique(sizes):
        same_size = sizes == size
        for c in range(n_categories):
            columns = same_size & held_out[c]
            if columns.any():
                reference[c, size] = np.nanmean(mean_scores[c, columns])

    eval_means = None
    if eval_scores is not None:
        eval_means = np.nanmean(eval_scores, axis=0)

    records = []
    for s in range(subsets.shape[1]):
        held = np.flatnonzero(held_out[:, s])
        if len(held) < MIN_HELD_OUT:
            continue
        raw = mean_scores[held, s]
        relative = raw - reference[held, sizes[s]]
        record = {
            "subset": np.flatnonzero(subsets[:, s]).tolist(),
            "held_out_mean": float(np.nanmean(raw)),
            "advantage": float(np.nanmean(relative)),
            "in_subset_mean": float(np.nanmean(mean_scores[subsets[:, s], s])),
        }
        if eval_means is not None:
            record["eval_held_out_mean"] = float(np.nanmean(eval_means[held, s]))
        records.append(record)
    records.sort(key=lambda record: -record["advantage"])
    return records[:top]


def held_out_by_size(scores, subsets):
    """Return {size: mean held-out score over all subsets of that size}.

    scores: (n_prompts, n_categories, n_subsets), averaged over the prompts.
    Each category is held out equally often for each size, so the sizes can be
    compared.
    """
    mean_scores = np.nanmean(scores, axis=0)
    sizes = subsets.sum(axis=0)
    result = {}
    for size in np.unique(sizes):
        columns = sizes == size
        held = mean_scores[:, columns][~subsets[:, columns]]
        result[int(size)] = float(np.nanmean(held))
    return result


def leave_one_out_scores(projections, y, categories, category_names, metric):
    """Return (loco, all), arrays (n_prompts, n_categories).

    loco[p, c]: score on category c of the mean direction over all other
    categories. all[p, c]: score on c of the mean direction over all categories
    (for c itself, the fold directions).
    """
    n = len(category_names)
    subsets = np.concatenate(
        [~np.eye(n, dtype=bool), np.ones((n, 1), dtype=bool)], axis=1
    )
    scores = subset_scores(projections, y, categories, category_names, subsets, metric)
    diagonal = np.arange(n)
    return scores[:, diagonal, diagonal], scores[:, :, n]


def quantise(projections, prompts):
    """Return (int16 array (n_prompts, n_items, n_categories), scales
    (n_prompts, n_categories)). value = int16 x scale."""
    values = np.stack([projections[p] for p in prompts])
    scales = np.abs(values).max(axis=1) / 32767.0
    scales[scales == 0] = 1.0
    # Little-endian, as the page reads it with Int16Array.
    quantised = np.round(values / scales[:, None, :]).astype("<i2")
    return quantised, scales
