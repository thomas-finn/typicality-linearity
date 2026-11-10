"""Train probes for each category and prompt format, and test how they transfer.

For one set of activations (one baseline), for each category c and prompt p:
- one probe is fitted on all items of c in p (the "full" probe)
- one probe is fitted for each fold f, on the items of c in p not in f

Category transfer, for each prompt p (rows: train category, columns: eval
category):
- off-diagonal (a, b): the full probe of (a, p), evaluated on all items of b in p
- diagonal (a, a): cross-validated. Each fold probe of (a, p) is evaluated on the
  held-out items of its fold in p.

Prompt transfer, for each category c (rows: train prompt, columns: eval prompt):
- all cells (pa, pb) are cross-validated. Each fold probe of (c, pa) is
  evaluated on the held-out items of its fold in pb. The items in the train and
  eval prompts are the same, so this prevents a probe from scoring well only
  because it recognises the items.

The folds are the same for all prompts, so the diagonals of the two matrices
agree.

Cosine similarity of probe directions uses the full probes, in activation
space (see probes.activation_direction):
- between categories, for each prompt: (n_categories, n_categories)
- between prompts, for each category: (n_prompts, n_prompts)

Cross-validated correlations are aggregated in one of two ways:
- "pooled": the correlation of all out-of-fold predictions with the targets
- "mean_over_folds": the mean of the correlations of each held-out fold
"""

import numpy as np

from typicality_linearity.probes import (
    METRICS,
    activation_direction,
    cosine_similarity,
    fit_probe,
)

AGGREGATES = ("pooled", "mean_over_folds")


def fit_rng(seed, category_index, prompt_index, fold):
    # fold is -1 for the full probe. Each fit gets its own random stream, so
    # the results do not depend on the order of the fits.
    return np.random.default_rng([seed, category_index, prompt_index, fold + 1])


def cv_correlation(predictions, y, folds, metric, aggregate):
    metric_function = METRICS[metric]
    if aggregate == "pooled":
        return metric_function(predictions, y)
    if aggregate == "mean_over_folds":
        values = []
        for fold in np.unique(folds):
            held_out = folds == fold
            values.append(metric_function(predictions[held_out], y[held_out]))
        return float(np.mean(values))
    raise ValueError(f"Unknown aggregate {aggregate!r}. Use one of {AGGREGATES}")


def fit_all_probes(x_by_prompt, y, categories, folds, probe_types, settings, seed):
    """Fit the full and fold probes for each category and prompt.

    Return (full, by_fold, alpha_records):
    - full[probe_type][category][prompt] = LinearProbe
    - by_fold[probe_type][category][prompt][fold] = LinearProbe
    - alpha_records: one dict per ridge fit
    """
    prompt_names = list(x_by_prompt)
    category_names = sorted(set(categories))
    categories = np.asarray(categories)
    n_folds = int(folds.max()) + 1
    full = {}
    by_fold = {}
    alpha_records = []
    for probe_type in probe_types:
        full[probe_type] = {}
        by_fold[probe_type] = {}
        for category_index, category in enumerate(category_names):
            rows = np.flatnonzero(categories == category)
            y_c = y[rows]
            folds_c = folds[rows]
            full[probe_type][category] = {}
            by_fold[probe_type][category] = {}
            for prompt_index, prompt in enumerate(prompt_names):
                x_c = x_by_prompt[prompt][rows]
                rng = fit_rng(seed, category_index, prompt_index, -1)
                probe, alpha = fit_probe(probe_type, x_c, y_c, settings, rng)
                full[probe_type][category][prompt] = probe
                records = [("all", alpha)]
                fold_probes = []
                for fold in range(n_folds):
                    train = folds_c != fold
                    rng = fit_rng(seed, category_index, prompt_index, fold)
                    probe, alpha = fit_probe(
                        probe_type, x_c[train], y_c[train], settings, rng
                    )
                    fold_probes.append(probe)
                    records.append((fold, alpha))
                by_fold[probe_type][category][prompt] = fold_probes
                if probe_type == "ridge":
                    for fold, alpha in records:
                        alpha_records.append(
                            {
                                "category": category,
                                "prompt": prompt,
                                "fold": fold,
                                "alpha": alpha,
                            }
                        )
    return full, by_fold, alpha_records


def cosine_matrices(directions, category_names, prompt_names):
    """Return (between_categories, between_prompts).

    directions: {probe_type: {category: {prompt: vector}}}
    - between_categories[probe_type][prompt]: (n_categories, n_categories)
    - between_prompts[probe_type][category]: (n_prompts, n_prompts)
    """
    between_categories = {}
    between_prompts = {}
    for probe_type, by_category in directions.items():
        between_categories[probe_type] = {}
        for prompt in prompt_names:
            matrix = np.empty((len(category_names), len(category_names)))
            for i, a in enumerate(category_names):
                for j, b in enumerate(category_names):
                    matrix[i, j] = cosine_similarity(
                        by_category[a][prompt], by_category[b][prompt]
                    )
            between_categories[probe_type][prompt] = matrix
        between_prompts[probe_type] = {}
        for category in category_names:
            matrix = np.empty((len(prompt_names), len(prompt_names)))
            for i, a in enumerate(prompt_names):
                for j, b in enumerate(prompt_names):
                    matrix[i, j] = cosine_similarity(
                        by_category[category][a], by_category[category][b]
                    )
            between_prompts[probe_type][category] = matrix
    return between_categories, between_prompts


def full_probe_directions(full):
    """Return {probe_type: {category: {prompt: direction}}} for the full probes."""
    return {
        probe_type: {
            category: {
                prompt: activation_direction(probe)
                for prompt, probe in by_prompt.items()
            }
            for category, by_prompt in by_category.items()
        }
        for probe_type, by_category in full.items()
    }


def out_of_fold_predictions(fold_probes, x_eval, folds):
    predictions = np.empty(len(x_eval))
    for fold, probe in enumerate(fold_probes):
        held_out = folds == fold
        predictions[held_out] = probe.predict(x_eval[held_out])
    return predictions


def transfer_matrices(
    x_by_prompt,
    y,
    categories,
    folds,
    probe_types,
    metrics,
    settings,
    aggregate,
    seed,
):
    """Return a dict with the transfer matrices and the ridge alphas.

    x_by_prompt: {prompt: array (n_rows, d_model)}, the same rows for each prompt
    y: typicality for each row (higher is more typical)
    categories: category of each row
    folds: fold id of each row (see probes.make_folds)

    Result:
    - "category_transfer"[probe_type][metric][prompt]: (n_categories, n_categories)
    - "prompt_transfer"[probe_type][metric][category]: (n_prompts, n_prompts)
    - "category_cosine"[probe_type][prompt]: (n_categories, n_categories)
    - "prompt_cosine"[probe_type][category]: (n_prompts, n_prompts)
    - "categories", "prompts": the order of the matrix rows and columns
    - "alphas": list of ridge alpha records
    - "full_probes"[probe_type][category][prompt]: LinearProbe
    - "fold_probes"[probe_type][category][prompt]: list of LinearProbe, one per fold
    """
    y = np.asarray(y, dtype=np.float64)
    categories = np.asarray(categories)
    prompt_names = list(x_by_prompt)
    category_names = sorted(set(categories.tolist()))
    rows_of = {c: np.flatnonzero(categories == c) for c in category_names}

    full, by_fold, alpha_records = fit_all_probes(
        x_by_prompt, y, categories, folds, probe_types, settings, seed
    )

    category_transfer = {}
    prompt_transfer = {}
    for probe_type in probe_types:
        category_transfer[probe_type] = {}
        prompt_transfer[probe_type] = {}
        for metric in metrics:
            by_prompt = {}
            for prompt in prompt_names:
                matrix = np.empty((len(category_names), len(category_names)))
                for i, train_category in enumerate(category_names):
                    for j, eval_category in enumerate(category_names):
                        rows = rows_of[eval_category]
                        x_eval = x_by_prompt[prompt][rows]
                        if i == j:
                            predictions = out_of_fold_predictions(
                                by_fold[probe_type][train_category][prompt],
                                x_eval,
                                folds[rows],
                            )
                            value = cv_correlation(
                                predictions, y[rows], folds[rows], metric, aggregate
                            )
                        else:
                            probe = full[probe_type][train_category][prompt]
                            value = METRICS[metric](probe.predict(x_eval), y[rows])
                        matrix[i, j] = value
                by_prompt[prompt] = matrix
            category_transfer[probe_type][metric] = by_prompt

            by_category = {}
            for category in category_names:
                rows = rows_of[category]
                matrix = np.empty((len(prompt_names), len(prompt_names)))
                for i, train_prompt in enumerate(prompt_names):
                    for j, eval_prompt in enumerate(prompt_names):
                        predictions = out_of_fold_predictions(
                            by_fold[probe_type][category][train_prompt],
                            x_by_prompt[eval_prompt][rows],
                            folds[rows],
                        )
                        matrix[i, j] = cv_correlation(
                            predictions, y[rows], folds[rows], metric, aggregate
                        )
                by_category[category] = matrix
            prompt_transfer[probe_type][metric] = by_category

    category_cosine, prompt_cosine = cosine_matrices(
        full_probe_directions(full), category_names, prompt_names
    )

    return {
        "category_transfer": category_transfer,
        "prompt_transfer": prompt_transfer,
        "category_cosine": category_cosine,
        "prompt_cosine": prompt_cosine,
        "categories": category_names,
        "prompts": prompt_names,
        "alphas": alpha_records,
        "full_probes": full,
        "fold_probes": by_fold,
    }
