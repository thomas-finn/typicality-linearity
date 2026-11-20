"""Difference-of-means probes pooled over categories or over prompts.

The points, activation types (`activation_types`, with the nonce contrast
from `nonce_csv`), settings (ambiguity, activation type, layer, n_groups),
folds, metrics and CV aggregation are the same as in
global_probe_experiment.py.

Baselines are used only for training. The probes are trained on the
activations of the activation type, and all scores (also on the evaluation
prompts) are computed on the raw activations. For the evaluation prompts,
only the raw activations are loaded, from `eval_cat_contrast_csv` (the
target rows).

Here the probes are pooled:

- Over categories (page 1): one probe per prompt, trained on the items of all
  categories together. `over_categories_groups` sets the groups:
  - "per_category": the most and least typical 1/n_groups of the training
    items of each category, put together. Each category has the same number
    of items in the top and the bottom group, so an offset between the
    categories cancels in the difference of means.
  - "all": the most and least typical 1/n_groups of all training items
    (typicality is a z-score in each category). If one category has more items
    in the top group than in the bottom group, its offset goes into the
    direction, and this can be larger than the typicality signal.
  Matrix rows train prompt, columns eval prompt: the training prompts, then
  the evaluation prompts (`eval_prompt_columns`, never used for training).
  Off-diagonal cells (and the evaluation prompt columns) are cross-validated
  with the same folds: each fold probe is evaluated on the held-out items in
  the eval prompt. The diagonal is cross-validated, or with `diagonal: full`
  the probe on all items is evaluated on the same items (in-sample). The
  metric is computed in
  each category (fold z-scored as in global_probe_experiment), then averaged
  over the categories. So the value measures typicality grading inside the
  categories, not the order of the categories.

- Over prompts (page 2): one probe per category, trained on the items of that
  category in all prompts together. The groups are the most and least typical
  1/n_groups of the items; the direction is the mean over the prompts of
  (mean of the top group - mean of the bottom group), which is the difference
  of means of all (item, prompt) rows. Matrix rows train category, columns
  eval category. Off-diagonal: the probe on all items of the train category,
  evaluated on the eval category. Diagonal: cross-validated, or with
  `diagonal: full` the probe on all items of the category (in-sample). The
  metric is computed in each prompt, then averaged over the prompts.

- Over categories and prompts (page 3, only with `eval_prompt_columns`): one
  probe per setting, trained on all items of all categories in all training
  prompts (the groups as for page 1; the direction is the mean over the
  prompts). It scores every item in each evaluation prompt. This probe is not
  cross-validated: the items are the same as in training, only the prompt
  format is new. The scores are stored raw and z-scored in each category, and
  the metrics are computed in each category, then averaged.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml, metadata.json
- matrices.npz and axes.json:
  - over_categories (setting..., metric, train prompt, eval prompt)
  - over_categories_by_category (..., eval category): before the mean
  - over_categories_eval and over_categories_eval_by_category: the same, with
    the evaluation prompts as the eval prompt axis
  - over_prompts (setting..., metric, train category, eval category)
  - over_prompts_by_prompt (..., eval prompt): before the mean
  - global_scores_raw and global_scores_z (setting..., eval prompt, point):
    page 3 scores of all 565 points (NaN for an excluded item), and
    global_eval_metrics (setting..., metric, eval prompt)
  - over_categories_cosine (setting..., prompt, prompt) and
    over_prompts_cosine (setting..., category, category): cosine of the
    directions trained on all items
- summary.csv: one row per setting and metric, with the means of the
  diagonal and off-diagonal cells of each matrix. `within_method` tells how
  the diagonal ("within") columns are made: "cv", or "in_sample" with
  diagonal "full". `global_eval_prompts_in_sample` is page 3: always
  in-sample for the items (only the prompt format is new).
- <html_name>: the page (see global_pooled_probe_plot.py). If `copy_html_to`
  is set, the page is also copied there.
"""

import csv
import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.activation_source import LazyModel
from typicality_linearity.global_pca_experiment import (
    ACTIVATION_FORMULAS,
    ACTIVATION_LABELS,
    extraction_config,
)
from typicality_linearity.global_pooled_probe_plot import write_html
from typicality_linearity.global_probe_experiment import (
    AMBIGUITY_LABELS,
    GROUP_LABELS,
    METRIC_CHANCE,
    METRIC_LABELS,
    check_groups,
    config_activation_types,
    config_diagonal,
    cosine_matrix,
    cv_value,
    diagonal_mean,
    keep_mask,
    load_points,
    metric_value,
    off_diagonal_mean,
    out_of_fold_scores,
    within_method,
)
from typicality_linearity.global_probe_experiment import (
    load_config as load_probe_config,
)
from typicality_linearity.probes import extreme_groups, fit_diff_of_means, make_folds
from typicality_linearity.rosch_prompts import EVAL_PROMPT_FORMATS, PROMPT_FORMATS
from typicality_linearity.run_metadata import (
    make_run_id,
    now_utc,
    set_seed,
    start_metadata,
    write_metadata,
)

SUMMARY_COLUMNS = (
    "ambiguity",
    "activation_type",
    "layer",
    "n_groups",
    "metric",
    "n_items",
    "within_method",
    "over_categories_within_prompt",
    "over_categories_between_prompts",
    "over_categories_eval_prompts",
    "global_eval_prompts_in_sample",
    "over_prompts_within_category",
    "over_prompts_between_categories",
    "over_categories_cosine_between",
    "over_prompts_cosine_between",
)

GROUP_MODES = ("per_category", "all")


def load_config(path):
    config = load_probe_config(path)
    if config.get("over_categories_groups") not in GROUP_MODES:
        raise ValueError(f"over_categories_groups must be one of {GROUP_MODES}")
    if config.get("eval_prompt_columns") and not config.get("eval_cat_contrast_csv"):
        raise ValueError("eval_prompt_columns needs eval_cat_contrast_csv")
    return config


def check_same_rows(point_rows, eval_rows):
    """The evaluation files must have the same points in the same order."""
    keys = [(r["item"], r["item_category"]) for r in point_rows]
    eval_keys = [(r["item"], r["item_category"]) for r in eval_rows]
    if keys != eval_keys:
        raise ValueError("The evaluation files do not have the same points in order")


def pooled_category_direction(x, rows, y, categories, n_groups, mode):
    """Difference of means of the rows in rows, over all their categories."""
    if mode == "all":
        return fit_diff_of_means(x[rows], y[rows], n_groups)
    if mode != "per_category":
        raise ValueError(f"Unknown over_categories_groups {mode!r}")
    most = []
    least = []
    for category in dict.fromkeys(categories[rows].tolist()):
        rows_c = rows[categories[rows] == category]
        least_c, most_c = extreme_groups(y[rows_c], n_groups)
        least.append(rows_c[least_c])
        most.append(rows_c[most_c])
    most = np.concatenate(most)
    least = np.concatenate(least)
    return x[most].mean(axis=0) - x[least].mean(axis=0)


# Pooled over categories


def over_categories_matrices(
    x_by_prompt,
    y,
    categories,
    folds,
    n_groups,
    metrics,
    aggregate,
    n_folds,
    mode,
    x_eval_by_prompt=(),
    diagonal="cv",
    x_score_by_prompt=None,
):
    """Return (transfer (n_m, n_p, n_p, n_c), eval transfer (n_m, n_p, n_e, n_c),
    cosine (n_p, n_p)).

    One probe per prompt on all categories, trained on x_by_prompt. Cells are
    CV, except the diagonal with diagonal "full" (the probe on all items,
    in-sample). The last axis is the eval category (mean over it gives the
    page value). x_score_by_prompt: the activations the training prompts are
    scored on (the raw activations; default x_by_prompt). x_eval_by_prompt:
    the activations the n_e evaluation prompts are scored on, same rows.
    """
    category_names = list(dict.fromkeys(categories))
    categories = np.asarray(categories)
    rows_of = [np.flatnonzero(categories == c) for c in category_names]
    for rows in rows_of:
        check_groups(len(rows), n_groups, n_folds)
    n_m, n_p, n_c = len(metrics), len(x_by_prompt), len(category_names)
    all_rows = np.arange(len(y))
    if x_score_by_prompt is None:
        x_score_by_prompt = x_by_prompt

    full = np.stack(
        [
            pooled_category_direction(x, all_rows, y, categories, n_groups, mode)
            for x in x_by_prompt
        ]
    )
    by_fold = []
    for x in x_by_prompt:
        by_fold.append(
            np.stack(
                [
                    pooled_category_direction(
                        x, all_rows[folds != f], y, categories, n_groups, mode
                    )
                    for f in range(n_folds)
                ]
            )
        )

    def cv_by_category(fold_directions, x):
        scores = out_of_fold_scores(fold_directions, x, folds)
        values = np.empty((n_m, n_c))
        for ci, rows in enumerate(rows_of):
            for mi, metric in enumerate(metrics):
                values[mi, ci] = cv_value(
                    metric, scores[rows], y[rows], folds[rows], n_groups, aggregate
                )
        return values

    def full_by_category(direction, x):
        scores = x @ direction
        values = np.empty((n_m, n_c))
        for ci, rows in enumerate(rows_of):
            for mi, metric in enumerate(metrics):
                values[mi, ci] = metric_value(metric, scores[rows], y[rows], n_groups)
        return values

    transfer = np.empty((n_m, n_p, n_p, n_c))
    eval_transfer = np.empty((n_m, n_p, len(x_eval_by_prompt), n_c))
    for ti in range(n_p):
        for ei, x in enumerate(x_score_by_prompt):
            if ti == ei and diagonal == "full":
                transfer[:, ti, ei] = full_by_category(full[ti], x)
            else:
                transfer[:, ti, ei] = cv_by_category(by_fold[ti], x)
        for ei, x in enumerate(x_eval_by_prompt):
            eval_transfer[:, ti, ei] = cv_by_category(by_fold[ti], x)
    return transfer, eval_transfer, cosine_matrix(full)


# Pooled over categories and prompts


def global_direction(x_by_prompt, y, categories, n_groups, mode):
    """One direction from all items of all categories in all prompts."""
    all_rows = np.arange(len(y))
    return np.mean(
        [
            pooled_category_direction(x, all_rows, y, categories, n_groups, mode)
            for x in x_by_prompt
        ],
        axis=0,
    )


def zscore_within_categories(scores, categories):
    result = np.empty(len(scores), dtype=np.float64)
    for category in dict.fromkeys(categories.tolist()):
        rows = categories == category
        std = scores[rows].std()
        result[rows] = (
            np.nan if std == 0 else (scores[rows] - scores[rows].mean()) / std
        )
    return result


def global_eval_scores(
    x_by_prompt, x_eval_by_prompt, y, categories, n_groups, metrics, mode
):
    """Return (raw scores (n_e, n), z scores (n_e, n), metrics (n_m, n_e)).

    The direction is trained on x_by_prompt and scored on x_eval_by_prompt.
    "raw scores" are the scores before the z-score. The metrics are computed in each category, then averaged.
    """
    categories = np.asarray(categories)
    direction = global_direction(x_by_prompt, y, categories, n_groups, mode)
    n_e = len(x_eval_by_prompt)
    raw = np.empty((n_e, len(y)))
    z = np.empty((n_e, len(y)))
    values = np.empty((len(metrics), n_e))
    for ei, x in enumerate(x_eval_by_prompt):
        raw[ei] = x @ direction
        z[ei] = zscore_within_categories(raw[ei], categories)
        for mi, metric in enumerate(metrics):
            per_category = [
                metric_value(
                    metric, raw[ei][categories == c], y[categories == c], n_groups
                )
                for c in dict.fromkeys(categories.tolist())
            ]
            values[mi, ei] = float(np.nanmean(per_category))
    return raw, z, values


# Pooled over prompts


def pooled_prompt_direction(x_by_prompt, rows, y, n_groups):
    """Direction for the items in rows, pooled over the prompts."""
    least, most = extreme_groups(y[rows], n_groups)
    least = rows[least]
    most = rows[most]
    differences = [x[most].mean(axis=0) - x[least].mean(axis=0) for x in x_by_prompt]
    return np.mean(differences, axis=0)


def over_prompts_matrices(
    x_by_prompt,
    y,
    categories,
    folds,
    n_groups,
    metrics,
    aggregate,
    n_folds,
    diagonal="cv",
    x_score_by_prompt=None,
):
    """Return (transfer (n_m, n_c, n_c, n_p), cosine (n_c, n_c)).

    One probe per category on all prompts, trained on x_by_prompt and scored
    on x_score_by_prompt (the raw activations; default x_by_prompt). The last
    axis is the eval prompt
    (mean over it gives the page value). The diagonal is CV, or with diagonal
    "full" the probe on all items of the category (in-sample).
    """
    category_names = list(dict.fromkeys(categories))
    categories = np.asarray(categories)
    rows_of = [np.flatnonzero(categories == c) for c in category_names]
    n_m, n_p, n_c = len(metrics), len(x_by_prompt), len(category_names)
    if x_score_by_prompt is None:
        x_score_by_prompt = x_by_prompt

    full = []
    by_fold = []
    for rows in rows_of:
        check_groups(len(rows), n_groups, n_folds)
        full.append(pooled_prompt_direction(x_by_prompt, rows, y, n_groups))
        by_fold.append(
            np.stack(
                [
                    pooled_prompt_direction(
                        x_by_prompt, rows[folds[rows] != f], y, n_groups
                    )
                    for f in range(n_folds)
                ]
            )
        )
    full = np.stack(full)

    transfer = np.empty((n_m, n_c, n_c, n_p))
    for ti in range(n_c):
        for ei, rows in enumerate(rows_of):
            for pi, x in enumerate(x_score_by_prompt):
                x_eval = x[rows]
                if ti == ei and diagonal == "cv":
                    scores = out_of_fold_scores(by_fold[ti], x_eval, folds[rows])
                    for mi, metric in enumerate(metrics):
                        transfer[mi, ti, ei, pi] = cv_value(
                            metric, scores, y[rows], folds[rows], n_groups, aggregate
                        )
                else:
                    scores = x_eval @ full[ti]
                    for mi, metric in enumerate(metrics):
                        transfer[mi, ti, ei, pi] = metric_value(
                            metric, scores, y[rows], n_groups
                        )
    return transfer, cosine_matrix(full)


def write_summary(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def run(config, config_path):
    run_id = make_run_id(config["experiment"])
    run_dir = Path(config["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = start_metadata(run_id, config, config_path, run_dir)
    write_metadata(metadata, run_dir / "metadata.json")

    try:
        set_seed(config["seed"])
        lazy_model = LazyModel(extraction_config(config), metadata)
        metadata["device"] = None
        metadata["activation_sources"] = {}
        metadata["score_activation_type"] = "raw"
        point_rows, types_by_prompt = load_points(config, lazy_model, metadata)
        eval_prompts = list(config.get("eval_prompt_columns") or [])
        eval_types_by_prompt = {}
        if eval_prompts:
            eval_rows, eval_types_by_prompt = load_points(
                config,
                lazy_model,
                metadata,
                config["eval_cat_contrast_csv"],
                prompt_columns=eval_prompts,
                types=["raw"],
                with_items=False,
            )
            check_same_rows(point_rows, eval_rows)

        prompts = list(config["prompt_columns"])
        layers = list(config["layers"])
        ambiguities = list(config["ambiguity"])
        group_counts = list(config["n_groups"])
        metrics = list(config["metrics"])
        types = config_activation_types(config)
        all_categories = [row["item_category"] for row in point_rows]
        category_names = list(dict.fromkeys(all_categories))
        target = np.array(
            [config["target_sign"] * float(row[config["target"]]) for row in point_rows]
        )

        shape = (len(ambiguities), len(types), len(layers), len(group_counts))
        n_m, n_p, n_c = len(metrics), len(prompts), len(category_names)
        n_e = len(eval_prompts)
        arrays = {
            "over_categories_by_category": np.empty(shape + (n_m, n_p, n_p, n_c)),
            "over_categories_eval_by_category": np.empty(shape + (n_m, n_p, n_e, n_c)),
            "global_scores_raw": np.full(shape + (n_e, len(point_rows)), np.nan),
            "global_scores_z": np.full(shape + (n_e, len(point_rows)), np.nan),
            "global_eval_metrics": np.full(shape + (n_m, n_e), np.nan),
            "over_categories_cosine": np.empty(shape + (n_p, n_p)),
            "over_prompts_by_prompt": np.empty(shape + (n_m, n_c, n_c, n_p)),
            "over_prompts_cosine": np.empty(shape + (n_c, n_c)),
        }
        n_items = {}
        summary_rows = []
        for ai, ambiguity in enumerate(ambiguities):
            keep = keep_mask(point_rows, ambiguity)
            categories = [c for c, k in zip(all_categories, keep) if k]
            if list(dict.fromkeys(categories)) != category_names:
                raise ValueError(f"A category has no items with {ambiguity}")
            y = target[keep]
            folds = make_folds(categories, y, config["n_folds"], config["seed"])
            n_items[ambiguity] = {
                c: int(sum(1 for x in categories if x == c)) for c in category_names
            }
            for ti, activation_type in enumerate(types):
                for li, layer in enumerate(layers):
                    x_by_prompt = [
                        types_by_prompt[p][activation_type][keep, li] for p in prompts
                    ]
                    x_raw_by_prompt = [
                        types_by_prompt[p]["raw"][keep, li] for p in prompts
                    ]
                    x_eval_by_prompt = [
                        eval_types_by_prompt[p]["raw"][keep, li] for p in eval_prompts
                    ]
                    for gi, n_groups in enumerate(group_counts):
                        arguments = (
                            x_by_prompt,
                            y,
                            categories,
                            folds,
                            n_groups,
                            metrics,
                            config["cv_aggregate"],
                            config["n_folds"],
                        )
                        oc, oc_eval, oc_cos = over_categories_matrices(
                            *arguments,
                            config["over_categories_groups"],
                            x_eval_by_prompt,
                            config_diagonal(config),
                            x_raw_by_prompt,
                        )
                        op, op_cos = over_prompts_matrices(
                            *arguments, config_diagonal(config), x_raw_by_prompt
                        )
                        index = (ai, ti, li, gi)
                        arrays["over_categories_by_category"][index] = oc
                        arrays["over_categories_eval_by_category"][index] = oc_eval
                        if n_e:
                            raw, z, values = global_eval_scores(
                                x_by_prompt,
                                x_eval_by_prompt,
                                y,
                                categories,
                                n_groups,
                                metrics,
                                config["over_categories_groups"],
                            )
                            arrays["global_scores_raw"][index][:, keep] = raw
                            arrays["global_scores_z"][index][:, keep] = z
                            arrays["global_eval_metrics"][index] = values
                        arrays["over_categories_cosine"][index] = oc_cos
                        arrays["over_prompts_by_prompt"][index] = op
                        arrays["over_prompts_cosine"][index] = op_cos
                        oc_mean = oc.mean(axis=-1)
                        op_mean = op.mean(axis=-1)
                        oc_eval_mean = oc_eval.mean(axis=-1)
                        for mi, metric in enumerate(metrics):
                            summary_rows.append(
                                {
                                    "ambiguity": ambiguity,
                                    "activation_type": activation_type,
                                    "layer": layer,
                                    "n_groups": n_groups,
                                    "metric": metric,
                                    "n_items": int(keep.sum()),
                                    "within_method": within_method(config),
                                    "over_categories_within_prompt": diagonal_mean(
                                        oc_mean[mi]
                                    ),
                                    "over_categories_between_prompts": off_diagonal_mean(
                                        oc_mean[mi]
                                    ),
                                    "over_categories_eval_prompts": (
                                        float(np.nanmean(oc_eval_mean[mi]))
                                        if n_e
                                        else float("nan")
                                    ),
                                    "global_eval_prompts_in_sample": (
                                        float(
                                            np.nanmean(
                                                arrays["global_eval_metrics"][index][mi]
                                            )
                                        )
                                        if n_e
                                        else float("nan")
                                    ),
                                    "over_prompts_within_category": diagonal_mean(
                                        op_mean[mi]
                                    ),
                                    "over_prompts_between_categories": off_diagonal_mean(
                                        op_mean[mi]
                                    ),
                                    "over_categories_cosine_between": off_diagonal_mean(
                                        oc_cos
                                    ),
                                    "over_prompts_cosine_between": off_diagonal_mean(
                                        op_cos
                                    ),
                                }
                            )
            print(f"Done {ambiguity} ({int(keep.sum())} items)")

        arrays["over_categories"] = np.nanmean(
            arrays["over_categories_by_category"], axis=-1
        )
        arrays["over_prompts"] = np.nanmean(arrays["over_prompts_by_prompt"], axis=-1)
        if n_e:
            arrays["over_categories_eval"] = np.nanmean(
                arrays["over_categories_eval_by_category"], axis=-1
            )
        else:
            arrays["over_categories_eval"] = np.empty(shape + (n_m, n_p, 0))
        axes = {
            "ambiguity": ambiguities,
            "activation_type": types,
            "layer": layers,
            "n_groups": group_counts,
            "metric": metrics,
            "prompt": prompts,
            "eval_prompt": eval_prompts,
            "category": category_names,
        }
        np.savez_compressed(run_dir / "matrices.npz", **arrays)
        with open(run_dir / "axes.json", "w") as f:
            json.dump(axes, f, indent=2)
        write_summary(summary_rows, run_dir / "summary.csv")
        metadata["n_items"] = n_items

        eval_texts = {}
        if eval_prompts:
            eval_texts = {p: [row[p] for row in eval_rows] for p in eval_prompts}
        html_path = write_html(
            run_dir / config["html_name"],
            title=config["title"],
            axes=axes,
            arrays=arrays,
            n_items=n_items,
            points={
                "item": [row["item"] for row in point_rows],
                "category": all_categories,
                "typicality": [round(float(v), 4) for v in target],
            },
            eval_texts=eval_texts,
            labels={
                "activation_type": ACTIVATION_LABELS,
                "formula": ACTIVATION_FORMULAS,
                "ambiguity": AMBIGUITY_LABELS,
                "n_groups": {str(k): v for k, v in GROUP_LABELS.items()},
                "metric": METRIC_LABELS,
            },
            metric_chance=METRIC_CHANCE,
            prompt_formats={
                **{p: PROMPT_FORMATS.get(p, "") for p in prompts},
                **{p: EVAL_PROMPT_FORMATS.get(p, "") for p in eval_prompts},
            },
            cv_aggregate=config["cv_aggregate"],
            n_folds=config["n_folds"],
            diagonal=config_diagonal(config),
        )
        if config["copy_html_to"] is not None:
            copy_dir = Path(config["copy_html_to"])
            copy_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(html_path, copy_dir / config["html_name"])
            metadata["html_copy"] = str(copy_dir / config["html_name"])
    except BaseException:
        metadata["status"] = "failed"
        metadata["error"] = traceback.format_exc()
        metadata["finished_at"] = now_utc()
        write_metadata(metadata, run_dir / "metadata.json")
        raise

    metadata["status"] = "completed"
    metadata["finished_at"] = now_utc()
    write_metadata(metadata, run_dir / "metadata.json")
    return run_dir
