"""Difference-of-means probes for the global direction experiments.

The points are the 565 rated pairs, for the four activation types of
global_pca_experiment.py (raw, mean category contrast, mean item contrast,
both contrasts). The target is typicality = target_sign x target (higher is
more typical).

Optional config keys:
- activation_types: the types to use (default: the four above). The type
  "nonce_contrast" is a(i, c) - a(i, nonce): the raw activation minus the
  mean activation of the same prompt with each nonce word as the category.
- nonce_csv: the nonce prompts (data/processed/rosch/rosch_nonce.csv), with
  the same prompt columns. Necessary for "nonce_contrast".
- diagonal: "cv" (default) or "full". The method for the diagonal cells of
  the transfer matrices. "cv": each fold probe on its held-out items. "full":
  the full probe on all items it was trained on (in-sample, not held out).

One setting is (ambiguity, activation type, layer, n_groups):
- ambiguity: "include" (all items), "exclude_core" or "exclude_extended"
  (remove the items with is_ambiguous_core / is_ambiguous_extended true from
  training and evaluation; the folds are made again on the other items)
- n_groups: 2 (halves), 3 (thirds), 4 (quarters). The probe direction is the
  mean of the most typical 1/n_groups of the training items minus the mean of
  the least typical 1/n_groups. The score of x is x . direction.

For each setting, for each category c and prompt p, one probe is fitted on
all items of c in p (the full probe) and one per fold on the items not in the
fold. The folds are made in each category, stratified by typicality, and are
the same for all prompts and activation types.

Matrices for each setting:
- category transfer [metric][prompt]: rows train category, columns eval
  category. Off-diagonal: the full probe on all items of the eval category.
  Diagonal: cross-validated, or the full probe with diagonal "full".
- category cosine [prompt]: cosine of the full probe directions.
- prompt transfer [metric][category]: rows train prompt, columns eval prompt.
  Off-diagonal cells are cross-validated with the same folds, so a probe is
  never evaluated on the items it was trained on. The diagonal is
  cross-validated, or the full probe with diagonal "full".
- prompt cosine [category]: cosine of the full probe directions.

Metrics: Spearman and Pearson correlation of score and typicality, and the
Wilcoxon rank-sum AUC of the most against the least typical 1/n_groups of the
eval items (see probes.wilcoxon_auc).

Cross-validated values are aggregated with `cv_aggregate`:
- "fold_zscored": the scores are z-scored inside each held-out fold, then the
  metric is computed on all held-out items. This removes the offset that each
  fold probe adds: with "pooled", the fold probes have different offsets, so
  the pooled metric is biased low. No labels are used.
- "pooled": the metric of all out-of-fold scores.
- "mean_over_folds": the mean of the metric of each held-out fold.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml, metadata.json
- matrices.npz: the four matrices for all settings, and axes.json with the
  order of each axis
- summary.csv: one row per setting and metric, with the means of the
  diagonal and off-diagonal cells of each matrix. `within_method` tells how
  the diagonal ("within") columns are made: "cv", or "in_sample" with
  diagonal "full".
- <html_name>: the page (see global_probe_plot.py). If `copy_html_to` is set,
  the page is also copied there.
"""

import csv
import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.activation_source import LazyModel, get_activations
from typicality_linearity.baselines import is_true, mean_contrast_groups
from typicality_linearity.global_pca_experiment import (
    ACTIVATION_FORMULAS,
    ACTIVATION_LABELS,
    ACTIVATION_TYPES,
    activation_types,
    check_same_points,
    extraction_config,
    prompt_activations,
    read_rows,
)
from typicality_linearity.global_probe_plot import write_html
from typicality_linearity.probes import (
    fit_diff_of_means,
    make_folds,
    pearson,
    spearman,
    wilcoxon_auc,
)
from typicality_linearity.rosch_prompts import PROMPT_FORMATS
from typicality_linearity.run_metadata import (
    make_run_id,
    now_utc,
    set_seed,
    start_metadata,
    write_metadata,
)

REQUIRED_KEYS = (
    "experiment",
    "title",
    "model",
    "cat_contrast_csv",
    "item_contrast_csv",
    "prompt_columns",
    "layers",
    "position",
    "ambiguity",
    "n_groups",
    "metrics",
    "n_folds",
    "cv_aggregate",
    "target",
    "target_sign",
    "batch_size",
    "device",
    "dtype",
    "prefix_token",
    "seed",
    "activation_cache_dir",
    "output_dir",
    "html_name",
    "copy_html_to",
)

# Ambiguity setting: the column that marks the items to remove (None: keep all).
AMBIGUITY_COLUMNS = {
    "include": None,
    "exclude_core": "is_ambiguous_core",
    "exclude_extended": "is_ambiguous_extended",
}

AMBIGUITY_LABELS = {
    "include": "Include all",
    "exclude_core": "Exclude core",
    "exclude_extended": "Exclude extended",
}

GROUP_LABELS = {2: "Halves", 3: "Thirds", 4: "Quarters"}

METRIC_NAMES = ("spearman", "pearson", "wilcoxon_auc")

METRIC_LABELS = {
    "spearman": "Spearman",
    "pearson": "Pearson",
    "wilcoxon_auc": "Wilcoxon AUC",
}

# The value of a metric for a probe with no signal.
METRIC_CHANCE = {"spearman": 0.0, "pearson": 0.0, "wilcoxon_auc": 0.5}

AGGREGATES = ("fold_zscored", "pooled", "mean_over_folds")

# The method for the diagonal cells of the transfer matrices.
DIAGONALS = ("cv", "full")

SUMMARY_COLUMNS = (
    "ambiguity",
    "activation_type",
    "layer",
    "n_groups",
    "metric",
    "n_items",
    "within_method",
    "category_within",
    "category_between",
    "prompt_within",
    "prompt_between",
    "category_cosine_between",
    "prompt_cosine_between",
)


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")
    for setting in config["ambiguity"]:
        if setting not in AMBIGUITY_COLUMNS:
            raise ValueError(f"Unknown ambiguity {setting!r}")
    for metric in config["metrics"]:
        if metric not in METRIC_NAMES:
            raise ValueError(f"Unknown metric {metric!r}")
    if config["cv_aggregate"] not in AGGREGATES:
        raise ValueError(f"Unknown cv_aggregate {config['cv_aggregate']!r}")
    if config_diagonal(config) not in DIAGONALS:
        raise ValueError(f"diagonal must be one of {DIAGONALS}")
    return config


def config_diagonal(config):
    return config.get("diagonal", "cv")


def within_method(config):
    """Return how the "within" summary columns are made: "cv" or "in_sample"."""
    if config_diagonal(config) == "full":
        return "in_sample"
    return "cv"


# Metrics and cross-validation


def metric_value(metric, scores, y, n_groups):
    if metric == "spearman":
        return spearman(scores, y)
    if metric == "pearson":
        return pearson(scores, y)
    if metric == "wilcoxon_auc":
        return wilcoxon_auc(scores, y, n_groups)
    raise ValueError(f"Unknown metric {metric!r}")


def zscore_within_folds(scores, folds):
    """Return the scores z-scored inside each fold. NaN for a constant fold."""
    result = np.empty(len(scores), dtype=np.float64)
    for fold in np.unique(folds):
        rows = folds == fold
        values = scores[rows]
        std = values.std()
        if std == 0:
            result[rows] = np.nan
        else:
            result[rows] = (values - values.mean()) / std
    return result


def cv_value(metric, scores, y, folds, n_groups, aggregate):
    """Return the metric of out-of-fold scores, aggregated over the folds."""
    if aggregate == "fold_zscored":
        return metric_value(metric, zscore_within_folds(scores, folds), y, n_groups)
    if aggregate == "pooled":
        return metric_value(metric, scores, y, n_groups)
    if aggregate == "mean_over_folds":
        values = []
        for fold in np.unique(folds):
            rows = folds == fold
            values.append(metric_value(metric, scores[rows], y[rows], n_groups))
        return float(np.mean(values))
    raise ValueError(f"Unknown aggregate {aggregate!r}")


def check_groups(n_items, n_groups, n_folds):
    """Each fold training set must have at least one item per group."""
    n_train = n_items - int(np.ceil(n_items / n_folds))
    if n_train // n_groups == 0:
        raise ValueError(f"{n_items} items are too few for {n_groups} groups")


# Probes


def fit_directions(x, y, folds, n_groups, n_folds):
    """Return (full direction (d,), fold directions (n_folds, d))."""
    full = fit_diff_of_means(x, y, n_groups)
    by_fold = np.stack(
        [
            fit_diff_of_means(x[folds != fold], y[folds != fold], n_groups)
            for fold in range(n_folds)
        ]
    )
    return full, by_fold


def out_of_fold_scores(fold_directions, x_eval, folds):
    """Return the score of each row with the probe of the fold it is held out in."""
    all_scores = x_eval @ fold_directions.T  # (n_rows, n_folds)
    return all_scores[np.arange(len(x_eval)), folds]


def cosine_matrix(vectors):
    """Return the cosine similarity of each pair of rows. NaN for a zero row."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = vectors / norms[:, None]
    return unit @ unit.T


def setting_matrices(
    x_by_prompt,
    y,
    categories,
    folds,
    n_groups,
    metrics,
    aggregate,
    n_folds,
    diagonal="cv",
):
    """Return the four matrices for one setting.

    x_by_prompt: list of arrays (n_rows, d_model), one per prompt, same rows
    y: typicality of each row; categories: category of each row (in order)
    folds: fold of each row
    diagonal: "cv" or "full" (the full probe on its training items)

    Result (n_m metrics, n_p prompts, n_c categories):
    - "category_transfer": (n_m, n_p, n_c, n_c)
    - "category_cosine": (n_p, n_c, n_c)
    - "prompt_transfer": (n_m, n_c, n_p, n_p)
    - "prompt_cosine": (n_c, n_p, n_p)
    """
    category_names = list(dict.fromkeys(categories))
    categories = np.asarray(categories)
    rows_of = [np.flatnonzero(categories == c) for c in category_names]
    n_m = len(metrics)
    n_p = len(x_by_prompt)
    n_c = len(category_names)

    full = {}
    by_fold = {}
    for ci, rows in enumerate(rows_of):
        check_groups(len(rows), n_groups, n_folds)
        for pi, x in enumerate(x_by_prompt):
            full[ci, pi], by_fold[ci, pi] = fit_directions(
                x[rows], y[rows], folds[rows], n_groups, n_folds
            )

    category_transfer = np.empty((n_m, n_p, n_c, n_c))
    category_cosine = np.empty((n_p, n_c, n_c))
    for pi, x in enumerate(x_by_prompt):
        category_cosine[pi] = cosine_matrix([full[ci, pi] for ci in range(n_c)])
        for ti in range(n_c):
            for ei, rows in enumerate(rows_of):
                x_eval = x[rows]
                if ti == ei and diagonal == "cv":
                    scores = out_of_fold_scores(by_fold[ti, pi], x_eval, folds[rows])
                    for mi, metric in enumerate(metrics):
                        category_transfer[mi, pi, ti, ei] = cv_value(
                            metric, scores, y[rows], folds[rows], n_groups, aggregate
                        )
                else:
                    scores = x_eval @ full[ti, pi]
                    for mi, metric in enumerate(metrics):
                        category_transfer[mi, pi, ti, ei] = metric_value(
                            metric, scores, y[rows], n_groups
                        )

    prompt_transfer = np.empty((n_m, n_c, n_p, n_p))
    prompt_cosine = np.empty((n_c, n_p, n_p))
    for ci, rows in enumerate(rows_of):
        prompt_cosine[ci] = cosine_matrix([full[ci, pi] for pi in range(n_p)])
        for ti in range(n_p):
            for ei, x in enumerate(x_by_prompt):
                if ti == ei and diagonal == "full":
                    scores = x[rows] @ full[ci, ti]
                    for mi, metric in enumerate(metrics):
                        prompt_transfer[mi, ci, ti, ei] = metric_value(
                            metric, scores, y[rows], n_groups
                        )
                    continue
                scores = out_of_fold_scores(by_fold[ci, ti], x[rows], folds[rows])
                for mi, metric in enumerate(metrics):
                    prompt_transfer[mi, ci, ti, ei] = cv_value(
                        metric, scores, y[rows], folds[rows], n_groups, aggregate
                    )

    return {
        "category_transfer": category_transfer,
        "category_cosine": category_cosine,
        "prompt_transfer": prompt_transfer,
        "prompt_cosine": prompt_cosine,
    }


def diagonal_mean(matrices):
    """Mean of the diagonals of an array (..., n, n)."""
    return float(np.nanmean(np.diagonal(matrices, axis1=-2, axis2=-1)))


def off_diagonal_mean(matrices):
    """Mean of the cells that are not on the diagonal of an array (..., n, n)."""
    n = matrices.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    return float(np.nanmean(matrices[..., mask]))


# Data


def keep_mask(point_rows, ambiguity):
    column = AMBIGUITY_COLUMNS[ambiguity]
    if column is None:
        return np.ones(len(point_rows), dtype=bool)
    return np.array([not is_true(row[column]) for row in point_rows])


def nonce_row_order(point_rows, nonce_rows):
    """Return the indices in nonce_rows of each point (item, item_category).

    There can be more than one nonce row for each point (one for each nonce
    word). All points must have the same number of nonce rows.
    """
    index = {}
    for i, row in enumerate(nonce_rows):
        index.setdefault((row["item"], row["category"]), []).append(i)
    keys = [(row["item"], row["item_category"]) for row in point_rows]
    missing = [key for key in keys if key not in index]
    if missing:
        raise ValueError(f"The nonce file has no rows for {missing[:5]}")
    counts = {len(index[key]) for key in keys}
    if len(counts) != 1:
        raise ValueError(f"Points have different numbers of nonce rows: {counts}")
    return [index[key] for key in keys]


def nonce_contrast(raw, nonce_values, order):
    """a(i, c) - mean over nonce words w of a(i, w).

    raw (n_points, ...), nonce_values (n_rows, ...), order from nonce_row_order.
    """
    return raw - nonce_values[np.array(order)].mean(axis=1)


def config_activation_types(config):
    types = list(config.get("activation_types") or ACTIVATION_TYPES)
    unknown = [t for t in types if t not in ACTIVATION_LABELS]
    if unknown:
        raise ValueError(f"Unknown activation types {unknown}")
    if "nonce_contrast" in types and not config.get("nonce_csv"):
        raise ValueError("The nonce_contrast type needs nonce_csv in the config")
    return types


def load_points(
    config,
    lazy_model,
    metadata,
    cat_csv=None,
    item_csv=None,
    prompt_columns=None,
    nonce_csv=None,
):
    """Return (point rows, {prompt: {type: array (n_points, n_layers, d)}}).

    The files and prompts default to the config's cat_contrast_csv,
    item_contrast_csv, prompt_columns and nonce_csv. Only the types of
    config_activation_types are kept. The nonce contrast uses the rows of
    nonce_csv (the same prompts with a nonce word as the category).
    """
    wanted = config_activation_types(config)
    cat_rows = read_rows(cat_csv or config["cat_contrast_csv"])
    item_rows = read_rows(item_csv or config["item_contrast_csv"])
    cat_groups = mean_contrast_groups(cat_rows)
    item_groups = mean_contrast_groups(item_rows)
    check_same_points(cat_rows, cat_groups, item_rows, item_groups)
    point_rows = [cat_rows[target] for target, _ in cat_groups]
    nonce_rows = None
    if "nonce_contrast" in wanted:
        nonce_rows = read_rows(nonce_csv or config["nonce_csv"])
        order = nonce_row_order(point_rows, nonce_rows)

    extraction = extraction_config(config)
    types_by_prompt = {}
    for prompt in prompt_columns or config["prompt_columns"]:
        cat_values, item_values, sources = prompt_activations(
            extraction, lazy_model, prompt, cat_rows, item_rows
        )
        types = activation_types(cat_values, cat_groups, item_values, item_groups)
        if nonce_rows is not None:
            nonce_extracted, sources["nonce_csv"] = get_activations(
                extraction,
                lazy_model,
                prompt,
                [row[prompt] for row in nonce_rows],
                [row["item_text"] for row in nonce_rows],
            )
            nonce_values = nonce_extracted.activations[config["position"]]
            types["nonce_contrast"] = nonce_contrast(types["raw"], nonce_values, order)
            del nonce_extracted, nonce_values
        metadata["activation_sources"][prompt] = sources
        types_by_prompt[prompt] = {
            name: types[name].astype(np.float32) for name in wanted
        }
        del cat_values, item_values
        print(f"Loaded {prompt}: {sources}")
    return point_rows, types_by_prompt


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
        point_rows, types_by_prompt = load_points(config, lazy_model, metadata)

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
        arrays = {
            "category_transfer": np.empty(shape + (n_m, n_p, n_c, n_c)),
            "category_cosine": np.empty(shape + (n_p, n_c, n_c)),
            "prompt_transfer": np.empty(shape + (n_m, n_c, n_p, n_p)),
            "prompt_cosine": np.empty(shape + (n_c, n_p, n_p)),
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
                    for gi, n_groups in enumerate(group_counts):
                        result = setting_matrices(
                            x_by_prompt,
                            y,
                            categories,
                            folds,
                            n_groups,
                            metrics,
                            config["cv_aggregate"],
                            config["n_folds"],
                            config_diagonal(config),
                        )
                        for name, values in result.items():
                            arrays[name][ai, ti, li, gi] = values
                        for mi, metric in enumerate(metrics):
                            ct = result["category_transfer"][mi]
                            pt = result["prompt_transfer"][mi]
                            summary_rows.append(
                                {
                                    "ambiguity": ambiguity,
                                    "activation_type": activation_type,
                                    "layer": layer,
                                    "n_groups": n_groups,
                                    "metric": metric,
                                    "n_items": int(keep.sum()),
                                    "within_method": within_method(config),
                                    "category_within": diagonal_mean(ct),
                                    "category_between": off_diagonal_mean(ct),
                                    "prompt_within": diagonal_mean(pt),
                                    "prompt_between": off_diagonal_mean(pt),
                                    "category_cosine_between": off_diagonal_mean(
                                        result["category_cosine"]
                                    ),
                                    "prompt_cosine_between": off_diagonal_mean(
                                        result["prompt_cosine"]
                                    ),
                                }
                            )
            print(f"Done {ambiguity} ({int(keep.sum())} items)")

        axes = {
            "ambiguity": ambiguities,
            "activation_type": types,
            "layer": layers,
            "n_groups": group_counts,
            "metric": metrics,
            "prompt": prompts,
            "category": category_names,
        }
        np.savez_compressed(run_dir / "matrices.npz", **arrays)
        with open(run_dir / "axes.json", "w") as f:
            json.dump(axes, f, indent=2)
        write_summary(summary_rows, run_dir / "summary.csv")
        metadata["n_items"] = n_items

        html_path = write_html(
            run_dir / config["html_name"],
            title=config["title"],
            axes=axes,
            arrays=arrays,
            n_items=n_items,
            labels={
                "activation_type": ACTIVATION_LABELS,
                "formula": ACTIVATION_FORMULAS,
                "ambiguity": AMBIGUITY_LABELS,
                "n_groups": {str(k): v for k, v in GROUP_LABELS.items()},
                "metric": METRIC_LABELS,
            },
            metric_chance=METRIC_CHANCE,
            prompt_formats={p: PROMPT_FORMATS.get(p, "") for p in prompts},
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
