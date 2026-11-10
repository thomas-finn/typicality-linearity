"""Remove prompt-format variance before difference-of-means probes (Rosch).

For each baseline, ambiguous option and preprocessing variant:

1. Preprocess the activations of the training prompts and the evaluation
   prompts (see format_erasure.py). Variants:
   - raw: no change
   - center: subtract the mean of each prompt format (over the items used)
   - inlp, leace: erase the subspace of a format probe. The format label of a
     row is its training prompt (7 classes). The eraser is fitted on the
     training prompts only and then applied to the evaluation prompts.
   - center_inlp, center_leace: center first, then fit and apply the eraser
   The evaluation prompts are centered with their own mean (no labels used).
2. Fit one difference-of-means probe per category on all training prompts
   together (top third minus bottom third of the items, with all their prompt
   rows). This is the same as a difference of means of the item activations
   averaged over the prompts. Fold probes use the same item folds in all
   prompts.
3. Score each item in each prompt. Correlations are computed two ways:
   - by prompt: inside each prompt, then the mean over the prompts
   - pooled: over the rows of all prompts of the category together. Only this
     value is affected by an offset between formats.

Statistics (mean over cells): within (CV), between categories, and the mean
direction of the other 9 categories ("loco"), on the training prompts and on
the evaluation prompts, by prompt and pooled. Cosine between the category
directions. Format diagnostics: the fraction of the variance from the format
means, the dev accuracy of a linear format classifier, the eraser rank.

Outputs in `<output_dir>/<run_id>/`:
- config.yaml, metadata.json
- summary.csv: one row per variant, baseline, ambiguous option and metric
- correlations.csv: one row per cell (train category, eval category, prompt)
- format_diagnostics.csv, inlp_history.csv
- probe_weights.npz: full-probe directions, keys
  "<variant>/<baseline>/<ambiguous>/<category>/<space>". space "transformed"
  is the direction for the preprocessed activations; "raw" gives the same
  scores (up to a constant) on raw activations of a training prompt.
- <html_name>: interactive page (see format_variance_plot.py)
"""

import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.diff_means_experiment import (
    AMBIGUOUS_OPTIONS,
    keep_mask,
    load_eval_inputs,
    load_probe_inputs,
    read_config,
)
from typicality_linearity.format_erasure import (
    between_group_variance_fraction,
    fit_inlp,
    fit_leace,
    group_accuracy,
    identity_eraser,
)
from typicality_linearity.format_variance_plot import write_run_page
from typicality_linearity.mean_direction import unit
from typicality_linearity.pca_experiment import LazyModel, read_prompt_rows
from typicality_linearity.probe_experiment import (
    baseline_name,
    point_indices_for,
    write_csv,
)
from typicality_linearity.probe_transfer import (
    AGGREGATES,
    cv_correlation,
    out_of_fold_predictions,
)
from typicality_linearity.probes import (
    METRICS,
    activation_direction,
    cosine_similarity,
    fit_diff_of_means,
    make_folds,
)
from typicality_linearity.rosch_prompts import EVAL_PROMPT_FORMATS, PROMPT_FORMATS
from typicality_linearity.run_metadata import (
    make_run_id,
    now_utc,
    set_seed,
    start_metadata,
    write_metadata,
)

VARIANTS = ("raw", "center", "inlp", "leace", "center_inlp", "center_leace")
POOLED = "pooled"
LOCO = "others"

REQUIRED_KEYS = (
    "experiment",
    "model",
    "prompts_csv",
    "prompt_columns",
    "layers",
    "positions",
    "probe_layer",
    "probe_position",
    "baselines",
    "target",
    "target_sign",
    "metrics",
    "n_folds",
    "cv_aggregate",
    "n_groups",
    "ambiguous",
    "variants",
    "inlp_iterations",
    "inlp_stop_accuracy",
    "classifier_c",
    "classifier_max_iter",
    "classifier_dev_fraction",
    "format_check",
    "eval_prompts_csv",
    "eval_prompt_columns",
    "batch_size",
    "device",
    "dtype",
    "prefix_token",
    "seed",
    "activation_cache_dir",
    "output_dir",
    "html_name",
)

STATISTICS = tuple(
    f"{kind}_{prompt_set}_{scope}"
    for kind in ("within", "between", "loco")
    for prompt_set in ("train", "eval")
    for scope in ("by_prompt", POOLED)
) + ("cosine_between_categories",)


def check_config(config):
    if config["probe_layer"] not in config["layers"]:
        raise ValueError("probe_layer must be one of layers")
    if config["probe_position"] not in config["positions"]:
        raise ValueError("probe_position must be one of positions")
    for metric in config["metrics"]:
        if metric not in METRICS:
            raise ValueError(f"Unknown metric {metric!r}")
    if config["cv_aggregate"] not in AGGREGATES:
        raise ValueError(f"cv_aggregate must be one of {AGGREGATES}")
    if config["target_sign"] not in (1, -1):
        raise ValueError("target_sign must be 1 or -1")
    if int(config["n_groups"]) < 2:
        raise ValueError("n_groups must be 2 or more")
    for option in config["ambiguous"]:
        if option not in AMBIGUOUS_OPTIONS:
            raise ValueError(f"ambiguous options must be in {AMBIGUOUS_OPTIONS}")
    for variant in config["variants"]:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Use one of {VARIANTS}")
    if set(config["eval_prompt_columns"]) & set(config["prompt_columns"]):
        raise ValueError("eval_prompt_columns must not be training prompts")
    if len(config["prompt_columns"]) < 2:
        raise ValueError("Format erasure needs 2 or more training prompts")
    if int(config["inlp_iterations"]) < 1:
        raise ValueError("inlp_iterations must be 1 or more")
    if not 0 < float(config["classifier_dev_fraction"]) < 1:
        raise ValueError("classifier_dev_fraction must be in (0, 1)")


def load_config(path):
    config = read_config(path, REQUIRED_KEYS)
    check_config(config)
    return config


# Preprocessing


def parse_variant(variant):
    """Return (centered, erasure method or None)."""
    centered = variant.startswith("center")
    method = variant.split("_")[-1]
    if method not in ("inlp", "leace"):
        method = None
    return centered, method


def stack(x_by_prompt):
    """Return (rows of all prompts, prompt index of each row)."""
    prompts = list(x_by_prompt)
    x = np.concatenate([x_by_prompt[p] for p in prompts], axis=0)
    groups = np.repeat(np.arange(len(prompts)), [len(x_by_prompt[p]) for p in prompts])
    return x, groups


def center_each(x_by_prompt):
    return {
        p: np.asarray(v, dtype=np.float64) - np.asarray(v, dtype=np.float64).mean(0)
        for p, v in x_by_prompt.items()
    }


def fit_eraser(x, groups, method, config):
    if method is None:
        return identity_eraser(x.shape[1])
    if method == "leace":
        return fit_leace(x, groups)
    stop = config["inlp_stop_accuracy"]
    return fit_inlp(
        x,
        groups,
        int(config["inlp_iterations"]),
        dev_fraction=float(config["classifier_dev_fraction"]),
        c_value=float(config["classifier_c"]),
        max_iter=int(config["classifier_max_iter"]),
        stop_accuracy=None if stop is None else float(stop),
        seed=config["seed"],
    )


def preprocess(x_train, x_eval, variant, config):
    """Return (x_train', x_eval', eraser) for one variant.

    x_train, x_eval: {prompt: (n_items, d_model)}. The eraser is fitted on the
    training prompts only.
    """
    centered, method = parse_variant(variant)
    x_train = {p: np.asarray(v, dtype=np.float64) for p, v in x_train.items()}
    x_eval = {p: np.asarray(v, dtype=np.float64) for p, v in x_eval.items()}
    if centered:
        x_train = center_each(x_train)
        x_eval = center_each(x_eval)
    stacked, groups = stack(x_train)
    eraser = fit_eraser(stacked, groups, method, config)
    if eraser.rank > 0:
        x_train = {p: eraser.apply(v) for p, v in x_train.items()}
        x_eval = {p: eraser.apply(v) for p, v in x_eval.items()}
    return x_train, x_eval, eraser


def format_diagnostics(x_train, x_eval, eraser, config):
    stacked, groups = stack(x_train)
    eval_stacked, eval_groups = stack(x_eval)
    result = {
        "eraser_rank": eraser.rank,
        "format_variance_train": between_group_variance_fraction(stacked, groups),
        "format_variance_eval": between_group_variance_fraction(
            eval_stacked, eval_groups
        ),
        "format_accuracy_train": float("nan"),
        "format_chance": 1.0 / len(x_train),
    }
    if config["format_check"]:
        result["format_accuracy_train"] = group_accuracy(
            stacked,
            groups,
            float(config["classifier_dev_fraction"]),
            float(config["classifier_c"]),
            int(config["classifier_max_iter"]),
            config["seed"],
        )
    return result


# Probes pooled over prompts


def prompt_mean(x_by_prompt):
    return np.mean([x_by_prompt[p] for p in x_by_prompt], axis=0)


def fit_pooled_probes(x_by_prompt, y, categories, folds, n_groups):
    """Return (full, by_fold): one difference-of-means probe per category.

    The probe uses all training prompts. With the same items in each prompt,
    the difference of means over all prompt rows is the difference of means of
    the item activations averaged over the prompts.
    """
    x_mean = prompt_mean(x_by_prompt)
    categories = np.asarray(categories)
    n_folds = int(folds.max()) + 1
    full = {}
    by_fold = {}
    for category in sorted(set(categories.tolist())):
        rows = np.flatnonzero(categories == category)
        full[category] = fit_diff_of_means(x_mean[rows], y[rows], n_groups)
        by_fold[category] = []
        for fold in range(n_folds):
            train = rows[folds[rows] != fold]
            by_fold[category].append(
                fit_diff_of_means(x_mean[train], y[train], n_groups)
            )
    return full, by_fold


# Correlations


def correlation(scores, y, folds, metric, aggregate, cross_validated):
    if cross_validated:
        return cv_correlation(scores, y, folds, metric, aggregate)
    return METRICS[metric](scores, y)


def correlation_cells(scores_by_prompt, y, folds, metric, aggregate, cross_validated):
    """Return {prompt or POOLED: value} for the scores of one category."""
    values = {
        p: correlation(s, y, folds, metric, aggregate, cross_validated)
        for p, s in scores_by_prompt.items()
    }
    n_prompts = len(scores_by_prompt)
    values[POOLED] = correlation(
        np.concatenate(list(scores_by_prompt.values())),
        np.tile(y, n_prompts),
        np.tile(folds, n_prompts),
        metric,
        aggregate,
        cross_validated,
    )
    return values


def loco_direction(full, held_out):
    """Return the mean unit direction of the probes of the other categories."""
    return np.mean(
        [unit(activation_direction(p)) for c, p in full.items() if c != held_out],
        axis=0,
    )


def score_rows(full, by_fold, x_by_prompt, y, categories, folds, metric, aggregate):
    """Yield one dict per (probe, train category, eval category, prompt).

    probe "single": the probe of the train category. On its own category it
    is cross-validated (fold probes). probe "loco": the mean direction of all
    other categories (train category LOCO).
    """
    categories = np.asarray(categories)
    category_names = sorted(full)
    rows_of = {c: np.flatnonzero(categories == c) for c in category_names}
    for train_category in category_names:
        for eval_category in category_names:
            rows = rows_of[eval_category]
            cross_validated = train_category == eval_category
            if cross_validated:
                scores = {
                    p: out_of_fold_predictions(
                        by_fold[train_category], x[rows], folds[rows]
                    )
                    for p, x in x_by_prompt.items()
                }
            else:
                probe = full[train_category]
                scores = {p: probe.predict(x[rows]) for p, x in x_by_prompt.items()}
            values = correlation_cells(
                scores, y[rows], folds[rows], metric, aggregate, cross_validated
            )
            for prompt, value in values.items():
                yield {
                    "probe": "single",
                    "train_category": train_category,
                    "eval_category": eval_category,
                    "prompt": prompt,
                    "value": value,
                }
    for eval_category in category_names:
        rows = rows_of[eval_category]
        direction = loco_direction(full, eval_category)
        scores = {p: x[rows] @ direction for p, x in x_by_prompt.items()}
        values = correlation_cells(
            scores, y[rows], folds[rows], metric, aggregate, False
        )
        for prompt, value in values.items():
            yield {
                "probe": "loco",
                "train_category": LOCO,
                "eval_category": eval_category,
                "prompt": prompt,
                "value": value,
            }


def cell_kind(cell):
    if cell["probe"] == "loco":
        return "loco"
    if cell["train_category"] == cell["eval_category"]:
        return "within"
    return "between"


def summarise_cells(cells):
    """Return {statistic: mean value} from the cells of one metric.

    Each cell has "prompt_set" ("train" or "eval").
    """
    groups = {}
    for cell in cells:
        scope = POOLED if cell["prompt"] == POOLED else "by_prompt"
        key = f"{cell_kind(cell)}_{cell['prompt_set']}_{scope}"
        groups.setdefault(key, []).append(cell["value"])
    return {key: float(np.nanmean(values)) for key, values in groups.items()}


def mean_cosine_between_categories(full):
    names = sorted(full)
    values = [
        cosine_similarity(activation_direction(full[a]), activation_direction(full[b]))
        for a in names
        for b in names
        if a != b
    ]
    return float(np.mean(values))


def evaluate_variant(x_train, x_eval, y, categories, folds, config):
    """Return (full probes, cells, {metric: statistics})."""
    full, by_fold = fit_pooled_probes(
        x_train, y, categories, folds, int(config["n_groups"])
    )
    cosine = mean_cosine_between_categories(full)
    cells = []
    statistics = {}
    for metric in config["metrics"]:
        metric_cells = []
        for prompt_set, x_by_prompt in (("train", x_train), ("eval", x_eval)):
            for cell in score_rows(
                full,
                by_fold,
                x_by_prompt,
                y,
                categories,
                folds,
                metric,
                config["cv_aggregate"],
            ):
                cell["prompt_set"] = prompt_set
                cell["metric"] = metric
                metric_cells.append(cell)
        values = summarise_cells(metric_cells)
        values["cosine_between_categories"] = cosine
        statistics[metric] = {key: values.get(key, float("nan")) for key in STATISTICS}
        cells.extend(metric_cells)
    return full, cells, statistics


def direction_arrays(prefix, full, eraser):
    arrays = {}
    for category, probe in full.items():
        direction = activation_direction(probe)
        arrays[f"{prefix}/{category}/transformed"] = direction.astype(np.float32)
        arrays[f"{prefix}/{category}/raw"] = eraser.raw_direction(direction).astype(
            np.float32
        )
    return arrays


def run(config, config_path):
    check_config(config)
    run_id = make_run_id(config["experiment"])
    run_dir = Path(config["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = start_metadata(run_id, config, config_path)
    write_metadata(metadata, run_dir / "metadata.json")

    try:
        seed = config["seed"]
        set_seed(seed)
        rows = read_prompt_rows(config["prompts_csv"], config.get("max_rows"))
        point_indices = point_indices_for(rows, config["baselines"])
        point_rows = [rows[i] for i in point_indices]
        all_categories = np.array([row["category"] for row in point_rows])
        all_y = config["target_sign"] * np.array(
            [float(row[config["target"]]) for row in point_rows]
        )
        lazy_model = LazyModel(config, metadata)
        metadata["device"] = None
        metadata["activation_sources"] = {}
        x = load_probe_inputs(
            config, lazy_model, rows, point_indices, metadata, run_dir
        )
        x_eval_all = load_eval_inputs(config, lazy_model, point_rows, metadata, run_dir)

        summary = []
        cells = []
        diagnostics = []
        history = []
        directions = {}
        for baseline in config["baselines"]:
            name = baseline_name(baseline)
            for option in config["ambiguous"]:
                mask = keep_mask(point_rows, option)
                y = all_y[mask]
                categories = all_categories[mask]
                folds = make_folds(categories, y, config["n_folds"], seed)
                x_train_raw = {p: v[mask] for p, v in x[name].items()}
                x_eval_raw = {p: v[mask] for p, v in x_eval_all.items()}
                for variant in config["variants"]:
                    print(f"Variant {variant}, baseline {name}, ambiguous {option}")
                    labels = {"variant": variant, "baseline": name, "ambiguous": option}
                    x_train, x_eval, eraser = preprocess(
                        x_train_raw, x_eval_raw, variant, config
                    )
                    diagnostic = format_diagnostics(x_train, x_eval, eraser, config)
                    diagnostics.append({**labels, **diagnostic})
                    for record in eraser.history:
                        history.append({**labels, **record})
                    full, variant_cells, statistics = evaluate_variant(
                        x_train, x_eval, y, categories, folds, config
                    )
                    for cell in variant_cells:
                        cells.append({**labels, **cell})
                    for metric, values in statistics.items():
                        summary.append(
                            {
                                **labels,
                                "metric": metric,
                                "n_items": int(mask.sum()),
                                **values,
                                **diagnostic,
                            }
                        )
                    directions.update(
                        direction_arrays(f"{variant}/{name}/{option}", full, eraser)
                    )
                    write_metadata(metadata, run_dir / "metadata.json")

        write_csv(summary, run_dir / "summary.csv")
        write_csv(cells, run_dir / "correlations.csv")
        write_csv(diagnostics, run_dir / "format_diagnostics.csv")
        if history:
            write_csv(history, run_dir / "inlp_history.csv")
        np.savez_compressed(run_dir / "probe_weights.npz", **directions)
        write_run_page(
            run_dir, config["html_name"], PROMPT_FORMATS, EVAL_PROMPT_FORMATS
        )
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
