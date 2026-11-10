"""Contrastive activation addition (CAA) directions on Rosch prompts.

For each baseline (none, mean contrast) and ambiguous option:

1. Fit one CAA direction per category, pooled over all training prompts: the
   mean activation of the top 1/n_groups most typical items minus the mean of
   the bottom 1/n_groups, with all their prompt rows. The evaluation prompts
   are never used for training. Fold directions use the same item folds in
   all prompts.
2. Mean directions:
   - all: the mean of the unit directions of all categories. For the items of
     category c, the term of c is its fold direction that did not see the item.
   - loco: the mean of the unit directions of the other categories.
3. Score each item with each direction in each prompt. The training prompts
   use the activations of the baseline. The evaluation prompts use raw
   activations (the score is linear, so a mean-contrast direction can score
   raw activations). The correlation with typicality is computed inside each
   prompt and category.

Outputs in `<output_dir>/<run_id>/`:
- config.yaml, metadata.json
- summary.csv: one row per baseline, ambiguous option, metric and prompt set
- correlations.csv: one row per (direction, eval category, prompt)
- cosine_similarity.csv: one row per pair of category directions, and the
  cosine between the no-baseline and mean-contrast direction of each category
- mean_all_scores.csv: score of each item with the mean direction "all"
- directions.npz: keys "<baseline>/<ambiguous>/<category>" (unnormalised CAA
  vector) and "<baseline>/<ambiguous>/mean_all" (mean of unit directions)
- <html_name>: interactive page (see cca_plot.py)
"""

import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.cca_plot import write_run_page
from typicality_linearity.diff_means_experiment import (
    AMBIGUOUS_OPTIONS,
    keep_mask,
    load_eval_inputs,
    load_probe_inputs,
    read_config,
)
from typicality_linearity.format_variance_experiment import (
    correlation,
    fit_pooled_probes,
)
from typicality_linearity.mean_direction import unit
from typicality_linearity.pca_experiment import LazyModel, read_prompt_rows
from typicality_linearity.probe_experiment import (
    baseline_name,
    point_indices_for,
    write_csv,
)
from typicality_linearity.probe_transfer import AGGREGATES
from typicality_linearity.probes import (
    METRICS,
    activation_direction,
    cosine_similarity,
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

MEAN_ALL = "mean_all"
MEAN_LOCO = "mean_loco"
PROMPT_SETS = ("train", "eval")

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

# Order of the statistics in summary.csv and on the page. Each is a mean over
# the prompts of the prompt set.
STATISTICS = (
    "within_cv",
    "between_categories",
    MEAN_ALL,
    MEAN_LOCO,
    "cosine_between_categories",
)


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
    if set(config["eval_prompt_columns"]) & set(config["prompt_columns"]):
        raise ValueError("eval_prompt_columns must not be training prompts")


def load_config(path):
    config = read_config(path, REQUIRED_KEYS)
    check_config(config)
    return config


# Directions


def unit_directions(full):
    return {c: unit(activation_direction(p)) for c, p in full.items()}


def mean_all_fold_directions(full, by_fold, category):
    """Return one mean direction "all" per fold, for the items of category.

    The term of category is its fold direction. The other terms are full.
    """
    others = [unit(activation_direction(p)) for c, p in full.items() if c != category]
    return [
        np.mean(others + [unit(activation_direction(fold_probe))], axis=0)
        for fold_probe in by_fold[category]
    ]


def mean_all_direction(full):
    """Return the mean unit direction of all categories (full probes)."""
    return np.mean(list(unit_directions(full).values()), axis=0)


def mean_loco_direction(full, held_out):
    return np.mean(
        [u for c, u in unit_directions(full).items() if c != held_out], axis=0
    )


def out_of_fold_scores(fold_directions, x, folds):
    scores = np.empty(len(x))
    for fold, direction in enumerate(fold_directions):
        held_out = folds == fold
        scores[held_out] = x[held_out] @ direction
    return scores


def direction_scores(full, by_fold, x, category, indices, folds):
    """Return {direction name: (scores, cross_validated)} for the items of one category.

    indices: rows of the items of category in x. Direction names are the train
    categories, MEAN_ALL and MEAN_LOCO.
    """
    x_rows = x[indices]
    folds_rows = folds[indices]
    result = {}
    for train_category, probe in full.items():
        if train_category == category:
            fold_directions = [activation_direction(p) for p in by_fold[train_category]]
            scores = out_of_fold_scores(fold_directions, x_rows, folds_rows)
            result[train_category] = (scores, True)
        else:
            result[train_category] = (x_rows @ activation_direction(probe), False)
    fold_directions = mean_all_fold_directions(full, by_fold, category)
    result[MEAN_ALL] = (out_of_fold_scores(fold_directions, x_rows, folds_rows), True)
    result[MEAN_LOCO] = (x_rows @ mean_loco_direction(full, category), False)
    return result


def category_indices(categories):
    """Return {category: row indices}, in sorted order."""
    categories = np.asarray(categories)
    return {
        c: np.flatnonzero(categories == c) for c in sorted(set(categories.tolist()))
    }


# Correlations


def correlation_cells(full, by_fold, x_by_prompt, y, categories, folds, config):
    """Yield one dict per (metric, direction, eval category, prompt)."""
    for prompt, x in x_by_prompt.items():
        for category, indices in category_indices(categories).items():
            scores = direction_scores(full, by_fold, x, category, indices, folds)
            for direction, (values, cross_validated) in scores.items():
                for metric in config["metrics"]:
                    yield {
                        "metric": metric,
                        "direction": direction,
                        "eval_category": category,
                        "prompt": prompt,
                        "cross_validated": cross_validated,
                        "value": correlation(
                            values,
                            y[indices],
                            folds[indices],
                            metric,
                            config["cv_aggregate"],
                            cross_validated,
                        ),
                    }


def cell_kind(cell):
    if cell["direction"] in (MEAN_ALL, MEAN_LOCO):
        return cell["direction"]
    if cell["direction"] == cell["eval_category"]:
        return "within_cv"
    return "between_categories"


def summarise_cells(cells):
    """Return {statistic: mean value} from the cells of one metric and prompt set."""
    groups = {}
    for cell in cells:
        groups.setdefault(cell_kind(cell), []).append(cell["value"])
    return {key: float(np.nanmean(values)) for key, values in groups.items()}


# Cosine similarity


def cosine_matrix(directions, names):
    return np.array(
        [
            [cosine_similarity(directions[a], directions[b]) for b in names]
            for a in names
        ]
    )


def off_diagonal_mean(matrix):
    mask = ~np.eye(len(matrix), dtype=bool)
    return float(np.nanmean(matrix[mask]))


def cosine_rows(directions_by_baseline, labels):
    """Return rows for cosine_similarity.csv.

    directions_by_baseline: {baseline: {name: vector}}, names are categories
    and MEAN_ALL.
    """
    rows = []
    for baseline, directions in directions_by_baseline.items():
        names = list(directions)
        for a in names:
            for b in names:
                rows.append(
                    {
                        **labels,
                        "kind": "between_categories",
                        "baseline": baseline,
                        "direction_a": a,
                        "direction_b": b,
                        "cosine": cosine_similarity(directions[a], directions[b]),
                    }
                )
    baselines = list(directions_by_baseline)
    for i, first in enumerate(baselines):
        for second in baselines[i + 1 :]:
            for name in directions_by_baseline[first]:
                if name not in directions_by_baseline[second]:
                    continue
                rows.append(
                    {
                        **labels,
                        "kind": "between_baselines",
                        "baseline": f"{first} vs {second}",
                        "direction_a": name,
                        "direction_b": name,
                        "cosine": cosine_similarity(
                            directions_by_baseline[first][name],
                            directions_by_baseline[second][name],
                        ),
                    }
                )
    return rows


# One baseline


def evaluate_baseline(x_train, x_eval, y, categories, folds, config):
    """Return (directions, cells, summary rows without labels)."""
    full, by_fold = fit_pooled_probes(
        x_train, y, categories, folds, int(config["n_groups"])
    )
    directions = {c: activation_direction(p) for c, p in full.items()}
    directions[MEAN_ALL] = mean_all_direction(full)
    category_names = sorted(full)
    cosine = off_diagonal_mean(cosine_matrix(directions, category_names))

    cells = []
    summary = []
    for prompt_set, x_by_prompt in (("train", x_train), ("eval", x_eval)):
        set_cells = list(
            correlation_cells(full, by_fold, x_by_prompt, y, categories, folds, config)
        )
        for cell in set_cells:
            cell["prompt_set"] = prompt_set
        cells.extend(set_cells)
        for metric in config["metrics"]:
            values = summarise_cells(c for c in set_cells if c["metric"] == metric)
            values["cosine_between_categories"] = cosine
            summary.append(
                {
                    "metric": metric,
                    "prompt_set": prompt_set,
                    **{key: values.get(key, float("nan")) for key in STATISTICS},
                }
            )
    return full, by_fold, directions, cells, summary


def mean_all_score_rows(full, by_fold, x_by_prompt, y, categories, folds, items):
    rows = []
    for prompt, x in x_by_prompt.items():
        for category, indices in category_indices(categories).items():
            fold_directions = mean_all_fold_directions(full, by_fold, category)
            scores = out_of_fold_scores(fold_directions, x[indices], folds[indices])
            for index, score in zip(indices, scores):
                rows.append(
                    {
                        "prompt": prompt,
                        "category": category,
                        "item": items[index],
                        "typicality": float(y[index]),
                        "fold": int(folds[index]),
                        "score": float(score),
                    }
                )
    return rows


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
        all_items = np.array([row["item"] for row in point_rows])
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
        cosines = []
        scores = []
        arrays = {}
        n_items = {}
        for option in config["ambiguous"]:
            mask = keep_mask(point_rows, option)
            n_items[option] = int(mask.sum())
            y = all_y[mask]
            categories = all_categories[mask]
            items = all_items[mask]
            folds = make_folds(categories, y, config["n_folds"], seed)
            x_eval = {p: v[mask] for p, v in x_eval_all.items()}
            directions_by_baseline = {}
            for baseline in config["baselines"]:
                name = baseline_name(baseline)
                print(f"Baseline {name}, ambiguous {option}")
                labels = {"baseline": name, "ambiguous": option}
                x_train = {p: v[mask] for p, v in x[name].items()}
                full, by_fold, directions, base_cells, base_summary = evaluate_baseline(
                    x_train, x_eval, y, categories, folds, config
                )
                cells.extend({**labels, **c} for c in base_cells)
                summary.extend(
                    {**labels, "n_items": n_items[option], **row}
                    for row in base_summary
                )
                for prompt_set, x_by_prompt in (("train", x_train), ("eval", x_eval)):
                    for row in mean_all_score_rows(
                        full, by_fold, x_by_prompt, y, categories, folds, items
                    ):
                        scores.append({**labels, "prompt_set": prompt_set, **row})
                directions_by_baseline[name] = directions
                for key, vector in directions.items():
                    arrays[f"{name}/{option}/{key}"] = vector.astype(np.float32)
            cosines.extend(cosine_rows(directions_by_baseline, {"ambiguous": option}))

        write_csv(summary, run_dir / "summary.csv")
        write_csv(cells, run_dir / "correlations.csv")
        write_csv(cosines, run_dir / "cosine_similarity.csv")
        write_csv(scores, run_dir / "mean_all_scores.csv")
        np.savez_compressed(run_dir / "directions.npz", **arrays)
        write_run_page(
            run_dir,
            config["html_name"],
            {p: PROMPT_FORMATS.get(p, "") for p in config["prompt_columns"]},
            {p: EVAL_PROMPT_FORMATS.get(p, "") for p in config["eval_prompt_columns"]},
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
