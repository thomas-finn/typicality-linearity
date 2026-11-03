"""Probe experiment on Rosch prompts: train probes per category and prompt,
measure transfer between categories and between prompts, write outputs.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml: copy of the config
- metadata.json: git commit, seed, environment, status, activation sources
- correlations.csv: one row per matrix cell (small summary metrics)
- cosine_similarity.csv: cosine similarity of the full-probe directions, one
  row per pair of categories (same prompt) and pair of prompts (same category)
- ridge_alphas.csv: the alpha chosen for each ridge fit
- probe_weights.npz: the full-data probes, key
  "<baseline>/<probe_type>/<category>/<prompt>/{weights,centre,scale,offset}"
- <html_name>: interactive heatmaps

Activations come from the shared activation cache when possible (see
activation_cache.py). `layers` and `positions` are the extraction settings and
are part of the cache key. `probe_layer` and `probe_position` select the
activations that the probes use.

If one of `baselines` is "mean_contrast", `prompts_csv` must have one row for
each item and prompt category (see baselines.py). The probes then use only the
target rows. With baseline null, the probes use the raw target-row activations.
"""

import csv
import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.baselines import apply_baseline, target_rows
from typicality_linearity.pca_experiment import (
    LazyModel,
    get_activations,
    read_prompt_rows,
)
from typicality_linearity.probe_plot import build_plot_data, write_html
from typicality_linearity.probe_transfer import (
    AGGREGATES,
    cosine_matrices,
    transfer_matrices,
)
from typicality_linearity.probes import METRICS, PROBE_TYPES, make_folds
from typicality_linearity.run_metadata import (
    make_run_id,
    now_utc,
    set_seed,
    start_metadata,
    write_metadata,
)

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
    "probe_types",
    "metrics",
    "n_folds",
    "cv_aggregate",
    "ridge_alphas",
    "ridge_inner_folds",
    "ridge_standardise",
    "batch_size",
    "device",
    "dtype",
    "prefix_token",
    "seed",
    "activation_cache_dir",
    "output_dir",
    "html_name",
)

BASELINE_LABELS = {
    None: "No baseline",
    "mean_contrast": "Mean contrast baseline",
}


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")
    check_config(config)
    return config


def check_config(config):
    if config["probe_layer"] not in config["layers"]:
        raise ValueError("probe_layer must be one of layers")
    if config["probe_position"] not in config["positions"]:
        raise ValueError("probe_position must be one of positions")
    for probe_type in config["probe_types"]:
        if probe_type not in PROBE_TYPES:
            raise ValueError(f"Unknown probe type {probe_type!r}")
    for metric in config["metrics"]:
        if metric not in METRICS:
            raise ValueError(f"Unknown metric {metric!r}")
    if config["cv_aggregate"] not in AGGREGATES:
        raise ValueError(f"cv_aggregate must be one of {AGGREGATES}")
    if config["target_sign"] not in (1, -1):
        raise ValueError("target_sign must be 1 or -1")


def baseline_name(baseline):
    """Name used in files and in the HTML. None becomes "none"."""
    if baseline is None:
        return "none"
    return baseline


def point_indices_for(rows, baselines):
    """Return the rows that are probe items. They are the same for all baselines."""
    if "mean_contrast" in baselines:
        return target_rows(rows, "mean_contrast")
    return list(range(len(rows)))


def probe_activations(extracted, rows, point_indices, baseline, layer_index, position):
    """Return the activations of the probe items, shape (n_points, d_model)."""
    if baseline is None:
        values = extracted.activations[position][point_indices]
    else:
        values = apply_baseline(extracted.activations, rows, baseline)[position]
    return values[:, layer_index, :]


def probe_settings(config):
    return {
        "ridge_alphas": config["ridge_alphas"],
        "ridge_inner_folds": config["ridge_inner_folds"],
        "ridge_standardise": config["ridge_standardise"],
    }


def correlation_rows(baseline, result):
    """Yield one dict per matrix cell."""
    categories = result["categories"]
    prompts = result["prompts"]
    for probe_type, by_metric in result["category_transfer"].items():
        for metric, by_prompt in by_metric.items():
            for prompt, matrix in by_prompt.items():
                for i, train_category in enumerate(categories):
                    for j, eval_category in enumerate(categories):
                        yield {
                            "baseline": baseline,
                            "probe_type": probe_type,
                            "metric": metric,
                            "matrix": "category_transfer",
                            "train_category": train_category,
                            "eval_category": eval_category,
                            "train_prompt": prompt,
                            "eval_prompt": prompt,
                            "cross_validated": i == j,
                            "value": float(matrix[i, j]),
                        }
    for probe_type, by_metric in result["prompt_transfer"].items():
        for metric, by_category in by_metric.items():
            for category, matrix in by_category.items():
                for i, train_prompt in enumerate(prompts):
                    for j, eval_prompt in enumerate(prompts):
                        yield {
                            "baseline": baseline,
                            "probe_type": probe_type,
                            "metric": metric,
                            "matrix": "prompt_transfer",
                            "train_category": category,
                            "eval_category": category,
                            "train_prompt": train_prompt,
                            "eval_prompt": eval_prompt,
                            "cross_validated": True,
                            "value": float(matrix[i, j]),
                        }


def cosine_rows(baseline, category_cosine, prompt_cosine, categories, prompts):
    """Yield one dict per pair. Diagonal pairs are included (value 1)."""
    for probe_type, by_prompt in category_cosine.items():
        for prompt, matrix in by_prompt.items():
            for i, category_a in enumerate(categories):
                for j, category_b in enumerate(categories):
                    yield {
                        "baseline": baseline,
                        "probe_type": probe_type,
                        "matrix": "between_categories",
                        "category_a": category_a,
                        "category_b": category_b,
                        "prompt_a": prompt,
                        "prompt_b": prompt,
                        "cosine": float(matrix[i, j]),
                    }
    for probe_type, by_category in prompt_cosine.items():
        for category, matrix in by_category.items():
            for i, prompt_a in enumerate(prompts):
                for j, prompt_b in enumerate(prompts):
                    yield {
                        "baseline": baseline,
                        "probe_type": probe_type,
                        "matrix": "between_prompts",
                        "category_a": category,
                        "category_b": category,
                        "prompt_a": prompt_a,
                        "prompt_b": prompt_b,
                        "cosine": float(matrix[i, j]),
                    }


def load_probe_directions(path):
    """Read probe_weights.npz. Return {baseline: {probe_type: {category: {prompt:
    direction in activation space}}}}. The direction is weights / scale."""
    directions = {}
    with np.load(path) as data:
        for key in data.files:
            if not key.endswith("/weights"):
                continue
            baseline, probe_type, category, prompt, _ = key.split("/")
            stem = key[: -len("/weights")]
            weights = data[key].astype(np.float64)
            scale = data[stem + "/scale"].astype(np.float64)
            by_probe = directions.setdefault(baseline, {})
            by_category = by_probe.setdefault(probe_type, {})
            by_category.setdefault(category, {})[prompt] = weights / scale
    return directions


def cosine_rows_from_directions(directions, prompt_order=None):
    """Return cosine rows for the output of load_probe_directions."""
    rows = []
    for baseline, by_probe in directions.items():
        first = next(iter(by_probe.values()))
        categories = sorted(first)
        prompts = list(first[categories[0]])
        if prompt_order is not None:
            prompts = [p for p in prompt_order if p in prompts]
        category_cosine, prompt_cosine = cosine_matrices(by_probe, categories, prompts)
        rows.extend(
            cosine_rows(baseline, category_cosine, prompt_cosine, categories, prompts)
        )
    return rows


def write_csv(rows, path):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def alpha_rows(baseline, result, alphas_grid):
    low = min(alphas_grid)
    high = max(alphas_grid)
    for record in result["alphas"]:
        row = {"baseline": baseline}
        row.update(record)
        row["at_grid_edge"] = record["alpha"] in (low, high)
        yield row


def probe_weight_arrays(baseline, result):
    arrays = {}
    for probe_type, by_category in result["full_probes"].items():
        for category, by_prompt in by_category.items():
            for prompt, probe in by_prompt.items():
                key = f"{baseline}/{probe_type}/{category}/{prompt}"
                arrays[key + "/weights"] = probe.weights.astype(np.float32)
                arrays[key + "/centre"] = probe.centre.astype(np.float32)
                arrays[key + "/scale"] = probe.scale.astype(np.float32)
                arrays[key + "/offset"] = np.array(probe.offset)
    return arrays


def page_notes(config, n_points):
    model_note = (
        f"Model {config['model']}, layer {config['probe_layer']}, "
        f"{config['probe_position']} token, {n_points} items."
    )
    target_note = (
        f"Target: typicality = {config['target_sign']} x {config['target']} "
        "(higher is more typical)."
    )
    cv_note = (
        f"Cross-validation: {config['n_folds']} folds per category, stratified by "
        "typicality, the same items in each fold for all prompts. "
        f"Aggregate: {config['cv_aggregate']}."
    )
    ridge_note = (
        f"Ridge alpha: chosen by inner {config['ridge_inner_folds']}-fold CV from "
        f"{min(config['ridge_alphas']):g} to {max(config['ridge_alphas']):g}"
    )
    if config["ridge_standardise"]:
        ridge_note += " on standardised activations."
    else:
        ridge_note += "."
    return [model_note, target_note, cv_note, ridge_note]


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
        item_texts = [row["item_text"] for row in rows]
        baselines = config["baselines"]
        point_indices = point_indices_for(rows, baselines)
        point_rows = [rows[i] for i in point_indices]
        categories = [row["category"] for row in point_rows]
        y = config["target_sign"] * np.array(
            [float(row[config["target"]]) for row in point_rows]
        )
        folds = make_folds(categories, y, config["n_folds"], seed)
        layer_index = config["layers"].index(config["probe_layer"])
        position = config["probe_position"]

        lazy_model = LazyModel(config, metadata)
        metadata["device"] = None
        metadata["activation_sources"] = {}

        # {baseline name: {prompt: (n_points, d_model)}}
        x = {baseline_name(b): {} for b in baselines}
        for prompt_name in config["prompt_columns"]:
            prompts = [row[prompt_name] for row in rows]
            extracted, source = get_activations(
                config, lazy_model, prompt_name, prompts, item_texts
            )
            print(f"Activations for {prompt_name}: from {source}")
            metadata["activation_sources"][prompt_name] = source
            write_metadata(metadata, run_dir / "metadata.json")
            for baseline in baselines:
                x[baseline_name(baseline)][prompt_name] = probe_activations(
                    extracted, rows, point_indices, baseline, layer_index, position
                )

        results = {}
        for baseline in baselines:
            name = baseline_name(baseline)
            print(f"Fitting probes: baseline {name}")
            results[name] = transfer_matrices(
                x[name],
                y,
                categories,
                folds,
                config["probe_types"],
                config["metrics"],
                probe_settings(config),
                config["cv_aggregate"],
                seed,
            )

        correlations = []
        cosines = []
        alphas = []
        weights = {}
        for name, result in results.items():
            correlations.extend(correlation_rows(name, result))
            cosines.extend(
                cosine_rows(
                    name,
                    result["category_cosine"],
                    result["prompt_cosine"],
                    result["categories"],
                    result["prompts"],
                )
            )
            alphas.extend(alpha_rows(name, result, config["ridge_alphas"]))
            weights.update(probe_weight_arrays(name, result))
        write_csv(correlations, run_dir / "correlations.csv")
        write_csv(cosines, run_dir / "cosine_similarity.csv")
        if alphas:
            write_csv(alphas, run_dir / "ridge_alphas.csv")
            metadata["ridge_alphas_at_grid_edge"] = sum(
                1 for row in alphas if row["at_grid_edge"]
            )
            metadata["ridge_fits"] = len(alphas)
        np.savez_compressed(run_dir / "probe_weights.npz", **weights)

        variants = {baseline_name(b): BASELINE_LABELS.get(b, str(b)) for b in baselines}
        n_items = {c: categories.count(c) for c in sorted(set(categories))}
        plot_data = build_plot_data(
            title=config.get("title", config["experiment"]),
            notes=page_notes(config, len(point_rows)),
            variants=variants,
            results=results,
            n_items=n_items,
        )
        write_html(plot_data, run_dir / config["html_name"])
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
