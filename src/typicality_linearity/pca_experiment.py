"""PCA experiment on Rosch prompts: extract activations, fit PCA, write outputs.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml: copy of the config
- metadata.json: git commit, seed, environment, status
- explained_variance.csv: small summary metrics
- pca_projections.npz: projections, key "<prompt>/<layer>/<scope>/<position>",
  and the row indices of each scope, key "rows/<scope>"
- <html_name>: interactive plots

The scope is "all" (PCA fitted on all rows) or a category name (PCA fitted only
on the rows of that category). Per-category fits are made only if
`per_category_pca` is true.

If `baseline` is "mean_contrast", `prompts_csv` must have one row for each item
and prompt category (see baselines.py). The baseline is subtracted before PCA,
and only the rows of the item's own category become points.

If `compare_no_baseline` is true (only with a baseline), a second PCA is
fitted on the same points without the baseline. Its outputs are
explained_variance_no_baseline.csv and pca_projections_no_baseline.npz, with
the same formats. The HTML shows the two PCAs side by side.

If `activation_cache_dir` is set, activations are read from the cache when
possible. The model is only loaded if one prompt format is not in the cache.
See activation_cache.py.
"""

import csv
import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.activation_cache import (
    cache_key,
    key_inputs,
    load_entry,
    save_entry,
)
from typicality_linearity.baselines import apply_baseline, target_rows
from typicality_linearity.pca import fit_pca
from typicality_linearity.pca_plot import ALL_SCOPE, build_plot_data, write_html
from typicality_linearity.rosch_prompts import read_ratings
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
    "n_components",
    "per_category_pca",
    "baseline",
    "color_by",
    "batch_size",
    "device",
    "dtype",
    "prefix_token",
    "seed",
    "activation_cache_dir",
    "output_dir",
    "html_name",
)


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")
    return config


# Variant names. A variant is one set of activations that PCA is fitted on.
WITH_BASELINE = "baseline"
NO_BASELINE = "no_baseline"


def variants_to_fit(config):
    """Return the variant names for the config, the main variant first."""
    compare = config.get("compare_no_baseline", False)
    if not compare:
        return [WITH_BASELINE]
    if config["baseline"] is None:
        raise ValueError("compare_no_baseline needs a baseline that is not null")
    return [WITH_BASELINE, NO_BASELINE]


def variant_labels(config, variants):
    """Return {variant: panel label} for the HTML. None if only one variant."""
    if len(variants) == 1:
        return None
    return {
        WITH_BASELINE: f"{config['baseline']} baseline",
        NO_BASELINE: "no baseline",
    }


def variant_activations(activations, rows, baseline, point_indices, variant):
    """Return {position: activations of the point rows} for one variant."""
    if variant == WITH_BASELINE:
        return apply_baseline(activations, rows, baseline)
    if variant == NO_BASELINE:
        return {
            position: values[point_indices] for position, values in activations.items()
        }
    raise ValueError(f"Unknown variant {variant!r}")


def output_suffix(variant):
    if variant == WITH_BASELINE:
        return ""
    return "_" + variant


def read_prompt_rows(path, max_rows=None):
    rows = read_ratings(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def point_info(rows, color_by):
    return {
        "item": [row["item"] for row in rows],
        "category": [row["category"] for row in rows],
        "color": [round(float(row[color_by]), 4) for row in rows],
    }


def scope_rows(categories, per_category):
    """Return {scope: row indices}. Scope "all" has all rows."""
    scopes = {ALL_SCOPE: np.arange(len(categories))}
    if not per_category:
        return scopes
    for category in sorted(set(categories)):
        if category == ALL_SCOPE:
            raise ValueError(f'A category has the reserved name "{ALL_SCOPE}"')
        scopes[category] = np.array(
            [i for i, value in enumerate(categories) if value == category]
        )
    return scopes


def fit_all_pca(activations, layers, positions, n_components, scopes):
    """Return {layer: {scope: {position: PcaResult}}} for one prompt format.

    scopes: {scope: row indices}. One PCA is fitted on the rows of each scope.
    """
    for scope, rows in scopes.items():
        if len(rows) <= n_components:
            raise ValueError(
                f"Scope {scope} has {len(rows)} rows. PCA with {n_components} "
                "components needs more rows than components."
            )
    results = {}
    for layer_index, layer in enumerate(layers):
        results[layer] = {}
        for scope, rows in scopes.items():
            results[layer][scope] = {}
            for position in positions:
                x = activations[position][rows, layer_index, :]
                results[layer][scope][position] = fit_pca(x, n_components)
    return results


def write_explained_variance(pca_results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = None
        for prompt_name, by_layer in pca_results.items():
            for layer, by_scope in by_layer.items():
                for scope, by_position in by_scope.items():
                    for position, result in by_position.items():
                        ratios = result.explained_variance_ratio
                        if header is None:
                            header = ["prompt", "layer", "scope", "position"]
                            header += [f"pc{k + 1}" for k in range(len(ratios))]
                            writer.writerow(header)
                        writer.writerow([prompt_name, layer, scope, position, *ratios])


def write_projections(pca_results, scopes, path):
    arrays = {}
    for scope, rows in scopes.items():
        arrays[f"rows/{scope}"] = rows
    for prompt_name, by_layer in pca_results.items():
        for layer, by_scope in by_layer.items():
            for scope, by_position in by_scope.items():
                for position, result in by_position.items():
                    key = f"{prompt_name}/{layer}/{scope}/{position}"
                    arrays[key] = result.projections
    np.savez_compressed(path, **arrays)


class LazyModel:
    """Load the model the first time it is necessary, then keep it."""

    def __init__(self, config, metadata):
        self.config = config
        self.metadata = metadata
        self.tokenizer = None
        self.model = None
        self.device = None

    def extract(self, prompts, item_texts):
        # Import here so that a run from the cache does not need torch.
        from typicality_linearity.activations import (
            choose_device,
            extract_activations,
            load_model,
        )

        if self.model is None:
            self.device = choose_device(self.config["device"])
            self.metadata["device"] = self.device
            self.tokenizer, self.model = load_model(
                self.config["model"], self.device, self.config["dtype"]
            )
        return extract_activations(
            self.model,
            self.tokenizer,
            prompts,
            item_texts,
            self.config["layers"],
            self.config["positions"],
            self.config["batch_size"],
            self.device,
            self.config["prefix_token"],
        )


def get_activations(config, lazy_model, prompt_name, prompts, item_texts):
    """Return (ExtractedActivations, source). Source is "cache" or "model"."""
    cache_dir = config["activation_cache_dir"]
    if cache_dir is None:
        return lazy_model.extract(prompts, item_texts), "model"

    inputs = key_inputs(
        config["model"],
        config["dtype"],
        config["layers"],
        config["positions"],
        prompts,
        item_texts,
        config["prefix_token"],
    )
    key = cache_key(inputs)
    cached = load_entry(cache_dir, prompt_name, key, config["positions"])
    if cached is not None:
        return cached, "cache"

    extracted = lazy_model.extract(prompts, item_texts)
    extra_info = {
        "device": lazy_model.device,
        "batch_size": config["batch_size"],
        "experiment": config["experiment"],
    }
    save_entry(cache_dir, prompt_name, key, inputs, extracted, extra_info)
    return extracted, "model"


def run(config, config_path):
    run_id = make_run_id(config["experiment"])
    run_dir = Path(config["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = start_metadata(run_id, config, config_path)
    write_metadata(metadata, run_dir / "metadata.json")

    try:
        set_seed(config["seed"])
        rows = read_prompt_rows(config["prompts_csv"], config.get("max_rows"))
        item_texts = [row["item_text"] for row in rows]
        layers = config["layers"]
        positions = config["positions"]
        baseline = config["baseline"]
        # The rows that become points. With a baseline, the other rows only
        # give the baseline.
        point_indices = target_rows(rows, baseline)
        point_rows = [rows[i] for i in point_indices]
        categories = [row["category"] for row in point_rows]
        scopes = scope_rows(categories, config["per_category_pca"])
        lazy_model = LazyModel(config, metadata)
        metadata["device"] = None
        metadata["activation_sources"] = {}
        variants = variants_to_fit(config)

        # {variant: {prompt_name: {layer: {scope: {position: PcaResult}}}}}
        pca_results = {variant: {} for variant in variants}
        prompts = {}
        token_spans = {}
        for prompt_name in config["prompt_columns"]:
            prompts[prompt_name] = [row[prompt_name] for row in rows]
            extracted, source = get_activations(
                config, lazy_model, prompt_name, prompts[prompt_name], item_texts
            )
            print(f"Activations for {prompt_name}: from {source}")
            metadata["activation_sources"][prompt_name] = source
            write_metadata(metadata, run_dir / "metadata.json")
            token_spans[prompt_name] = {
                position: [spans[i] for i in point_indices]
                for position, spans in extracted.token_spans.items()
            }
            # Show only the prompts of the points in the HTML.
            prompts[prompt_name] = [prompts[prompt_name][i] for i in point_indices]
            for variant in variants:
                activations = variant_activations(
                    extracted.activations, rows, baseline, point_indices, variant
                )
                pca_results[variant][prompt_name] = fit_all_pca(
                    activations,
                    layers,
                    positions,
                    config["n_components"],
                    scopes,
                )

        for variant in variants:
            suffix = output_suffix(variant)
            write_explained_variance(
                pca_results[variant], run_dir / f"explained_variance{suffix}.csv"
            )
            write_projections(
                pca_results[variant], scopes, run_dir / f"pca_projections{suffix}.npz"
            )
        labels = variant_labels(config, variants)
        if labels is None:
            plot_results = pca_results[WITH_BASELINE]
        else:
            plot_results = pca_results
        plot_data = build_plot_data(
            title=config.get("title", config["experiment"]),
            layers=layers,
            prompt_names=config["prompt_columns"],
            positions=positions,
            points=point_info(point_rows, config["color_by"]),
            prompts=prompts,
            token_spans=token_spans,
            pca_results=plot_results,
            scopes=scopes,
            color_label=config["color_by"],
            variants=labels,
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
