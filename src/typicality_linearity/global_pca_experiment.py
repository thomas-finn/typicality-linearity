"""PCA of four activation types for the global direction experiments.

The four activation types, for item i rated in category c, prompt format p and
layer l:

- raw: a(i, c)
- cat_contrast: a(i, c) - mean over c' != c of a(i, c')
  (rows of `cat_contrast_csv`, see baselines.py)
- item_contrast: a(i, c) - mean over the contrast items j of a(j, c)
  (the contrast rows of `item_contrast_csv`, see baselines.py)
- both_contrasts: a(i, c) - mean_c' a(i, c') - mean_j a(j, c)

There is one point per rated pair (565). For each type, layer, prompt format
and scope, a PCA is fitted. The scope is "all" (all points) or one category
(only the points of that category). For each fit we also count the components
needed to explain `variance_threshold` of the variance. A scope with n points
has at most n - 1 components that are not zero.

Activations come from the activation cache (see activation_extraction.py). The
model is only loaded if an entry is missing.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml: copy of the config
- metadata.json: git commit, seed, environment, status
- variance_components.csv: one row per fit: type, layer, prompt, scope,
  number of points, components for the threshold, first explained ratios
- <html_name>: the interactive page (see global_pca_plot.py)
If `copy_html_to` is set, the page is also copied to that directory.
"""

import csv
import shutil
import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.baselines import mean_contrast, mean_contrast_groups
from typicality_linearity.global_pca_plot import write_html
from typicality_linearity.pca import (
    explained_variance_ratio_all,
    fit_pca,
    n_components_for_variance,
)
from typicality_linearity.pca_experiment import LazyModel, get_activations
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
    "n_components",
    "variance_threshold",
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

ACTIVATION_TYPES = ("raw", "cat_contrast", "item_contrast", "both_contrasts")

ACTIVATION_LABELS = {
    "raw": "Raw",
    "cat_contrast": "Mean category contrast",
    "item_contrast": "Mean item contrast",
    "both_contrasts": "Both contrasts",
}

ACTIVATION_FORMULAS = {
    "raw": "a(i, c)",
    "cat_contrast": "a(i, c) − E_c′[a(i, c′)]",
    "item_contrast": "a(i, c) − E_i′[a(i′, c)]",
    "both_contrasts": "a(i, c) − E_c′[a(i, c′)] − E_i′[a(i′, c)]",
}

ALL_SCOPE = "all"

CSV_COLUMNS = (
    "activation_type",
    "layer",
    "prompt",
    "scope",
    "n_points",
    "n_components_threshold",
    "explained_ratio_1",
    "explained_ratio_2",
    "explained_ratio_3",
)


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")
    if not 0 < config["variance_threshold"] <= 1:
        raise ValueError("variance_threshold must be in (0, 1]")
    if config["n_components"] not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    return config


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_keys(rows, groups):
    return [(rows[t]["item"], rows[t]["item_category"]) for t, _ in groups]


def check_same_points(cat_rows, cat_groups, item_rows, item_groups):
    """The two files must give the same points in the same order."""
    if group_keys(cat_rows, cat_groups) != group_keys(item_rows, item_groups):
        raise ValueError(
            "The category and item contrast files do not have the same "
            "(item, item_category) pairs in the same order"
        )


def activation_types(cat_values, cat_groups, item_values, item_groups):
    """Return {type: array (n_points, ...)} for the four activation types.

    cat_values, item_values: activations of all rows of the two files, with
    shape (n_rows, ...). The raw activation is the target row of the category
    contrast file. The item contrast mean is subtracted from it, so the result
    does not depend on the anchor rows of the item file.
    """
    targets = [target for target, _ in cat_groups]
    raw = cat_values[targets]
    cat = mean_contrast(cat_values, cat_groups)
    item_means = np.stack(
        [item_values[others].mean(axis=0) for _, others in item_groups]
    )
    return {
        "raw": raw,
        "cat_contrast": cat,
        "item_contrast": raw - item_means,
        "both_contrasts": cat - item_means,
    }


def scope_rows(categories):
    """Return {scope: point indices}. "all" first, then the categories in order
    of first appearance."""
    scopes = {ALL_SCOPE: list(range(len(categories)))}
    for index, category in enumerate(categories):
        scopes.setdefault(category, []).append(index)
    return scopes


def fit_scope(x, n_components, threshold):
    """Return a dict with the projections, the first ratios and the count."""
    pca = fit_pca(x, n_components)
    ratio_all = explained_variance_ratio_all(x)
    return {
        "projections": pca.projections,
        "explained_ratio": pca.explained_variance_ratio,
        "n_components_threshold": n_components_for_variance(ratio_all, threshold),
        "n_points": len(x),
    }


def fit_prompt(types, layers, scopes, n_components, threshold):
    """Return {type: {layer: {scope: fit}}} for one prompt format.

    types: {type: array (n_points, n_layers, d_model)}
    """
    fits = {}
    for activation_type, values in types.items():
        fits[activation_type] = {}
        for layer_index, layer in enumerate(layers):
            fits[activation_type][layer] = {}
            for scope, indices in scopes.items():
                x = values[indices, layer_index, :]
                fits[activation_type][layer][scope] = fit_scope(
                    x, n_components, threshold
                )
    return fits


def csv_rows_for_prompt(prompt, fits):
    rows = []
    for activation_type, by_layer in fits.items():
        for layer, by_scope in by_layer.items():
            for scope, fit in by_scope.items():
                ratios = list(fit["explained_ratio"]) + [None] * 3
                rows.append(
                    {
                        "activation_type": activation_type,
                        "layer": layer,
                        "prompt": prompt,
                        "scope": scope,
                        "n_points": fit["n_points"],
                        "n_components_threshold": fit["n_components_threshold"],
                        "explained_ratio_1": ratios[0],
                        "explained_ratio_2": ratios[1],
                        "explained_ratio_3": ratios[2],
                    }
                )
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def prompt_activations(config, lazy_model, prompt, cat_rows, item_rows):
    """Return (cat values, item values) at the position, all rows."""
    position = config["position"]
    cat_extracted, cat_source = get_activations(
        config,
        lazy_model,
        prompt,
        [row[prompt] for row in cat_rows],
        [row["item_text"] for row in cat_rows],
    )
    item_extracted, item_source = get_activations(
        config,
        lazy_model,
        prompt,
        [row[prompt] for row in item_rows],
        [row["item_text"] for row in item_rows],
    )
    sources = {"cat_contrast_csv": cat_source, "item_contrast_csv": item_source}
    return (
        cat_extracted.activations[position],
        item_extracted.activations[position],
        sources,
    )


def extraction_config(config):
    """Return the config that get_activations and LazyModel use."""
    extraction = dict(config)
    extraction["positions"] = [config["position"]]
    return extraction


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
        extraction = extraction_config(config)
        lazy_model = LazyModel(extraction, metadata)
        metadata["device"] = None
        metadata["activation_sources"] = {}

        cat_rows = read_rows(config["cat_contrast_csv"])
        item_rows = read_rows(config["item_contrast_csv"])
        cat_groups = mean_contrast_groups(cat_rows)
        item_groups = mean_contrast_groups(item_rows)
        check_same_points(cat_rows, cat_groups, item_rows, item_groups)

        targets = [target for target, _ in cat_groups]
        point_rows = [cat_rows[t] for t in targets]
        categories = [row["item_category"] for row in point_rows]
        scopes = scope_rows(categories)

        all_fits = {}
        csv_rows = []
        for number, prompt in enumerate(config["prompt_columns"], 1):
            cat_values, item_values, sources = prompt_activations(
                extraction, lazy_model, prompt, cat_rows, item_rows
            )
            metadata["activation_sources"][prompt] = sources
            types = activation_types(cat_values, cat_groups, item_values, item_groups)
            del cat_values, item_values
            fits = fit_prompt(
                types,
                config["layers"],
                scopes,
                config["n_components"],
                config["variance_threshold"],
            )
            all_fits[prompt] = fits
            csv_rows.extend(csv_rows_for_prompt(prompt, fits))
            print(f"[{number}/{len(config['prompt_columns'])}] {prompt}: {sources}")
            write_metadata(metadata, run_dir / "metadata.json")

        write_csv(csv_rows, run_dir / "variance_components.csv")

        points = {
            "item": [row["item"] for row in point_rows],
            "item_text": [row["item_text"] for row in point_rows],
            "category": categories,
            "rating_zscore": [float(row["rating_zscore"]) for row in point_rows],
        }
        prompt_texts = {
            prompt: [row[prompt] for row in point_rows]
            for prompt in config["prompt_columns"]
        }
        html_path = write_html(
            run_dir / config["html_name"],
            title=config["title"],
            layers=config["layers"],
            prompts=config["prompt_columns"],
            prompt_formats={
                p: PROMPT_FORMATS.get(p, "") for p in config["prompt_columns"]
            },
            activation_types=list(ACTIVATION_TYPES),
            activation_labels=ACTIVATION_LABELS,
            activation_formulas=ACTIVATION_FORMULAS,
            scopes=scopes,
            points=points,
            prompt_texts=prompt_texts,
            fits=all_fits,
            variance_threshold=config["variance_threshold"],
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
