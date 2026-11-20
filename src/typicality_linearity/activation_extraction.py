"""Extract activations for all prompt columns of a set of CSVs into the cache.

This step only fills the activation cache (see activation_cache.py). It does
no analysis. Later experiments with the same model, dtype, layers, positions,
prefix token and prompts read the activations from the cache.

Each (CSV, prompt column) pair is one cache entry. An entry that is already in
the cache is not made again. The model is loaded only if one entry must be
extracted.

Derived entries: `derive_from` maps a target CSV to a source CSV. Each row of
the target must have the same prompt and item text as a row of the source
(for example, every prompt in rosch_mean_item_contrast.csv is also in
rosch_mean_cat_contrast.csv). Then the target entry is made from the cached
source entry of the same column, without the model. The target entry has its
own cache key, so experiments find it as a normal entry. The activations are
the ones the model gave for the source row; the prompt and item text are the
same, so only the batch is different.

Extended entries: `extend_from` maps a target CSV to a source CSV, as
derive_from, but the target can have rows that are not in the source (for
example, rosch_mean_cat_contrast_extended.csv has the rows of
rosch_mean_cat_contrast.csv and rows for 10 more categories). The rows that
are in the source are copied from the cached source entry; only the other
rows are extracted with the model.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml: copy of the config
- metadata.json: git commit, seed, environment, status
- manifest.csv: one row per entry: CSV, column, number of prompts, cache key,
  source ("cache", "model", "derived" or "extended"), the key of the source
  entry for a derived or extended entry, the number of rows extracted with
  the model, and the path of the .npz file
"""

import csv
import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.activation_cache import (
    ExtractedActivations,
    cache_key,
    entry_paths,
    key_inputs,
    load_entry,
    model_cache_dir,
    save_entry,
)
from typicality_linearity.activation_source import LazyModel
from typicality_linearity.run_metadata import (
    make_run_id,
    now_utc,
    set_seed,
    start_metadata,
    write_metadata,
)
from typicality_linearity.token_positions import period_char_index

REQUIRED_KEYS = (
    "experiment",
    "model",
    "prompt_csvs",
    "prompt_columns",
    "layers",
    "positions",
    "batch_size",
    "device",
    "dtype",
    "prefix_token",
    "seed",
    "activation_cache_dir",
    "output_dir",
)

# The prefix of the prompt columns in the Rosch CSVs.
PROMPT_COLUMN_PREFIX = "rosch_"

MANIFEST_COLUMNS = (
    "prompts_csv",
    "prompt_column",
    "n_prompts",
    "key",
    "source",
    "source_key",
    "n_model_rows",
    "npz_path",
)


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")
    if config["activation_cache_dir"] is None:
        raise ValueError("activation_cache_dir must be set: this step fills the cache")
    check_derive_from(config)
    return config


def derive_from_of(config):
    """Return {target csv: source csv}. Empty if the config has no derive_from."""
    return config.get("derive_from") or {}


def extend_from_of(config):
    """Return {target csv: source csv}. Empty if the config has no extend_from."""
    return config.get("extend_from") or {}


def sources_of(config):
    """Return {target csv: (source csv, "derive" or "extend")}."""
    sources = {}
    for target, source in derive_from_of(config).items():
        sources[target] = (source, "derive")
    for target, source in extend_from_of(config).items():
        if target in sources:
            raise ValueError(f"{target} is in both derive_from and extend_from")
        sources[target] = (source, "extend")
    return sources


def check_derive_from(config):
    csvs = set(config["prompt_csvs"])
    sources = sources_of(config)
    for target, (source, mode) in sources.items():
        name = f"{mode}_from"
        if target not in csvs or source not in csvs:
            raise ValueError(
                f"{name} {target} -> {source}: both files must be in prompt_csvs"
            )
        if source in sources:
            raise ValueError(f"{name} source {source} is also derived or extended")


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def prompt_columns_of(rows, prompt_columns):
    """Return the prompt columns to extract.

    prompt_columns: "all" (all columns that start with "rosch_") or a list.
    """
    header = list(rows[0].keys())
    if prompt_columns == "all":
        columns = [c for c in header if c.startswith(PROMPT_COLUMN_PREFIX)]
    else:
        columns = list(prompt_columns)
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"Prompt columns not in the CSV: {missing}")
    if not columns:
        raise ValueError("No prompt columns found")
    return columns


def check_prompts(prompts, positions):
    """Fail early if a prompt has no final period and "period" is a position."""
    if "period" not in positions:
        return
    for prompt in prompts:
        period_char_index(prompt)


def extraction_jobs(config):
    """Return a list of job dicts, one per (CSV, prompt column).

    The jobs of the CSVs that are not derived come first, in config order, so
    a source entry is always made before the entries derived from it.
    """
    sources = sources_of(config)
    ordered_csvs = [p for p in config["prompt_csvs"] if p not in sources]
    ordered_csvs += [p for p in config["prompt_csvs"] if p in sources]

    jobs = []
    for csv_path in ordered_csvs:
        rows = read_rows(csv_path)
        item_texts = [row["item_text"] for row in rows]
        for column in prompt_columns_of(rows, config["prompt_columns"]):
            prompts = [row[column] for row in rows]
            check_prompts(prompts, config["positions"])
            jobs.append(
                {
                    "csv": csv_path,
                    "column": column,
                    "prompts": prompts,
                    "item_texts": item_texts,
                    "source_csv": sources.get(csv_path, (None, None))[0],
                    "mode": sources.get(csv_path, (None, None))[1],
                }
            )
    return jobs


def find_job(jobs, csv_path, column):
    for job in jobs:
        if job["csv"] == csv_path and job["column"] == column:
            return job
    raise ValueError(f"No job for column {column!r} of {csv_path}")


def entry_key(config, prompts, item_texts):
    inputs = key_inputs(
        config["model"],
        config["dtype"],
        config["layers"],
        config["positions"],
        prompts,
        item_texts,
        config["prefix_token"],
    )
    return inputs, cache_key(inputs)


def source_row_indices(
    source_prompts, source_item_texts, prompts, item_texts, allow_missing=False
):
    """Return, for each target row, the index of a source row with the same
    prompt and item text. With allow_missing, a row that is not in the source
    gets None; otherwise it is an error."""
    lookup = {}
    for index, pair in enumerate(zip(source_prompts, source_item_texts)):
        lookup.setdefault(pair, index)
    indices = []
    for prompt, item_text in zip(prompts, item_texts):
        if (prompt, item_text) in lookup:
            indices.append(lookup[(prompt, item_text)])
        elif allow_missing:
            indices.append(None)
        else:
            raise ValueError(
                f"Prompt {prompt!r} with item text {item_text!r} is not in the source"
            )
    return indices


def select_rows(extracted, indices):
    """Return the ExtractedActivations of the rows in indices, in that order."""
    activations = {}
    token_spans = {}
    token_texts = {}
    for position, values in extracted.activations.items():
        activations[position] = values[indices]
        token_spans[position] = [extracted.token_spans[position][i] for i in indices]
        token_texts[position] = [extracted.token_texts[position][i] for i in indices]
    return ExtractedActivations(activations, token_spans, token_texts)


def merge_rows(n_rows, parts):
    """Return one ExtractedActivations from parts.

    parts: list of (target row indices, ExtractedActivations of those rows).
    Each target row must be in exactly one part.
    """
    first = parts[0][1]
    activations = {}
    token_spans = {}
    token_texts = {}
    for position, values in first.activations.items():
        activations[position] = np.empty(
            (n_rows,) + values.shape[1:], dtype=values.dtype
        )
        token_spans[position] = [None] * n_rows
        token_texts[position] = [None] * n_rows
    for rows, extracted in parts:
        for position, values in activations.items():
            values[rows] = extracted.activations[position]
            for k, row in enumerate(rows):
                token_spans[position][row] = extracted.token_spans[position][k]
                token_texts[position][row] = extracted.token_texts[position][k]
    for position in activations:
        if any(span is None for span in token_spans[position]):
            raise ValueError("merge_rows: a row is in no part")
    return ExtractedActivations(activations, token_spans, token_texts)


def cache_dir_of(config):
    return model_cache_dir(config["activation_cache_dir"], config["model"])


def entry_exists(config, prompt_name, key):
    # Only check that the files exist, so a large entry is not loaded.
    npz_path, json_path = entry_paths(cache_dir_of(config), prompt_name, key)
    return npz_path.exists() and json_path.exists()


def extract_job(config, lazy_model, job):
    """Put one entry in the cache with the model. Return a manifest row."""
    inputs, key = entry_key(config, job["prompts"], job["item_texts"])
    if not entry_exists(config, job["column"], key):
        extracted = lazy_model.extract(job["prompts"], job["item_texts"])
        extra_info = {
            "device": lazy_model.device,
            "batch_size": config["batch_size"],
            "experiment": config["experiment"],
        }
        save_entry(
            cache_dir_of(config),
            job["column"],
            key,
            inputs,
            extracted,
            extra_info,
        )
        return manifest_row(config, job, key, "model", "", len(job["prompts"]))
    return manifest_row(config, job, key, "cache", "", 0)


def derive_job(config, job, source_job, lazy_model=None):
    """Put one entry in the cache from the source entry. Return a manifest row.

    With job mode "extend", the rows that are not in the source are extracted
    with lazy_model.
    """
    extend = job["mode"] == "extend"
    inputs, key = entry_key(config, job["prompts"], job["item_texts"])
    _source_inputs, source_key = entry_key(
        config, source_job["prompts"], source_job["item_texts"]
    )
    if entry_exists(config, job["column"], key):
        return manifest_row(config, job, key, "cache", source_key, 0)

    source = load_entry(
        cache_dir_of(config),
        source_job["column"],
        source_key,
        config["positions"],
    )
    if source is None:
        raise ValueError(
            f"Source entry {source_job['column']}_{source_key} is not in the cache"
        )
    indices = source_row_indices(
        source_job["prompts"],
        source_job["item_texts"],
        job["prompts"],
        job["item_texts"],
        allow_missing=extend,
    )
    copied = [row for row, index in enumerate(indices) if index is not None]
    missing = [row for row, index in enumerate(indices) if index is None]
    parts = [(copied, select_rows(source, [indices[row] for row in copied]))]
    if missing:
        extracted = lazy_model.extract(
            [job["prompts"][row] for row in missing],
            [job["item_texts"][row] for row in missing],
        )
        parts.append((missing, extracted))
    extra_info = {
        "experiment": config["experiment"],
        "derived_from": {
            "prompts_csv": source_job["csv"],
            "prompt_name": source_job["column"],
            "key": source_key,
        },
    }
    if extend:
        extra_info["n_model_rows"] = len(missing)
        extra_info["device"] = lazy_model.device
        extra_info["batch_size"] = config["batch_size"]
    save_entry(
        cache_dir_of(config),
        job["column"],
        key,
        inputs,
        merge_rows(len(job["prompts"]), parts),
        extra_info,
    )
    source_name = "extended" if extend else "derived"
    return manifest_row(config, job, key, source_name, source_key, len(missing))


def manifest_row(config, job, key, source, source_key, n_model_rows):
    npz_path, _json_path = entry_paths(cache_dir_of(config), job["column"], key)
    return {
        "prompts_csv": job["csv"],
        "prompt_column": job["column"],
        "n_prompts": len(job["prompts"]),
        "key": key,
        "source": source,
        "source_key": source_key,
        "n_model_rows": n_model_rows,
        "npz_path": str(npz_path),
    }


def write_manifest(manifest_rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(manifest_rows)


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
        jobs = extraction_jobs(config)
        lazy_model = LazyModel(config, metadata)
        metadata["device"] = None
        metadata["n_entries"] = len(jobs)
        manifest_rows = []
        for number, job in enumerate(jobs, 1):
            if job["source_csv"] is None:
                row = extract_job(config, lazy_model, job)
            else:
                source_job = find_job(jobs, job["source_csv"], job["column"])
                row = derive_job(config, job, source_job, lazy_model)
            print(
                f"[{number}/{len(jobs)}] {job['csv']} {job['column']}: "
                f"{len(job['prompts'])} prompts, {row['source']}"
            )
            manifest_rows.append(row)
            write_manifest(manifest_rows, run_dir / "manifest.csv")
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
