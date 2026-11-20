import csv
import json

import numpy as np
import pytest

from typicality_linearity import activation_extraction
from typicality_linearity.activation_cache import (
    ExtractedActivations,
    cache_key,
    key_inputs,
    load_entry,
    model_cache_dir,
)
from typicality_linearity.activation_extraction import (
    check_derive_from,
    extraction_jobs,
    prompt_columns_of,
    run,
    source_row_indices,
)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prompt_rows():
    return [
        {
            "item": "doll",
            "item_text": "doll",
            "rosch_example": "A doll is an example of a toy.",
            "rosch_simple": "Doll: toy.",
        },
        {
            "item": "robin",
            "item_text": "robin",
            "rosch_example": "A robin is an example of a bird.",
            "rosch_simple": "Robin: bird.",
        },
    ]


def make_config(tmp_path, csv_paths):
    return {
        "experiment": "test-extraction",
        "model": "fake-model",
        "prompt_csvs": [str(p) for p in csv_paths],
        "prompt_columns": "all",
        "layers": [1, 2],
        "positions": ["period"],
        "batch_size": 2,
        "device": "cpu",
        "dtype": "float32",
        "prefix_token": None,
        "seed": 0,
        "activation_cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "out"),
    }


class FakeModel:
    """Stands in for LazyModel. Counts the calls to extract."""

    calls = 0

    def __init__(self, config, metadata):
        self.config = config
        self.device = "cpu"

    def extract(self, prompts, item_texts):
        FakeModel.calls += 1
        n_layers = len(self.config["layers"])
        # Each prompt gets its own values, so rows can be told apart.
        values = np.zeros((len(prompts), n_layers, 4), dtype=np.float32)
        for row, prompt in enumerate(prompts):
            values[row] = sum(prompt.encode("utf-8")) + len(prompt)
        spans = [(len(p) - 1, len(p)) for p in prompts]
        return ExtractedActivations(
            {"period": values}, {"period": spans}, {"period": ["."] * len(prompts)}
        )


def test_prompt_columns_all_and_list():
    rows = prompt_rows()
    assert prompt_columns_of(rows, "all") == ["rosch_example", "rosch_simple"]
    assert prompt_columns_of(rows, ["rosch_simple"]) == ["rosch_simple"]
    with pytest.raises(ValueError):
        prompt_columns_of(rows, ["rosch_missing"])


def test_prompt_without_period_fails(tmp_path):
    rows = prompt_rows()
    rows[0]["rosch_simple"] = "Doll: toy"
    path = tmp_path / "p.csv"
    write_csv(path, rows)
    with pytest.raises(ValueError):
        extraction_jobs(make_config(tmp_path, [path]))


def test_run_fills_cache_then_uses_it(tmp_path, monkeypatch):
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    write_csv(path_a, prompt_rows())
    write_csv(path_b, prompt_rows()[:1])
    config = make_config(tmp_path, [path_a, path_b])
    monkeypatch.setattr(activation_extraction, "LazyModel", FakeModel)
    FakeModel.calls = 0

    run_dir = run(config, "config.yaml")
    assert FakeModel.calls == 4
    with open(run_dir / "manifest.csv") as f:
        manifest = list(csv.DictReader(f))
    assert [(r["prompt_column"], r["n_prompts"]) for r in manifest] == [
        ("rosch_example", "2"),
        ("rosch_simple", "2"),
        ("rosch_example", "1"),
        ("rosch_simple", "1"),
    ]
    assert all(r["source"] == "model" for r in manifest)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    # The entries are in the subdirectory of the model.
    model_dir = tmp_path / "cache" / "fake-model"
    assert len(list(model_dir.glob("*.npz"))) == 4
    assert list((tmp_path / "cache").glob("*.npz")) == []
    assert all(r["npz_path"].startswith(str(model_dir)) for r in manifest)

    # A second run finds all entries in the cache and does not extract.
    config["output_dir"] = str(tmp_path / "out2")
    run_dir = run(config, "config.yaml")
    assert FakeModel.calls == 4
    with open(run_dir / "manifest.csv") as f:
        assert all(r["source"] == "cache" for r in csv.DictReader(f))


def contrast_rows():
    """Rows like a category contrast file: two items, two categories."""
    rows = []
    for item in ["doll", "robin"]:
        for category in ["toy", "bird"]:
            rows.append(
                {
                    "item": item,
                    "item_text": item,
                    "rosch_example": f"A {item} is an example of a {category}.",
                }
            )
    return rows


def item_contrast_rows():
    """Rows like an item contrast file: every prompt is in contrast_rows()."""
    rows = []
    for anchor, category, other in [
        ("doll", "toy", "robin"),
        ("robin", "bird", "doll"),
    ]:
        for prompt_item in [anchor, other]:
            rows.append(
                {
                    "item": anchor,
                    "item_text": prompt_item,
                    "rosch_example": f"A {prompt_item} is an example of a {category}.",
                }
            )
    return rows


def entry_of(config, csv_rows, column):
    prompts = [row[column] for row in csv_rows]
    item_texts = [row["item_text"] for row in csv_rows]
    inputs = key_inputs(
        config["model"],
        config["dtype"],
        config["layers"],
        config["positions"],
        prompts,
        item_texts,
        config["prefix_token"],
    )
    cache_dir = model_cache_dir(config["activation_cache_dir"], config["model"])
    return load_entry(cache_dir, column, cache_key(inputs), config["positions"])


def test_source_row_indices():
    assert source_row_indices(
        ["a.", "b.", "a."], ["x", "y", "x"], ["a.", "b."], ["x", "y"]
    ) == [0, 1]
    with pytest.raises(ValueError):
        source_row_indices(["a."], ["x"], ["a."], ["other"])


def test_derived_entry_equals_direct_extraction(tmp_path, monkeypatch):
    source = tmp_path / "cat.csv"
    target = tmp_path / "item.csv"
    write_csv(source, contrast_rows())
    write_csv(target, item_contrast_rows())
    # The target is listed first: the source must still be made first.
    config = make_config(tmp_path, [target, source])
    config["derive_from"] = {str(target): str(source)}
    monkeypatch.setattr(activation_extraction, "LazyModel", FakeModel)
    FakeModel.calls = 0

    run_dir = run(config, "config.yaml")
    assert FakeModel.calls == 1
    with open(run_dir / "manifest.csv") as f:
        manifest = list(csv.DictReader(f))
    assert [(r["prompts_csv"], r["source"]) for r in manifest] == [
        (str(source), "model"),
        (str(target), "derived"),
    ]
    assert manifest[1]["source_key"] == manifest[0]["key"]

    derived = entry_of(config, item_contrast_rows(), "rosch_example")
    prompts = [row["rosch_example"] for row in item_contrast_rows()]
    direct = FakeModel(config, {}).extract(prompts, None)
    np.testing.assert_array_equal(
        derived.activations["period"], direct.activations["period"]
    )
    assert derived.token_spans["period"] == direct.token_spans["period"]

    # A second run finds the derived entry in the cache.
    config["output_dir"] = str(tmp_path / "out2")
    run_dir = run(config, "config.yaml")
    with open(run_dir / "manifest.csv") as f:
        assert [r["source"] for r in csv.DictReader(f)] == ["cache", "cache"]


def test_derived_prompt_not_in_source_fails(tmp_path, monkeypatch):
    source = tmp_path / "cat.csv"
    target = tmp_path / "item.csv"
    write_csv(source, contrast_rows())
    rows = item_contrast_rows()
    rows[0]["rosch_example"] = "A doll is an example of a vehicle."
    write_csv(target, rows)
    config = make_config(tmp_path, [source, target])
    config["derive_from"] = {str(target): str(source)}
    monkeypatch.setattr(activation_extraction, "LazyModel", FakeModel)
    with pytest.raises(ValueError):
        run(config, "config.yaml")


def test_check_derive_from(tmp_path):
    config = make_config(tmp_path, ["a.csv", "b.csv"])
    config["derive_from"] = {"a.csv": "c.csv"}
    with pytest.raises(ValueError):
        check_derive_from(config)
    config["derive_from"] = {"a.csv": "b.csv", "b.csv": "a.csv"}
    with pytest.raises(ValueError):
        check_derive_from(config)


def extended_contrast_rows():
    """Rows like an extended contrast file: contrast_rows() and new categories."""
    rows = []
    for item in ("doll", "robin"):
        for category in ("toy", "bird", "fish", "drink"):
            rows.append(
                {
                    "item": item,
                    "item_text": item,
                    "rosch_example": f"A {item} is an example of a {category}.",
                }
            )
    return rows


def test_extended_entry_copies_source_rows_and_extracts_the_others(
    tmp_path, monkeypatch
):
    source = tmp_path / "cat.csv"
    target = tmp_path / "cat_extended.csv"
    write_csv(source, contrast_rows())
    write_csv(target, extended_contrast_rows())
    config = make_config(tmp_path, [target, source])
    config["extend_from"] = {str(target): str(source)}
    extracted_prompts = []

    class CountingModel(FakeModel):
        def extract(self, prompts, item_texts):
            extracted_prompts.append(list(prompts))
            return super().extract(prompts, item_texts)

    monkeypatch.setattr(activation_extraction, "LazyModel", CountingModel)
    run_dir = run(config, "config.yaml")

    source_prompts = {row["rosch_example"] for row in contrast_rows()}
    new_prompts = [
        row["rosch_example"]
        for row in extended_contrast_rows()
        if row["rosch_example"] not in source_prompts
    ]
    # One call for the source, one call with only the new rows of the target.
    assert len(extracted_prompts) == 2
    assert extracted_prompts[1] == new_prompts
    with open(run_dir / "manifest.csv") as f:
        manifest = {r["prompts_csv"]: r for r in csv.DictReader(f)}
    assert manifest[str(target)]["source"] == "extended"
    assert manifest[str(target)]["n_model_rows"] == str(len(new_prompts))

    extended = entry_of(config, extended_contrast_rows(), "rosch_example")
    prompts = [row["rosch_example"] for row in extended_contrast_rows()]
    direct = FakeModel(config, {}).extract(prompts, None)
    np.testing.assert_array_equal(
        extended.activations["period"], direct.activations["period"]
    )
    assert extended.token_spans["period"] == direct.token_spans["period"]


def test_a_file_cannot_be_derived_and_extended(tmp_path):
    config = make_config(tmp_path, ["a.csv", "b.csv"])
    config["derive_from"] = {"a.csv": "b.csv"}
    config["extend_from"] = {"a.csv": "b.csv"}
    with pytest.raises(ValueError):
        check_derive_from(config)
