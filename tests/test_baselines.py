import csv
import json

import numpy as np
import pytest

from typicality_linearity import pca_experiment
from typicality_linearity.activation_cache import ExtractedActivations
from typicality_linearity.baselines import (
    apply_baseline,
    mean_contrast,
    mean_contrast_groups,
    target_rows,
)
from typicality_linearity.pca import fit_pca

CATEGORIES = ["bird", "fruit", "toy"]


def contrast_rows(items):
    """Rows like rosch_mean_contrast.csv. items: [(item, item_category, zscore)]."""
    rows = []
    for item, item_category, zscore in items:
        for category in CATEGORIES:
            own = category == item_category
            rows.append(
                {
                    "item": item,
                    "item_category": item_category,
                    "rating_zscore": str(zscore) if own else "",
                    "item_text": item,
                    "category": category,
                    "is_item_category": str(own),
                    "is_rated_member": str(own),
                    "rosch_simple": f"{item.capitalize()}: {category}.",
                }
            )
    return rows


def test_groups_find_target_and_others():
    rows = contrast_rows([("doll", "toy", -1.0), ("robin", "bird", 0.5)])
    groups = mean_contrast_groups(rows)
    assert groups == [(2, [0, 1]), (3, [4, 5])]
    assert target_rows(rows, "mean_contrast") == [2, 3]
    assert target_rows(rows, None) == list(range(6))


def test_same_item_in_two_categories_is_two_groups():
    rows = contrast_rows([("tomato", "fruit", 0.0), ("tomato", "toy", 1.0)])
    assert mean_contrast_groups(rows) == [(1, [0, 2]), (5, [3, 4])]


def test_group_without_target_fails():
    rows = contrast_rows([("doll", "toy", -1.0)])[:2]
    with pytest.raises(ValueError):
        mean_contrast_groups(rows)


def test_mean_contrast_subtracts_mean_of_other_categories():
    x = np.array([[1.0, 0.0], [3.0, 2.0], [10.0, 10.0]])
    result = mean_contrast(x, [(2, [0, 1])])
    assert np.allclose(result, [[8.0, 9.0]])


def test_apply_baseline_keeps_layers_and_features():
    rows = contrast_rows([("doll", "toy", -1.0), ("robin", "bird", 0.5)])
    x = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    result = apply_baseline({"period": x}, rows, "mean_contrast")
    assert result["period"].shape == (2, 2, 3)
    assert np.allclose(result["period"][0], x[2] - x[[0, 1]].mean(axis=0))
    assert apply_baseline({"period": x}, rows, None)["period"] is x


def test_unknown_baseline_fails():
    with pytest.raises(ValueError):
        target_rows([], "other")


def write_contrast_csv(tmp_path):
    items = [(f"t{k}", "toy", k - 2.0) for k in range(5)]
    items += [(f"b{k}", "bird", 2.0 - k) for k in range(5)]
    rows = contrast_rows(items)
    csv_path = tmp_path / "prompts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def contrast_config(tmp_path, csv_path):
    return {
        "experiment": "test",
        "model": "none",
        "prompts_csv": str(csv_path),
        "prompt_columns": ["rosch_simple"],
        "layers": [10],
        "positions": ["period"],
        "n_components": 2,
        "per_category_pca": True,
        "baseline": "mean_contrast",
        "color_by": "rating_zscore",
        "batch_size": 4,
        "device": "cpu",
        "dtype": "float32",
        "prefix_token": None,
        "seed": 0,
        "activation_cache_dir": None,
        "output_dir": str(tmp_path / "out"),
        "html_name": "page.html",
    }


def fake_extract(self, prompts, item_texts):
    x = np.random.default_rng(0).normal(size=(len(prompts), 1, 8))
    spans = [(0, 1)] * len(prompts)
    return ExtractedActivations(
        {"period": x}, {"period": spans}, {"period": ["."] * len(prompts)}
    )


def test_run_with_mean_contrast(tmp_path, monkeypatch):
    csv_path = write_contrast_csv(tmp_path)
    monkeypatch.setattr(pca_experiment.LazyModel, "extract", fake_extract)
    config = contrast_config(tmp_path, csv_path)
    run_dir = pca_experiment.run(config, "config.yaml")

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    with np.load(run_dir / "pca_projections.npz") as data:
        assert data["rosch_simple/10/all/period"].shape == (10, 2)
        assert data["rosch_simple/10/toy/period"].shape == (5, 2)
    page = (run_dir / "page.html").read_text()
    assert "T0: toy." in page
    assert "T0: bird." not in page
    assert not (run_dir / "pca_projections_no_baseline.npz").exists()


def test_run_with_mean_contrast_and_no_baseline_comparison(tmp_path, monkeypatch):
    csv_path = write_contrast_csv(tmp_path)
    monkeypatch.setattr(pca_experiment.LazyModel, "extract", fake_extract)
    config = contrast_config(tmp_path, csv_path)
    config["compare_no_baseline"] = True
    run_dir = pca_experiment.run(config, "config.yaml")

    rows = read_contrast_rows(csv_path)
    x = fake_extract(None, [None] * len(rows), None).activations["period"]
    targets = target_rows(rows, "mean_contrast")
    expected = fit_pca(x[targets, 0, :], 2).projections
    with np.load(run_dir / "pca_projections_no_baseline.npz") as data:
        assert np.allclose(data["rosch_simple/10/all/period"], expected)
    with np.load(run_dir / "pca_projections.npz") as data:
        assert not np.allclose(data["rosch_simple/10/all/period"], expected)
    assert (run_dir / "explained_variance_no_baseline.csv").exists()
    page = (run_dir / "page.html").read_text()
    assert "no baseline" in page
    assert "mean_contrast baseline" in page


def test_compare_no_baseline_needs_a_baseline(tmp_path, monkeypatch):
    csv_path = write_contrast_csv(tmp_path)
    config = contrast_config(tmp_path, csv_path)
    config["baseline"] = None
    config["compare_no_baseline"] = True
    with pytest.raises(ValueError):
        pca_experiment.run(config, "config.yaml")


def read_contrast_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))
