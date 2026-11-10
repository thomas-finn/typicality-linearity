import csv

import numpy as np
import pytest

from typicality_linearity import global_pca_experiment
from typicality_linearity.activation_cache import ExtractedActivations
from typicality_linearity.baselines import mean_contrast_groups
from typicality_linearity.global_pca_experiment import (
    activation_types,
    check_same_points,
    fit_scope,
    run,
    scope_rows,
)
from typicality_linearity.pca import (
    explained_variance_ratio_all,
    n_components_for_variance,
)

CATEGORIES = ["toy", "bird"]
ITEMS = [
    ("doll", "toy", -1.0),
    ("ball", "toy", 0.5),
    ("kite", "toy", 1.0),
    ("robin", "bird", -1.2),
    ("crow", "bird", 0.2),
    ("emu", "bird", 1.5),
]


def cat_rows():
    rows = []
    for item, item_category, z in ITEMS:
        for category in CATEGORIES:
            own = category == item_category
            rows.append(
                {
                    "item": item,
                    "item_category": item_category,
                    "rating_zscore": str(z) if own else "",
                    "item_text": item,
                    "category": category,
                    "is_item_category": str(own),
                    "rosch_example": f"A {item} is an example of a {category}.",
                }
            )
    return rows


def item_rows():
    rows = []
    for item, category, z in ITEMS:
        others = [i for i, c, _ in ITEMS if c != category][:2]
        for prompt_item in [item] + others:
            anchor = prompt_item == item
            rows.append(
                {
                    "item": item,
                    "item_category": category,
                    "rating_zscore": str(z) if anchor else "",
                    "item_text": prompt_item,
                    "category": category,
                    "is_anchor": str(anchor),
                    "rosch_example": f"A {prompt_item} is an example of a {category}.",
                }
            )
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_n_components_for_variance():
    assert n_components_for_variance(np.array([0.5, 0.3, 0.15, 0.05]), 0.95) == 3
    assert n_components_for_variance(np.array([0.5, 0.45, 0.05]), 0.95) == 2
    assert n_components_for_variance(np.array([1.0, 0.0]), 0.95) == 1


def test_explained_variance_ratio_all_sums_to_one():
    x = np.random.default_rng(0).normal(size=(10, 6))
    ratio = explained_variance_ratio_all(x)
    assert ratio.shape == (6,)
    assert np.isclose(ratio.sum(), 1.0)
    assert np.all(np.diff(ratio) <= 1e-12)


def test_activation_types():
    cat = cat_rows()
    item = item_rows()
    cat_groups = mean_contrast_groups(cat)
    item_groups = mean_contrast_groups(item)
    check_same_points(cat, cat_groups, item, item_groups)
    rng = np.random.default_rng(1)
    cat_values = rng.normal(size=(len(cat), 2, 3))
    item_values = rng.normal(size=(len(item), 2, 3))
    types = activation_types(cat_values, cat_groups, item_values, item_groups)
    target, others = cat_groups[0]
    _item_target, item_others = item_groups[0]
    raw = cat_values[target]
    cat_mc = raw - cat_values[others].mean(axis=0)
    item_mc = raw - item_values[item_others].mean(axis=0)
    np.testing.assert_allclose(types["raw"][0], raw)
    np.testing.assert_allclose(types["cat_contrast"][0], cat_mc)
    np.testing.assert_allclose(types["item_contrast"][0], item_mc)
    np.testing.assert_allclose(
        types["both_contrasts"][0],
        raw - cat_values[others].mean(axis=0) - item_values[item_others].mean(axis=0),
    )


def test_check_same_points_fails_on_other_order():
    cat = cat_rows()
    item = item_rows()
    item = item[3:] + item[:3]
    with pytest.raises(ValueError):
        check_same_points(
            cat, mean_contrast_groups(cat), item, mean_contrast_groups(item)
        )


def test_scope_rows_and_fit_scope():
    scopes = scope_rows(["toy", "toy", "bird"])
    assert scopes == {"all": [0, 1, 2], "toy": [0, 1], "bird": [2]}
    assert list(scopes) == ["all", "toy", "bird"]
    x = np.random.default_rng(2).normal(size=(8, 5))
    fit = fit_scope(x, 3, 0.95)
    assert fit["projections"].shape == (8, 3)
    assert fit["n_points"] == 8
    assert 1 <= fit["n_components_threshold"] <= 7


class FakeModel:
    def __init__(self, config, metadata):
        self.config = config
        self.device = "cpu"

    def extract(self, prompts, item_texts):
        rng = np.random.default_rng(len(prompts))
        values = rng.normal(size=(len(prompts), len(self.config["layers"]), 6))
        spans = [(len(p) - 1, len(p)) for p in prompts]
        return ExtractedActivations(
            {"period": values.astype(np.float32)},
            {"period": spans},
            {"period": ["."] * len(prompts)},
        )


def test_run_writes_csv_html_and_copy(tmp_path, monkeypatch):
    cat_path = tmp_path / "cat.csv"
    item_path = tmp_path / "item.csv"
    write_csv(cat_path, cat_rows())
    write_csv(item_path, item_rows())
    config = {
        "experiment": "test-global-pca",
        "title": "Test",
        "model": "fake",
        "cat_contrast_csv": str(cat_path),
        "item_contrast_csv": str(item_path),
        "prompt_columns": ["rosch_example"],
        "layers": [1, 2],
        "position": "period",
        "n_components": 3,
        "variance_threshold": 0.95,
        "batch_size": 2,
        "device": "cpu",
        "dtype": "float32",
        "prefix_token": None,
        "seed": 0,
        "activation_cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "out"),
        "html_name": "page.html",
        "copy_html_to": str(tmp_path / "copy"),
    }
    monkeypatch.setattr(global_pca_experiment, "LazyModel", FakeModel)
    run_dir = run(config, "config.yaml")

    with open(run_dir / "variance_components.csv") as f:
        rows = list(csv.DictReader(f))
    # 4 types x 2 layers x 1 prompt x 3 scopes (all, toy, bird)
    assert len(rows) == 4 * 2 * 1 * 3
    assert {r["scope"] for r in rows} == {"all", "toy", "bird"}
    page = (run_dir / "page.html").read_text()
    assert "Mean item contrast" in page
    assert (tmp_path / "copy" / "page.html").read_text() == page
