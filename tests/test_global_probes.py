import csv

import numpy as np
import pytest

from typicality_linearity import global_probe_experiment
from typicality_linearity.activation_cache import ExtractedActivations
from typicality_linearity.global_probe_experiment import (
    cv_value,
    diagonal_mean,
    keep_mask,
    off_diagonal_mean,
    out_of_fold_scores,
    run,
    setting_matrices,
    zscore_within_folds,
)
from typicality_linearity.global_probe_plot import cosine_linkage, linkage_lists
from typicality_linearity.probes import (
    fit_diff_of_means,
    make_folds,
    spearman,
    wilcoxon_auc,
)


def test_wilcoxon_auc():
    y = np.arange(9.0)
    assert wilcoxon_auc(y, y, 3) == 1.0
    assert wilcoxon_auc(-y, y, 3) == 0.0
    assert wilcoxon_auc(np.zeros(9), y, 3) == 0.5
    # Only the extreme groups count: the middle items do not change the AUC.
    scores = y.copy()
    scores[3:6] = [100.0, -100.0, 0.0]
    assert wilcoxon_auc(scores, y, 3) == 1.0
    assert np.isnan(wilcoxon_auc([np.nan] * 9, y, 3))


def test_zscore_within_folds_removes_fold_offsets():
    scores = np.array([1.0, 2.0, 3.0, 101.0, 102.0, 103.0])
    folds = np.array([0, 0, 0, 1, 1, 1])
    z = zscore_within_folds(scores, folds)
    np.testing.assert_allclose(z[:3], z[3:])
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert cv_value("spearman", scores, y, folds, 3, "fold_zscored") == pytest.approx(1)
    assert cv_value("spearman", scores, y, folds, 3, "pooled") < 1


def test_out_of_fold_scores_use_the_fold_probe():
    x = np.eye(3)
    directions = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    folds = np.array([0, 1, 2])
    np.testing.assert_allclose(out_of_fold_scores(directions, x, folds), [1, 2, 3])


def synthetic(n_per_category=30, d=8, seed=0):
    """Two categories; typicality is linear along axis 0 in all prompts."""
    rng = np.random.default_rng(seed)
    categories = ["toy"] * n_per_category + ["bird"] * n_per_category
    y = rng.normal(size=len(categories))
    x_by_prompt = []
    for _ in range(2):
        x = rng.normal(scale=0.3, size=(len(y), d))
        x[:, 0] += y
        x_by_prompt.append(x)
    folds = make_folds(categories, y, 5, seed)
    return x_by_prompt, y, categories, folds


def test_setting_matrices_find_a_linear_signal():
    x_by_prompt, y, categories, folds = synthetic()
    metrics = ["spearman", "pearson", "wilcoxon_auc"]
    result = setting_matrices(
        x_by_prompt, y, categories, folds, 3, metrics, "fold_zscored", 5
    )
    assert result["category_transfer"].shape == (3, 2, 2, 2)
    assert result["category_cosine"].shape == (2, 2, 2)
    assert result["prompt_transfer"].shape == (3, 2, 2, 2)
    assert result["prompt_cosine"].shape == (2, 2, 2)
    assert diagonal_mean(result["category_transfer"][0]) > 0.8
    assert off_diagonal_mean(result["category_transfer"][0]) > 0.8
    assert off_diagonal_mean(result["prompt_transfer"][2]) > 0.9
    np.testing.assert_allclose(
        np.diagonal(result["category_cosine"], axis1=1, axis2=2), 1
    )
    assert off_diagonal_mean(result["category_cosine"]) > 0.7


def test_setting_matrices_no_signal_is_near_chance():
    x_by_prompt, y, categories, folds = synthetic(n_per_category=60)
    rng = np.random.default_rng(5)
    x_by_prompt = [rng.normal(size=x.shape) for x in x_by_prompt]
    result = setting_matrices(
        x_by_prompt,
        y,
        categories,
        folds,
        3,
        ["spearman", "wilcoxon_auc"],
        "fold_zscored",
        5,
    )
    assert abs(off_diagonal_mean(result["category_transfer"][0])) < 0.25
    assert abs(off_diagonal_mean(result["category_transfer"][1]) - 0.5) < 0.15


def test_setting_matrices_full_diagonal_is_in_sample():
    x_by_prompt, y, categories, folds = synthetic()
    arguments = (x_by_prompt, y, categories, folds, 3, ["spearman"], "fold_zscored", 5)
    cv = setting_matrices(*arguments, diagonal="cv")
    full = setting_matrices(*arguments, diagonal="full")
    toy = np.flatnonzero(np.asarray(categories) == "toy")
    for pi, x in enumerate(x_by_prompt):
        direction = fit_diff_of_means(x[toy], y[toy], 3)
        expected = spearman(x[toy] @ direction, y[toy])
        assert full["category_transfer"][0, pi, 0, 0] == pytest.approx(expected)
        assert full["prompt_transfer"][0, 0, pi, pi] == pytest.approx(expected)
    # Only the diagonal changes.
    for name in ("category_transfer", "prompt_transfer"):
        off = ~np.eye(2, dtype=bool)
        np.testing.assert_allclose(full[name][..., off], cv[name][..., off])
        assert not np.allclose(
            np.diagonal(full[name], axis1=-2, axis2=-1),
            np.diagonal(cv[name], axis1=-2, axis2=-1),
        )


def test_keep_mask():
    rows = [
        {"is_ambiguous_extended": "True", "is_ambiguous_core": "False"},
        {"is_ambiguous_extended": "True", "is_ambiguous_core": "True"},
        {"is_ambiguous_extended": "False", "is_ambiguous_core": "False"},
    ]
    assert keep_mask(rows, "include").tolist() == [True, True, True]
    assert keep_mask(rows, "exclude_core").tolist() == [True, False, True]
    assert keep_mask(rows, "exclude_extended").tolist() == [False, False, True]


# End-to-end run with a fake model and small CSVs.

CATEGORIES = ["toy", "bird"]


def items():
    rng = np.random.default_rng(3)
    result = []
    for category in CATEGORIES:
        for k in range(20):
            result.append(
                (f"{category}{k}", category, float(rng.normal()), k == 0, k < 2)
            )
    return result


def cat_rows():
    rows = []
    for item, item_category, z, core, extended in items():
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
                    "is_ambiguous_extended": str(extended),
                    "is_ambiguous_core": str(core),
                    "rosch_example": f"A {item} is an example of a {category}.",
                    "rosch_neg": f"A {item} is not an example of a {category}.",
                }
            )
    return rows


def item_rows():
    rows = []
    all_items = items()
    for item, category, z, core, extended in all_items:
        others = [i for i, c, *_ in all_items if c != category][:3]
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
                    "rosch_neg": f"A {prompt_item} is not an example of a {category}.",
                }
            )
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def test_run_end_to_end(tmp_path, monkeypatch):
    cat_path = tmp_path / "cat.csv"
    item_path = tmp_path / "item.csv"
    write_csv(cat_path, cat_rows())
    write_csv(item_path, item_rows())
    config = {
        "experiment": "test-global-probes",
        "title": "Test",
        "model": "fake",
        "cat_contrast_csv": str(cat_path),
        "item_contrast_csv": str(item_path),
        "prompt_columns": ["rosch_example", "rosch_neg"],
        "layers": [1, 2],
        "position": "period",
        "ambiguity": ["include", "exclude_core", "exclude_extended"],
        "n_groups": [2, 3],
        "metrics": ["spearman", "pearson", "wilcoxon_auc"],
        "n_folds": 5,
        "cv_aggregate": "fold_zscored",
        "target": "rating_zscore",
        "target_sign": -1,
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
    monkeypatch.setattr(global_probe_experiment, "LazyModel", FakeModel)
    run_dir = run(config, "config.yaml")

    with np.load(run_dir / "matrices.npz") as data:
        assert data["category_transfer"].shape == (3, 4, 2, 2, 3, 2, 2, 2)
        assert data["prompt_cosine"].shape == (3, 4, 2, 2, 2, 2, 2)
    with open(run_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 4 * 2 * 2 * 3
    n_items = {r["ambiguity"]: r["n_items"] for r in rows}
    assert n_items == {"include": "40", "exclude_core": "38", "exclude_extended": "36"}
    assert {r["within_method"] for r in rows} == {"cv"}
    page = (run_dir / "page.html").read_text()
    assert 'texttemplate: "%{z:.2f}"' in page
    assert "Cosine between prompts" in page
    assert "NaN" not in page
    assert (tmp_path / "copy" / "page.html").exists()


def test_cosine_linkage_joins_the_closest_pair_first():
    # Directions 0 and 1 are close, 2 and 3 are close, the two pairs are far.
    vectors = np.array([[1.0, 0.1, 0], [1.0, 0, 0], [0, 1.0, 0], [0.1, 1.0, 0]])
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = unit @ unit.T
    merges, correlation = cosine_linkage(cosine)
    assert len(merges) == 3
    first_pairs = {frozenset(m[:2]) for m in merges[:2]}
    assert first_pairs == {frozenset({0, 1}), frozenset({2, 3})}
    # Root height is 1 - mean cosine between the two pairs (average linkage).
    expected_root = 1.0 - cosine[np.ix_([0, 1], [2, 3])].mean()
    assert merges[-1][2] == pytest.approx(expected_root, abs=1e-3)
    assert merges[-1][:2] == [4, 5]
    assert correlation > 0.9


def test_cosine_linkage_nan_gives_no_tree():
    cosine = np.eye(3)
    cosine[0, 1] = cosine[1, 0] = np.nan
    assert cosine_linkage(cosine) == (None, None)


def test_linkage_lists_keeps_the_leading_axes():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(2, 3, 5, 4))
    unit = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    cosines = unit @ np.swapaxes(unit, -1, -2)
    cosines[1, 2, 0, 1] = np.nan
    merges, correlations, largest = linkage_lists(cosines)
    assert len(merges) == 2 and len(merges[0]) == 3
    assert len(merges[0][0]) == 4
    assert merges[1][2] is None and correlations[1][2] is None
    assert 0 < largest <= 2


NONCE_WORDS = ("wugawrar", "njonjobura", "duplambi")


def nonce_rows():
    rows = []
    # Reverse order, so the test checks that the rows are matched by key.
    for item, category, *_ in reversed(items()):
        for word in NONCE_WORDS:
            rows.append(
                {
                    "item": item,
                    "category": category,
                    "nonce_category": word,
                    "item_text": item,
                    "rosch_example": f"A {item} is an example of a {word}.",
                    "rosch_neg": f"A {item} is not an example of a {word}.",
                }
            )
    return rows


def test_nonce_contrast_matches_rows_by_key():
    from typicality_linearity.global_probe_experiment import (
        nonce_contrast,
        nonce_row_order,
    )

    points = [
        {"item": "doll", "item_category": "toy"},
        {"item": "robin", "item_category": "bird"},
    ]
    nonce = [
        {"item": "robin", "category": "bird"},
        {"item": "doll", "category": "toy"},
        {"item": "robin", "category": "bird"},
        {"item": "doll", "category": "toy"},
    ]
    order = nonce_row_order(points, nonce)
    assert order == [[1, 3], [0, 2]]
    raw = np.array([[10.0, 10.0], [20.0, 20.0]])
    nonce_values = np.array([[2.0, 0.0], [1.0, 0.0], [4.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        nonce_contrast(raw, nonce_values, order), [[8, 8], [17, 19]]
    )
    with pytest.raises(ValueError):
        nonce_row_order(points, nonce[:1])
    with pytest.raises(ValueError):
        nonce_row_order(points, nonce[:3])


def test_run_with_nonce_contrast(tmp_path, monkeypatch):
    cat_path = tmp_path / "cat.csv"
    item_path = tmp_path / "item.csv"
    nonce_path = tmp_path / "nonce.csv"
    write_csv(cat_path, cat_rows())
    write_csv(item_path, item_rows())
    write_csv(nonce_path, nonce_rows())
    config = {
        "experiment": "test-global-probes-nonce",
        "title": "Test",
        "model": "fake",
        "cat_contrast_csv": str(cat_path),
        "item_contrast_csv": str(item_path),
        "nonce_csv": str(nonce_path),
        "activation_types": ["raw", "cat_contrast", "nonce_contrast"],
        "prompt_columns": ["rosch_example", "rosch_neg"],
        "layers": [1, 2],
        "position": "period",
        "ambiguity": ["include", "exclude_core"],
        "n_groups": [3],
        "metrics": ["spearman"],
        "n_folds": 5,
        "cv_aggregate": "fold_zscored",
        "target": "rating_zscore",
        "target_sign": -1,
        "batch_size": 2,
        "device": "cpu",
        "dtype": "float32",
        "prefix_token": None,
        "seed": 0,
        "activation_cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "out"),
        "html_name": "page.html",
        "copy_html_to": None,
    }
    monkeypatch.setattr(global_probe_experiment, "LazyModel", FakeModel)
    run_dir = run(config, "config.yaml")

    with np.load(run_dir / "matrices.npz") as data:
        assert data["category_transfer"].shape == (2, 3, 2, 1, 1, 2, 2, 2)
    with open(run_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["activation_type"] for r in rows} == {
        "raw",
        "cat_contrast",
        "nonce_contrast",
    }
    page = (run_dir / "page.html").read_text()
    assert "Nonce contrast" in page
    assert "NaN" not in page


def test_nonce_contrast_needs_nonce_csv():
    from typicality_linearity.global_probe_experiment import config_activation_types

    with pytest.raises(ValueError):
        config_activation_types({"activation_types": ["raw", "nonce_contrast"]})
    with pytest.raises(ValueError):
        config_activation_types({"activation_types": ["unknown"]})
    assert len(config_activation_types({})) == 4


def test_within_method():
    from typicality_linearity.global_probe_experiment import within_method

    assert within_method({}) == "cv"
    assert within_method({"diagonal": "cv"}) == "cv"
    assert within_method({"diagonal": "full"}) == "in_sample"
