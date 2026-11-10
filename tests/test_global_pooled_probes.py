import csv

import numpy as np
from test_global_probes import FakeModel, cat_rows, item_rows, synthetic, write_csv

from typicality_linearity import global_pooled_probe_experiment
from typicality_linearity.global_pooled_probe_experiment import (
    global_direction,
    global_eval_scores,
    over_categories_matrices,
    over_prompts_matrices,
    pooled_category_direction,
    pooled_prompt_direction,
    run,
)
from typicality_linearity.probes import fit_diff_of_means

METRICS = ["spearman", "pearson", "wilcoxon_auc"]


def test_pooled_prompt_direction_is_mean_of_prompt_directions():
    x_by_prompt, y, _categories, _ = synthetic()
    rows = np.arange(30)
    direction = pooled_prompt_direction(x_by_prompt, rows, y, 3)
    per_prompt = [fit_diff_of_means(x[rows], y[rows], 3).weights for x in x_by_prompt]
    np.testing.assert_allclose(direction, np.mean(per_prompt, axis=0))


def test_over_categories_finds_a_shared_signal():
    x_by_prompt, y, categories, folds = synthetic()
    transfer, _eval, cosine = over_categories_matrices(
        x_by_prompt, y, categories, folds, 3, METRICS, "fold_zscored", 5, "per_category"
    )
    assert transfer.shape == (3, 2, 2, 2)
    assert cosine.shape == (2, 2)
    assert transfer[0].mean() > 0.8
    assert transfer[2].mean() > 0.9
    np.testing.assert_allclose(np.diag(cosine), 1)


def shifted_categories():
    """Toy items are moved by +50 along the typicality axis, and there are
    fewer toy items in the top third of all items than in the bottom third."""
    x_by_prompt, y, categories, folds = synthetic()
    y = y.copy()
    y[:30] = np.sort(y[:30]) - 0.5
    shifted = []
    for x in x_by_prompt:
        x = x.copy()
        x[:, 0] = y + np.random.default_rng(1).normal(scale=0.3, size=len(y))
        x[:30, 0] += 50.0
        shifted.append(x)
    return x_by_prompt, shifted, y, categories, folds


def test_per_category_groups_cancel_category_offsets():
    _, shifted, y, categories, folds = shifted_categories()
    per_category, _, _ = over_categories_matrices(
        shifted,
        y,
        categories,
        folds,
        3,
        ["spearman"],
        "fold_zscored",
        5,
        "per_category",
    )
    assert per_category.mean() > 0.8


def test_all_groups_take_the_category_offset():
    _, shifted, y, categories, folds = shifted_categories()
    pooled, _, _ = over_categories_matrices(
        shifted, y, categories, folds, 3, ["spearman"], "fold_zscored", 5, "all"
    )
    # The offset is in the direction and is larger than the signal.
    assert pooled.mean() < 0.5


def test_pooled_category_direction_modes_agree_without_offsets():
    x_by_prompt, y, categories, _ = synthetic()
    rows = np.arange(len(y))
    categories = np.asarray(categories)
    a = pooled_category_direction(
        x_by_prompt[0], rows, y, categories, 3, "per_category"
    )
    b = pooled_category_direction(x_by_prompt[0], rows, y, categories, 3, "all")
    cosine = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    assert cosine > 0.9


def test_over_prompts_finds_a_shared_signal_and_no_signal_is_chance():
    x_by_prompt, y, categories, folds = synthetic()
    transfer, _cosine = over_prompts_matrices(
        x_by_prompt, y, categories, folds, 3, METRICS, "fold_zscored", 5
    )
    assert transfer.shape == (3, 2, 2, 2)
    assert transfer[0].mean() > 0.8
    rng = np.random.default_rng(7)
    noise = [rng.normal(size=x.shape) for x in x_by_prompt]
    transfer, _ = over_prompts_matrices(
        noise, y, categories, folds, 3, ["spearman"], "fold_zscored", 5
    )
    assert abs(transfer.mean()) < 0.25


def test_run_end_to_end(tmp_path, monkeypatch):
    cat_path = tmp_path / "cat.csv"
    item_path = tmp_path / "item.csv"
    write_csv(cat_path, cat_rows())
    write_csv(item_path, item_rows())
    config = {
        "experiment": "test-pooled",
        "title": "Test",
        "model": "fake",
        "cat_contrast_csv": str(cat_path),
        "item_contrast_csv": str(item_path),
        "prompt_columns": ["rosch_example", "rosch_neg"],
        "layers": [1, 2],
        "position": "period",
        "ambiguity": ["include", "exclude_lenient", "exclude_strict"],
        "n_groups": [2, 3],
        "metrics": METRICS,
        "n_folds": 5,
        "over_categories_groups": "per_category",
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
    monkeypatch.setattr(global_pooled_probe_experiment, "LazyModel", FakeModel)
    run_dir = run(config, "config.yaml")
    with np.load(run_dir / "matrices.npz") as data:
        # (ambiguity, type, layer, groups, metric, ...)
        assert data["over_categories"].shape == (3, 4, 2, 2, 3, 2, 2)
        assert data["over_categories_by_category"].shape == (3, 4, 2, 2, 3, 2, 2, 2)
        assert data["over_prompts"].shape == (3, 4, 2, 2, 3, 2, 2)
        assert data["over_prompts_by_prompt"].shape == (3, 4, 2, 2, 3, 2, 2, 2)
        np.testing.assert_allclose(
            data["over_categories"], data["over_categories_by_category"].mean(axis=-1)
        )
    with open(run_dir / "summary.csv") as f:
        assert len(list(csv.DictReader(f))) == 3 * 4 * 2 * 2 * 3
    page = (run_dir / "page.html").read_text()
    assert "Pooled over prompts" in page
    assert "NaN" not in page
    assert (tmp_path / "copy" / "page.html").exists()


def test_eval_columns_equal_the_same_prompt_as_a_training_column():
    x_by_prompt, y, categories, folds = synthetic()
    transfer, eval_transfer, _ = over_categories_matrices(
        x_by_prompt,
        y,
        categories,
        folds,
        3,
        METRICS,
        "fold_zscored",
        5,
        "per_category",
        [x_by_prompt[1]],
    )
    assert eval_transfer.shape == (3, 2, 1, 2)
    np.testing.assert_allclose(eval_transfer[:, :, 0], transfer[:, :, 1])


def test_run_with_eval_prompts(tmp_path, monkeypatch):
    cat_path = tmp_path / "cat.csv"
    item_path = tmp_path / "item.csv"
    write_csv(cat_path, cat_rows())
    write_csv(item_path, item_rows())
    config = {
        "experiment": "test-pooled-eval",
        "title": "Test",
        "model": "fake",
        "cat_contrast_csv": str(cat_path),
        "item_contrast_csv": str(item_path),
        "prompt_columns": ["rosch_example"],
        "layers": [1],
        "eval_cat_contrast_csv": str(cat_path),
        "eval_item_contrast_csv": str(item_path),
        "eval_prompt_columns": ["rosch_neg"],
        "position": "period",
        "ambiguity": ["include", "exclude_strict"],
        "n_groups": [3],
        "metrics": METRICS,
        "n_folds": 5,
        "over_categories_groups": "per_category",
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
    monkeypatch.setattr(global_pooled_probe_experiment, "LazyModel", FakeModel)
    run_dir = run(config, "config.yaml")
    with np.load(run_dir / "matrices.npz") as data:
        assert data["over_categories_eval"].shape == (2, 4, 1, 1, 3, 1, 1)
        assert np.all(np.isfinite(data["over_categories_eval"]))
        assert data["global_scores_raw"].shape == (2, 4, 1, 1, 1, 40)
        assert np.all(np.isfinite(data["global_scores_z"][0]))
        # exclude_strict removes 2 items per category: their scores are NaN.
        assert np.isnan(data["global_scores_z"][1]).sum() == 4 * 4
        assert data["global_eval_metrics"].shape == (2, 4, 1, 1, 3, 1)
    with open(run_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert all(r["over_categories_eval_prompts"] not in ("", "nan") for r in rows)
    page = (run_dir / "page.html").read_text()
    assert '"eval_prompt":["rosch_neg"]' in page
    assert "Pooled over categories and prompts" in page
    assert all(r["global_eval_prompts"] not in ("", "nan") for r in rows)


def test_global_direction_is_mean_of_prompt_directions():
    x_by_prompt, y, categories, _ = synthetic()
    categories = np.asarray(categories)
    rows = np.arange(len(y))
    expected = np.mean(
        [
            pooled_category_direction(x, rows, y, categories, 3, "per_category")
            for x in x_by_prompt
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        global_direction(x_by_prompt, y, categories, 3, "per_category"), expected
    )


def test_global_eval_scores():
    x_by_prompt, y, categories, _ = synthetic()
    raw, z, values = global_eval_scores(
        x_by_prompt, [x_by_prompt[0]], y, categories, 3, METRICS, "per_category"
    )
    assert raw.shape == z.shape == (1, len(y))
    assert values.shape == (3, 1)
    assert values[0, 0] > 0.8
    assert values[2, 0] > 0.9
    categories = np.asarray(categories)
    for c in ("toy", "bird"):
        np.testing.assert_allclose(z[0][categories == c].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(z[0][categories == c].std(), 1)
