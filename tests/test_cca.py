import numpy as np
import pytest

from typicality_linearity.cca_experiment import (
    MEAN_ALL,
    MEAN_LOCO,
    STATISTICS,
    check_config,
    cosine_rows,
    direction_scores,
    evaluate_baseline,
    load_config,
    mean_all_fold_directions,
    out_of_fold_scores,
)
from typicality_linearity.cca_plot import build_plot_data, render_html
from typicality_linearity.format_variance_experiment import fit_pooled_probes
from typicality_linearity.mean_direction import unit
from typicality_linearity.probes import activation_direction, make_folds

N_ITEMS = 60
D_MODEL = 8
CATEGORIES = ("a", "b", "c")
CONFIG = {
    "metrics": ["spearman", "pearson"],
    "cv_aggregate": "pooled",
    "n_groups": 3,
    "n_folds": 5,
}


def shared_direction_data(seed=0):
    """Items of 3 categories with one shared typicality direction."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=N_ITEMS)
    categories = np.array(list(CATEGORIES) * (N_ITEMS // 3))
    direction = np.zeros(D_MODEL)
    direction[0] = 1.0
    item = rng.normal(scale=0.3, size=(N_ITEMS, D_MODEL)) + np.outer(y, direction)
    x_train = {p: item + rng.normal(scale=0.1, size=item.shape) for p in ("p0", "p1")}
    x_eval = {"e0": item + rng.normal(scale=0.1, size=item.shape)}
    folds = make_folds(categories, y, 5, 0)
    return x_train, x_eval, y, categories, folds


def test_mean_all_fold_direction_uses_fold_probe_for_own_category():
    x_train, _, y, categories, folds = shared_direction_data()
    full, by_fold = fit_pooled_probes(x_train, y, categories, folds, 3)
    fold_directions = mean_all_fold_directions(full, by_fold, "a")
    assert len(fold_directions) == 5
    expected = np.mean(
        [
            unit(activation_direction(full["b"])),
            unit(activation_direction(full["c"])),
            unit(activation_direction(by_fold["a"][2])),
        ],
        axis=0,
    )
    np.testing.assert_allclose(fold_directions[2], expected)


def test_out_of_fold_scores_uses_direction_of_each_fold():
    x = np.eye(3)
    folds = np.array([0, 1, 0])
    directions = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])]
    np.testing.assert_allclose(out_of_fold_scores(directions, x, folds), [1, 20, 3])


def test_direction_scores_marks_cross_validated_directions():
    x_train, _, y, categories, folds = shared_direction_data()
    full, by_fold = fit_pooled_probes(x_train, y, categories, folds, 3)
    indices = np.flatnonzero(categories == "b")
    scores = direction_scores(full, by_fold, x_train["p0"], "b", indices, folds)
    assert set(scores) == {"a", "b", "c", MEAN_ALL, MEAN_LOCO}
    assert scores["b"][1] and scores[MEAN_ALL][1]
    assert not scores["a"][1] and not scores[MEAN_LOCO][1]
    assert all(len(values) == len(indices) for values, _ in scores.values())


def test_evaluate_baseline_finds_shared_direction():
    x_train, x_eval, y, categories, folds = shared_direction_data()
    _, _, directions, cells, summary = evaluate_baseline(
        x_train, x_eval, y, categories, folds, CONFIG
    )
    assert set(directions) == {"a", "b", "c", MEAN_ALL}
    assert len(summary) == 4
    eval_spearman = next(
        row
        for row in summary
        if row["prompt_set"] == "eval" and row["metric"] == "spearman"
    )
    for statistic in STATISTICS:
        assert eval_spearman[statistic] > 0.8
    # 2 metrics x (3 + 2 directions) x 3 categories x (2 train + 1 eval prompts)
    assert len(cells) == 2 * 5 * 3 * 3


def test_cosine_rows_compare_baselines():
    directions = {
        "none": {"a": np.array([1.0, 0.0]), MEAN_ALL: np.array([1.0, 1.0])},
        "mean_contrast": {"a": np.array([0.0, 1.0]), MEAN_ALL: np.array([1.0, 1.0])},
    }
    rows = cosine_rows(directions, {"ambiguous": "include"})
    between = {
        r["direction_a"]: r["cosine"] for r in rows if r["kind"] == "between_baselines"
    }
    assert between["a"] == pytest.approx(0.0)
    assert between[MEAN_ALL] == pytest.approx(1.0)
    assert sum(r["kind"] == "between_categories" for r in rows) == 8


def test_check_config_rejects_eval_prompt_in_training():
    config = load_config("experiments/probes/rosch/qwen2-5-1-5B-cca/config.yaml")
    config["eval_prompt_columns"] = [config["prompt_columns"][0]]
    with pytest.raises(ValueError):
        check_config(config)


def test_page_data_has_both_baselines():
    x_train, x_eval, y, categories, folds = shared_direction_data()
    summary, cells, by_baseline = [], [], {}
    for baseline in ("none", "mean_contrast"):
        labels = {"baseline": baseline, "ambiguous": "include"}
        _, _, directions, base_cells, base_summary = evaluate_baseline(
            x_train, x_eval, y, categories, folds, CONFIG
        )
        cells += [{**labels, **c} for c in base_cells]
        summary += [{**labels, "n_items": N_ITEMS, **r} for r in base_summary]
        by_baseline[baseline] = directions
    cosines = cosine_rows(by_baseline, {"ambiguous": "include"})
    config = {
        "experiment": "test",
        "model": "m",
        "probe_layer": 1,
        "probe_position": "period",
        "n_groups": 3,
        "prompt_columns": ["p0", "p1"],
        "eval_prompt_columns": ["e0"],
        "target_sign": -1,
        "target": "rating_zscore",
        "n_folds": 5,
        "cv_aggregate": "pooled",
        "metrics": ["spearman", "pearson"],
    }
    as_text = [{k: str(v) for k, v in row.items()} for row in summary]
    cells_text = [{k: str(v) for k, v in row.items()} for row in cells]
    cosines_text = [{k: str(v) for k, v in row.items()} for row in cosines]
    data = build_plot_data(config, as_text, cells_text, cosines_text, {}, {})
    assert data["baselines"] == ["none", "mean_contrast"]
    assert "include|spearman|none|mean_eval" in data["matrices"]
    assert len(data["matrices"]["include|spearman|none|e0"]) == 5
    assert data["cosine_between_baselines"]["include"][MEAN_ALL] == pytest.approx(1.0)
    assert "<html" in render_html(data)
