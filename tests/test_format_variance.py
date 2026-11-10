import numpy as np
import pytest

from typicality_linearity.format_erasure import (
    between_group_variance_fraction,
    center_by_group,
    fit_inlp,
    fit_leace,
    group_accuracy,
)
from typicality_linearity.format_variance_experiment import (
    POOLED,
    STATISTICS,
    VARIANTS,
    check_config,
    correlation_cells,
    evaluate_variant,
    fit_pooled_probes,
    parse_variant,
    preprocess,
)
from typicality_linearity.probes import fit_diff_of_means, make_folds, spearman

N_ITEMS = 60
D_MODEL = 12
PROMPTS = ("p0", "p1", "p2")


def format_data(seed=0, offset=5.0):
    """Items with a typicality direction, and a large offset for each prompt."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=N_ITEMS)
    categories = np.array(["a", "b", "c"] * (N_ITEMS // 3))
    typicality_direction = np.zeros(D_MODEL)
    typicality_direction[0] = 1.0
    item = rng.normal(size=(N_ITEMS, D_MODEL)) + np.outer(y, typicality_direction)
    x = {}
    for k, prompt in enumerate(PROMPTS):
        shift = np.zeros(D_MODEL)
        shift[1 + k] = offset
        x[prompt] = item + shift + 0.1 * rng.normal(size=(N_ITEMS, D_MODEL))
    return x, y, categories


def small_config(**changes):
    config = {
        "inlp_iterations": 5,
        "inlp_stop_accuracy": None,
        "classifier_c": 1.0,
        "classifier_max_iter": 500,
        "classifier_dev_fraction": 0.3,
        "format_check": True,
        "seed": 0,
        "n_groups": 3,
        "metrics": ["spearman"],
        "cv_aggregate": "pooled",
    }
    config.update(changes)
    return config


def stacked(x):
    values = np.concatenate([x[p] for p in PROMPTS])
    groups = np.repeat(np.arange(len(PROMPTS)), N_ITEMS)
    return values, groups


def group_means(values, groups):
    return np.array([values[groups == g].mean(axis=0) for g in np.unique(groups)])


def test_center_by_group_removes_means():
    values, groups = stacked(format_data()[0])
    centred = center_by_group(values, groups)
    assert np.allclose(group_means(centred, groups), 0)


def test_variance_fraction():
    values, groups = stacked(format_data()[0])
    assert between_group_variance_fraction(values, groups) > 0.5
    centred = center_by_group(values, groups)
    assert between_group_variance_fraction(centred, groups) == pytest.approx(0)


def test_leace_makes_group_means_equal():
    values, groups = stacked(format_data()[0])
    eraser = fit_leace(values, groups)
    assert eraser.rank == len(PROMPTS) - 1
    erased = eraser.apply(values)
    means = group_means(erased, groups)
    assert np.allclose(means, means[0], atol=1e-8)


def test_leace_after_centering_is_identity():
    values, groups = stacked(format_data()[0])
    centred = center_by_group(values, groups)
    assert fit_leace(centred, groups).rank == 0


def test_inlp_removes_format_information():
    values, groups = stacked(format_data()[0])
    before = group_accuracy(values, groups, 0.3, 1.0, 500, 0)
    eraser = fit_inlp(values, groups, n_iterations=5, max_iter=500)
    after = group_accuracy(eraser.apply(values), groups, 0.3, 1.0, 500, 0)
    assert before > 0.95
    assert after < 0.6
    basis = eraser.left
    assert np.allclose(basis.T @ basis, np.eye(eraser.rank), atol=1e-8)
    assert len(eraser.history) >= 1


def test_raw_direction_gives_same_scores():
    values, groups = stacked(format_data()[0])
    eraser = fit_leace(values, groups)
    direction = np.random.default_rng(1).normal(size=D_MODEL)
    erased_scores = eraser.apply(values) @ direction
    raw_scores = values @ eraser.raw_direction(direction)
    difference = erased_scores - raw_scores
    assert np.allclose(difference, difference[0])


def test_parse_variant():
    assert parse_variant("raw") == (False, None)
    assert parse_variant("center") == (True, None)
    assert parse_variant("leace") == (False, "leace")
    assert parse_variant("center_inlp") == (True, "inlp")
    for variant in VARIANTS:
        parse_variant(variant)


def test_preprocess_centers_eval_with_own_mean():
    x, _, _ = format_data()
    x_eval = {"e0": x["p0"] + 3.0}
    x_train, x_eval_out, eraser = preprocess(x, x_eval, "center", small_config())
    assert eraser.rank == 0
    assert np.allclose(x_eval_out["e0"].mean(axis=0), 0)
    assert np.allclose(x_train["p1"].mean(axis=0), 0)


def test_pooled_probe_is_mean_over_prompts():
    x, y, categories = format_data()
    folds = make_folds(categories, y, 5, 0)
    full, by_fold = fit_pooled_probes(x, y, categories, folds, 3)
    rows = categories == "a"
    x_mean = np.mean([x[p][rows] for p in PROMPTS], axis=0)
    expected = fit_diff_of_means(x_mean, y[rows], 3).weights
    assert np.allclose(full["a"].weights, expected)
    assert len(by_fold["a"]) == 5


def test_pooled_correlation_sees_format_offset():
    rng = np.random.default_rng(0)
    y = rng.normal(size=30)
    folds = np.arange(30) % 5
    scores = {"p0": y.copy(), "p1": y + 100.0}
    values = correlation_cells(scores, y, folds, "spearman", "pooled", False)
    assert values["p0"] == pytest.approx(1.0)
    assert values["p1"] == pytest.approx(1.0)
    assert values[POOLED] < 0.9
    assert values[POOLED] == pytest.approx(
        spearman(np.concatenate([y, y + 100.0]), np.tile(y, 2))
    )


def test_centering_improves_pooled_but_not_by_prompt():
    x, y, categories = format_data()
    folds = make_folds(categories, y, 5, 0)
    x_eval = {"e0": x["p0"] + 2.0, "e1": x["p1"] - 2.0}
    config = small_config()
    results = {}
    for variant in ("raw", "center"):
        x_train, x_eval_out, _ = preprocess(x, x_eval, variant, config)
        _, _, statistics = evaluate_variant(
            x_train, x_eval_out, y, categories, folds, config
        )
        assert set(statistics["spearman"]) == set(STATISTICS)
        results[variant] = statistics["spearman"]
    raw = results["raw"]
    center = results["center"]
    # A full probe scores each prompt with one direction, so an offset of the
    # prompt does not change the values by prompt. (Within CV by prompt can
    # change: the fold probes give the offset different constants.)
    assert center["between_train_by_prompt"] == pytest.approx(
        raw["between_train_by_prompt"]
    )
    assert center["loco_eval_by_prompt"] == pytest.approx(raw["loco_eval_by_prompt"])
    assert center["within_eval_pooled"] > raw["within_eval_pooled"]


def test_check_config_rejects_unknown_variant():
    config = {
        "probe_layer": 15,
        "layers": [15],
        "probe_position": "period",
        "positions": ["period"],
        "metrics": ["spearman"],
        "cv_aggregate": "pooled",
        "target_sign": -1,
        "n_groups": 3,
        "ambiguous": ["include"],
        "variants": ["raw", "whiten"],
        "eval_prompt_columns": ["e"],
        "prompt_columns": ["a", "b"],
        "inlp_iterations": 5,
        "classifier_dev_fraction": 0.3,
    }
    with pytest.raises(ValueError, match="Unknown variant"):
        check_config(config)
    config["variants"] = ["raw"]
    check_config(config)


def test_plot_data_from_run_rows():
    from typicality_linearity.format_variance_plot import build_plot_data, render_html

    config = {
        "experiment": "e",
        "model": "m",
        "probe_layer": 15,
        "probe_position": "period",
        "target": "t",
        "target_sign": -1,
        "n_groups": 3,
        "n_folds": 5,
        "cv_aggregate": "pooled",
        "inlp_iterations": 2,
        "inlp_stop_accuracy": 0.2,
        "prompt_columns": ["p0", "p1"],
        "eval_prompt_columns": ["e0"],
    }
    labels = {"variant": "raw", "baseline": "none", "ambiguous": "include"}
    statistics = {name: 0.5 for name in STATISTICS}
    summary = [{**labels, "metric": "spearman", "n_items": 6, **statistics}]
    cells = []
    for prompt_set, prompts in (
        ("train", ["p0", "p1", POOLED]),
        ("eval", ["e0", POOLED]),
    ):
        for prompt in prompts:
            for a in ("a", "b"):
                for b in ("a", "b"):
                    cells.append(
                        {
                            **labels,
                            "probe": "single",
                            "train_category": a,
                            "eval_category": b,
                            "prompt": prompt,
                            "value": 0.25,
                            "prompt_set": prompt_set,
                            "metric": "spearman",
                        }
                    )
                cells.append(
                    {
                        **labels,
                        "probe": "loco",
                        "train_category": "others",
                        "eval_category": a,
                        "prompt": prompt,
                        "value": float("nan"),
                        "prompt_set": prompt_set,
                        "metric": "spearman",
                    }
                )
    diagnostics = [
        {
            **labels,
            "eraser_rank": 0,
            "format_variance_train": 0.7,
            "format_variance_eval": 0.7,
            "format_accuracy_train": 1.0,
            "format_chance": 0.5,
        }
    ]
    weights = {
        "raw/none/include/a/transformed": np.array([1.0, 0.0]),
        "raw/none/include/b/transformed": np.array([1.0, 1.0]),
        "raw/none/include/a/raw": np.array([1.0, 0.0]),
    }
    data = build_plot_data(config, summary, cells, diagnostics, [], weights)
    matrices = data["matrices"]["raw|none|include|spearman"]
    assert data["prompt_rows"][2] == "mean, train prompts"
    assert matrices["mean, train prompts"][0] == [0.25, 0.25]
    assert matrices["pooled, eval prompts"][2] == [None, None]
    assert data["cosine"]["raw|none|include"][0][1] == pytest.approx(0.7071, abs=1e-4)
    page = render_html(data)
    for element_id in ("variant", "summary", "diagnostics", "category-panels"):
        assert f'id="{element_id}"' in page
