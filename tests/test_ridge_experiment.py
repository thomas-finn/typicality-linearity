import numpy as np
import pytest

from typicality_linearity.diff_means_experiment import evaluate, summary_values
from typicality_linearity.diff_means_plot import build_plot_data, render_html
from typicality_linearity.probe_transfer import transfer_matrices
from typicality_linearity.probes import fit_probe, fit_ridge, make_folds
from typicality_linearity.ridge_experiment import (
    CV_ALPHA,
    alpha_text,
    check_config,
    fit_settings,
    ridge_page_parameters,
    settings_grid,
)


def ridge_config(**changes):
    config = {
        "probe_layer": 15,
        "layers": [15],
        "probe_position": "period",
        "positions": ["period"],
        "metrics": ["spearman"],
        "cv_aggregate": "pooled",
        "target_sign": -1,
        "ambiguous": ["include", "exclude"],
        "eval_prompt_columns": ["e1"],
        "prompt_columns": ["p1"],
        "ridge_alphas": ["cv", 1000.0, 0.01, 1.0e6],
        "ridge_cv_alphas": [1.0, 10.0],
        "ridge_inner_folds": 3,
        "ridge_standardise": True,
        "null_settings": ["a1000_include"],
        "n_permutations": 2,
    }
    config.update(changes)
    return config


def test_alpha_text():
    assert alpha_text(CV_ALPHA) == "cv"
    assert alpha_text(0.01) == "0.01"
    assert alpha_text(1000.0) == "1000"
    assert alpha_text(3.16e6) == "3.16e6"
    assert alpha_text(1e8) == "1e8"


def test_settings_grid_order_and_keys():
    grid = settings_grid(ridge_config())
    keys = [s["key"] for s in grid]
    assert keys[:4] == ["acv_include", "a0.01_include", "a1000_include", "a1e6_include"]
    assert len(set(keys)) == 8
    check_config(ridge_config())


def test_check_config_errors():
    with pytest.raises(ValueError):
        check_config(ridge_config(ridge_alphas=[0.0]))
    with pytest.raises(ValueError):
        check_config(ridge_config(ridge_alphas=[1.0, 1.0]))
    with pytest.raises(ValueError):
        check_config(ridge_config(null_settings=["a5_include"]))
    config = ridge_config()
    del config["ridge_cv_alphas"]
    with pytest.raises(ValueError):
        check_config(config)


def test_fixed_alpha_is_used():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 8))
    y = x[:, 0] + 0.1 * rng.normal(size=30)
    setting = {"key": "a10_include", "alpha": 10.0, "ambiguous": "include"}
    settings = fit_settings(setting, ridge_config())
    probe, alpha = fit_probe("ridge", x, y, settings, rng)
    assert alpha == 10.0
    expected = fit_ridge(x, y, 10.0, True)
    assert np.allclose(probe.weights, expected.weights)

    cv_setting = {"key": "acv_include", "alpha": CV_ALPHA, "ambiguous": "include"}
    settings = fit_settings(cv_setting, ridge_config())
    _, alpha = fit_probe("ridge", x, y, settings, rng)
    assert alpha in (1.0, 10.0)


def test_large_alpha_direction_is_covariance():
    # For a large alpha the ridge weights are proportional to Z^T (y - mean y).
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    probe = fit_ridge(x, y, 1e10, True)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    expected = z.T @ (y - y.mean())
    cosine = probe.weights @ expected
    cosine /= np.linalg.norm(probe.weights) * np.linalg.norm(expected)
    assert cosine > 0.999999


def test_ridge_family_evaluate_and_page():
    rng = np.random.default_rng(0)
    categories = np.array(["a"] * 20 + ["b"] * 20)
    y = rng.normal(size=40)
    x_by_prompt = {}
    for prompt in ("p1", "p2", "e1"):
        x = rng.normal(size=(40, 6)) * 0.3
        x[:, 0] += 2 * y
        x_by_prompt[prompt] = x
    folds = make_folds(categories, y, 5, seed=0)
    x_train = {p: x_by_prompt[p] for p in ("p1", "p2")}
    x_eval = {"e1": x_by_prompt["e1"]}
    setting = {"key": "a1_include", "alpha": 1.0, "ambiguous": "include"}
    result = transfer_matrices(
        x_train,
        y,
        categories,
        folds,
        ["ridge"],
        ["spearman"],
        fit_settings(setting, ridge_config()),
        "pooled",
        seed=0,
    )
    assert summary_values(result, "spearman", "ridge")["within_cv"] > 0.8
    evaluated = evaluate(
        result, x_train, x_eval, y, categories, folds, ["spearman"], "pooled", "ridge"
    )
    statistics = evaluated["statistics"]["spearman"]
    assert statistics["mean_loco_eval"] > 0.8

    data = build_plot_data(
        "t",
        ["n"],
        [setting],
        {"none": "No baseline"},
        {"a1_include": {"none": result}},
        [dict(setting="a1_include", baseline="none", metric="spearman", **statistics)],
        {"include": {"a": 20, "b": 20}},
        ridge_page_parameters(ridge_config()),
        probe_type="ridge",
    )
    assert data["settings"][0]["selector"] == {"alpha": "1", "ambiguous": "include"}
    assert data["parameters"][0]["labels"] == {"1": "alpha = 1"}
    page = render_html(data)
    assert 'id="alpha"' in page
    assert 'id="ambiguous"' in page
    assert 'id="split"' not in page
