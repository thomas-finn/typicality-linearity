import numpy as np
import pytest

from typicality_linearity.probe_experiment import (
    cosine_rows_from_directions,
    load_probe_directions,
    probe_weight_arrays,
)
from typicality_linearity.probe_plot import build_plot_data, render_html
from typicality_linearity.probe_transfer import cv_correlation, transfer_matrices
from typicality_linearity.probes import (
    activation_direction,
    average_ranks,
    choose_ridge_alpha,
    cosine_similarity,
    fit_diff_of_means,
    fit_ridge,
    make_folds,
    pearson,
    spearman,
)

SETTINGS = {
    "ridge_alphas": [1.0, 10.0, 100.0, 1000.0],
    "ridge_inner_folds": 3,
    "ridge_standardise": True,
}


def test_average_ranks_with_ties():
    assert average_ranks([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]


def test_spearman_and_pearson():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert pearson(a, a**3) < 1.0
    assert pearson(a, -a) == pytest.approx(-1.0)
    assert np.isnan(pearson(a, np.ones(4)))


def test_diff_of_means_finds_typicality_direction():
    rng = np.random.default_rng(0)
    direction = np.zeros(20)
    direction[3] = 1.0
    y = rng.normal(size=60)
    x = y[:, None] * direction + rng.normal(size=(60, 20)) * 0.1
    probe = fit_diff_of_means(x, y)
    assert np.argmax(np.abs(probe.weights)) == 3
    assert probe.weights[3] > 0
    assert spearman(probe.predict(x), y) > 0.9


def test_ridge_matches_direct_solution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 5))
    y = rng.normal(size=30)
    alpha = 3.0
    probe = fit_ridge(x, y, alpha, standardise=True)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    expected = np.linalg.solve(z.T @ z + alpha * np.eye(5), z.T @ (y - y.mean()))
    assert np.allclose(probe.weights, expected)
    assert np.allclose(probe.predict(x), z @ expected + y.mean())


def test_choose_ridge_alpha_is_from_grid():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 100))
    y = rng.normal(size=40)
    alpha = choose_ridge_alpha(x, y, np.array(SETTINGS["ridge_alphas"]), 3, True, rng)
    assert alpha in SETTINGS["ridge_alphas"]


def test_make_folds_are_per_category_balanced_and_repeatable():
    categories = ["a"] * 23 + ["b"] * 17
    values = np.arange(40.0)
    folds = make_folds(categories, values, 5, seed=0)
    assert np.array_equal(folds, make_folds(categories, values, 5, seed=0))
    for category, n in (("a", 23), ("b", 17)):
        counts = np.bincount(folds[np.array(categories) == category], minlength=5)
        assert counts.sum() == n
        assert counts.max() - counts.min() <= 1


def test_cv_correlation_aggregates():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    folds = np.array([0, 0, 0, 1, 1, 1])
    predictions = y.copy()
    assert cv_correlation(predictions, y, folds, "pearson", "pooled") == 1.0
    assert cv_correlation(predictions, y, folds, "pearson", "mean_over_folds") == 1.0


def synthetic_data(shared_direction, n_per_category=30, d_model=10, seed=0):
    """Two categories, two prompts. Typicality is along one feature.

    If shared_direction is false, each category uses a different feature.
    """
    rng = np.random.default_rng(seed)
    categories = ["a"] * n_per_category + ["b"] * n_per_category
    y = rng.normal(size=2 * n_per_category)
    feature = np.where(np.array(categories) == "a", 0, 0 if shared_direction else 1)
    x_by_prompt = {}
    for prompt in ("p1", "p2"):
        x = rng.normal(size=(2 * n_per_category, d_model)) * 0.3
        x[np.arange(len(y)), feature] += 2 * y
        x_by_prompt[prompt] = x
    return x_by_prompt, y, categories


def run_transfer(x_by_prompt, y, categories, aggregate="pooled"):
    folds = make_folds(categories, y, 5, seed=0)
    return transfer_matrices(
        x_by_prompt,
        y,
        categories,
        folds,
        ["diff_of_means", "ridge"],
        ["spearman", "pearson"],
        SETTINGS,
        aggregate,
        seed=0,
    )


def test_transfer_shapes_and_shared_direction_transfers():
    result = run_transfer(*synthetic_data(shared_direction=True))
    assert result["categories"] == ["a", "b"]
    assert result["prompts"] == ["p1", "p2"]
    for probe_type in ("diff_of_means", "ridge"):
        category_matrix = result["category_transfer"][probe_type]["spearman"]["p1"]
        prompt_matrix = result["prompt_transfer"][probe_type]["spearman"]["a"]
        assert category_matrix.shape == (2, 2)
        assert prompt_matrix.shape == (2, 2)
        assert category_matrix.min() > 0.8
        assert prompt_matrix.min() > 0.8
    assert len(result["alphas"]) == 2 * 2 * 6  # categories x prompts x (all + 5 folds)


def test_different_directions_do_not_transfer_between_categories():
    result = run_transfer(*synthetic_data(shared_direction=False))
    matrix = result["category_transfer"]["diff_of_means"]["pearson"]["p1"]
    assert matrix[0, 0] > 0.8 and matrix[1, 1] > 0.8
    assert abs(matrix[0, 1]) < 0.5 and abs(matrix[1, 0]) < 0.5


def test_diagonals_of_both_matrices_agree():
    result = run_transfer(*synthetic_data(shared_direction=True))
    category_matrix = result["category_transfer"]["ridge"]["pearson"]["p2"]
    prompt_matrix = result["prompt_transfer"]["ridge"]["pearson"]["b"]
    assert category_matrix[1, 1] == pytest.approx(prompt_matrix[1, 1])


def test_cross_validation_does_not_fit_noise():
    # Pure noise with more features than items. An in-sample fit would give a
    # correlation near 1. Out-of-fold predictions must not.
    rng = np.random.default_rng(1)
    categories = ["a"] * 40
    y = rng.normal(size=40)
    x = rng.normal(size=(40, 300))
    result = run_transfer({"p1": x, "p2": x}, y, categories)
    for probe_type in ("diff_of_means", "ridge"):
        value = result["prompt_transfer"][probe_type]["pearson"]["a"][0, 1]
        assert value < 0.6


def test_render_html_contains_selectors_and_data():
    result = run_transfer(*synthetic_data(shared_direction=True))
    plot_data = build_plot_data(
        title="Test <probes>",
        notes=["note"],
        variants={"none": "No baseline", "mean_contrast": "Mean contrast"},
        results={"none": result, "mean_contrast": result},
        n_items={"a": 30, "b": 30},
    )
    matrix = plot_data["category_transfer"]["none"]["ridge"]["spearman"]["p1"]
    assert len(matrix) == 2 and len(matrix[0]) == 2
    page = render_html(plot_data)
    assert "Test &lt;probes&gt;" in page
    for element_id in ('id="probe"', 'id="metric"', 'id="prompt"', 'id="category"'):
        assert element_id in page
    assert 'id="category-cosine-panels"' in page
    assert 'id="prompt-cosine-panels"' in page
    assert len(plot_data["prompt_cosine"]["none"]["ridge"]["a"]) == 2
    assert "</script>" not in page.split('id="data">')[1].split("</script>")[0]


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)
    assert np.isnan(cosine_similarity([0.0, 0.0], [1.0, 0.0]))


def test_ridge_direction_is_in_activation_space():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4)) * np.array([1.0, 10.0, 0.1, 5.0])
    y = rng.normal(size=30)
    probe = fit_ridge(x, y, 2.0, standardise=True)
    direction = activation_direction(probe)
    # Moving x along any vector v changes the score by v . direction.
    v = rng.normal(size=4)
    change = probe.predict(x[:1] + v) - probe.predict(x[:1])
    assert change[0] == pytest.approx(v @ direction)


def test_cosine_matrices_shared_and_different_directions():
    shared = run_transfer(*synthetic_data(shared_direction=True))
    different = run_transfer(*synthetic_data(shared_direction=False))
    for probe_type in ("diff_of_means", "ridge"):
        matrix = shared["category_cosine"][probe_type]["p1"]
        assert matrix.shape == (2, 2)
        assert np.allclose(np.diag(matrix), 1.0)
        assert np.allclose(matrix, matrix.T)
        assert matrix[0, 1] > 0.5
        assert shared["prompt_cosine"][probe_type]["a"][0, 1] > 0.5
    assert abs(different["category_cosine"]["diff_of_means"]["p1"][0, 1]) < 0.3


def test_cosine_from_saved_weights_matches_run(tmp_path):
    result = run_transfer(*synthetic_data(shared_direction=False))
    path = tmp_path / "probe_weights.npz"
    np.savez_compressed(path, **probe_weight_arrays("none", result))
    rows = cosine_rows_from_directions(load_probe_directions(path), ["p1", "p2"])
    assert len(rows) == 2 * (2 * 2 * 2 + 2 * 2 * 2)  # probes x (categories + prompts)
    for row in rows:
        if row["matrix"] == "between_categories":
            matrix = result["category_cosine"][row["probe_type"]][row["prompt_a"]]
            i = result["categories"].index(row["category_a"])
            j = result["categories"].index(row["category_b"])
        else:
            matrix = result["prompt_cosine"][row["probe_type"]][row["category_a"]]
            i = result["prompts"].index(row["prompt_a"])
            j = result["prompts"].index(row["prompt_b"])
        assert row["cosine"] == pytest.approx(matrix[i, j], abs=1e-5)
