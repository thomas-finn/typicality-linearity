import numpy as np
import pytest

from typicality_linearity.diff_means_experiment import (
    STATISTICS,
    check_config,
    eval_page,
    eval_point_rows,
    evaluate,
    keep_mask,
    load_config,
    settings_grid,
    summary_values,
)
from typicality_linearity.diff_means_plot import (
    build_plot_data,
    dom_parameters,
    render_html,
)
from typicality_linearity.eval_transfer import (
    eval_transfer,
    null_summary,
    permute_within,
)
from typicality_linearity.probe_transfer import transfer_matrices
from typicality_linearity.probes import (
    fit_diff_of_means,
    fit_probe,
    group_weights,
    make_folds,
    spearman,
)


def test_default_is_unweighted_thirds():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 5))
    y = rng.normal(size=30)
    order = np.argsort(y, kind="stable")
    expected = x[order[-10:]].mean(axis=0) - x[order[:10]].mean(axis=0)
    assert np.allclose(fit_diff_of_means(x, y).weights, expected)
    probe, alpha = fit_probe("diff_of_means", x, y, {}, rng)
    assert alpha is None
    assert np.allclose(probe.weights, expected)


@pytest.mark.parametrize("n_groups", [2, 3, 4])
def test_split_uses_top_and_bottom_groups(n_groups):
    y = np.arange(12, dtype=float)
    x = np.zeros((12, 1))
    x[:, 0] = y
    n = 12 // n_groups
    expected = y[-n:].mean() - y[:n].mean()
    probe = fit_diff_of_means(x, y, n_groups=n_groups)
    assert probe.weights[0] == pytest.approx(expected)


def test_group_weights_are_stable_and_normalised():
    weights = group_weights([0.0, 1.0, 1000.0], 1.0)
    assert np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(group_weights([1.0, 2.0, 3.0], 0.0), 1 / 3)


def test_weighting_moves_means_to_the_ends():
    y = np.arange(12, dtype=float)
    x = y[:, None].copy()
    plain = fit_diff_of_means(x, y, n_groups=2).weights[0]
    both = fit_diff_of_means(x, y, n_groups=2, weight_constant=1.0).weights[0]
    top = fit_diff_of_means(
        x, y, n_groups=2, weight_constant=1.0, weight_groups="top"
    ).weights[0]
    # With weighting, the top mean moves up and the bottom mean moves down.
    assert plain < top < both
    top_mean = group_weights(y[6:], 1.0) @ y[6:]
    assert top == pytest.approx(top_mean - y[:6].mean())


def test_weight_groups_must_be_known():
    with pytest.raises(ValueError):
        fit_diff_of_means(np.zeros((6, 2)), np.arange(6.0), weight_groups="bottom")


def test_keep_mask_and_settings_grid():
    rows = [{"is_ambiguous_strict": "True"}, {"is_ambiguous_strict": "False"}]
    assert keep_mask(rows, "include").tolist() == [True, True]
    assert keep_mask(rows, "exclude").tolist() == [False, True]
    config = {
        "ambiguous": ["include", "exclude"],
        "n_groups": [2, 3],
        "weight_constants": [0.0, 0.5],
    }
    grid = settings_grid(config)
    assert len(grid) == 8
    assert len({s["key"] for s in grid}) == 8
    assert grid[0]["key"] == "g2_c0_include"


def toy_data(seed=0, prompts=("p1", "p2", "e1")):
    rng = np.random.default_rng(seed)
    categories = np.array(["a"] * 20 + ["b"] * 20)
    y = rng.normal(size=40)
    x_by_prompt = {}
    for prompt in prompts:
        x = rng.normal(size=(40, 6)) * 0.3
        x[:, 0] += 2 * y
        x_by_prompt[prompt] = x
    folds = make_folds(categories, y, 5, seed=0)
    return x_by_prompt, y, categories, folds


def fit_toy(x_train, y, categories, folds):
    return transfer_matrices(
        x_train,
        y,
        categories,
        folds,
        ["diff_of_means"],
        ["spearman"],
        {"dom_n_groups": 4, "dom_weight_constant": 0.5},
        "pooled",
        seed=0,
    )


def test_summary_and_page():
    x_by_prompt, y, categories, folds = toy_data()
    x_train = {p: x_by_prompt[p] for p in ("p1", "p2")}
    x_eval = {"e1": x_by_prompt["e1"]}
    result = fit_toy(x_train, y, categories, folds)
    values = summary_values(result, "spearman")
    assert values["within_cv"] > 0.8
    assert values["between_categories"] > 0.8

    evaluated = evaluate(
        result, x_train, x_eval, y, categories, folds, ["spearman"], "pooled"
    )
    statistics = evaluated["statistics"]["spearman"]
    assert list(statistics) == list(STATISTICS)
    for name in ("eval_within", "eval_between_categories", "mean_loco_eval"):
        assert statistics[name] > 0.8
    assert set(evaluated["projections"]) == {"p1", "p2", "e1"}
    matrices = eval_page(evaluated["eval_transfer"])["spearman"]
    assert matrices["prompts"].shape == (2, 1)
    assert matrices["categories"].shape == (2, 2)

    settings = [
        {
            "key": "g4_c0.5_include",
            "n_groups": 4,
            "weight_constant": 0.5,
            "ambiguous": "include",
        }
    ]
    summary = [
        dict(
            setting="g4_c0.5_include", baseline="none", metric="spearman", **statistics
        )
    ]
    null = [
        dict(
            setting="g4_c0.5_include",
            baseline="none",
            metric="spearman",
            statistic="within_cv",
            **null_summary(0.9, [0.0, 0.1, float("nan")]),
        )
    ]
    data = build_plot_data(
        "t",
        ["n"],
        settings,
        {"none": "No baseline"},
        {"g4_c0.5_include": {"none": result}},
        summary,
        {"include": {"a": 20, "b": 20}},
        dom_parameters({4: "Quarters"}),
        {"p1": "A ITEM is an example of a CATEGORY."},
        eval_prompts=["e1"],
        eval_prompt_formats={"e1": "A ITEM is a kind of CATEGORY."},
        eval_transfer={
            "g4_c0.5_include": {"none": eval_page(evaluated["eval_transfer"])}
        },
        null=null,
    )
    assert data["prompt_formats"] == {"p1": "A ITEM is an example of a CATEGORY."}
    assert data["mean_prompts"] == ["p1", "p2", "e1"]
    assert data["null"][0]["n_permutations"] == 2
    page = render_html(data)
    for element_id in (
        "split",
        "constant",
        "ambiguous",
        "metric",
        "summary",
        "prompt-format",
        "prompt-format-list",
        "eval-format-list",
        "eval-prompt-panels",
        "eval-category-panels",
        "null-panels",
    ):
        assert f'id="{element_id}"' in page


def test_eval_transfer_uses_fold_probes_for_own_category():
    # Pure noise: if the own category used the full probe, the score on the
    # training prompt itself would be high. With fold probes it is near 0.
    rng = np.random.default_rng(1)
    categories = np.array(["a"] * 30 + ["b"] * 30)
    y = rng.normal(size=60)
    x = rng.normal(size=(60, 200))
    folds = make_folds(categories, y, 5, seed=0)
    result = fit_toy({"p": x}, y, categories, folds)
    matrix = eval_transfer(
        result["full_probes"]["diff_of_means"],
        result["fold_probes"]["diff_of_means"],
        {"p": x},
        y,
        categories,
        folds,
        result["categories"],
        result["prompts"],
        "spearman",
        "pooled",
    )
    assert matrix.shape == (1, 1, 2, 2)
    diagonal = np.diag(matrix[0, 0])
    assert np.allclose(
        diagonal, np.diag(result["category_transfer"]["diff_of_means"]["spearman"]["p"])
    )
    assert np.all(np.abs(diagonal) < 0.5)


def test_permute_within_keeps_values_per_category():
    y = np.arange(10.0)
    categories = np.array(["a"] * 5 + ["b"] * 5)
    shuffled = permute_within(y, categories, np.random.default_rng(0))
    assert sorted(shuffled[:5]) == list(y[:5])
    assert sorted(shuffled[5:]) == list(y[5:])
    assert not np.array_equal(shuffled, y)


def test_null_summary():
    summary = null_summary(0.5, [0.0, 0.6, 0.1, 0.2])
    assert summary["p_value"] == pytest.approx(2 / 5)
    assert summary["null_max"] == pytest.approx(0.6)
    assert summary["n_permutations"] == 4


def test_eval_point_rows_must_match():
    point_rows = [
        {"item": "doll", "item_category": "toy"},
        {"item": "bat", "item_category": "bird"},
    ]
    eval_rows = [
        {"item": "doll", "item_category": "toy", "is_item_category": "True"},
        {"item": "doll", "item_category": "toy", "is_item_category": "False"},
        {"item": "bat", "item_category": "bird", "is_item_category": "True"},
    ]
    assert eval_point_rows(eval_rows, point_rows) == [eval_rows[0], eval_rows[2]]
    with pytest.raises(ValueError):
        eval_point_rows(eval_rows[:1], point_rows)


def test_config_has_eval_and_null_keys():
    config = load_config("experiments/probes/rosch/qwen2-5-1-5B-diff-means/config.yaml")
    bad = dict(config, null_settings=["g9_c0_include"])
    with pytest.raises(ValueError):
        check_config(bad)
    bad = dict(config, eval_prompt_columns=["rosch_example"])
    with pytest.raises(ValueError):
        check_config(bad)


def test_leave_one_out_scores():
    from typicality_linearity.mean_direction import (
        item_projections,
        leave_one_out_scores,
    )

    x_by_prompt, y, categories, folds = toy_data(prompts=("p1",))
    result = fit_toy(x_by_prompt, y, categories, folds)
    projections = item_projections(
        x_by_prompt,
        categories,
        folds,
        result["full_probes"]["diff_of_means"],
        result["fold_probes"]["diff_of_means"],
    )
    loco, all_categories = leave_one_out_scores(
        projections, y, categories, ["a", "b"], "spearman"
    )
    assert loco.shape == (1, 2)
    # With two categories, the held-out direction of "a" is the direction of "b".
    rows = categories == "a"
    expected = spearman(projections["p1"][rows, 1], y[rows])
    assert loco[0, 0] == pytest.approx(expected)
    assert np.all(all_categories > 0.8)


def test_mean_direction_projections_and_ranking():
    from typicality_linearity.mean_direction import (
        all_subsets,
        column_correlations,
        item_projections,
        quantise,
        rank_subsets,
        subset_scores,
    )
    from typicality_linearity.probes import pearson, spearman

    rng = np.random.default_rng(1)
    names = ["a", "b", "c", "d", "e"]
    categories = np.repeat(names, 20)
    y = rng.normal(size=100)
    x = rng.normal(size=(100, 8)) * 0.3
    x[:, 0] += y
    folds = make_folds(categories, y, 5, seed=0)
    result = transfer_matrices(
        {"p": x},
        y,
        categories,
        folds,
        ["diff_of_means"],
        ["spearman"],
        {},
        "pooled",
        seed=0,
    )
    full = result["full_probes"]["diff_of_means"]
    by_fold = result["fold_probes"]["diff_of_means"]
    projections = item_projections({"p": x}, categories, folds, full, by_fold)
    assert projections["p"].shape == (100, 5)

    # Own category: the fold probe that did not see the item.
    probe = by_fold["a"]["p"][folds[0]]
    direction = probe.weights / np.linalg.norm(probe.weights)
    assert projections["p"][0, 0] == pytest.approx(x[0] @ direction)
    # Other category: the full probe.
    direction = full["b"]["p"].weights / np.linalg.norm(full["b"]["p"].weights)
    assert projections["p"][0, 1] == pytest.approx(x[0] @ direction)

    subsets = all_subsets(5)
    assert subsets.shape == (5, 30)
    scores = subset_scores(projections, y, categories, names, subsets, "spearman")
    # Check one value against the direct computation.
    s = 3
    rows = categories == "c"
    direct = projections["p"][rows][:, subsets[:, s]].mean(axis=1)
    assert scores[0, 2, s] == pytest.approx(spearman(direct, y[rows]))
    pearson_value = column_correlations(direct[:, None], y[rows], "pearson")[0]
    assert pearson_value == pytest.approx(pearson(direct, y[rows]))

    best = rank_subsets(scores, subsets, top=5)
    assert len(best) == 5
    assert all(len(r["subset"]) <= 2 for r in best)  # 3 or more held out
    assert best[0]["advantage"] >= best[-1]["advantage"]

    quantised, scales = quantise(projections, ["p"])
    assert quantised.dtype == np.dtype("<i2")
    assert np.allclose(quantised[0] * scales[0], projections["p"], atol=1e-3)


def test_held_out_by_size():
    from typicality_linearity.mean_direction import all_subsets, held_out_by_size

    subsets = all_subsets(4)
    # Score = 1 for held-out category 0, else 0, in one prompt.
    scores = np.zeros((1, 4, subsets.shape[1]))
    scores[0, 0, :] = 1.0
    result = held_out_by_size(scores, subsets)
    assert sorted(result) == [1, 2, 3]
    # With size k, category 0 is 1 of the 4 - k held-out categories.
    for size in (1, 2, 3):
        assert result[size] == pytest.approx(1 / 4)


def test_mean_direction_averages_over_prompts():
    from typicality_linearity.mean_direction import item_projections, unit

    rng = np.random.default_rng(2)
    names = ["a", "b", "c"]
    categories = np.repeat(names, 15)
    y = rng.normal(size=45)
    x_by_prompt = {}
    for prompt, feature in (("p", 0), ("q", 1)):
        x = rng.normal(size=(45, 6)) * 0.3
        x[:, feature] += y
        x_by_prompt[prompt] = x
    folds = make_folds(categories, y, 5, seed=0)
    result = transfer_matrices(
        x_by_prompt,
        y,
        categories,
        folds,
        ["diff_of_means"],
        ["spearman"],
        {},
        "pooled",
        seed=0,
    )
    full = result["full_probes"]["diff_of_means"]
    by_fold = result["fold_probes"]["diff_of_means"]
    projections = item_projections(x_by_prompt, categories, folds, full, by_fold)

    # Item 0 is in category a. Category b: mean of the full probes of (b, p), (b, q).
    m_b = (unit(full["b"]["p"].weights) + unit(full["b"]["q"].weights)) / 2
    assert projections["q"][0, 1] == pytest.approx(x_by_prompt["q"][0] @ m_b)
    # Category a: mean of the fold probes that did not see item 0, in both prompts.
    f = folds[0]
    m_a = (unit(by_fold["a"]["p"][f].weights) + unit(by_fold["a"]["q"][f].weights)) / 2
    assert projections["p"][0, 0] == pytest.approx(x_by_prompt["p"][0] @ m_a)
