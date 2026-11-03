import numpy as np
import pytest

from typicality_linearity.pca import fit_pca
from typicality_linearity.pca_experiment import fit_all_pca, scope_rows
from typicality_linearity.pca_plot import build_plot_data, render_html
from typicality_linearity.rosch_prompts import read_ratings
from typicality_linearity.token_positions import (
    item_char_span,
    target_char_index,
    token_index_at_char,
)

PROMPTS_PATH = "data/processed/rosch/rosch_prompts.csv"
PROMPT_COLUMNS = [
    "rosch_example",
    "rosch_neg",
    "rosch_simple",
    "rosch_fav",
    "rosch_story_1",
    "rosch_story_2",
    "rosch_story_neg",
]


def test_item_span_is_whole_word():
    prompt = "When she looked back at her younger years, an ear always came to mind."
    start, end = item_char_span(prompt, "ear")
    assert prompt[start:end] == "ear"
    assert prompt[start - 3 : start] == "an "


def test_item_span_ignores_case():
    prompt = "Doll: toy."
    assert item_char_span(prompt, "doll") == (0, 4)


def test_item_is_found_before_category_in_all_prompts():
    for row in read_ratings(PROMPTS_PATH):
        for column in PROMPT_COLUMNS:
            prompt = row[column]
            start, end = item_char_span(prompt, row["item_text"])
            assert prompt[start:end].lower() == row["item_text"].lower()
            assert target_char_index(prompt, row["item_text"], "period") == (
                len(prompt) - 1
            )


def test_token_index_at_char():
    # "A doll is a toy." as tokens "A", " doll", " is", " a", " toy", "."
    offsets = [(0, 1), (1, 6), (6, 9), (9, 11), (11, 15), (15, 16), (0, 0)]
    prompt = "A doll is a toy."
    assert token_index_at_char(offsets, target_char_index(prompt, "doll", "item")) == 1
    assert (
        token_index_at_char(offsets, target_char_index(prompt, "doll", "period")) == 5
    )


def test_token_index_at_char_fails_when_no_token():
    with pytest.raises(ValueError):
        token_index_at_char([(0, 0)], 3)


def test_fit_pca_finds_main_direction():
    rng = np.random.default_rng(0)
    direction = np.array([3.0, 4.0, 0.0]) / 5.0
    x = rng.normal(size=(200, 1)) * 10 * direction + rng.normal(size=(200, 3)) * 0.1
    result = fit_pca(x, 2)
    assert result.projections.shape == (200, 2)
    assert np.allclose(np.abs(result.components[0]), direction, atol=0.01)
    assert result.components[0][np.abs(result.components[0]).argmax()] > 0
    assert result.explained_variance_ratio[0] > 0.99


def test_render_html_contains_data():
    result = fit_pca(np.random.default_rng(0).normal(size=(4, 5)), 3)
    plot_data = build_plot_data(
        title="Test <page>",
        layers=[10],
        prompt_names=["rosch_simple"],
        positions=["item", "period"],
        points={"item": ["a"] * 4, "category": ["toy"] * 4, "color": [0.0] * 4},
        prompts={"rosch_simple": ["A: toy."] * 4},
        token_spans={"rosch_simple": {"item": [(0, 1)] * 4, "period": [(6, 7)] * 4}},
        pca_results={"rosch_simple": {10: {"all": {"item": result, "period": result}}}},
        color_label="rating_zscore",
    )
    page = render_html(plot_data)
    assert "Test &lt;page&gt;" in page
    assert '"rosch_simple"' in page
    assert "</script>" not in page.split('id="data">')[1].split("</script>")[0]


def test_render_html_has_category_selector_and_prompt_hover():
    result = fit_pca(np.random.default_rng(0).normal(size=(4, 5)), 3)
    plot_data = build_plot_data(
        title="Test",
        layers=[10],
        prompt_names=["rosch_simple"],
        positions=["item"],
        points={"item": ["a"] * 4, "category": ["toy"] * 4, "color": [0.0] * 4},
        prompts={"rosch_simple": ["A: toy."] * 4},
        token_spans={"rosch_simple": {"item": [(0, 1)] * 4}},
        pca_results={"rosch_simple": {10: {"all": {"item": result}}}},
        color_label="rating_zscore",
    )
    page = render_html(plot_data)
    assert 'id="category"' in page
    assert "All categories" in page
    assert "hoverText(position, row)" in page


def test_scope_rows_has_all_and_each_category():
    scopes = scope_rows(["toy", "bird", "toy", "bird", "toy"], per_category=True)
    assert list(scopes) == ["all", "bird", "toy"]
    assert scopes["all"].tolist() == [0, 1, 2, 3, 4]
    assert scopes["bird"].tolist() == [1, 3]
    assert scopes["toy"].tolist() == [0, 2, 4]
    assert list(scope_rows(["toy", "bird"], per_category=False)) == ["all"]


def test_per_category_pca_uses_only_rows_of_that_category():
    # Category "a" varies along feature 0, category "b" along feature 1.
    # The pooled PCA is dominated by the offset between the two categories.
    rng = np.random.default_rng(0)
    n = 50
    a = np.zeros((n, 3))
    a[:, 0] = rng.normal(size=n)
    b = np.zeros((n, 3))
    b[:, 1] = rng.normal(size=n)
    b[:, 2] = 100.0
    x = np.concatenate([a, b]) + rng.normal(size=(2 * n, 3)) * 0.01
    activations = {"item": x[:, None, :]}  # (rows, layers, features)
    categories = ["a"] * n + ["b"] * n
    scopes = scope_rows(categories, per_category=True)

    results = fit_all_pca(activations, [10], ["item"], 2, scopes)

    assert np.argmax(np.abs(results[10]["all"]["item"].components[0])) == 2
    assert np.argmax(np.abs(results[10]["a"]["item"].components[0])) == 0
    assert np.argmax(np.abs(results[10]["b"]["item"].components[0])) == 1
    assert results[10]["a"]["item"].projections.shape == (n, 2)


def test_fit_all_pca_fails_when_scope_has_too_few_rows():
    activations = {"item": np.zeros((4, 1, 5))}
    scopes = scope_rows(["a", "a", "a", "b"], per_category=True)
    with pytest.raises(ValueError):
        fit_all_pca(activations, [10], ["item"], 2, scopes)


def test_render_html_has_per_category_scopes():
    rng = np.random.default_rng(0)
    pooled = fit_pca(rng.normal(size=(6, 5)), 2)
    toy = fit_pca(rng.normal(size=(3, 5)), 2)
    bird = fit_pca(rng.normal(size=(3, 5)), 2)
    scopes = scope_rows(["toy", "bird", "toy", "bird", "toy", "bird"], True)
    plot_data = build_plot_data(
        title="Test",
        layers=[10],
        prompt_names=["rosch_simple"],
        positions=["item"],
        points={
            "item": ["x"] * 6,
            "category": ["toy", "bird"] * 3,
            "color": [0.0] * 6,
        },
        prompts={"rosch_simple": ["X: toy."] * 6},
        token_spans={"rosch_simple": {"item": [(0, 1)] * 6}},
        pca_results={
            "rosch_simple": {
                10: {
                    "all": {"item": pooled},
                    "toy": {"item": toy},
                    "bird": {"item": bird},
                }
            }
        },
        scopes=scopes,
        color_label="rating_zscore",
    )
    assert plot_data["scope_rows"]["toy"] == [0, 2, 4]
    by_scope = plot_data["pca"]["main"]["rosch_simple"]["10"]
    assert set(by_scope) == {"all", "toy", "bird"}
    assert "scope_rows" in render_html(plot_data)


def test_render_html_has_panel_for_each_variant():
    rng = np.random.default_rng(0)
    with_baseline = fit_pca(rng.normal(size=(4, 5)), 2)
    without_baseline = fit_pca(rng.normal(size=(4, 5)), 2)
    plot_data = build_plot_data(
        title="Test",
        layers=[10],
        prompt_names=["rosch_simple"],
        positions=["period"],
        points={"item": ["a"] * 4, "category": ["toy"] * 4, "color": [0.0] * 4},
        prompts={"rosch_simple": ["A: toy."] * 4},
        token_spans={"rosch_simple": {"period": [(6, 7)] * 4}},
        pca_results={
            "baseline": {"rosch_simple": {10: {"all": {"period": with_baseline}}}},
            "no_baseline": {
                "rosch_simple": {10: {"all": {"period": without_baseline}}}
            },
        },
        color_label="rating_zscore",
        variants={"baseline": "mean_contrast baseline", "no_baseline": "no baseline"},
    )
    assert plot_data["variants"] == ["baseline", "no_baseline"]
    coords = plot_data["pca"]["no_baseline"]["rosch_simple"]["10"]["all"]["period"]
    assert np.allclose(
        coords["coords"][0], without_baseline.projections[:, 0], atol=1e-4
    )
    page = render_html(plot_data)
    assert "no baseline" in page
    assert "showPromptInAllPanels" in page
