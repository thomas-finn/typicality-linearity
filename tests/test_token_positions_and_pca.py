import numpy as np
import pytest

from typicality_linearity.pca import fit_pca
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
