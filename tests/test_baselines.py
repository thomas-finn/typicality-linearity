import numpy as np
import pytest

from typicality_linearity.baselines import mean_contrast, mean_contrast_groups

CATEGORIES = ["bird", "fruit", "toy"]


def contrast_rows(items):
    """Rows like rosch_mean_cat_contrast.csv. items: [(item, item_category, zscore)]."""
    rows = []
    for item, item_category, zscore in items:
        for category in CATEGORIES:
            own = category == item_category
            rows.append(
                {
                    "item": item,
                    "item_category": item_category,
                    "rating_zscore": str(zscore) if own else "",
                    "item_text": item,
                    "category": category,
                    "is_item_category": str(own),
                    "is_rated_member": str(own),
                    "rosch_simple": f"{item.capitalize()}: {category}.",
                }
            )
    return rows


def test_groups_find_target_and_others():
    rows = contrast_rows([("doll", "toy", -1.0), ("robin", "bird", 0.5)])
    groups = mean_contrast_groups(rows)
    assert groups == [(2, [0, 1]), (3, [4, 5])]


def test_same_item_in_two_categories_is_two_groups():
    rows = contrast_rows([("tomato", "fruit", 0.0), ("tomato", "toy", 1.0)])
    assert mean_contrast_groups(rows) == [(1, [0, 2]), (5, [3, 4])]


def test_group_without_target_fails():
    rows = contrast_rows([("doll", "toy", -1.0)])[:2]
    with pytest.raises(ValueError):
        mean_contrast_groups(rows)


def item_contrast_rows(groups):
    """Rows like rosch_mean_item_contrast.csv.

    groups: [(item, category, [contrast items])]
    """
    rows = []
    for item, category, contrast_items in groups:
        for prompt_item in [item] + contrast_items:
            anchor = prompt_item == item
            rows.append(
                {
                    "item": item,
                    "item_category": category,
                    "category": category,
                    "prompt_item": prompt_item,
                    "item_text": prompt_item,
                    "is_anchor": str(anchor),
                }
            )
    return rows


def test_item_contrast_groups_use_is_anchor():
    rows = item_contrast_rows(
        [("doll", "toy", ["robin", "apple"]), ("robin", "bird", ["doll", "pear"])]
    )
    assert mean_contrast_groups(rows) == [(0, [1, 2]), (3, [4, 5])]


def test_item_contrast_anchor_not_first_row():
    rows = item_contrast_rows([("doll", "toy", ["robin", "apple"])])
    rows = [rows[1], rows[0], rows[2]]
    assert mean_contrast_groups(rows) == [(1, [0, 2])]


def test_item_contrast_group_without_anchor_fails():
    rows = item_contrast_rows([("doll", "toy", ["robin", "apple"])])[1:]
    with pytest.raises(ValueError):
        mean_contrast_groups(rows)


def test_rows_need_one_target_column():
    rows = contrast_rows([("doll", "toy", -1.0)])
    for row in rows:
        row["is_anchor"] = row["is_item_category"]
    with pytest.raises(ValueError):
        mean_contrast_groups(rows)
    for row in rows:
        del row["is_anchor"]
        del row["is_item_category"]
    with pytest.raises(ValueError):
        mean_contrast_groups(rows)


def test_item_contrast_groups_on_generated_file():
    from typicality_linearity.rosch_prompts import (
        build_mean_item_contrast_prompts,
        read_ratings,
    )

    ratings = read_ratings("data/raw/rosch/rosch1975_ratings.csv")
    rows = [
        {key: str(value) for key, value in row.items()}
        for row in build_mean_item_contrast_prompts(ratings, seed=0)
    ]
    groups = mean_contrast_groups(rows)
    assert len(groups) == len(ratings)
    for target, others in groups:
        assert rows[target]["is_anchor"] == "True"
        assert len(others) == 9
        assert all(rows[i]["category"] == rows[target]["category"] for i in others)

    x = np.arange(len(rows), dtype=float)[:, None]
    result = mean_contrast(x, groups)
    target, others = groups[0]
    assert result[0, 0] == x[target, 0] - x[others, 0].mean()


def test_mean_contrast_subtracts_mean_of_other_categories():
    x = np.array([[1.0, 0.0], [3.0, 2.0], [10.0, 10.0]])
    result = mean_contrast(x, [(2, [0, 1])])
    assert np.allclose(result, [[8.0, 9.0]])
