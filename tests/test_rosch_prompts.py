from typicality_linearity.rosch_prompts import (
    CATEGORY_FORMS,
    TEMPLATES,
    build_mean_contrast_prompts,
    build_prompts,
    item_phrase,
    read_ratings,
)

RATINGS_PATH = "data/raw/rosch/rosch1975_ratings.csv"


def render(template_name, item, category):
    template = TEMPLATES[template_name]
    return template(item_phrase(item, category), CATEGORY_FORMS[category])


def test_count_noun_takes_article():
    assert render("rosch_example", "doll", "toy") == "A doll is an example of a toy."
    assert (
        render("rosch_example", "apple", "fruit")
        == "An apple is an example of a fruit."
    )


def test_plural_item_takes_are():
    assert (
        render("rosch_fav", "marbles", "toy") == "Marbles are my favourite kind of toy."
    )


def test_lemmatised_plural_is_restored():
    assert (
        render("rosch_neg", "pant", "clothing")
        == "Pants are not an example of clothing."
    )


def test_sport_has_no_article():
    assert (
        render("rosch_story_1", "tennis", "sport")
        == "When she looked back at her younger years, tennis always came to mind "
        "when she thought about sports."
    )


def test_mass_category():
    assert (
        render("rosch_story_neg", "chair", "furniture")
        == "Are you deluded? A chair is clearly not a kind of furniture."
    )


def test_simple():
    assert render("rosch_simple", "doll", "toy") == "Doll: toy."


def test_one_row_per_item_and_one_column_per_template():
    rows = read_ratings(RATINGS_PATH)
    prompt_rows = build_prompts(rows)
    assert len(prompt_rows) == len(rows)
    for prompt_row in prompt_rows:
        for name in TEMPLATES:
            assert "  " not in prompt_row[name]
            assert prompt_row[name].endswith(".")


def test_mean_contrast_has_all_categories_per_item():
    rows = read_ratings(RATINGS_PATH)
    prompt_rows = build_mean_contrast_prompts(rows)
    assert len(prompt_rows) == len(rows) * len(CATEGORY_FORMS)
    own = [r for r in prompt_rows if r["is_item_category"]]
    assert len(own) == len(rows)
    assert [r["rosch_example"] for r in own] == [
        r["rosch_example"] for r in build_prompts(rows)
    ]


def test_mean_contrast_keeps_item_grammar_and_flags_members():
    rows = read_ratings(RATINGS_PATH)
    prompt_rows = build_mean_contrast_prompts(rows)
    pants_bird = next(
        r for r in prompt_rows if r["item"] == "pant" and r["category"] == "bird"
    )
    assert pants_bird["rosch_example"] == "Pants are an example of a bird."
    tomato_veg = next(
        r
        for r in prompt_rows
        if r["item"] == "tomato"
        and r["item_category"] == "fruit"
        and r["category"] == "vegetable"
    )
    assert tomato_veg["is_rated_member"]
    assert not tomato_veg["is_item_category"]


def test_mean_contrast_rating_only_for_item_category():
    rows = read_ratings(RATINGS_PATH)
    for prompt_row in build_mean_contrast_prompts(rows):
        if prompt_row["is_item_category"]:
            assert prompt_row["rating"] is not None
            assert prompt_row["rating_zscore"] is not None
        else:
            assert prompt_row["rating"] is None
            assert prompt_row["rating_zscore"] is None
