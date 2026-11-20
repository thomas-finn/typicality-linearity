from typicality_linearity.rosch_prompts import (
    CATEGORY_FORMS,
    TEMPLATES,
    build_mean_cat_contrast_prompts,
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
    prompt_rows = build_mean_cat_contrast_prompts(rows)
    assert len(prompt_rows) == len(rows) * len(CATEGORY_FORMS)
    own = [r for r in prompt_rows if r["is_item_category"]]
    assert len(own) == len(rows)
    assert [r["rosch_example"] for r in own] == [
        r["rosch_example"] for r in build_prompts(rows)
    ]


def test_mean_contrast_keeps_item_grammar_and_flags_members():
    rows = read_ratings(RATINGS_PATH)
    prompt_rows = build_mean_cat_contrast_prompts(rows)
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
    for prompt_row in build_mean_cat_contrast_prompts(rows):
        if prompt_row["is_item_category"]:
            assert prompt_row["rating"] is not None
            assert prompt_row["rating_zscore"] is not None
        else:
            assert prompt_row["rating"] is None
            assert prompt_row["rating_zscore"] is None


def test_ambiguous_flag_in_both_suites():
    from typicality_linearity.rosch_prompts import AMBIGUOUS_EXTENDED, AMBIGUOUS_ITEMS

    ratings = read_ratings(RATINGS_PATH)
    rated = {(row["category"], row["item"]) for row in ratings}
    assert set(AMBIGUOUS_ITEMS) <= rated
    assert AMBIGUOUS_EXTENDED <= set(AMBIGUOUS_ITEMS)

    rows = build_prompts(ratings)
    flagged = {(r["category"], r["item"]) for r in rows if r["is_ambiguous_extended"]}
    assert flagged == AMBIGUOUS_EXTENDED
    assert len(flagged) == 41
    assert ("bird", "bat") in flagged
    assert ("toy", "car") not in flagged

    contrast = build_mean_cat_contrast_prompts(ratings)
    # The flag follows the rated category, so it is the same for all rows of an item.
    bat_rows = [
        r for r in contrast if (r["item_category"], r["item"]) == ("bird", "bat")
    ]
    assert len(bat_rows) == len(CATEGORY_FORMS)
    assert all(r["is_ambiguous_extended"] for r in bat_rows)


def test_prompt_formats_cover_templates():
    from typicality_linearity.rosch_prompts import PROMPT_FORMATS

    assert set(PROMPT_FORMATS) == set(TEMPLATES)


def test_eval_templates():
    from typicality_linearity.rosch_prompts import (
        CATEGORY_FORMS,
        EVAL_PROMPT_FORMATS,
        EVAL_TEMPLATES,
        item_phrase,
    )

    assert set(EVAL_PROMPT_FORMATS) == set(EVAL_TEMPLATES)
    assert not set(EVAL_TEMPLATES) & set(TEMPLATES)

    doll = item_phrase("doll", "toy")
    toy = CATEGORY_FORMS["toy"]
    assert EVAL_TEMPLATES["rosch_eval_1"](doll, toy) == "A doll is a kind of toy."
    assert EVAL_TEMPLATES["rosch_eval_2"](doll, toy) == (
        "When you think about toys, what comes to mind first for me is a doll."
    )
    assert EVAL_TEMPLATES["rosch_eval_3"](doll, toy) == (
        "Slow down there, I do not think that a doll is a toy, do you think that."
    )

    pants = item_phrase("pant", "clothing")
    clothing = CATEGORY_FORMS["clothing"]
    assert EVAL_TEMPLATES["rosch_eval_1"](pants, clothing) == (
        "Pants are a kind of clothing."
    )
    assert EVAL_TEMPLATES["rosch_eval_3"](pants, clothing) == (
        "Slow down there, I do not think that pants are clothing, do you think that."
    )


def test_build_eval_prompts():
    from typicality_linearity.rosch_prompts import (
        CATEGORY_FORMS,
        EVAL_TEMPLATES,
        build_eval_prompts,
    )

    rows = [
        {"item": "doll", "category": "toy", "rating": "1.4", "rating_zscore": "-1.5"},
        {"item": "bat", "category": "bird", "rating": "6.0", "rating_zscore": "2.0"},
    ]
    prompt_rows = build_eval_prompts(rows)
    assert len(prompt_rows) == 2 * len(CATEGORY_FORMS)
    for row in prompt_rows:
        assert set(EVAL_TEMPLATES) <= set(row)
        assert not set(TEMPLATES) & set(row)
        assert row["rosch_eval_1"].endswith(".")
    own = [row for row in prompt_rows if row["is_item_category"]]
    assert [row["item"] for row in own] == ["doll", "bat"]
    assert [row["is_ambiguous_extended"] for row in own] == [False, True]


def test_new_frames_count_noun():
    assert render("rosch_typ_frame_1", "doll", "toy") == "A doll is a typical toy."
    assert (
        render("rosch_typ_frame_2", "doll", "toy")
        == "A doll is a good example of a toy."
    )
    assert render("rosch_typ_frame_3", "doll", "toy") == "A doll is an unusual toy."
    assert render("rosch_hedge_1", "doll", "toy") == "Technically, a doll is a toy."
    assert (
        render("rosch_hedge_2", "doll", "toy") == "Strictly speaking, a doll is a toy."
    )
    assert render("rosch_hedge_3", "doll", "toy") == "A doll is a toy, sort of."
    assert render("rosch_cat_first_1", "doll", "toy") == "One kind of toy is a doll."
    assert (
        render("rosch_cat_first_2", "doll", "toy") == "An example of a toy is a doll."
    )


def test_new_frames_mass_category_and_plural_item():
    assert render("rosch_typ_frame_1", "pant", "clothing") == (
        "Pants are typical clothing."
    )
    assert render("rosch_typ_frame_3", "chair", "furniture") == (
        "A chair is unusual furniture."
    )
    assert render("rosch_hedge_1", "pant", "clothing") == (
        "Technically, pants are clothing."
    )
    assert render("rosch_cat_first_2", "pant", "clothing") == (
        "An example of clothing is pants."
    )
    assert render("rosch_cat_first_1", "tennis", "sport") == (
        "One kind of sport is tennis."
    )


def test_core_ambiguous_is_subset_of_extended():
    from typicality_linearity.rosch_prompts import AMBIGUOUS_CORE

    rows = build_prompts(read_ratings(RATINGS_PATH))
    extended = {(r["category"], r["item"]) for r in rows if r["is_ambiguous_extended"]}
    core = {(r["category"], r["item"]) for r in rows if r["is_ambiguous_core"]}
    assert core == AMBIGUOUS_CORE
    assert core <= extended
    assert len(core) == 34
    assert ("bird", "bat") in core
    assert ("bird", "turkey") in core
    assert ("sport", "squash") in core
    assert ("bird", "lark") not in core
    assert ("bird", "lark") in extended
    assert ("bird", "duck") not in extended


def test_ambiguous_flags_are_in_the_review_list():
    from typicality_linearity.rosch_prompts import (
        AMBIGUOUS_CORE,
        AMBIGUOUS_EXTENDED,
        AMBIGUOUS_ITEMS,
    )

    assert AMBIGUOUS_CORE <= AMBIGUOUS_EXTENDED <= set(AMBIGUOUS_ITEMS)


def test_mean_item_contrast_groups():
    from typicality_linearity.rosch_prompts import build_mean_item_contrast_prompts

    ratings = read_ratings(RATINGS_PATH)
    rated_pairs = {(row["item"].lower(), row["category"]) for row in ratings}
    prompt_rows = build_mean_item_contrast_prompts(ratings, seed=0)
    assert len(prompt_rows) == len(ratings) * len(CATEGORY_FORMS)

    groups = {}
    for row in prompt_rows:
        groups.setdefault((row["item"], row["item_category"]), []).append(row)
    assert len(groups) == len(ratings)
    for (item, category), rows in groups.items():
        anchors = [r for r in rows if r["is_anchor"]]
        contrast = [r for r in rows if not r["is_anchor"]]
        assert len(anchors) == 1
        assert anchors[0]["prompt_item"] == item
        assert len(contrast) == len(CATEGORY_FORMS) - 1
        assert all(r["category"] == category for r in rows)
        other_categories = {r["prompt_item_category"] for r in contrast}
        assert other_categories == set(CATEGORY_FORMS) - {category}
        for r in contrast:
            assert (r["prompt_item"].lower(), category) not in rated_pairs
        words = [r["prompt_item"].lower() for r in rows]
        assert len(set(words)) == len(words)


def test_mean_item_contrast_anchor_matches_prompts_and_seed():
    from typicality_linearity.rosch_prompts import build_mean_item_contrast_prompts

    ratings = read_ratings(RATINGS_PATH)
    rows_a = build_mean_item_contrast_prompts(ratings, seed=0)
    assert rows_a == build_mean_item_contrast_prompts(ratings, seed=0)
    assert rows_a != build_mean_item_contrast_prompts(ratings, seed=1)

    anchors = [r for r in rows_a if r["is_anchor"]]
    assert [r["rosch_example"] for r in anchors] == [
        r["rosch_example"] for r in build_prompts(ratings)
    ]
    # A contrast item keeps the grammar of its own category.
    pants_rows = [
        r for r in rows_a if not r["is_anchor"] and r["prompt_item"] == "pant"
    ]
    for r in pants_rows:
        assert r["rosch_example"].startswith("Pants are an example of")


def test_mean_item_contrast_rating_only_for_anchor():
    from typicality_linearity.rosch_prompts import build_mean_item_contrast_prompts

    rows = read_ratings(RATINGS_PATH)
    for prompt_row in build_mean_item_contrast_prompts(rows, seed=0):
        if prompt_row["is_anchor"]:
            assert prompt_row["rating"] is not None
            assert prompt_row["rating_zscore"] is not None
        else:
            assert prompt_row["rating"] is None
            assert prompt_row["rating_zscore"] is None


def test_nonce_prompts():
    from typicality_linearity.rosch_prompts import (
        NONCE_CATEGORIES,
        build_nonce_prompts,
    )

    assert NONCE_CATEGORIES == ("wugawrar", "njonjobura", "duplambi")
    ratings = read_ratings(RATINGS_PATH)
    rows = build_nonce_prompts(ratings)
    normal = build_prompts(ratings)
    assert len(rows) == 3 * len(normal)
    assert "rating" not in rows[0] and "rating_zscore" not in rows[0]
    assert set(TEMPLATES) <= set(rows[0])
    for i, row in enumerate(rows):
        normal_row = normal[i // 3]
        assert (row["item"], row["category"]) == (
            normal_row["item"],
            normal_row["category"],
        )
        assert row["nonce_category"] == NONCE_CATEGORIES[i % 3]
    dolls = [r for r in rows if (r["item"], r["category"]) == ("doll", "toy")]
    assert [r["nonce_category"] for r in dolls] == list(NONCE_CATEGORIES)
    doll = dolls[0]
    assert doll["rosch_example"] == "A doll is an example of a wugawrar."
    assert doll["rosch_story_1"].endswith("thought about wugawrars.")
    assert doll["rosch_typ_frame_3"] == "A doll is an unusual wugawrar."
    assert doll["rosch_cat_first_2"] == "An example of a wugawrar is a doll."
    assert dolls[1]["rosch_example"] == "A doll is an example of a njonjobura."
    assert dolls[2]["rosch_example"] == "A doll is an example of a duplambi."
    tennis = next(r for r in rows if (r["item"], r["category"]) == ("tennis", "sport"))
    assert tennis["rosch_simple"] == "Tennis: wugawrar."
    assert tennis["rosch_hedge_1"] == "Technically, tennis is a wugawrar."


def test_nonce_eval_prompts():
    from typicality_linearity.rosch_prompts import (
        EVAL_TEMPLATES,
        build_nonce_prompts,
    )

    ratings = read_ratings(RATINGS_PATH)
    rows = build_nonce_prompts(ratings, templates=EVAL_TEMPLATES)
    assert len(rows) == 3 * len(ratings)
    assert set(EVAL_TEMPLATES) <= set(rows[0])
    assert not set(TEMPLATES) & set(rows[0])
    doll = next(r for r in rows if (r["item"], r["category"]) == ("doll", "toy"))
    assert doll["rosch_eval_1"] == "A doll is a kind of wugawrar."
    assert doll["rosch_eval_3"].endswith("a doll is a wugawrar, do you think that.")


def test_nonce_prompts_agree_with_plural_items():
    from typicality_linearity.rosch_prompts import (
        EVAL_TEMPLATES,
        nonce_category_forms,
    )

    marbles = item_phrase("marbles", "toy")
    doll = item_phrase("doll", "toy")
    nonce = nonce_category_forms("wugawrar")
    assert TEMPLATES["rosch_typ_frame_1"](marbles, nonce) == (
        "Marbles are typical wugawrars."
    )
    assert TEMPLATES["rosch_typ_frame_3"](marbles, nonce) == (
        "Marbles are unusual wugawrars."
    )
    assert TEMPLATES["rosch_hedge_1"](marbles, nonce) == (
        "Technically, marbles are wugawrars."
    )
    assert (
        TEMPLATES["rosch_hedge_3"](marbles, nonce) == "Marbles are wugawrars, sort of."
    )
    assert "marbles are wugawrars," in EVAL_TEMPLATES["rosch_eval_3"](marbles, nonce)
    # "an example of" keeps the singular.
    assert TEMPLATES["rosch_example"](marbles, nonce) == (
        "Marbles are an example of a wugawrar."
    )
    # A singular item does not change.
    assert (
        TEMPLATES["rosch_typ_frame_1"](doll, nonce) == "A doll is a typical wugawrar."
    )


def test_real_prompts_agree_with_plural_items():
    marbles = item_phrase("marbles", "toy")
    pants = item_phrase("pant", "clothing")
    assert render("rosch_hedge_3", "marbles", "toy") == "Marbles are toys, sort of."
    assert render("rosch_hedge_1", "plier", "tool") == "Technically, pliers are tools."
    assert TEMPLATES["rosch_typ_frame_1"](marbles, CATEGORY_FORMS["toy"]) == (
        "Marbles are typical toys."
    )
    assert TEMPLATES["rosch_typ_frame_3"](marbles, CATEGORY_FORMS["sport"]) == (
        "Marbles are unusual sports."
    )
    # A mass category has no plural form, so it does not change.
    assert TEMPLATES["rosch_typ_frame_1"](pants, CATEGORY_FORMS["clothing"]) == (
        "Pants are typical clothing."
    )
    assert (
        render("rosch_example", "marbles", "toy") == "Marbles are an example of a toy."
    )


def test_mean_cat_contrast_extended_prompts():
    from typicality_linearity.rosch_prompts import (
        EXTENDED_CATEGORY_FORMS,
        build_mean_cat_contrast_extended_prompts,
    )

    ratings = read_ratings(RATINGS_PATH)
    rows = build_mean_cat_contrast_extended_prompts(ratings)
    base = build_mean_cat_contrast_prompts(ratings)
    n_categories = len(CATEGORY_FORMS) + len(EXTENDED_CATEGORY_FORMS)
    assert len(rows) == len(ratings) * n_categories
    # The rated-category rows are the base rows, in the same order.
    rated = [r for r in rows if not r["is_extended_category"]]
    assert len(rated) == len(base)
    for row, base_row in zip(rated, base):
        assert all(row[key] == value for key, value in base_row.items())
    # Extended rows: never a target, never rated, no rating.
    extended = [r for r in rows if r["is_extended_category"]]
    assert not any(r["is_item_category"] or r["is_rated_member"] for r in extended)
    assert all(r["rating"] is None for r in extended)
    doll = {r["category"]: r for r in extended if r["item"] == "doll"}
    assert (
        doll["occupation"]["rosch_example"] == "A doll is an example of an occupation."
    )
    assert doll["fish"]["rosch_story_1"].endswith("when she thought about fish.")
    marbles = {r["category"]: r for r in extended if r["item"] == "marbles"}
    assert marbles["illness"]["rosch_typ_frame_1"] == "Marbles are typical illnesses."
    members = {(r["category"], r["item"]) for r in rows if r["is_plausible_member"]}
    assert ("musical instrument", "drum") in members
    assert ("drink", "squash") in members
    assert all(category in EXTENDED_CATEGORY_FORMS for category, _ in members)
