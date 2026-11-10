"""Build the Rosch (1975) suite of prompts from the item-category ratings.

Each prompt template needs a grammatical noun phrase for the item and the
category. Most items are singular count nouns ("a doll"). The exceptions
(mass nouns, plurals, sports, lemmatised plurals such as "pant") are listed
in ITEM_OVERRIDES.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

# Determiner values for a noun phrase.
ARTICLE = "article"  # "a" or "an", chosen from the first letter
NONE = ""  # mass noun, plural, or activity: no determiner


@dataclass(frozen=True)
class NounPhrase:
    text: str
    determiner: str  # ARTICLE or NONE
    plural: bool = False

    def with_determiner(self):
        if self.determiner == ARTICLE:
            return indefinite_article(self.text) + " " + self.text
        return self.text

    def verb_be(self):
        if self.plural:
            return "are"
        return "is"


@dataclass(frozen=True)
class CategoryForms:
    bare: str  # "toy", used after "kind of"
    with_determiner: str  # "a toy", used after "an example of"
    general: str  # "toys", used to talk about the category in general


CATEGORY_FORMS = {
    "bird": CategoryForms("bird", "a bird", "birds"),
    "clothing": CategoryForms("clothing", "clothing", "clothing"),
    "fruit": CategoryForms("fruit", "a fruit", "fruit"),
    "furniture": CategoryForms("furniture", "furniture", "furniture"),
    "sport": CategoryForms("sport", "a sport", "sports"),
    "tool": CategoryForms("tool", "a tool", "tools"),
    "toy": CategoryForms("toy", "a toy", "toys"),
    "vegetable": CategoryForms("vegetable", "a vegetable", "vegetables"),
    "vehicle": CategoryForms("vehicle", "a vehicle", "vehicles"),
    "weapon": CategoryForms("weapon", "a weapon", "weapons"),
}

# Categories where items are activities and take no determiner by default.
CATEGORIES_WITHOUT_ARTICLE = {"sport"}


def mass(text):
    return NounPhrase(text, NONE, plural=False)


def plural(text):
    return NounPhrase(text, NONE, plural=True)


def count(text):
    return NounPhrase(text, ARTICLE, plural=False)


# Keyed by (category, raw item). The same item can need a different form in
# a different category, for example "checkers" (toy: plural, sport: game name).
ITEM_OVERRIDES = {
    # toy
    ("toy", "marbles"): plural("marbles"),
    ("toy", "jacks"): plural("jacks"),
    ("toy", "paper dolls"): plural("paper dolls"),
    ("toy", "clay"): mass("clay"),
    ("toy", "crayons"): plural("crayons"),
    ("toy", "skates"): plural("skates"),
    ("toy", "monopoly"): mass("Monopoly"),
    ("toy", "stilts"): plural("stilts"),
    ("toy", "checkers"): plural("checkers"),
    ("toy", "dishes"): plural("dishes"),
    ("toy", "cards"): plural("cards"),
    ("toy", "animals"): plural("animals"),
    ("toy", "books"): plural("books"),
    # vegetable
    ("vegetable", "spinach"): mass("spinach"),
    ("vegetable", "broccoli"): mass("broccoli"),
    ("vegetable", "asparagus"): mass("asparagus"),
    ("vegetable", "corn"): mass("corn"),
    ("vegetable", "cauliflower"): mass("cauliflower"),
    ("vegetable", "brussels sprouts"): plural("Brussels sprouts"),
    ("vegetable", "lettuce"): mass("lettuce"),
    ("vegetable", "celery"): mass("celery"),
    ("vegetable", "green"): plural("greens"),
    ("vegetable", "romaine"): mass("romaine"),
    ("vegetable", "okra"): mass("okra"),
    ("vegetable", "turnip green"): plural("turnip greens"),
    ("vegetable", "collard"): plural("collard greens"),
    ("vegetable", "watercress"): mass("watercress"),
    ("vegetable", "black eyed pea"): count("black-eyed pea"),
    ("vegetable", "parsley"): mass("parsley"),
    ("vegetable", "kale"): mass("kale"),
    ("vegetable", "escarole"): mass("escarole"),
    ("vegetable", "rhubarb"): mass("rhubarb"),
    ("vegetable", "sauerkraut"): mass("sauerkraut"),
    ("vegetable", "baked bean"): plural("baked beans"),
    ("vegetable", "seaweed"): mass("seaweed"),
    ("vegetable", "garlic"): mass("garlic"),
    ("vegetable", "rice"): mass("rice"),
    # tool
    ("tool", "sandpaper"): mass("sandpaper"),
    ("tool", "plier"): plural("pliers"),
    ("tool", "wood"): mass("wood"),
    ("tool", "lumber"): mass("lumber"),
    ("tool", "chalk"): mass("chalk"),
    ("tool", "glue"): mass("glue"),
    ("tool", "varnish"): mass("varnish"),
    ("tool", "plaster"): mass("plaster"),
    ("tool", "cement"): mass("cement"),
    ("tool", "scissor"): plural("scissors"),
    # clothing
    ("clothing", "pant"): plural("pants"),
    ("clothing", "slack"): plural("slacks"),
    ("clothing", "underpant"): plural("underpants"),
    ("clothing", "panty"): plural("panties"),
    ("clothing", "pajama"): plural("pajamas"),
    ("clothing", "nylon"): plural("nylons"),
    ("clothing", "earmuff"): plural("earmuffs"),
    # vehicle
    ("vehicle", "feet"): plural("feet"),
    # weapon
    ("weapon", "Atom bomb"): count("atom bomb"),
    ("weapon", "brass knuckle"): plural("brass knuckles"),
    ("weapon", "teargas"): mass("tear gas"),
    ("weapon", "judo"): mass("judo"),
    ("weapon", "poison"): mass("poison"),
    ("weapon", "gas"): mass("gas"),
    ("weapon", "scissor"): plural("scissors"),
    ("weapon", "glass"): mass("glass"),
}


# Items whose word, as written in the prompt, has a common second meaning that
# is not the meaning rated in the category. Keyed by (category, raw item). The
# value is the other meaning. Items that are only weak members of the category
# with the same meaning ("car" as a toy, "hammer" as a weapon) are not ambiguous.
AMBIGUOUS_ITEMS = {
    # bird
    ("bird", "bat"): "baseball bat",
    ("bird", "crane"): "lifting machine",
    ("bird", "cardinal"): "church official",
    ("bird", "swallow"): "verb, to swallow",
    ("bird", "lark"): "a bit of fun",
    ("bird", "duck"): "verb, to duck",
    ("bird", "turkey"): "the country",
    ("bird", "dove"): "past tense of dive",
    # toy
    ("toy", "top"): "upper part",
    ("toy", "block"): "city block",
    ("toy", "ball"): "formal dance",
    ("toy", "bat"): "the animal",
    ("toy", "kite"): "the bird",
    ("toy", "jacks"): "car jacks",
    ("toy", "swing"): "verb, to swing",
    # sport
    ("sport", "squash"): "the vegetable",
    ("sport", "cricket"): "the insect",
    ("sport", "pool"): "swimming pool",
    ("sport", "fencing"): "fences",
    ("sport", "track"): "path or music track",
    ("sport", "polo"): "polo shirt",
    # vegetable
    ("vegetable", "squash"): "the sport",
    ("vegetable", "green"): "the colour",
    ("vegetable", "pepper"): "the spice",
    # fruit
    ("fruit", "orange"): "the colour",
    ("fruit", "date"): "calendar date",
    ("fruit", "apple"): "the company",
    ("fruit", "nut"): "hardware nut",
    ("fruit", "squash"): "the sport",
    # tool
    ("tool", "plane"): "aircraft",
    ("tool", "file"): "document",
    ("tool", "saw"): "past tense of see",
    ("tool", "nail"): "fingernail",
    ("tool", "ruler"): "monarch",
    ("tool", "level"): "stage or adjective",
    ("tool", "square"): "the shape",
    ("tool", "bolt"): "lightning bolt",
    ("tool", "nut"): "food nut",
    ("tool", "crane"): "the bird",
    ("tool", "wood"): "forest",
    # clothing
    ("clothing", "suit"): "lawsuit",
    ("clothing", "slip"): "verb, to slip",
    ("clothing", "tie"): "verb, to tie, or a draw",
    ("clothing", "ring"): "sound or boxing ring",
    ("clothing", "watch"): "verb, to watch",
    ("clothing", "cape"): "headland",
    ("clothing", "jumper"): "a person who jumps",
    # vehicle
    ("vehicle", "train"): "verb, to train",
    ("vehicle", "tank"): "container",
    ("vehicle", "jet"): "stream of liquid",
    ("vehicle", "skate"): "the fish",
    ("vehicle", "trailer"): "film trailer",
    ("vehicle", "subway"): "the restaurant",
    # furniture
    ("furniture", "chest"): "part of the body",
    ("furniture", "bureau"): "government office",
    ("furniture", "bar"): "pub or metal bar",
    ("furniture", "counter"): "a device that counts",
    ("furniture", "lounge"): "room, or verb",
    ("furniture", "vanity"): "pride",
    ("furniture", "rocker"): "rock musician",
    ("furniture", "buffet"): "meal",
    ("furniture", "cabinet"): "government cabinet",
    ("furniture", "stool"): "faeces",
    ("furniture", "fan"): "supporter",
    ("furniture", "ottoman"): "the empire",
    ("furniture", "picture"): "image or photograph",
    # weapon
    ("weapon", "club"): "organisation",
    ("weapon", "bow"): "verb, to bow, or ribbon",
    ("weapon", "mortar"): "building material",
    ("weapon", "blackjack"): "card game",
    ("weapon", "tank"): "container",
    ("weapon", "gas"): "fuel",
}


# Items in AMBIGUOUS_ITEMS that are not ambiguous under the lenient rule. The
# rated meaning is the most common meaning of the word in a category context
# ("duck" as a bird, "apple" as a fruit). Chosen by hand, item by item.
NOT_AMBIGUOUS_LENIENT = {
    ("bird", "duck"),
    ("bird", "turkey"),
    ("bird", "dove"),
    ("toy", "ball"),
    ("sport", "fencing"),
    ("sport", "polo"),
    ("vegetable", "pepper"),
    ("fruit", "apple"),
    ("tool", "saw"),
    ("tool", "nail"),
    ("tool", "wood"),
    ("clothing", "suit"),
    ("clothing", "watch"),
    ("clothing", "cape"),
    ("clothing", "jumper"),
    ("vehicle", "train"),
    ("vehicle", "jet"),
    ("vehicle", "skate"),
    ("vehicle", "subway"),
    ("furniture", "chest"),
    ("furniture", "counter"),
    ("furniture", "rocker"),
    ("furniture", "cabinet"),
    ("furniture", "stool"),
    ("furniture", "ottoman"),
    ("furniture", "picture"),
    ("weapon", "club"),
    ("weapon", "bow"),
}


def is_ambiguous(item, category):
    return (category, item) in AMBIGUOUS_ITEMS


def is_ambiguous_lenient(item, category):
    if (category, item) in NOT_AMBIGUOUS_LENIENT:
        return False
    return is_ambiguous(item, category)


def indefinite_article(text):
    # The data has no "hour"/"unicorn" type words, so the first letter is enough.
    # "T-square" and "Honda" start with a consonant sound, as the letter shows.
    if text[0].lower() in "aeiou":
        return "an"
    return "a"


def item_phrase(item, category):
    key = (category, item)
    if key in ITEM_OVERRIDES:
        return ITEM_OVERRIDES[key]
    if category in CATEGORIES_WITHOUT_ARTICLE:
        return mass(item)
    return count(item)


def capitalise_first(text):
    return text[0].upper() + text[1:]


# Each template takes (item phrase, category forms) and returns one prompt.
def prompt_example(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} an example of {cat.with_determiner}."


def prompt_neg(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} not an example of {cat.with_determiner}."


def prompt_simple(item, cat):
    return f"{capitalise_first(item.text)}: {cat.bare}."


def prompt_fav(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} my favourite kind of {cat.bare}."


def prompt_story_1(item, cat):
    return (
        "When she looked back at her younger years, "
        f"{item.with_determiner()} always came to mind "
        f"when she thought about {cat.general}."
    )


def prompt_story_2(item, cat):
    return (
        f"The problem with {item.with_determiner()} is that "
        f"he really did not like {cat.general}."
    )


def prompt_story_neg(item, cat):
    subject = capitalise_first(item.with_determiner())
    return (
        f"Are you deluded? {subject} {item.verb_be()} clearly not a kind of {cat.bare}."
    )


def category_with_adjective(cat, adjective):
    """Return "a typical toy", or "typical clothing" for a mass category."""
    if cat.with_determiner == cat.bare:
        return f"{adjective} {cat.bare}"
    return f"{indefinite_article(adjective)} {adjective} {cat.bare}"


def prompt_typ_frame_1(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} {category_with_adjective(cat, 'typical')}."


def prompt_typ_frame_2(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} a good example of {cat.with_determiner}."


def prompt_typ_frame_3(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} {category_with_adjective(cat, 'unusual')}."


def prompt_hedge_1(item, cat):
    return (
        f"Technically, {item.with_determiner()} {item.verb_be()} {cat.with_determiner}."
    )


def prompt_hedge_2(item, cat):
    return (
        f"Strictly speaking, {item.with_determiner()} {item.verb_be()} "
        f"{cat.with_determiner}."
    )


def prompt_hedge_3(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} {cat.with_determiner}, sort of."


def prompt_cat_first_1(item, cat):
    return f"One kind of {cat.bare} is {item.with_determiner()}."


def prompt_cat_first_2(item, cat):
    return f"An example of {cat.with_determiner} is {item.with_determiner()}."


# The format of each template, for display. The grammar of ITEM and CATEGORY
# changes with the item and category (see the template functions).
PROMPT_FORMATS = {
    "rosch_example": "A ITEM is an example of a CATEGORY.",
    "rosch_neg": "A ITEM is not an example of a CATEGORY.",
    "rosch_simple": "ITEM: CATEGORY.",
    "rosch_fav": "A ITEM is my favourite kind of CATEGORY.",
    "rosch_story_1": (
        "When she looked back at her younger years, a ITEM always came to mind "
        "when she thought about CATEGORIES."
    ),
    "rosch_story_2": "The problem with a ITEM is that he really did not like CATEGORIES.",
    "rosch_story_neg": "Are you deluded? A ITEM is clearly not a kind of CATEGORY.",
    "rosch_typ_frame_1": "A ITEM is a typical CATEGORY.",
    "rosch_typ_frame_2": "A ITEM is a good example of a CATEGORY.",
    "rosch_typ_frame_3": "A ITEM is an unusual CATEGORY.",
    "rosch_hedge_1": "Technically, a ITEM is a CATEGORY.",
    "rosch_hedge_2": "Strictly speaking, a ITEM is a CATEGORY.",
    "rosch_hedge_3": "A ITEM is a CATEGORY, sort of.",
    "rosch_cat_first_1": "One kind of CATEGORY is a ITEM.",
    "rosch_cat_first_2": "An example of a CATEGORY is a ITEM.",
}


# Evaluation templates. Probes are never trained on these prompts; they are
# only used to evaluate probes trained on TEMPLATES.
def prompt_eval_1(item, cat):
    subject = capitalise_first(item.with_determiner())
    return f"{subject} {item.verb_be()} a kind of {cat.bare}."


def prompt_eval_2(item, cat):
    return (
        f"When you think about {cat.general}, "
        f"what comes to mind first for me is {item.with_determiner()}."
    )


def prompt_eval_3(item, cat):
    return (
        "Slow down there, I do not think that "
        f"{item.with_determiner()} {item.verb_be()} {cat.with_determiner}, "
        "do you think that."
    )


TEMPLATES = {
    "rosch_example": prompt_example,
    "rosch_neg": prompt_neg,
    "rosch_simple": prompt_simple,
    "rosch_fav": prompt_fav,
    "rosch_story_1": prompt_story_1,
    "rosch_story_2": prompt_story_2,
    "rosch_story_neg": prompt_story_neg,
    "rosch_typ_frame_1": prompt_typ_frame_1,
    "rosch_typ_frame_2": prompt_typ_frame_2,
    "rosch_typ_frame_3": prompt_typ_frame_3,
    "rosch_hedge_1": prompt_hedge_1,
    "rosch_hedge_2": prompt_hedge_2,
    "rosch_hedge_3": prompt_hedge_3,
    "rosch_cat_first_1": prompt_cat_first_1,
    "rosch_cat_first_2": prompt_cat_first_2,
}


EVAL_PROMPT_FORMATS = {
    "rosch_eval_1": "A ITEM is a kind of CATEGORY.",
    "rosch_eval_2": (
        "When you think about CATEGORIES, what comes to mind first for me is a ITEM."
    ),
    "rosch_eval_3": (
        "Slow down there, I do not think that a ITEM is a CATEGORY, do you think that."
    ),
}

EVAL_TEMPLATES = {
    "rosch_eval_1": prompt_eval_1,
    "rosch_eval_2": prompt_eval_2,
    "rosch_eval_3": prompt_eval_3,
}


def read_ratings(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_prompts(rows):
    """Return one row per item, with one column of prompts per template."""
    prompt_rows = []
    for row in rows:
        item = item_phrase(row["item"], row["category"])
        cat = CATEGORY_FORMS[row["category"]]
        prompt_row = {
            "item": row["item"],
            "category": row["category"],
            "rating": row["rating"],
            "rating_zscore": row["rating_zscore"],
            "item_text": item.text,
            "is_ambiguous_strict": is_ambiguous(row["item"], row["category"]),
            "is_ambiguous_lenient": is_ambiguous_lenient(row["item"], row["category"]),
        }
        for name, template in TEMPLATES.items():
            prompt_row[name] = template(item, cat)
        prompt_rows.append(prompt_row)
    return prompt_rows


def build_mean_cat_contrast_prompts(rows, templates=None):
    """Return one row per item and category, for all categories.

    templates: {column name: template}. The default is TEMPLATES.

    The item keeps the grammar of the category it was rated in
    (`item_category`). `category` is the category in the prompt.
    `is_item_category` is true when the two are the same. `rating` and
    `rating_zscore` are None when `is_item_category` is false.
    `is_rated_member` is true when the raw data has a rating for the item in
    `category`, for example "tomato" in both "fruit" and "vegetable".
    `is_ambiguous_strict` and `is_ambiguous_lenient` are for the item in
    `item_category` (see AMBIGUOUS_ITEMS), so they are the same for all rows of
    the item.
    """
    if templates is None:
        templates = TEMPLATES
    rated_pairs = set()
    for row in rows:
        rated_pairs.add((row["item"].lower(), row["category"]))

    prompt_rows = []
    for row in rows:
        item = item_phrase(row["item"], row["category"])
        for category, cat in CATEGORY_FORMS.items():
            is_item_category = category == row["category"]
            rating = None
            rating_zscore = None
            if is_item_category:
                rating = row["rating"]
                rating_zscore = row["rating_zscore"]
            prompt_row = {
                "item": row["item"],
                "item_category": row["category"],
                "rating": rating,
                "rating_zscore": rating_zscore,
                "item_text": item.text,
                "category": category,
                "is_item_category": is_item_category,
                "is_rated_member": (row["item"].lower(), category) in rated_pairs,
                "is_ambiguous_strict": is_ambiguous(row["item"], row["category"]),
                "is_ambiguous_lenient": is_ambiguous_lenient(
                    row["item"], row["category"]
                ),
            }
            for name, template in templates.items():
                prompt_row[name] = template(item, cat)
            prompt_rows.append(prompt_row)
    return prompt_rows


def build_mean_item_contrast_prompts(rows, seed, templates=None):
    """Return 10 rows per item-category pair: the pair, and 9 contrast items.

    templates: {column name: template}. The default is TEMPLATES.

    The category stays the same in all rows of a group. A group is all rows of
    one (`item`, `item_category`) pair. `category` is always `item_category`.
    The anchor row has `is_anchor` true and `prompt_item` equal to `item`.
    Each contrast row has one random item from each of the other 9 categories
    (`prompt_item`, `prompt_item_category`). A contrast item is never rated in
    `category` ("tomato" is never a contrast item for "vegetable"), and the
    item words in a group are all different.

    `item_text` is the text of `prompt_item` in the prompt. The prompt item
    keeps the grammar of the category it was rated in. `rating` and
    `rating_zscore` are for the anchor pair, and are None when `is_anchor` is
    false (a contrast item has no rating in `category`).
    `is_ambiguous_strict` and `is_ambiguous_lenient` are for the anchor pair,
    so they are the same for all rows of the group. The
    `prompt_item_is_ambiguous_*` columns are for the prompt item in
    `prompt_item_category`.

    seed: seed for the random choice of contrast items.
    """
    if templates is None:
        templates = TEMPLATES
    rng = random.Random(seed)
    rated_pairs = set()
    rows_by_category = {}
    for row in rows:
        rated_pairs.add((row["item"].lower(), row["category"]))
        rows_by_category.setdefault(row["category"], []).append(row)

    prompt_rows = []
    for anchor in rows:
        category = anchor["category"]
        group = [anchor]
        used_words = {anchor["item"].lower()}
        for other_category in CATEGORY_FORMS:
            if other_category == category:
                continue
            candidates = []
            for row in rows_by_category[other_category]:
                word = row["item"].lower()
                if (word, category) in rated_pairs:
                    continue
                if word in used_words:
                    continue
                candidates.append(row)
            choice = rng.choice(candidates)
            used_words.add(choice["item"].lower())
            group.append(choice)

        cat = CATEGORY_FORMS[category]
        for prompt_row_source in group:
            prompt_item = prompt_row_source["item"]
            prompt_item_category = prompt_row_source["category"]
            item = item_phrase(prompt_item, prompt_item_category)
            is_anchor = prompt_row_source is anchor
            rating = None
            rating_zscore = None
            if is_anchor:
                rating = anchor["rating"]
                rating_zscore = anchor["rating_zscore"]
            prompt_row = {
                "item": anchor["item"],
                "item_category": category,
                "rating": rating,
                "rating_zscore": rating_zscore,
                "category": category,
                "prompt_item": prompt_item,
                "prompt_item_category": prompt_item_category,
                "item_text": item.text,
                "is_anchor": is_anchor,
                "is_ambiguous_strict": is_ambiguous(anchor["item"], category),
                "is_ambiguous_lenient": is_ambiguous_lenient(anchor["item"], category),
                "prompt_item_is_ambiguous_strict": is_ambiguous(
                    prompt_item, prompt_item_category
                ),
                "prompt_item_is_ambiguous_lenient": is_ambiguous_lenient(
                    prompt_item, prompt_item_category
                ),
            }
            for name, template in templates.items():
                prompt_row[name] = template(item, cat)
            prompt_rows.append(prompt_row)
    return prompt_rows


def build_eval_prompts(rows):
    """Return the evaluation prompts, in the layout of the mean contrast file.

    All categories are included, so the mean contrast baseline can be applied.
    For no baseline, use the rows where is_item_category is true.
    """
    return build_mean_cat_contrast_prompts(rows, EVAL_TEMPLATES)


def write_prompts(prompt_rows, path):
    with open(path, "w", newline="") as f:
        # Quote all fields, so prompts with and without commas look the same.
        writer = csv.DictWriter(
            f, fieldnames=list(prompt_rows[0].keys()), quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(prompt_rows)


def write_suite(ratings_path, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prompts(build_prompts(read_ratings(ratings_path)), path)
    return path


def write_mean_cat_contrast_suite(ratings_path, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prompts(build_mean_cat_contrast_prompts(read_ratings(ratings_path)), path)
    return path


def write_mean_item_contrast_suite(ratings_path, out_path, seed):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_ratings(ratings_path)
    write_prompts(build_mean_item_contrast_prompts(rows, seed), path)
    return path


def write_mean_item_contrast_eval_suite(ratings_path, out_path, seed):
    """The item contrast layout with the evaluation templates.

    The random contrast items do not depend on the templates, so with the same
    seed they are the same as in the item contrast file of TEMPLATES.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_ratings(ratings_path)
    write_prompts(build_mean_item_contrast_prompts(rows, seed, EVAL_TEMPLATES), path)
    return path


def write_eval_suite(ratings_path, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prompts(build_eval_prompts(read_ratings(ratings_path)), path)
    return path
