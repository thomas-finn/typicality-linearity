"""Build the Rosch (1975) suite of prompts from the item-category ratings.

Each prompt template needs a grammatical noun phrase for the item and the
category. Most items are singular count nouns ("a doll"). The exceptions
(mass nouns, plurals, sports, lemmatised plurals such as "pant") are listed
in ITEM_OVERRIDES.
"""

import csv
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


TEMPLATES = {
    "rosch_example": prompt_example,
    "rosch_neg": prompt_neg,
    "rosch_simple": prompt_simple,
    "rosch_fav": prompt_fav,
    "rosch_story_1": prompt_story_1,
    "rosch_story_2": prompt_story_2,
    "rosch_story_neg": prompt_story_neg,
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
        }
        for name, template in TEMPLATES.items():
            prompt_row[name] = template(item, cat)
        prompt_rows.append(prompt_row)
    return prompt_rows


def build_mean_contrast_prompts(rows):
    """Return one row per item and category, for all categories.

    The item keeps the grammar of the category it was rated in
    (`item_category`). `category` is the category in the prompt.
    `is_item_category` is true when the two are the same. `rating` and
    `rating_zscore` are None when `is_item_category` is false.
    `is_rated_member` is true when the raw data has a rating for the item in
    `category`, for example "tomato" in both "fruit" and "vegetable".
    """
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
            }
            for name, template in TEMPLATES.items():
                prompt_row[name] = template(item, cat)
            prompt_rows.append(prompt_row)
    return prompt_rows


def write_prompts(prompt_rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prompt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prompt_rows)


def write_suite(ratings_path, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prompts(build_prompts(read_ratings(ratings_path)), path)
    return path


def write_mean_contrast_suite(ratings_path, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prompts(build_mean_contrast_prompts(read_ratings(ratings_path)), path)
    return path
