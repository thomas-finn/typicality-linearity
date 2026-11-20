## Creating the Rosch suite of prompts

The Rosch dataset is stored in `data/raw/rosch1975_ratings.csv` and contains ten categories with items rated between 1 (highly typical) and 7 (highly atypical) by human evaluators, with the score normalised to a z-score to ease comparison between categories.

Here in `data/processed/rosch/` we create a suite of prompts that use the item-category pairs from Rosch.
All prompts are in one file, `rosch_prompts.csv`. Each name in the table below is a column in that file.
This suite of prompts will allow us to extract activations at intermediate layers of LLMs to see how typicality may be encoded.

| Name | Prompt format |
|:-|:-|
| rosch_example | "A ITEM is an example of a CATEGORY." |
| rosch_neg | "A ITEM is not an example of a CATEGORY." |
| rosch_simple | "ITEM: CATEGORY." |
| rosch_fav | "A ITEM is my favourite kind of CATEGORY." |
| rosch_story_1 | "When she looked back at her younger years, a ITEM always came to mind when she thought about CATEGORY." |
| rosch_story_2 | "The problem with ITEM is that he really did not like CATEGORY." |
| rosch_story_neg | "Are you deluded, a ITEM is clearly not a kind of CATEGORY." |
| rosch_typ_frame_1 | "A ITEM is a typical CATEGORY." |
| rosch_typ_frame_2 | "A ITEM is a good example of a CATEGORY." |
| rosch_typ_frame_3 | "A ITEM is an unusual CATEGORY." |
| rosch_hedge_1 | "Technically, a ITEM is a CATEGORY." |
| rosch_hedge_2 | "Strictly speaking, a ITEM is a CATEGORY." |
| rosch_hedge_3 | "A ITEM is a CATEGORY, sort of." |
| rosch_cat_first_1 | "One kind of CATEGORY is a ITEM." |
| rosch_cat_first_2 | "An example of a CATEGORY is a ITEM." |

**Notes**
 - The prompts are created by `scripts/make_rosch_prompts.py`, which uses `src/typicality_linearity/rosch_prompts.py`. Run it from the repository root with `uv run python scripts/make_rosch_prompts.py`.
 - `rosch_prompts.csv` has one row per item-category pair. The columns are `item`, `category`, `rating`, `rating_zscore` (copied from the raw data), `item_text` (the item as written in the prompts), `is_ambiguous_extended`, `is_ambiguous_core`, then one column for each prompt format in the table.
 - `is_ambiguous_extended` is true when the item word has a second meaning that is not the rated meaning and that can interfere ("bat" in bird, "lark" in bird). Weak members with the same meaning ("car" as a toy) are not ambiguous. 41 items are true: the 34 core items, plus lark (bird), ring (clothing), cabinet, chest, ottoman and stool (furniture), club (weapon). The list is `AMBIGUOUS_EXTENDED` in `rosch_prompts.py`. `AMBIGUOUS_ITEMS` is the list of 72 candidate items from the first review, with their second meanings; the flags do not use it directly.
 - `is_ambiguous_core` is a subset of `is_ambiguous_extended`. It is true only for the items where the second meaning is common, also in a category context. 34 items are true (28 words; a word in the list is core ambiguous in all its categories, for example squash in sport, vegetable and fruit). The list is `AMBIGUOUS_CORE` in `rosch_prompts.py`.
 - History: before 2026-11-12 this column was `is_ambiguous_lenient` (with config option `exclude_lenient`). On 2026-11-10, 5 items were moved to strict only (lark, ring, orange, lounge, bolt): 44 → 39 items. On 2026-11-12, after a manual review, the column was renamed to `is_ambiguous_core` and the list was replaced: turkey (bird) was added, and swallow (bird), slip and tie (clothing), swing and top (toy), green (vegetable) were removed: 39 → 34 items. Experiment runs before these dates use the old flags. On 2026-11-12, `is_ambiguous_strict` (72 items, all of `AMBIGUOUS_ITEMS`; config option `exclude_strict`) was replaced by `is_ambiguous_extended` (41 items; config option `exclude_extended`).
 - Each prompt may require alterations to make the sentence grammatically correct. This may change the sentence format slightly but being grammatically correct is important.
 - Grammar changes: items get "a"/"an" by default. Mass nouns ("clay", "rice"), plural items ("marbles", with "are") and sports ("tennis") get no article. Lemmatised plurals in the raw data are restored ("pant" → "pants", "plier" → "pliers"), and a few spellings are normalised ("Atom bomb" → "atom bomb", "teargas" → "tear gas"). `clothing` and `furniture` are mass nouns ("an example of clothing"). The full list is `ITEM_OVERRIDES` in `rosch_prompts.py`.
 - `rosch_typ_frame_1` and `rosch_typ_frame_3` put the adjective before the category ("a typical toy", "an unusual toy"). For the mass categories there is no article ("typical clothing", "unusual furniture").
 - `rosch_cat_first_1` and `rosch_cat_first_2` put the category before the item, so the item (not the category) is the last word before the period.
 - `rosch_story_neg` uses "Are you deluded? A ITEM is…" (question mark, not a comma). `rosch_story_1` and `rosch_story_2` use the general form of the category ("toys", "fruit", "furniture").

## Mean category contrast prompts

In `rosch_mean_cat_contrast.csv` we have the same prompts as in `rosch_prompts.csv` but for each item we have prompts for all the categories. This facilitates the mean contrast baseline experiments.

 - `scripts/make_rosch_prompts.py` creates this file together with `rosch_prompts.csv`.
 - One row per raw item-category pair and prompt category: 565 × 10 = 5650 rows.
 - Columns: `item`, `item_category` (the category the item was rated in), `rating`, `rating_zscore` (for `item` in `item_category`; empty when `is_item_category` is false), `item_text`, `category` (the category in the prompt), `is_item_category`, `is_rated_member`, `is_ambiguous_extended`, `is_ambiguous_core` (for `item` in `item_category`, the same for all rows of the item), then one column for each prompt format.
 - The item keeps the grammar from its rated category ("Pants are an example of a bird.").
 - Some items are rated in more than one category ("tomato" in fruit and vegetable, "car" in toy, vehicle and weapon). These items have one set of rows for each rated category. `is_rated_member` is true when the raw data has a rating for the item in the prompt category, so you can exclude these rows from the "non-member" side of a contrast if necessary.
 - Some shared items have different senses ("bat" as bird and toy, "crane" as bird and tool, "nut" as tool and fruit).

## Mean item contrast prompts

In `rosch_mean_item_contrast.csv` the category is fixed and the item changes. For each item-category pair there is the prompt for the pair (the anchor) and 9 prompts with contrast items that are not members of the category. This facilitates a mean contrast over items for each prompt format.

 - `scripts/make_rosch_prompts.py` creates this file together with the other files. The contrast items are random, with `--seed` (default 0).
 - One group per raw item-category pair, 10 rows per group: 565 × 10 = 5650 rows.
 - Each contrast item comes from a different one of the other 9 categories, so all other categories have the same weight in the contrast mean.
 - A contrast item is never rated in the anchor category ("tomato" is never a contrast item for "vegetable"). The item words in a group are all different.
 - Ambiguous items can be contrast items. Use the `prompt_item_is_ambiguous_*` columns to filter them if necessary.
 - Columns: `item`, `item_category` (the anchor pair; the group key), `rating`, `rating_zscore` (for the anchor pair; empty when `is_anchor` is false), `category` (the category in the prompt, always `item_category`), `prompt_item`, `prompt_item_category` (the item in the prompt and the category it was rated in), `item_text` (the prompt item as written in the prompt), `is_anchor` (true when `prompt_item` is `item`), `is_ambiguous_extended`, `is_ambiguous_core` (for the anchor pair), `prompt_item_is_ambiguous_extended`, `prompt_item_is_ambiguous_core` (for the prompt item in `prompt_item_category`), then one column for each prompt format.
 - The prompt item keeps the grammar from its rated category ("Pants are an example of a bird.").
 - Each prompt in this file is also a row in `rosch_mean_cat_contrast.csv` (that file has every item with every category). The activation cache is keyed on the prompt text, so cached activations can be used again.

## Evaluation prompts

`rosch_eval.csv` has prompt formats that are never used to train probes. They are only used to evaluate probes trained on the formats above, to test transfer to new prompt formats.

| Name | Prompt format |
|:-|:-|
| rosch_eval_1 | "A ITEM is a kind of CATEGORY." |
| rosch_eval_2 | "When you think about CATEGORIES, what comes to mind first for me is a ITEM." |
| rosch_eval_3 | "Slow down there, I do not think that a ITEM is a CATEGORY, do you think that." |

 - `scripts/make_rosch_prompts.py` creates this file together with the other files.
 - The rows and the columns before the prompts are the same as in `rosch_mean_cat_contrast.csv` (5650 rows, all categories), so the mean contrast baseline can be applied. For no baseline, use the rows where `is_item_category` is true (565 rows).
 - The grammar rules are the same as above. `rosch_eval_2` uses the general form of the category ("toys"). `rosch_eval_3` uses the category with its determiner ("a toy", "clothing") and "are" for plural items ("pants are clothing").
 - `rosch_eval_3` ends with a period, as given, not a question mark, so the period token is the final token, as in the other formats.
 - The templates are `EVAL_TEMPLATES` in `rosch_prompts.py`. They are separate from `TEMPLATES`, so the training code does not use them.

## Item contrast evaluation prompts

Removed on 2026-11-20. `rosch_mean_item_contrast_eval.csv` had the layout of `rosch_mean_item_contrast.csv` (5650 rows, 10 per pair) with the evaluation templates (`rosch_eval_1` to `rosch_eval_3`). The evaluation prompts use only the raw activations, which come from the target rows of `rosch_eval.csv`, so this file is not necessary.
 - Each prompt is also a row in `rosch_eval.csv`, so its activations can be derived from the cache (see `activation_extraction.py`, `derive_from`).

## Extended mean category contrast

`rosch_mean_cat_contrast_extended.csv` is `rosch_mean_cat_contrast.csv` with 10 more categories, so the mean over the other categories covers more of the category space. The 10 rated categories are all concrete (7 artifact or activity categories, fruit, vegetable, bird); the new categories add other animals, plants, artifacts, food, people and abstract categories.

- New categories (`EXTENDED_CATEGORY_FORMS` in `rosch_prompts.py`): fish, insect, flower, tree, musical instrument, building, drink, occupation, emotion, illness. All are count nouns ("an example of an occupation", "thought about illnesses", "Marbles are typical fish."). None is a supercategory of a rated category (as "animal" is of "bird"), and few rated items are members of them.
- 11,300 rows: for each of the 565 rated pairs, the 10 rows of `rosch_mean_cat_contrast.csv` (the same values, in the same order), then one row for each new category. The 15 training prompt formats.
- Columns: the columns of `rosch_mean_cat_contrast.csv`, and:
  - `is_extended_category`: true for the 10 new categories.
  - `is_plausible_member`: true when the item is a member of the new category in a second meaning (`PLAUSIBLE_EXTENDED_MEMBERS`, chosen by hand on 2026-11-18): skate (fish), cricket (insect), dandelion (flower), drum and piano (musical instrument), squash (drink; rated in 3 categories, so 3 rows). 8 rows. False for the rated categories; use `is_rated_member` for them.
- The new categories have no ratings: `rating` and `rating_zscore` are empty, and `is_item_category` and `is_rated_member` are false.
- With the mean contrast baseline (`baselines.py`), the target row is the same as in `rosch_mean_cat_contrast.csv`, and the mean is over 19 categories in place of 9.
- No evaluation version yet.
- Activations: the rows of the 10 rated categories are copied from the cached entries of `rosch_mean_cat_contrast.csv` (`extend_from` in the activations config, see `activation_extraction.py`); only the rows of the new categories are extracted with the model.
- Made by `scripts/make_rosch_prompts.py` (`build_mean_cat_contrast_extended_prompts`).

## Nonce category prompts

`rosch_nonce.csv` has the same 565 items and the same 15 prompt formats as `rosch_prompts.csv`, but the category in each prompt is a nonce word (a made-up word with no meaning). There are 3 nonce words, "wugawrar", "njonjobura" and "duplambi", and one row for each item and word (1,695 rows, in the order of `rosch_prompts.csv`, then the word order above). Use it to see what part of the activations comes from the item alone, without a real category.

- Example: "A doll is an example of a wugawrar.", "The problem with a doll is that he really did not like njonjoburas.", "One kind of duplambi is a doll."
- The nonce words are count nouns in all prompts: "a wugawrar" after "an example of", "wugawrars" for the general form (`rosch_story_1`, `rosch_story_2`), "a typical wugawrar". This is also true for the rows of the mass categories clothing and furniture.
- The item keeps the grammar of its rated category ("Tennis is an example of a wugawrar.", "Marbles are my favourite kind of wugawrar.").
- After a plural item, the category after "are" is plural, so the verb and the noun agree: "Marbles are typical wugawrars.", "Technically, pants are wugawrars.", "Marbles are wugawrars, sort of." (`rosch_typ_frame_1`, `rosch_typ_frame_3`, `rosch_hedge_1` to `rosch_hedge_3`; 28 plural items, 84 rows). "An example of" and "a kind of" keep the singular ("Marbles are an example of a wugawrar."). Before 2026-11-18 these prompts had "are a wugawrar". The real category prompts have the same rule (see "Plural items" below).
- Columns: `item`, `category` (the rated category; it is not in the prompts), `nonce_category` (the nonce word in the prompts), `item_text`, `is_ambiguous_extended`, `is_ambiguous_core`, then one column for each prompt format. There are no `rating` and `rating_zscore` columns; join on `item` and `category` to get them from `rosch_prompts.csv`.
- The evaluation formats (`rosch_eval_*`) are not in this file, because they are not in `rosch_prompts.csv`. Until 2026-11-20 they were in `rosch_nonce_eval.csv`. That file is removed: the evaluation prompts use only the raw activations, so they need no nonce prompts.
- Made by `scripts/make_rosch_prompts.py` (`build_nonce_prompts` in `rosch_prompts.py`).

## Plural items

After a plural item, a count category is plural, so the verb and the noun agree (`category_predicate` in `rosch_prompts.py`): "Marbles are typical toys.", "Marbles are unusual toys.", "Technically, pliers are tools.", "Marbles are toys, sort of.", "...I do not think that marbles are toys, do you think that." This is in `rosch_typ_frame_1`, `rosch_typ_frame_3`, `rosch_hedge_1` to `rosch_hedge_3` and `rosch_eval_3`, in all files. A mass category does not change ("Pants are typical clothing."), and "an example of" and "a kind of" keep the singular ("Marbles are an example of a toy.").

Before 2026-11-18, these prompts had a singular category ("Marbles are a typical toy."). Rows changed: `rosch_prompts.csv` 21 per column, `rosch_mean_cat_contrast.csv` 224, `rosch_mean_item_contrast.csv` 235, `rosch_eval.csv` 224, `rosch_mean_item_contrast_eval.csv` 235, the nonce files 84. Activations and results made before this change used the old prompts.
