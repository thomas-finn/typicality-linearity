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
 - `rosch_prompts.csv` has one row per item-category pair. The columns are `item`, `category`, `rating`, `rating_zscore` (copied from the raw data), `item_text` (the item as written in the prompts), `is_ambiguous_strict`, `is_ambiguous_lenient`, then one column for each prompt format in the table.
 - `is_ambiguous_strict` is true when the item word has a common second meaning that is not the rated meaning ("bat" in bird, "plane" in tool). Weak members with the same meaning ("car" as a toy) are not ambiguous. The list and the second meanings are `AMBIGUOUS_ITEMS` in `rosch_prompts.py`.
 - `is_ambiguous_lenient` is a subset of `is_ambiguous_strict`. It is also false for 28 items where the rated meaning is the most common meaning in a category context ("duck" as a bird, "apple" as a fruit, "stool" as furniture). 44 items are true. The list is `NOT_AMBIGUOUS_LENIENT` in `rosch_prompts.py`.
 - Each prompt may require alterations to make the sentence grammatically correct. This may change the sentence format slightly but being grammatically correct is important.
 - Grammar changes: items get "a"/"an" by default. Mass nouns ("clay", "rice"), plural items ("marbles", with "are") and sports ("tennis") get no article. Lemmatised plurals in the raw data are restored ("pant" → "pants", "plier" → "pliers"), and a few spellings are normalised ("Atom bomb" → "atom bomb", "teargas" → "tear gas"). `clothing` and `furniture` are mass nouns ("an example of clothing"). The full list is `ITEM_OVERRIDES` in `rosch_prompts.py`.
 - `rosch_typ_frame_1` and `rosch_typ_frame_3` put the adjective before the category ("a typical toy", "an unusual toy"). For the mass categories there is no article ("typical clothing", "unusual furniture").
 - `rosch_cat_first_1` and `rosch_cat_first_2` put the category before the item, so the item (not the category) is the last word before the period.
 - `rosch_story_neg` uses "Are you deluded? A ITEM is…" (question mark, not a comma). `rosch_story_1` and `rosch_story_2` use the general form of the category ("toys", "fruit", "furniture").

## Mean category contrast prompts

In `rosch_mean_cat_contrast.csv` we have the same prompts as in `rosch_prompts.csv` but for each item we have prompts for all the categories. This facilitates the mean contrast baseline experiments.

 - `scripts/make_rosch_prompts.py` creates this file together with `rosch_prompts.csv`.
 - One row per raw item-category pair and prompt category: 565 × 10 = 5650 rows.
 - Columns: `item`, `item_category` (the category the item was rated in), `rating`, `rating_zscore` (for `item` in `item_category`; empty when `is_item_category` is false), `item_text`, `category` (the category in the prompt), `is_item_category`, `is_rated_member`, `is_ambiguous_strict`, `is_ambiguous_lenient` (for `item` in `item_category`, the same for all rows of the item), then one column for each prompt format.
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
 - Columns: `item`, `item_category` (the anchor pair; the group key), `rating`, `rating_zscore` (for the anchor pair; empty when `is_anchor` is false), `category` (the category in the prompt, always `item_category`), `prompt_item`, `prompt_item_category` (the item in the prompt and the category it was rated in), `item_text` (the prompt item as written in the prompt), `is_anchor` (true when `prompt_item` is `item`), `is_ambiguous_strict`, `is_ambiguous_lenient` (for the anchor pair), `prompt_item_is_ambiguous_strict`, `prompt_item_is_ambiguous_lenient` (for the prompt item in `prompt_item_category`), then one column for each prompt format.
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

`rosch_mean_item_contrast_eval.csv` has the layout of `rosch_mean_item_contrast.csv` (5650 rows, 10 per pair) with the evaluation templates (`rosch_eval_1` to `rosch_eval_3`) in place of the training templates.

 - `scripts/make_rosch_prompts.py` creates this file together with the other files, with the same `--seed` (default 0).
 - The contrast items are the same as in `rosch_mean_item_contrast.csv`: the random choice does not depend on the templates.
 - Each prompt is also a row in `rosch_eval.csv`, so its activations can be derived from the cache (see `activation_extraction.py`, `derive_from`).
