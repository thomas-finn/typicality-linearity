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

**Notes**
 - The prompts are created by `scripts/make_rosch_prompts.py`, which uses `src/typicality_linearity/rosch_prompts.py`. Run it from the repository root with `uv run python scripts/make_rosch_prompts.py`.
 - `rosch_prompts.csv` has one row per item-category pair. The columns are `item`, `category`, `rating`, `rating_zscore` (copied from the raw data), `item_text` (the item as written in the prompts), then one column for each prompt format in the table.
 - Each prompt may require alterations to make the sentence grammatically correct. This may change the sentence format slightly but being grammatically correct is important.
 - Grammar changes: items get "a"/"an" by default. Mass nouns ("clay", "rice"), plural items ("marbles", with "are") and sports ("tennis") get no article. Lemmatised plurals in the raw data are restored ("pant" → "pants", "plier" → "pliers"), and a few spellings are normalised ("Atom bomb" → "atom bomb", "teargas" → "tear gas"). `clothing` and `furniture` are mass nouns ("an example of clothing"). The full list is `ITEM_OVERRIDES` in `rosch_prompts.py`.
 - `rosch_story_neg` uses "Are you deluded? A ITEM is…" (question mark, not a comma). `rosch_story_1` and `rosch_story_2` use the general form of the category ("toys", "fruit", "furniture").

## Mean contrast prompts

In `rosch_mean_contrast.csv` we have the same prompts as in `rosch_prompts.csv` but for each item we have prompts for all the categories. This facilitates the mean contrast baseline experiments.

 - `scripts/make_rosch_prompts.py` creates this file together with `rosch_prompts.csv`.
 - One row per raw item-category pair and prompt category: 565 × 10 = 5650 rows.
 - Columns: `item`, `item_category` (the category the item was rated in), `rating`, `rating_zscore` (for `item` in `item_category`; empty when `is_item_category` is false), `item_text`, `category` (the category in the prompt), `is_item_category`, `is_rated_member`, then one column for each prompt format.
 - The item keeps the grammar from its rated category ("Pants are an example of a bird.").
 - Some items are rated in more than one category ("tomato" in fruit and vegetable, "car" in toy, vehicle and weapon). These items have one set of rows for each rated category. `is_rated_member` is true when the raw data has a rating for the item in the prompt category, so you can exclude these rows from the "non-member" side of a contrast if necessary.
 - Some shared items have different senses ("bat" as bird and toy, "crane" as bird and tool, "nut" as tool and fruit).
