# Probes pooled over categories and over prompts: Qwen2.5-1.5B

## Question

Does one difference-of-means probe trained on all categories together grade typicality inside each category, and does it transfer between prompt formats? Does one probe per category trained on all prompt formats together transfer between categories? How do the four activation types, the layer, the removal of ambiguous items and the size of the typicality groups change this? (See `../../../README.md`, "Probing over categories and prompts".)

## Dataset

The same as `../../per-category-prompt/qwen2-5-1-5B/`: the 565 rated pairs from `data/processed/rosch/rosch_mean_cat_contrast.csv` and `data/processed/rosch/rosch_mean_item_contrast.csv` (seed 0), the 15 training prompt formats, and the cached activations (layers 10, 15, 20, 25, period token).

Evaluation prompts (page 1 only, never used for training): `rosch_eval_1` to `rosch_eval_3` from `data/processed/rosch/rosch_eval.csv` (raw and category contrast) and `data/processed/rosch/rosch_mean_item_contrast_eval.csv` (item contrast; the same contrast items as the training file, seed 0). The item contrast evaluation activations are derived from the `rosch_eval.csv` cache entries (activation run `20261108T202641Z`).

## Configuration

`config.yaml`. The settings, folds, metrics and CV aggregation are the same as in the per-category experiment: 3 ambiguity settings × 4 activation types × 4 layers × 3 group sizes = 144 settings; 5 folds in each category (seed 0); Spearman, Pearson, Wilcoxon AUC; CV scores z-scored in each held-out fold.

| Page | Probe | Matrix | Cells | Value |
|:-|:-|:-|:-|:-|
| 1, pooled over categories | one per prompt, trained on all categories | train prompt × eval prompt (15 × 15, then 3 evaluation prompt columns) | all CV with the same folds | metric in each category, mean over the 10 categories |
| 2, pooled over prompts | one per category, trained on all 15 prompts | train × eval category (10 × 10) | off-diagonal: probe on all items of the train category; diagonal: CV | metric in each prompt, mean over the 15 prompts |
| 3, pooled over categories and prompts | one per setting, trained on all items of all categories in all 15 prompts | scatter: typicality (−rating z) against the probe score of each item, per evaluation prompt | not CV: all items are training items, in a new prompt format | metric in each category, mean over the 10 categories |

- Page 1 groups (`over_categories_groups: per_category`): the most and least typical 1/n of the training items of each category, put together. Each category has the same number of items in the top and bottom group, so an offset between categories cancels. With `all` (the most and least typical 1/n of all training items), a category with more items in one group puts its offset into the direction: in a synthetic test with an offset of 50 this reverses the direction. On the real data the two give almost the same value (raw, layer 15, thirds, 3 prompts: 0.447 against 0.437), because typicality is a z-score in each category.
- Page 2 direction: the mean over the 15 prompts of (mean of the top group − mean of the bottom group), with the groups chosen on the items. This is the difference of means of all (item, prompt) rows of the category.
- Page 3: the direction is the mean over the 15 training prompts of the page 1 direction (groups per category). The scores are raw (x · direction, with the offsets between categories) or z-scored in each category; a selector on the page changes between them. A second selector highlights one category. Because the probe is trained on the same items (in other prompt formats), its values are a little higher than cross-validated values. `matrices.npz`: `global_scores_raw`, `global_scores_z` (NaN for excluded items) and `global_eval_metrics`; `summary.csv`: `global_eval_prompts`.
- Evaluation prompt columns: each fold probe of the train prompt is evaluated on its held-out items in the evaluation prompt, as for the other columns. On the page they are right of a line, and each panel shows their mean.
- The value is computed inside a category (page 1) or inside a prompt (page 2), so a direction does not score well only because it orders the categories or the prompts.
- `matrices.npz` has the evaluation columns in `over_categories_eval` (and `over_categories_eval_by_category`), and also the values before the mean (`over_categories_by_category`, `over_prompts_by_prompt`) and the cosine between the directions trained on all items (`over_categories_cosine` between prompts, `over_prompts_cosine` between categories). The page does not show them.

## Command

From the repository root:

    uv run python scripts/run_global_pooled_probes_rosch.py experiments/probes/global-direction/probes/over-categories-prompts/qwen2-5-1-5B/config.yaml

Outputs in `outputs/probes/global-direction/over-categories-prompts/qwen2-5-1-5B/<run_id>/`: `config.yaml`, `metadata.json`, `summary.csv` (one row per setting and metric, with the mean diagonal and off-diagonal cells, and `over_categories_eval_prompts`: the mean of the evaluation columns), `matrices.npz` and `axes.json`, and `qwen2-5-1-5B-probes-02.html`, with a copy in `experiments/probes/global-direction/probes/over-categories-prompts/` (in `.gitignore`).

The page has three pages (tabs). Pages 1 and 2 are 2×2 grids of heatmaps, one per activation type, with the selectors layer, ambiguous items, difference of means (halves/thirds/quarters) and metric. Page 3 is a 2×2 grid of scatter plots with the selectors evaluation prompt, layer, ambiguous items, difference of means, score (z-scored in each category or raw) and highlighted category.

## Result

Run `20261108T195834Z` (2026-11-08, 3 min 12 s, all activations from the cache), status completed. This run is from before the evaluation prompt columns were added; its values for the other cells do not change.

Evaluation columns: run `20261108T202948Z` (2026-11-08, 3 min 36 s), status completed, no NaN values. All other cells are the same as in run `20261108T195834Z`. Page: 1.0 MB.

Pooled over categories, evaluation prompts (thirds, include all, Spearman, mean over the 15 train prompts). In brackets: the mean transfer between the training prompts (off-diagonal).

| Layer | Type | `rosch_eval_1` | `rosch_eval_2` | `rosch_eval_3` | Mean |
|:-|:-|-:|-:|-:|-:|
| 10 | Raw | 0.41 | 0.49 | 0.25 | 0.38 (0.37) |
| 10 | Category contrast | 0.46 | 0.60 | 0.48 | 0.51 (0.46) |
| 15 | Raw | 0.47 | 0.55 | 0.27 | 0.43 (0.41) |
| 15 | Category contrast | 0.44 | 0.57 | 0.46 | 0.49 (0.44) |
| 15 | Item contrast | 0.43 | 0.51 | 0.25 | 0.39 (0.38) |
| 15 | Both contrasts | 0.41 | 0.53 | 0.41 | 0.45 (0.41) |
| 20 | Category contrast | 0.35 | 0.44 | 0.14 | 0.31 (0.31) |
| 25 | Category contrast | 0.30 | 0.44 | 0.21 | 0.32 (0.26) |

Page 3 (pooled over categories and prompts): run `20261108T204617Z` (2026-11-08, 3 min 41 s), status completed, no NaN values; all other arrays are the same as in run `20261108T202948Z`. Page: 4.3 MB.

Thirds, include all, Spearman in each category, mean. In brackets: the mean over the evaluation prompts of the page 1 probes (one per training prompt, cross-validated).

| Layer | Type | `rosch_eval_1` | `rosch_eval_2` | `rosch_eval_3` | Mean (page 1, CV) |
|:-|:-|-:|-:|-:|-:|
| 10 | Raw | 0.46 | 0.59 | 0.34 | 0.46 (0.38) |
| 10 | Category contrast | 0.47 | 0.66 | 0.51 | 0.55 (0.51) |
| 15 | Raw | 0.56 | 0.70 | 0.39 | 0.55 (0.43) |
| 15 | Category contrast | 0.47 | 0.70 | 0.55 | 0.57 (0.49) |
| 15 | Item contrast | 0.51 | 0.66 | 0.36 | 0.51 (0.39) |
| 15 | Both contrasts | 0.45 | 0.67 | 0.49 | 0.54 (0.45) |
| 20 | Raw | 0.59 | 0.69 | 0.28 | 0.52 (0.32) |
| 20 | Category contrast | 0.46 | 0.61 | 0.30 | 0.45 (0.31) |
| 25 | Raw | 0.67 | 0.67 | 0.42 | 0.59 (0.37) |
| 25 | Category contrast | 0.50 | 0.61 | 0.33 | 0.48 (0.32) |

Category contrast, layer 15, thirds: AUC 0.77 / 0.91 / 0.83; mean 0.57 (include), 0.61 (exclude lenient and strict). By category on `rosch_eval_2`: clothing 0.87, weapon 0.86, furniture 0.85, vehicle 0.80, sport and fruit 0.77, vegetable 0.71, toy 0.62, tool 0.37, bird 0.32. Raw activations, raw score (with the category offsets), Spearman over all 565 items: 0.48 / 0.62 / 0.23.

Category contrast, layer 15, thirds: Wilcoxon AUC 0.76 / 0.85 / 0.78. Mean over the evaluation prompts 0.49 (include), 0.52 (exclude lenient), 0.53 (exclude strict); halves / thirds / quarters 0.49 / 0.49 / 0.50. By train prompt: best `rosch_typ_frame_2`, `rosch_example`, `rosch_typ_frame_1` (0.55 to 0.56); worst `rosch_story_neg` (0.41) and `rosch_neg` (0.36).

Layer 15, thirds, include all, Spearman. In brackets: the same statistic from the per-category experiment (`../../per-category-prompt/qwen2-5-1-5B/`, run `20261108T192615Z`), with one probe per category and prompt.

| Type | Pooled over categories: within prompt (CV) | between prompts | Pooled over prompts: within category (CV) | between categories | Cosine between prompts | Cosine between categories |
|:-|-:|-:|-:|-:|-:|-:|
| Raw | 0.53 | 0.41 (0.41) | 0.52 | 0.30 (0.28) | 0.32 | 0.26 |
| Category contrast | 0.53 | 0.44 (0.44) | 0.55 | 0.41 (0.39) | 0.35 | 0.38 |
| Item contrast | 0.50 | 0.38 (0.39) | 0.51 | 0.26 (0.26) | 0.31 | 0.23 |
| Both contrasts | 0.51 | 0.41 (0.42) | 0.54 | 0.37 (0.36) | 0.34 | 0.34 |

By layer (thirds, include all), pooled over categories between prompts / pooled over prompts between categories:

| Type | L10 | L15 | L20 | L25 |
|:-|:-|:-|:-|:-|
| Raw | 0.37 / 0.20 | 0.41 / 0.30 | 0.30 / 0.25 | 0.34 / 0.15 |
| Category contrast | 0.46 / 0.40 | 0.44 / 0.41 | 0.31 / 0.30 | 0.26 / 0.21 |
| Item contrast | 0.35 / 0.18 | 0.38 / 0.26 | 0.27 / 0.21 | 0.30 / 0.12 |
| Both contrasts | 0.42 / 0.35 | 0.41 / 0.37 | 0.28 / 0.27 | 0.22 / 0.18 |

Category contrast, layer 15: removing ambiguous items increases both values from 0.41 to 0.44 to 0.46 to 0.48. The group size changes them by 0.03 or less.

Category contrast, layer 15, thirds, include all, by category: pooled over prompts, within-category CV: weapon 0.81, vehicle 0.77, clothing 0.73, sport 0.70, fruit 0.58, toy 0.57, furniture 0.57, bird 0.49, vegetable 0.27, tool −0.02. Pooled over categories, between prompts: weapon 0.75, vehicle 0.66, clothing 0.64, furniture 0.53, sport 0.49, toy 0.48, fruit 0.34, vegetable 0.24, bird 0.17, tool 0.11.

## Interpretation

Preliminary, from the summary only (the page is not yet analysed):

- **One probe for all categories grades typicality inside the categories almost as well as one probe per category** (within prompt, CV: category contrast 0.53 pooled against 0.56 per category; raw 0.53 against 0.52). Its transfer between prompts is the same (0.44 and 0.44). So a large part of the typicality direction is shared by the categories.
- **Pooling over prompts gives a small increase in the transfer between categories** (category contrast 0.39 → 0.41, raw 0.28 → 0.30) and more similar category directions (cosine 0.29 → 0.38 with category contrast).
- **The category contrast is again the best type**, and the item contrast does not help. Layers 10 and 15 are best; with the contrasts, the transfer falls at layers 20 and 25.
- **Removing ambiguous items helps** (0.41 → 0.47 with category contrast); strict and lenient are the same.
- **The pooled probe transfers to the evaluation prompts as well as between the training prompts** (category contrast, layer 15: 0.49 against 0.44). The evaluation prompts are not harder. `rosch_eval_2` ("what comes to mind first … is ITEM") is the best (0.57 to 0.60), as in the earlier diff-means experiment.
- **The category contrast is needed for the negated evaluation prompt** (`rosch_eval_3`: raw 0.27, category contrast 0.46 at layer 15). Probes trained on the negated training prompts (`rosch_neg`, `rosch_story_neg`) transfer worst to the evaluation prompts.
- **Layers 10 and 15 are best for the evaluation prompts too; at layer 20 `rosch_eval_3` falls to 0.12 to 0.15** for all types.
- **Page 3 values are higher than the cross-validated page 1 values, but a part of this is because the probe saw the items.** The page 3 probe is trained on the same items in 15 other prompt formats. The difference to page 1 is small with the category contrast at layers 10 to 15 (0.55 against 0.51, 0.57 against 0.49), but large for raw activations at layers 20 and 25 (0.52 against 0.32, 0.59 against 0.37). The category contrast removes the part of the activation that is fixed for an item, so it removes most of the item identity that the probe can learn. A large gain for raw activations at late layers is then most probably item identity, not a better typicality direction. A cross-validated version of page 3 (fold probes over all categories and prompts) is necessary to separate the two.
- **With the category contrast at layer 15, one direction for all categories and prompts grades typicality in new prompt formats at 0.57** (AUC up to 0.91 on `rosch_eval_2`), and inside 8 of 10 categories at 0.62 or more on `rosch_eval_2`.
- **Bird is graded well inside the category (0.49) but the probe pooled over categories grades it badly (0.17)**: the bird direction is different from the shared direction. Tool has no signal in either case.
