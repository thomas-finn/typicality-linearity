# Probes pooled over categories and over prompts: Qwen2.5-7B

## Question

The same question as `../qwen2-5-1-5B/README.md`, for Qwen2.5-7B: does one difference-of-means probe trained on all categories grade typicality inside each category and transfer between prompt formats and to the evaluation prompts? Does one probe per category trained on all prompt formats transfer between categories? How do the results compare with Qwen2.5-1.5B?

## Dataset

The same as `../qwen2-5-1-5B/README.md`: 565 rated pairs, 15 training prompt formats, and the evaluation prompts `rosch_eval_1` to `rosch_eval_3` (`rosch_eval.csv` and `rosch_mean_item_contrast_eval.csv`). Activations from the cache made by `experiments/activations/qwen2-5-7B/` (run `20261110T145109Z`, `outputs/activation_cache/Qwen2.5-7B/`, layers 10, 15, 20, 25, period token, float32). The item contrast evaluation entries are derived from the `rosch_eval.csv` entries, as for 1.5B.

## Configuration

`config.yaml`. The same as the 1.5B config, except `experiment`, `title`, `model` (`Qwen/Qwen2.5-7B`), `output_dir` and `html_name`. 144 settings, 5 folds, `over_categories_groups: per_category`, `cv_aggregate: fold_zscored`, seed 0. See `../qwen2-5-1-5B/README.md` for the three pages and the probes.

- `diagonal: full` (added 2026-11-18): the diagonal cells of the transfer matrices use the probe trained on all items, evaluated on the same items (in-sample, not cross-validated). The off-diagonal cells do not change. The results below are from runs before this change, with a CV diagonal (`diagonal: cv`, the default when the key is not set).

## Command

From the repository root:

    uv run python scripts/run_global_pooled_probes_rosch.py experiments/probes/over-categories-prompts/qwen2-5-7B/config.yaml

Outputs in `outputs/probes/global-direction/over-categories-prompts/qwen2-5-7B/<run_id>/`: `config.yaml`, `metadata.json`, `summary.csv`, `matrices.npz`, `axes.json` and `qwen2-5-7B-probes-02.html`, with a copy in `experiments/probes/over-categories-prompts/` (in `.gitignore`). The page has the same format as the 1.5B page (three tabs).

## Result

Run `20261112T164642Z` (2026-11-12, 4 min 0 s, local CPU, all activations from the cache), status completed. The same as run `20261112T152036Z` (`summary.csv` is identical), but each heatmap cell now shows its value (2 decimals). The page copy in `experiments/probes/` is from this run.

Run `20261112T152036Z` (2026-11-12, 3 min 59 s, local CPU, all activations from the cache, model not loaded), status completed, 144 settings, no NaN values. Page: 4.4 MB. Rerun after the change to the ambiguous items: the ambiguity settings are now `exclude_core` (34 items, `is_ambiguous_core`) and `exclude_extended` (41 items, `is_ambiguous_extended`). They replace `exclude_lenient` (39 items) and `exclude_strict` (72 items); see `data/processed/rosch/README.md`. Items: 565 / 531 / 524 (include / exclude core / exclude extended). The include values are the same as in the earlier run (max difference 0). Git `88c73a3` with uncommitted changes (`dirty: true`): the ambiguity change. The values for exclude core and exclude extended below are from this run. All other values do not change.

Run `20261110T160250Z` (2026-11-10, 4 min 5 s, local CPU, all activations from the cache), status completed, no NaN values. Page: 4.4 MB. Git `88c73a3` with uncommitted changes (`dirty: true`): the change that puts the activation cache in one subdirectory for each model. This change does not change the activations or the probes.

In brackets in all tables: 1.5B (`../qwen2-5-1-5B/README.md`).

Layer 15, thirds, include all, Spearman:

| Type | Pooled over categories: within prompt (CV) | between prompts | Pooled over prompts: within category (CV) | between categories | Cosine between prompts | Cosine between categories |
|:-|-:|-:|-:|-:|-:|-:|
| Raw | 0.57 (0.53) | 0.44 (0.41) | 0.61 (0.52) | 0.38 (0.30) | 0.29 (0.32) | 0.30 (0.26) |
| Category contrast | 0.58 (0.53) | 0.46 (0.44) | 0.60 (0.55) | 0.40 (0.41) | 0.30 (0.35) | 0.32 (0.38) |
| Item contrast | 0.54 (0.50) | 0.42 (0.38) | 0.59 (0.51) | 0.37 (0.26) | 0.29 (0.31) | 0.29 (0.23) |
| Both contrasts | 0.56 (0.51) | 0.44 (0.41) | 0.60 (0.54) | 0.38 (0.37) | 0.29 (0.34) | 0.30 (0.34) |

The same statistics from the 7B per-category experiment (`../../per-category-prompt/qwen2-5-7B/`, run `20261110T155341Z`), category contrast: category CV 0.63, between prompts 0.50, between categories 0.43. Raw: 0.60 / 0.48 / 0.37.

By layer (thirds, include all), pooled over categories between prompts / pooled over prompts between categories:

| Type | L10 | L15 | L20 | L25 |
|:-|:-|:-|:-|:-|
| Raw | 0.48 / 0.32 (0.37 / 0.20) | 0.44 / 0.38 (0.41 / 0.30) | 0.34 / 0.34 (0.30 / 0.25) | 0.34 / 0.22 (0.34 / 0.15) |
| Category contrast | 0.50 / 0.38 (0.46 / 0.40) | 0.46 / 0.40 (0.44 / 0.41) | 0.33 / 0.32 (0.31 / 0.30) | 0.24 / 0.22 (0.26 / 0.21) |
| Item contrast | 0.45 / 0.30 (0.35 / 0.18) | 0.42 / 0.37 (0.38 / 0.26) | 0.31 / 0.32 (0.27 / 0.21) | 0.31 / 0.18 (0.30 / 0.12) |
| Both contrasts | 0.47 / 0.37 (0.42 / 0.35) | 0.44 / 0.38 (0.41 / 0.37) | 0.30 / 0.30 (0.28 / 0.27) | 0.21 / 0.18 (0.22 / 0.18) |

Category contrast, layer 15: removing ambiguous items increases the values from 0.46 / 0.40 to 0.47 / 0.43 (core) and 0.47 / 0.44 (extended); before: 0.48 / 0.45 to 0.46 (lenient and strict). The group size changes them by 0.04 or less.

Category contrast, layer 15, thirds, include all, by category. Pooled over prompts, within-category CV: weapon 0.85, vehicle 0.82, clothing 0.74, sport 0.71, furniture 0.70, fruit 0.66, toy 0.64, bird 0.50, vegetable 0.36, tool 0.05. Pooled over categories, between prompts: weapon 0.81, vehicle 0.68, clothing 0.66, furniture 0.62, toy 0.55, sport 0.43, fruit 0.42, vegetable 0.28, tool 0.11, bird 0.08.

Pooled over categories, evaluation prompts (page 1, thirds, include all, Spearman, mean over the 15 train prompts). In brackets after the mean: the mean transfer between the training prompts, then 1.5B.

| Layer | Type | `rosch_eval_1` | `rosch_eval_2` | `rosch_eval_3` | Mean (between train; 1.5B mean) |
|:-|:-|-:|-:|-:|-:|
| 10 | Raw | 0.51 | 0.51 | 0.35 | 0.46 (0.48; 0.38) |
| 10 | Category contrast | 0.48 | 0.59 | 0.49 | 0.52 (0.50; 0.51) |
| 15 | Raw | 0.44 | 0.53 | 0.21 | 0.39 (0.44; 0.43) |
| 15 | Category contrast | 0.40 | 0.60 | 0.31 | 0.44 (0.46; 0.49) |
| 15 | Item contrast | 0.39 | 0.50 | 0.19 | 0.36 (0.42; 0.39) |
| 15 | Both contrasts | 0.39 | 0.56 | 0.28 | 0.41 (0.44; 0.45) |
| 20 | Category contrast | 0.24 | 0.43 | 0.13 | 0.27 (0.33; 0.31) |
| 25 | Category contrast | 0.20 | 0.39 | 0.15 | 0.25 (0.24; 0.32) |

Category contrast, layer 15, thirds: Wilcoxon AUC 0.73 / 0.85 / 0.69. Mean over the evaluation prompts 0.44 (include), 0.44 (exclude core and extended; before: 0.45 lenient, 0.46 strict); halves / thirds / quarters 0.44 / 0.44 / 0.43. By train prompt: best `rosch_typ_frame_1`, `rosch_story_2`, `rosch_typ_frame_3` (0.50 to 0.51); worst `rosch_neg` and `rosch_cat_first_2` (0.33).

Page 3 (pooled over categories and prompts, not CV). Thirds, include all, Spearman in each category, mean. In brackets: the page 1 mean over the evaluation prompts (CV), then the 1.5B page 3 mean.

| Layer | Type | `rosch_eval_1` | `rosch_eval_2` | `rosch_eval_3` | Mean (page 1 CV; 1.5B) |
|:-|:-|-:|-:|-:|-:|
| 10 | Raw | 0.56 | 0.59 | 0.47 | 0.54 (0.46; 0.46) |
| 10 | Category contrast | 0.48 | 0.61 | 0.52 | 0.54 (0.52; 0.55) |
| 15 | Raw | 0.59 | 0.65 | 0.31 | 0.52 (0.39; 0.55) |
| 15 | Category contrast | 0.47 | 0.68 | 0.48 | 0.54 (0.44; 0.57) |
| 15 | Item contrast | 0.54 | 0.62 | 0.31 | 0.49 (0.36; 0.51) |
| 15 | Both contrasts | 0.47 | 0.66 | 0.45 | 0.53 (0.41; 0.54) |
| 20 | Raw | 0.37 | 0.63 | 0.24 | 0.41 (0.26; 0.52) |
| 20 | Category contrast | 0.41 | 0.65 | 0.33 | 0.47 (0.27; 0.45) |
| 25 | Raw | 0.28 | 0.62 | 0.32 | 0.41 (0.30; 0.59) |
| 25 | Category contrast | 0.46 | 0.59 | 0.26 | 0.43 (0.25; 0.48) |

Category contrast, layer 15, thirds: AUC 0.77 / 0.89 / 0.80; mean 0.54 (include), 0.55 (exclude core), 0.56 (exclude extended); before: 0.57 (lenient and strict). By category on `rosch_eval_2`: weapon 0.90, vehicle 0.88, furniture 0.84, clothing 0.81, fruit and sport 0.77, vegetable 0.67, toy 0.60, tool 0.38, bird 0.18. Raw activations, raw score (with the category offsets), Spearman over all 565 items: 0.49 / 0.62 / 0.23.

## Interpretation

Preliminary, from the summary only (the page is not yet analysed):

- **On the training prompts, 7B is a little better than 1.5B**, most for raw and item contrast activations (pooled over prompts, between categories: raw 0.30 → 0.38, item contrast 0.26 → 0.37). With the category contrast the gain is small (0.44 → 0.46 between prompts) or zero (0.41 → 0.40 between categories). So for 7B the four activation types are almost the same, and the category contrast is no longer clearly the best type.
- **Pooling does not help for 7B.** For 1.5B, pooling over prompts increased the transfer between categories a little (category contrast 0.39 → 0.41). For 7B it decreases it (0.43 per category and prompt → 0.40 pooled), and the probe pooled over categories transfers less between prompts than the per-category probes (0.46 against 0.50).
- **7B does not transfer better to the evaluation prompts.** At layer 15 it is worse than 1.5B (category contrast 0.44 against 0.49), and it is lower than the transfer between the training prompts (0.44 against 0.46). For 1.5B it was higher (0.49 against 0.44). The difference is mostly `rosch_eval_3`, the negated prompt: category contrast 0.31 against 0.46 for 1.5B, raw 0.21 against 0.27.
- **For 7B, layer 10 is best for the evaluation prompts** (category contrast 0.52, raw 0.46), and the transfer falls quickly after layer 15. At layers 20 and 25 the evaluation transfer is 0.22 to 0.30 for all types.
- **Page 3 (not CV) is a little lower for 7B than for 1.5B** at layer 15 (category contrast 0.54 against 0.57). The difference between page 3 and page 1 is larger for 7B at all layers and for all types (for example category contrast layer 20: 0.47 against 0.27). For 1.5B this difference was large only for raw activations at late layers, which we explained as item identity. For 7B the category contrast does not remove it, so a part of the page 3 value for 7B is probably also item identity (or a direction that the CV folds cannot find with fewer training items). A cross-validated page 3 is necessary to separate the two.
- **Bird is graded worst by the shared direction** (between prompts 0.08, page 3 `rosch_eval_2` 0.18), but well inside the category (0.50), the same as for 1.5B. Tool has no signal in either case.
