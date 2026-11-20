# Probes per category and prompt: Qwen2.5-7B

## Question

The same question as `../qwen2-5-1-5B/README.md`, for Qwen2.5-7B: how well do difference-of-means probes at the period token grade typicality within a category, and how do they transfer between categories and between prompt formats? How do the activation type, layer, ambiguity and group size change this? How do the results compare with Qwen2.5-1.5B?

## Dataset

The same as `../qwen2-5-1-5B/README.md` (565 target rows, 15 training prompt formats; 34 core and 41 extended ambiguous items; runs before 2026-11-12: 72 strict and 44 lenient). Activations from the cache made by `experiments/activations/qwen2-5-7B/` (`outputs/activation_cache/Qwen2.5-7B/`, layers 10, 15, 20, 25, period token, float32).

## Configuration

`config.yaml`. The same as the 1.5B config, except `experiment`, `title`, `model` (`Qwen/Qwen2.5-7B`), `output_dir` and `html_name`. 144 settings (3 ambiguity × 4 types × 4 layers × 3 group sizes), 5 folds, `cv_aggregate: fold_zscored`, seed 0. See `../qwen2-5-1-5B/README.md` for the probes, metrics and the four matrices.

- `diagonal: full` (added 2026-11-18): the diagonal cells of the transfer matrices use the probe trained on all items, evaluated on the same items (in-sample, not cross-validated). The off-diagonal cells do not change. The results below are from runs before this change, with a CV diagonal (`diagonal: cv`, the default when the key is not set).


**Scoring change (2026-11-20).** Baselines are now used only to train the probes. All scores (transfer cells, held-out items, in-sample diagonals and evaluation prompts) are x_raw · direction. Before, a probe was scored on activations of its own type (for example a category contrast probe on category contrast activations). The results below are from runs before this change. Run the experiment again to get results with raw scoring. The raw rows do not change.

## Command

From the repository root:

    uv run python scripts/run_global_probes_rosch.py experiments/probes/per-category-prompt/qwen2-5-7B/config.yaml

Outputs in `outputs/probes/global-direction/per-category-prompt/qwen2-5-7B/<run_id>/`: `summary.csv`, `matrices.npz`, `axes.json` and `qwen2-5-7B-probes-01.html`. A copy of the page goes to `experiments/probes/per-category-prompt/qwen2-5-7B-probes-01.html` (in `.gitignore`). The page has the same format as the 1.5B page (four tabs, dendrograms on pages 2 and 4).

## Result

Run `20261112T164255Z` (2026-11-12, 3 min 46 s, local CPU, all activations from the cache), status completed. The same as run `20261112T151316Z` (`summary.csv` is identical), but each heatmap cell now shows its value (2 decimals). The page copy in `experiments/probes/` is from this run.

Run `20261112T151316Z` (2026-11-12, 3 min 42 s, local CPU, all activations from the cache, model not loaded), status completed, 144 settings, no NaN values. Page: 13.7 MB. Rerun after the change to the ambiguous items: the ambiguity settings are now `exclude_core` (34 items, `is_ambiguous_core`) and `exclude_extended` (41 items, `is_ambiguous_extended`). They replace `exclude_lenient` (39 items) and `exclude_strict` (72 items); see `data/processed/rosch/README.md`. Items: 565 / 531 / 524 (include / exclude core / exclude extended). The include values are the same as in the earlier run (max difference 0). Git `88c73a3` with uncommitted changes (`dirty: true`): the ambiguity change. The values for exclude core and exclude extended below are from this run. All other values do not change.

Run `20261110T155341Z` (2026-11-10, 3 min 37 s, local CPU, all activations from the cache), status completed. 144 settings, no NaN values. Items: 565 / 521 / 493 (include / exclude lenient / exclude strict). Page: 13.7 MB. Git `88c73a3` with uncommitted changes (`dirty: true`): the change that puts the activation cache in one subdirectory for each model. This change does not change the activations or the probes.

Means over all prompts and categories (Spearman, fold z-scored CV), layer 15, thirds, include all. In brackets: 1.5B.

| Type | Category CV | Between categories | Between prompts | Cosine categories | Cosine prompts |
|:-|-:|-:|-:|-:|-:|
| Raw | 0.60 (0.52) | 0.37 (0.28) | 0.48 (0.41) | 0.24 (0.20) | 0.22 (0.25) |
| Category contrast | 0.63 (0.56) | 0.43 (0.39) | 0.50 (0.44) | 0.27 (0.29) | 0.22 (0.24) |
| Item contrast | 0.58 (0.51) | 0.35 (0.26) | 0.45 (0.39) | 0.23 (0.18) | 0.22 (0.25) |
| Both contrasts | 0.62 (0.56) | 0.41 (0.36) | 0.48 (0.42) | 0.26 (0.27) | 0.22 (0.24) |

Wilcoxon AUC, the same setting: category CV 0.87 / 0.88 / 0.86 / 0.88 (1.5B: 0.82 / 0.84 / 0.82 / 0.84), between categories 0.73 / 0.76 / 0.72 / 0.75 (1.5B: 0.67 / 0.74 / 0.66 / 0.72).

By layer (thirds, include all), category CV / between categories / between prompts:

| Type | L10 | L15 | L20 | L25 |
|:-|:-|:-|:-|:-|
| Raw | 0.54 / 0.29 / 0.50 | 0.60 / 0.37 / 0.48 | 0.58 / 0.35 / 0.37 | 0.57 / 0.25 / 0.46 |
| Category contrast | 0.58 / 0.38 / 0.54 | 0.63 / 0.43 / 0.50 | 0.62 / 0.38 / 0.39 | 0.59 / 0.29 / 0.36 |
| Item contrast | 0.52 / 0.27 / 0.48 | 0.58 / 0.35 / 0.45 | 0.55 / 0.32 / 0.34 | 0.55 / 0.22 / 0.44 |
| Both contrasts | 0.57 / 0.36 / 0.52 | 0.62 / 0.41 / 0.48 | 0.60 / 0.36 / 0.36 | 0.57 / 0.26 / 0.33 |

Layer 15, category contrast: between categories 0.44 / 0.43 / 0.45 (halves / thirds / quarters, include all) and 0.45 to 0.48 with ambiguous items removed (core 0.45 / 0.46 / 0.48, extended 0.45 / 0.47 / 0.48; before: 0.46 to 0.49) (1.5B: 0.40 / 0.39 / 0.42 and 0.41 to 0.45). Category CV 0.62 to 0.63 (include) and 0.64 to 0.65 (exclude). Cosine between categories 0.27 (halves) to 0.31 (quarters). Raw layer 25, cosine between prompts: 0.38 (1.5B: 0.48).

Category CV by category (layer 15, thirds, include, category contrast): weapon 0.86, vehicle 0.83, clothing 0.77, sport 0.72, furniture 0.70, toy 0.69, fruit 0.68, bird 0.57, vegetable 0.32, tool 0.13. By prompt: 0.47 (`rosch_simple`) and 0.56 (`rosch_fav`) to 0.68 (`rosch_hedge_1`, `rosch_story_2`).

## Interpretation

Preliminary, from the summary only (the page is not yet analysed):

- **7B is better than 1.5B for all types and layers**: about +0.06 to +0.08 for category CV, and +0.04 to +0.09 between categories. The increase between categories is largest for raw (0.28 → 0.37) and item contrast (0.26 → 0.35), so the gain from the category contrast is smaller for 7B (raw 0.37 → 0.43 against 0.28 → 0.39).
- **The order of the types is the same**: category contrast > both > raw > item contrast. Layer 15 is best between categories and layer 10 is best between prompts, the same as for 1.5B.
- **The cosines are not larger for 7B** (categories 0.23 to 0.27, prompts 0.22), although the transfer is better. The hidden size is 2.3 times larger, so random directions have a smaller cosine; compare the cosines between models with care.
- **Removing ambiguous items and using quarters help a little**, the same as for 1.5B.
- **Tool (0.13) and vegetable (0.32) are still the weakest categories**, and `rosch_simple` is still the weakest prompt. The category-first prompts are no longer the strongest; `rosch_hedge_1` and `rosch_story_2` are.
