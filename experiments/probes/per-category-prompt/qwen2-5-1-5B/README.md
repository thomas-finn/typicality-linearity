# Probes per category and prompt: Qwen2.5-1.5B

## Question

How well do difference-of-means probes at the period token grade typicality within a category, and how do they transfer between categories and between prompt formats? How similar are the probe directions? How do the four activation types, the layer, the removal of ambiguous items and the size of the typicality groups change this? (See `../../../README.md`, "Probing per category and prompts".)

## Dataset

- `data/processed/rosch/rosch_mean_cat_contrast.csv` (raw activations from the 565 target rows, and the category contrast) and `data/processed/rosch/rosch_mean_item_contrast.csv` (item contrast, seed 0).
- The 15 training prompt formats. The 3 evaluation formats have no item contrast file, so they are not used.
- Activations from the cache made by `experiments/activations/qwen2-5-1-5B/` (layers 10, 15, 20, 25, period token).
- Ambiguous items: 34 core, 41 extended (see `data/processed/rosch/README.md`). Runs before 2026-11-12 used 72 strict and 44 (later 39) lenient items.

## Configuration

`config.yaml`. One setting is (ambiguity, activation type, layer, groups): 3 × 4 × 4 × 3 = 144 settings.

- Activation types: raw, mean category contrast, mean item contrast, both contrasts (formulas in `../../../pca/qwen2-5-1-5B/README.md`).
- Probe: the mean of the most typical 1/n of the training items minus the mean of the least typical 1/n, n = 2 (halves), 3 (thirds), 4 (quarters). Typicality = −rating z-score. Score = x · direction.
- Ambiguity: include all, exclude core, exclude extended. Excluded items are removed from training and evaluation, and the folds are made again on the other items (seed 0).
- 5 folds in each category, stratified by typicality. The folds are the same for all prompts and activation types.
- Metrics: Spearman, Pearson, and the Wilcoxon rank-sum AUC (most typical 1/n against least typical 1/n of the eval items, with the same n as the probe; 0.5 is chance).
- Cross-validated cells: the out-of-fold scores are z-scored inside each fold before the metric (`cv_aggregate: fold_zscored`). This removes the offset that each fold probe adds; pooled scores are biased low (see `experiments/probes/rosch/qwen2-5-1-5B-ridge/README.md`).
- `diagonal: full` (added 2026-11-18): the diagonal cells of the transfer matrices use the probe trained on all items, evaluated on the same items (in-sample, not cross-validated). The off-diagonal cells do not change. The results below are from runs before this change, with a CV diagonal (`diagonal: cv`, the default when the key is not set).

Matrices for each setting:

| Page | Matrix | Rows × columns | Cells |
|:-|:-|:-|:-|
| 1 | Transfer between categories, per prompt and metric | train × eval category (10 × 10) | off-diagonal: probe on all items of the train category; diagonal: CV, or in-sample with `diagonal: full` |
| 2 | Cosine between categories, per prompt | 10 × 10 | probe directions on all items |
| 3 | Transfer between prompts, per category and metric | train × eval prompt (15 × 15) | off-diagonal: CV with the same folds, so a probe is never evaluated on its training items; diagonal: CV, or in-sample with `diagonal: full` |
| 4 | Cosine between prompts, per category | 15 × 15 | probe directions on all items |


**Scoring change (2026-11-20).** Baselines are now used only to train the probes. All scores (transfer cells, held-out items, in-sample diagonals and evaluation prompts) are x_raw · direction. Before, a probe was scored on activations of its own type (for example a category contrast probe on category contrast activations). Run `20261120T150505Z` is the first run with raw scoring (see Result). The older results are kept under it for comparison.

## Command

From the repository root:

    uv run python scripts/run_global_probes_rosch.py experiments/probes/per-category-prompt/qwen2-5-1-5B/config.yaml

Outputs in `outputs/probes/global-direction/per-category-prompt/qwen2-5-1-5B/<run_id>/`:
- `summary.csv`: one row per setting and metric, with the mean diagonal and off-diagonal cells (columns `category_within`, `prompt_within`; before 2026-11-18 `category_within_cv`, `prompt_within_cv`) of each matrix. From 2026-11-18 the column `within_method` is `cv` or `in_sample` (with `diagonal: full`), so the method of the within columns is in the file
- `matrices.npz` and `axes.json`: all matrices, with the order of each axis
- `qwen2-5-1-5B-probes-01.html`, and a copy in `experiments/probes/per-category-prompt/` (in `.gitignore`)

The page has four pages (tabs). Each is a 2×2 grid of heatmaps, one per activation type, with the selectors of the README: layer, ambiguous items, difference of means (halves/thirds/quarters), prompt format, and metric (transfer pages); pages 3 and 4 also have the category, and outline the selected prompt format. The colour scale is diverging around chance, with the same range for all settings of a page and metric.

Pages 2 and 4 also have a 2×2 grid of dendrograms under the heatmaps, one per activation type. Each is average linkage (UPGMA) on the distance 1 − cosine of the heatmap above (scipy). The cosine heatmap above each tree has its rows and columns in the same leaf order, so the order changes with the selectors. Each panel gives the cophenetic correlation and the root height. The height axis is the same for all settings of a page. The trees add about 0.6 MB to the page.

## Result

### Raw scoring (2026-11-20)

Run `20261120T150505Z` (2026-11-20, local CPU, all activations from the cache), status completed. The same config as `20261118T161202Z`, with all scores on raw activations. Run `20261120T145218Z` is the same command, run a second time by mistake; its summary is the same (max difference 0). Checks against `20261118T161202Z`: the raw rows and all cosine columns are the same (max difference 0), as expected.

L15, thirds, include, Spearman. Matched scoring (before) → raw scoring:

| Activation type (training) | Category within | Between categories | Between prompts |
|:-|:-|:-|:-|
| raw | 0.673 → 0.673 | 0.277 → 0.277 | 0.405 → 0.405 |
| cat contrast | 0.698 → 0.632 | 0.392 → 0.363 | 0.443 → 0.361 |
| item contrast | 0.671 → 0.662 | 0.256 → 0.271 | 0.387 → 0.402 |
| both contrasts | 0.704 → 0.611 | 0.357 → 0.344 | 0.422 → 0.353 |

Cat contrast between categories without ambiguous items: 0.388 (exclude core, before 0.420), 0.401 (exclude extended, before 0.435).

Include, thirds, Spearman, raw scoring, layers 10 / 15 / 20 / 25:
- Between categories: raw 0.19 / 0.28 / 0.25 / 0.17; cat contrast 0.30 / 0.36 / 0.31 / 0.22; item contrast 0.18 / 0.27 / 0.24 / 0.16; both 0.27 / 0.34 / 0.29 / 0.20.
- Between prompts: raw 0.40 / 0.41 / 0.33 / 0.45; cat contrast 0.35 / 0.36 / 0.26 / 0.22; item contrast 0.39 / 0.40 / 0.32 / 0.44; both 0.34 / 0.35 / 0.26 / 0.20.

The interpretation below is for the matched scoring. See "Interpretation with raw scoring" for this run.

### Matched scoring (before 2026-11-20)

Run `20261118T161202Z` (2026-11-18, 3 min 5 s, local CPU, all activations from the cache, model not loaded), status completed. First run with the plural agreement fix of the prompts (see "Plural items" in `data/processed/rosch/README.md`) and `diagonal: full` (`within_method` = `in_sample`). Compared with run `20261118T112304Z` (the same config, old prompts): the largest difference over all settings and summary columns is 0.003 (mean 0.0005). The fix does not change the results.

L15, thirds, include, Spearman (category within is in-sample):

| Activation type | Category within | Between categories | Between prompts | Cosine between categories | Cosine between prompts |
|:-|:-|:-|:-|:-|:-|
| raw | 0.673 | 0.277 | 0.405 | 0.198 | 0.255 |
| cat contrast | 0.698 | 0.392 | 0.443 | 0.292 | 0.241 |
| item contrast | 0.671 | 0.256 | 0.387 | 0.184 | 0.253 |
| both contrasts | 0.704 | 0.357 | 0.422 | 0.267 | 0.243 |

Cat contrast between categories without ambiguous items: 0.420 (exclude core), 0.435 (exclude extended).

Run `20261112T151002Z` (2026-11-12, 3 min 14 s, local CPU, all activations from the cache, model not loaded), status completed, 144 settings, no NaN values. Page: 13.8 MB. Rerun after the change to the ambiguous items: the ambiguity settings are now `exclude_core` (34 items, `is_ambiguous_core`) and `exclude_extended` (41 items, `is_ambiguous_extended`). They replace `exclude_lenient` (39 items) and `exclude_strict` (72 items); see `data/processed/rosch/README.md`. Items: 565 / 531 / 524 (include / exclude core / exclude extended). The include values are the same as in the earlier run (max difference 0). Git `88c73a3` with uncommitted changes (`dirty: true`): the ambiguity change. The values for exclude core and exclude extended below are from this run. All other values do not change.

Run `20261109T142816Z` (2026-11-09, completed) re-made the page with the dendrograms on pages 2 and 4, and the cosine heatmaps sorted by dendrogram leaf order (run `20261109T122137Z` had the dendrograms only). Its matrices are the same as run `20261108T192615Z`, so the results below do not change. Page: 13.8 MB, in `outputs/probes/global-direction/per-category-prompt/qwen2-5-1-5B/` and copied to `../qwen2-5-1-5B-probes-01.html`. Mean cophenetic correlation (layer 15, thirds, include all): category trees 0.83 (raw), 0.92 (category contrast), 0.83 (item contrast), 0.89 (both); prompt trees 0.85 to 0.86 for all types.

Run `20261108T192615Z` (2026-11-08, 3 min 11 s, all activations from the cache), status completed. This run is from before the restructure: its outputs are in `outputs/probes/global-direction/qwen2-5-1-5B/`, and its page was named `qwen2-5-1-5B-probes.html` (the copy is now `../qwen2-5-1-5B-probes-01.html`). The results do not change. 144 settings, no NaN values. Items: 565 (include), 521 (exclude lenient), 493 (exclude strict). Page: 13 MB.

Means over all prompts and categories (Spearman, fold z-scored CV). Columns: category CV (diagonal of the category matrix), between categories (off-diagonal), between prompts (off-diagonal of the prompt matrix, CV), and the mean off-diagonal cosine between categories and between prompts.

Layer 15, thirds, include all:

| Type | Category CV | Between categories | Between prompts | Cosine categories | Cosine prompts |
|:-|-:|-:|-:|-:|-:|
| Raw | 0.52 | 0.28 | 0.41 | 0.20 | 0.25 |
| Category contrast | 0.56 | 0.39 | 0.44 | 0.29 | 0.24 |
| Item contrast | 0.51 | 0.26 | 0.39 | 0.18 | 0.25 |
| Both contrasts | 0.56 | 0.36 | 0.42 | 0.27 | 0.24 |

Wilcoxon AUC, the same setting: category CV 0.82 / 0.84 / 0.82 / 0.84, between categories 0.67 / 0.74 / 0.66 / 0.72.

By layer (thirds, include all), category CV / between categories / between prompts:

| Type | L10 | L15 | L20 | L25 |
|:-|:-|:-|:-|:-|
| Raw | 0.46 / 0.19 / 0.40 | 0.52 / 0.28 / 0.41 | 0.49 / 0.25 / 0.33 | 0.52 / 0.17 / 0.45 |
| Category contrast | 0.53 / 0.37 / 0.47 | 0.56 / 0.39 / 0.44 | 0.56 / 0.35 / 0.35 | 0.54 / 0.28 / 0.33 |

Layer 15, category contrast: between categories 0.40 / 0.39 / 0.42 (halves / thirds / quarters, include all) and 0.41 to 0.43 (exclude core) and 0.42 to 0.45 (exclude extended); before: 0.43 to 0.46 (lenient and strict). Category CV 0.55 to 0.56 (include) and 0.57 to 0.59 (exclude). Cosine between categories increases from 0.27 (halves) to 0.34 (quarters).

Category CV by category (layer 15, thirds, include, category contrast): weapon 0.83, vehicle 0.78, clothing 0.76, sport 0.70, fruit 0.60, furniture 0.58, toy 0.57, bird 0.48, vegetable 0.25, tool 0.07. By prompt: 0.43 (`rosch_simple`) and 0.46 (`rosch_fav`) to 0.66 (`rosch_cat_first_1`).

## Interpretation

### Interpretation with raw scoring

Preliminary, from the summary only:

- **A category contrast direction still transfers better between categories on raw activations** (0.36 against 0.28 for a raw-trained probe at layer 15, and higher at all layers). So training on the category contrast finds a direction that reads typicality from a single prompt in other categories better than training on raw. About a quarter of the earlier gain (0.39 against 0.28) came from the subtraction at evaluation.
- **Between prompts, the category contrast direction is now worse than raw** (0.36 against 0.41 at layer 15; 0.22 against 0.45 at layer 25). The earlier gain between prompts came from the subtraction at evaluation. The category contrast direction probably has a part that cancels in a(i, c) − a(i, c′) but changes between prompt formats on raw activations.
- **The within-category value falls with the category contrast** (0.70 → 0.63, in-sample). Without the subtraction, the item-level variation that the contrast removed is noise along the direction.
- **The item contrast now equals raw or is a little better** (between categories 0.27 against 0.28, between prompts 0.40 against 0.41). This agrees with the expectation: its baseline is almost constant inside a category, so the scoring change removes only the noise of the random contrast items.
- **Both contrasts behave like the category contrast**, a little lower.

### Interpretation with matched scoring (before 2026-11-20)

Preliminary, from the summary only (the page is not yet analysed):

- **The category contrast is the best activation type.** It increases the transfer between categories most (0.28 → 0.39 at layer 15) and makes the category directions more similar (cosine 0.20 → 0.29). The item contrast alone does not help (0.26) and makes the both-contrasts type a little worse than the category contrast alone. Within a category, all types are similar (0.51 to 0.56).
- **Layer 15 is best for transfer between categories**; layer 10 is best for transfer between prompts with the category contrast (0.47). For raw activations, layer 25 has the lowest transfer between categories (0.17) but the most similar directions between prompts (cosine 0.48).
- **Removing ambiguous items helps a little** (category contrast, between categories 0.39 → 0.42 core, 0.44 extended; before 0.45 to 0.46 with lenient and strict). The 7 items that are only in the extended list add +0.01 to +0.02. The smaller gain than before shows that a part of the earlier gain came from items that are not in the new lists.
- **Quarters transfer a little better than halves** between categories, with more similar directions, and the same CV.
- **Tool (0.07) and vegetable (0.25) have almost no signal within the category**, as in the earlier experiments. `rosch_simple` and `rosch_fav` are the weakest prompt formats; the category-first prompts (item as last word) are the strongest.
