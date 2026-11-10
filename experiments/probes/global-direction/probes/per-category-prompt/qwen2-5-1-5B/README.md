# Probes per category and prompt: Qwen2.5-1.5B

## Question

How well do difference-of-means probes at the period token grade typicality within a category, and how do they transfer between categories and between prompt formats? How similar are the probe directions? How do the four activation types, the layer, the removal of ambiguous items and the size of the typicality groups change this? (See `../../../README.md`, "Probing per category and prompts".)

## Dataset

- `data/processed/rosch/rosch_mean_cat_contrast.csv` (raw activations from the 565 target rows, and the category contrast) and `data/processed/rosch/rosch_mean_item_contrast.csv` (item contrast, seed 0).
- The 15 training prompt formats. The 3 evaluation formats have no item contrast file, so they are not used.
- Activations from the cache made by `experiments/probes/global-direction/activations/qwen2-5-1-5B/` (layers 10, 15, 20, 25, period token).
- Ambiguous items: 72 strict, 44 lenient (see `data/processed/rosch/README.md`).

## Configuration

`config.yaml`. One setting is (ambiguity, activation type, layer, groups): 3 × 4 × 4 × 3 = 144 settings.

- Activation types: raw, mean category contrast, mean item contrast, both contrasts (formulas in `../../../pca/qwen2-5-1-5B/README.md`).
- Probe: the mean of the most typical 1/n of the training items minus the mean of the least typical 1/n, n = 2 (halves), 3 (thirds), 4 (quarters). Typicality = −rating z-score. Score = x · direction.
- Ambiguity: include all, exclude lenient, exclude strict. Excluded items are removed from training and evaluation, and the folds are made again on the other items (seed 0).
- 5 folds in each category, stratified by typicality. The folds are the same for all prompts and activation types.
- Metrics: Spearman, Pearson, and the Wilcoxon rank-sum AUC (most typical 1/n against least typical 1/n of the eval items, with the same n as the probe; 0.5 is chance).
- Cross-validated cells: the out-of-fold scores are z-scored inside each fold before the metric (`cv_aggregate: fold_zscored`). This removes the offset that each fold probe adds; pooled scores are biased low (see `experiments/probes/rosch/qwen2-5-1-5B-ridge/README.md`).

Matrices for each setting:

| Page | Matrix | Rows × columns | Cells |
|:-|:-|:-|:-|
| 1 | Transfer between categories, per prompt and metric | train × eval category (10 × 10) | off-diagonal: probe on all items of the train category; diagonal: CV |
| 2 | Cosine between categories, per prompt | 10 × 10 | probe directions on all items |
| 3 | Transfer between prompts, per category and metric | train × eval prompt (15 × 15) | all cells CV with the same folds, so a probe is never evaluated on its training items |
| 4 | Cosine between prompts, per category | 15 × 15 | probe directions on all items |

## Command

From the repository root:

    uv run python scripts/run_global_probes_rosch.py experiments/probes/global-direction/probes/per-category-prompt/qwen2-5-1-5B/config.yaml

Outputs in `outputs/probes/global-direction/per-category-prompt/qwen2-5-1-5B/<run_id>/`:
- `summary.csv`: one row per setting and metric, with the mean diagonal (CV) and off-diagonal cells of each matrix
- `matrices.npz` and `axes.json`: all matrices, with the order of each axis
- `qwen2-5-1-5B-probes-01.html`, and a copy in `experiments/probes/global-direction/probes/per-category-prompt/` (in `.gitignore`)

The page has four pages (tabs). Each is a 2×2 grid of heatmaps, one per activation type, with the selectors of the README: layer, ambiguous items, difference of means (halves/thirds/quarters), prompt format, and metric (transfer pages); pages 3 and 4 also have the category, and outline the selected prompt format. The colour scale is diverging around chance, with the same range for all settings of a page and metric.

Pages 2 and 4 also have a 2×2 grid of dendrograms under the heatmaps, one per activation type. Each is average linkage (UPGMA) on the distance 1 − cosine of the heatmap above (scipy). The cosine heatmap above each tree has its rows and columns in the same leaf order, so the order changes with the selectors. Each panel gives the cophenetic correlation and the root height. The height axis is the same for all settings of a page. The trees add about 0.6 MB to the page.

## Result

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

Layer 15, category contrast: between categories 0.40 / 0.39 / 0.42 (halves / thirds / quarters, include all) and 0.43 to 0.46 with ambiguous items removed. Category CV 0.55 to 0.56 (include) and 0.58 to 0.60 (exclude). Cosine between categories increases from 0.27 (halves) to 0.34 (quarters).

Category CV by category (layer 15, thirds, include, category contrast): weapon 0.83, vehicle 0.78, clothing 0.76, sport 0.70, fruit 0.60, furniture 0.58, toy 0.57, bird 0.48, vegetable 0.25, tool 0.07. By prompt: 0.43 (`rosch_simple`) and 0.46 (`rosch_fav`) to 0.66 (`rosch_cat_first_1`).

## Interpretation

Preliminary, from the summary only (the page is not yet analysed):

- **The category contrast is the best activation type.** It increases the transfer between categories most (0.28 → 0.39 at layer 15) and makes the category directions more similar (cosine 0.20 → 0.29). The item contrast alone does not help (0.26) and makes the both-contrasts type a little worse than the category contrast alone. Within a category, all types are similar (0.51 to 0.56).
- **Layer 15 is best for transfer between categories**; layer 10 is best for transfer between prompts with the category contrast (0.47). For raw activations, layer 25 has the lowest transfer between categories (0.17) but the most similar directions between prompts (cosine 0.48).
- **Removing ambiguous items helps a little** (category contrast, between categories 0.39 → 0.45 to 0.46). Strict and lenient give almost the same values, so the 28 items that are only in the strict list do not change much.
- **Quarters transfer a little better than halves** between categories, with more similar directions, and the same CV.
- **Tool (0.07) and vegetable (0.25) have almost no signal within the category**, as in the earlier experiments. `rosch_simple` and `rosch_fav` are the weakest prompt formats; the category-first prompts (item as last word) are the strongest.
