# PCA by activation type: Qwen2.5-1.5B

## Question

How is typicality visible in the first principal components of the residual stream at the period token, for four activation types (see `../../README.md`, "PCA plots")? How many components explain 95% of the variance, by category, prompt format and layer?

## Dataset

- `data/processed/rosch/rosch_mean_cat_contrast.csv`: raw activations (the 565 target rows) and the category contrast.
- `data/processed/rosch/rosch_mean_item_contrast.csv`: the item contrast (9 contrast items per pair, one from each other category, seed 0).
- The 15 training prompt formats. The 3 evaluation formats have no item contrast file, so they are not used.
- Activations from the cache made by `experiments/probes/global-direction/activations/qwen2-5-1-5B/` (layers 10, 15, 20, 25, period token).

## Configuration

`config.yaml`. The four activation types, for item i rated in category c:

| Type | Formula |
|:-|:-|
| Raw | a(i, c) |
| Mean category contrast | a(i, c) − E_c′[a(i, c′)], over the other 9 categories |
| Mean item contrast | a(i, c) − E_i′[a(i′, c)], over the 9 contrast items |
| Both contrasts | a(i, c) − E_c′[a(i, c′)] − E_i′[a(i′, c)] |

- One PCA for each type, layer, prompt format and scope. The scope is all 565 points or the points of one category (50 to 60 points). 4 × 4 × 15 × 11 = 2640 fits.
- For each fit: the projections on 3 components and the smallest number of components for 95% of the variance (`variance_threshold`). A scope with n points has at most n − 1 components that are not zero, so the counts for one category are at most 49 to 59.
- Both contrasts subtracts the item contrast mean from the raw activation, so it does not use the anchor rows of the item contrast file.

## Command

From the repository root:

    uv run python scripts/run_global_pca_rosch.py experiments/probes/global-direction/pca/qwen2-5-1-5B/config.yaml

Outputs in `outputs/pca/global-direction/qwen2-5-1-5B/<run_id>/`: `config.yaml`, `metadata.json`, `variance_components.csv` (one row per fit) and `qwen2-5-1-5B-rosch-pca.html`. A copy of the page goes to `experiments/probes/global-direction/pca/qwen2-5-1-5B-rosch-pca.html` (in `.gitignore`).

The page:
- One row of selectors for both pages: layer, prompt format, category (or all), number of components (2 or 3, page 1 only).
- Page 1: 2×2 grid of PCA projections, one panel per type. Colour is the rating z-score (red typical, blue atypical).
- Page 2: 2×2 grid of heatmaps (rows: all and each category; columns: prompt formats; cell: components for 95%) for the selected layer, with a log colour scale that is the same for all panels and layers. Click a cell to select that category and prompt. Below: a line chart of the count over layers for the selected category and prompt, one line per type.

## Result

Run `20261108T183911Z` (2026-11-08, 54 s, all activations from the cache), status completed. 2640 fits, all counts in [1, n − 1]. Page: 5.9 MB.

Components for 95% of the variance, all categories (565 points), median over the 15 prompt formats [min–max]:

| Type | Layer 10 | Layer 15 | Layer 20 | Layer 25 |
|:-|:-|:-|:-|:-|
| Raw | 109 [89–164] | 114 [89–175] | 92 [79–145] | 135 [104–171] |
| Mean category contrast | 69 [40–168] | 100 [58–197] | 87 [67–151] | 91 [47–160] |
| Mean item contrast | 155 [133–221] | 149 [129–215] | 124 [100–178] | 172 [140–205] |
| Both contrasts | 177 [137–255] | 164 [137–243] | 142 [109–202] | 191 [144–231] |

One category (50 to 60 points), median over 10 categories × 15 prompts: 30 to 39 for all types and layers (raw 34 / 33 / 30 / 37).

PC1 explained ratio, all categories, layer 15, median over prompts: raw 0.19, category contrast 0.22, item contrast 0.18, both 0.20.

The projection plots are not yet analysed.

## Interpretation

Preliminary, from the counts only:

- The item contrast removes the mean of each category (the contrast items are all in the same prompt category). Between-category differences are then mostly gone, and the pooled activations need more components (raw 114 → 149 at layer 15). The category contrast removes the item's mean over categories and needs fewer components at layer 10 (109 → 69).
- Layer 20 needs the fewest components for all four types. Layer 25 needs the most.
- The spread over prompt formats is large (for example raw layer 15: 89 to 175), so the prompt format changes the dimensionality much.
- In one category the counts (30 to 39) are about 60% of the maximum (n − 1 = 49 to 59) for all types. With so few points, the count does not separate the types well. Use the pooled counts or the projections for comparisons.
