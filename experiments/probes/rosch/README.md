# Linear probes on Rosch data

## Layer 15 of Qwen2.5-1.5B

Our goal in this experiment is to train linear probes at layer 15 of Qwen2.5-1.5B to grade the typicality of exemplars in a category. 

### Probing per category and per prompt

We train probes over each category and each prompt in the Rosch dataset. 
Our aim is to investigate how well a probe is able to grade typicality both within that category and prompt format, but also whether that probe generalises to other categories and other prompt formats.

For each category and prompt format we train:
 - A difference-of-means probe: Take the mean of the activations of the top third most typical items and the mean of the activations of the bottom third most typical items and extract a direction from the difference of the means.
 - Ridge regression: Take a heavily regularised ridge regression to fit a linear probe against typicality.

Have a range of plots that have results side-by-side for activations from no-baseline and mean-contrast baselines on the end-of-sentence period token. The results should outline correlations for the probe:
 - How well does the prompt transfer between categories? Have a dropdown menu for the prompt and a matrix of categories on the vertical axis that is what the probe is trained on and the horizontal axis the category that is being evaluated on.
 - How well does the prompt transfer between prompts? Have a dropdown menu for the category and a matrix of the different prompt formats on the vertical that is what the probe was trained on and on the horizontal axis the prompt that is being evaluated on.
 - Have a dropdown for the metric used (spearman or pearson) for plots.
 - Have a dropdown for difference-of-means or ridge regression for plots.
 - For diagonals of plots use cross-validation (5-fold).
 - Cosine similarity between directions across categories and prompt formats.

Results should be in `qwen2-5-1-5B-probes.html`.

### Decisions

- **Typicality target.** Typicality is $-1 \times$ `rating_zscore`. A low Rosch rating is typical, so a higher value is more typical. A positive correlation means that the probe grades typicality in the correct direction. The "most typical third" for difference-of-means is the third with the lowest rating.
- **Folds.** 5 folds in each category, stratified by typicality (sort the items, cut them into blocks of 5, and give each item in a block a different random fold). The same items are in each fold for all prompts. Seed 0.
- **Transfer between categories.** Off-diagonal cell (a, b): the probe trained on all items of category a, evaluated on all items of category b, same prompt. Diagonal cell (a, a): 5-fold cross-validated.
- **Transfer between prompts.** All cells are cross-validated, not only the diagonal. Each fold probe is trained on the training items in prompt A and evaluated on the held-out items in prompt B. The items are the same in all prompts, so a probe could otherwise score well only because it recognises the items. The diagonals of the two matrices are the same numbers.
- **Cross-validated correlation.** Pooled: the correlation of all out-of-fold predictions with typicality. `cv_aggregate: mean_over_folds` gives the mean of the correlations of each fold.
- **Ridge alpha.** Chosen by inner 5-fold CV (smallest squared error) on the training items only, from a log grid of 1 to 1e6 (half decades). Each feature is standardised with the mean and std of the training items. The chosen alphas are in `ridge_alphas.csv`. The number of fits at the edge of the grid is in `metadata.json`.
- **Difference-of-means.** On the raw activations (no standardisation). The score is the projection onto the direction.
- **Baselines.** No baseline and mean contrast, side by side. Both use the 565 target rows of `rosch_mean_contrast.csv`. Mean contrast is $a(i, c) - \text{mean}_{c' \neq c}\, a(i, c')$ (see `src/typicality_linearity/baselines.py`).
- **Cosine similarity.** Between the directions of the probes trained on all items of each category and prompt. Between categories: for each prompt, a category × category matrix. Between prompts: for each category, a prompt × prompt matrix. Directions are in activation space: difference-of-means is the raw direction, and ridge is weights / feature std (the direction that the probe reads in the unstandardised activations). Computed separately for each baseline and probe type. The metric dropdown does not change these plots.
- **Plots.** Fixed diverging colour scale from −1 (red) to 1 (blue). In the category matrix, the cross-validated diagonal cells have an outline. Dropdowns: probe, metric, prompt (category matrix) and category (prompt matrix).
- **Output.** `outputs/probes/rosch/qwen2-5-1-5B/<run_id>/qwen2-5-1-5B-probes.html`. Not committed.

### Dataset

`data/processed/rosch/rosch_mean_contrast.csv`: 565 item-category pairs from Rosch (1975), 10 categories (50 to 60 items each), seven prompt formats, and all 10 prompt categories for each item (5650 rows). See `data/processed/rosch/README.md`.

### Configuration

`qwen2-5-1-5B/config.yaml`. `layers` and `positions` are the extraction settings and are part of the activation cache key. They are the same as the PCA mean-contrast run, so all activations come from `outputs/activation_cache/` and the model is not loaded. `probe_layer: 15` and `probe_position: period` select the activations for the probes.

Code:
- `src/typicality_linearity/probes.py`: probes, folds, metrics
- `src/typicality_linearity/probe_transfer.py`: transfer matrices
- `src/typicality_linearity/probe_plot.py`: HTML page
- `src/typicality_linearity/probe_experiment.py`: run, outputs, metadata
- `tests/test_probes.py`

### Command

From the repository root:

    uv run python scripts/run_probes_rosch.py experiments/probes/rosch/qwen2-5-1-5B/config.yaml

Outputs go to `outputs/probes/rosch/qwen2-5-1-5B/<run_id>/`:
- `metadata.json`, `config.yaml`
- `correlations.csv`: one row per matrix cell (baseline, probe type, metric, matrix, train and eval category and prompt, cross-validated, value)
- `cosine_similarity.csv`: one row per pair of directions (baseline, probe type, matrix, category a and b, prompt a and b, cosine)
- `ridge_alphas.csv`: the alpha of each ridge fit
- `probe_weights.npz`: the probes trained on all items of each category and prompt
- `qwen2-5-1-5B-probes.html`

To compute `cosine_similarity.csv` again from the saved probes of a run, without fitting them again:

    uv run python scripts/probe_cosine_similarity.py outputs/probes/rosch/qwen2-5-1-5B/<run_id>

### Result

Run `20261103T135510Z_probes-rosch-qwen2-5-1-5B` (2026-11-03). Status completed, seed 0. All activations came from the cache. The git commit is not recorded (`commit: null`, `dirty: true`): the working tree had changes that were not committed.

All values below are Spearman correlations. Pearson values are within about 0.02 of them. "Mean" is the mean over the cells of the matrices (10 categories × 7 prompts).

**Within category and prompt (5-fold CV, diagonal cells)**

| Baseline | Diff-of-means | Ridge |
|:-|-:|-:|
| None | 0.36 | 0.60 |
| Mean contrast | 0.49 | 0.62 |

By category (mean over prompts):

| Category | None, DoM | None, ridge | MC, DoM | MC, ridge |
|:-|-:|-:|-:|-:|
| weapon | 0.60 | 0.78 | 0.76 | 0.82 |
| clothing | 0.59 | 0.83 | 0.71 | 0.84 |
| vehicle | 0.57 | 0.75 | 0.72 | 0.81 |
| sport | 0.52 | 0.73 | 0.63 | 0.74 |
| fruit | 0.35 | 0.65 | 0.54 | 0.67 |
| toy | 0.35 | 0.58 | 0.50 | 0.56 |
| furniture | 0.28 | 0.55 | 0.50 | 0.61 |
| bird | 0.27 | 0.57 | 0.36 | 0.58 |
| vegetable | 0.07 | 0.43 | 0.17 | 0.34 |
| tool | 0.01 | 0.09 | 0.05 | 0.19 |

By prompt, the differences are smaller. `rosch_simple` is the lowest for ridge (0.52) and `rosch_fav` is the lowest for difference-of-means with no baseline (0.18). `rosch_story_1` is the highest with mean contrast (DoM 0.57, ridge 0.69).

**Transfer between categories (same prompt, off-diagonal cells)**

| Baseline | Diff-of-means | Ridge | Cells > 0.2 (DoM / ridge) | Cells < 0 (DoM / ridge) |
|:-|-:|-:|-:|-:|
| None | 0.25 | 0.37 | 55% / 74% | 19% / 6% |
| Mean contrast | 0.39 | 0.42 | 75% / 76% | 8% / 5% |

- The evaluation category controls most of the variation. Probes from all categories grade weapon, clothing and vehicle well (MC: 0.55 to 0.64) and grade tool badly (0.06 to 0.13).
- Probes trained on tool transfer badly (MC ridge 0.27, MC DoM 0.13). Probes trained on vegetable transfer well (MC 0.42 to 0.49), although they are weak within vegetable.
- `rosch_fav` gives the lowest transfer (none DoM 0.06, MC DoM 0.22).

**Transfer between prompts (same category, all cells cross-validated)**

| Baseline | Probe | Diagonal | Off-diagonal |
|:-|:-|-:|-:|
| None | DoM | 0.36 | 0.19 |
| None | Ridge | 0.60 | 0.22 |
| Mean contrast | DoM | 0.49 | 0.40 |
| Mean contrast | Ridge | 0.62 | 0.28 |

- With mean contrast, difference-of-means transfers between prompts with a small loss (0.49 to 0.40). Ridge loses more than half of its correlation (0.62 to 0.28).
- Some pairs of prompts transfer better: `rosch_example` ↔ `rosch_neg` ↔ `rosch_story_neg` (for example none ridge, `rosch_neg` → `rosch_story_neg` 0.45), and `rosch_story_1` ↔ `rosch_story_2` (MC ridge 0.42 to 0.44).
- By category, prompt transfer follows the within-category result: weapon, clothing and vehicle are high (MC DoM 0.62 to 0.73), tool and vegetable are near 0.

**Cosine similarity of directions (off-diagonal mean)**

| Baseline | Probe | Between categories | Between prompts |
|:-|:-|-:|-:|
| None | DoM | 0.18 | 0.21 |
| None | Ridge | 0.13 | 0.12 |
| Mean contrast | DoM | 0.31 | 0.21 |
| Mean contrast | Ridge | 0.09 | 0.09 |

- With mean contrast, the difference-of-means directions of weapon, clothing, furniture, vehicle and toy are the most similar to the other categories (mean 0.39 to 0.45). Tool (0.04) and bird (0.13) are the least similar.
- Between prompts, the highest similarities are between the same prompt pairs that transfer well: `rosch_example`–`rosch_neg` (0.45 to 0.51 DoM) and `rosch_neg`–`rosch_story_neg` (0.44 to 0.50 DoM).

**Ridge alpha.** 132 of 840 fits (16%) are at the edge of the grid. 106 are at the lowest alpha (1.0), mostly in clothing, sport, vehicle and fruit. 26 are at the highest alpha (1e6), all in tool and vegetable.

### Interpretation

1. **Typicality can be decoded linearly at layer 15, at the period token, for most categories.** Cross-validated ridge probes get a Spearman correlation of about 0.6 on average and about 0.8 for weapon, clothing and vehicle. Tool is the exception: no probe finds a signal (ridge at most 0.19, and the inner CV often chooses the largest alpha, which makes the prediction nearly constant). Vegetable is weak. These two categories possibly have Rosch ratings that the model does not represent, or a small spread of typicality that is difficult to predict.

2. **Mean contrast helps, mostly for difference-of-means.** Subtracting the mean activation of the item in the other categories increases DoM from 0.36 to 0.49 within category, from 0.25 to 0.39 between categories, and from 0.19 to 0.40 between prompts. Ridge changes less, because it can learn to ignore the item-specific directions that mean contrast removes. This agrees with the hypothesis that the raw activation contains a large item identity component, and that the typicality signal is clearer when we remove it.

3. **There is a direction that transfers between categories, but the probes do not find the same direction.** Probes trained on one category grade typicality in other categories (MC: about 0.4, three quarters of the cells above 0.2). But the cosine similarity between directions is low, specially for ridge (about 0.1). Thus many different directions in the 1536-dimensional space have a shared component that grades typicality. Difference-of-means with mean contrast has the most similar directions (0.31) and also the best transfer. This is a possible shared "typicality" direction. It is also possible that it is a correlate of typicality, for example word frequency or familiarity. This experiment does not separate them.

4. **Ridge fits prompt-specific features.** Ridge is the best probe within a prompt but it loses most of its correlation in a different prompt, and its directions are almost orthogonal between prompts (0.09 to 0.12). Difference-of-means with mean contrast loses little between prompts. Thus a part of the ridge result comes from directions that are specific to the prompt format. The prompts that transfer well share a sentence structure (`rosch_example`, `rosch_neg`, `rosch_story_neg`; the two stories). `rosch_fav` is the most different from the others.

5. **Cautions.**
   - The off-diagonal cells of the category matrix use probes trained on all items. The diagonal cells use 5-fold CV (80% of the items). Thus the two are not fully comparable: the off-diagonal cells have more training data.
   - With 50 to 60 items in each category, a single correlation has a large error (approximately ±0.25 at 95% for ρ = 0.4). The means over many cells are more reliable than individual cells.
   - The ridge grid is too narrow at the low end: 106 fits chose alpha = 1.0, mostly in the categories with the strongest signal. A grid that starts lower can change the ridge results for those categories.
   - Only one layer (15) and one position (period) were probed.

**Possible next steps:** extend the ridge alpha grid below 1; probe layers 10, 20 and 25 (already in the activation cache); control for word frequency; train one probe on all categories (or all prompts) together and test it on held-out categories.