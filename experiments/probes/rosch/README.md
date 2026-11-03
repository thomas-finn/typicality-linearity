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

Not run yet.

### Interpretation

Not analysed yet.