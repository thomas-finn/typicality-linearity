# Ridge probes over a grid of regularisation values

The same experiment as `qwen2-5-1-5B-diff-means` (`experiments/probes/rosch/qwen2-5-1-5B-diff-means/`), with ridge probes in place of difference-of-means probes. The ridge alpha is a setting of the grid, in place of the split and the weight constant.

### Question

1. How does the ridge alpha change the probes within a category, and in transfer between categories, between prompts and to new prompt formats? The parent experiment (`qwen2-5-1-5B`) chose alpha by inner CV: ridge was better than difference of means within a category (about 0.6 against 0.49) but transferred worse between prompts (0.28 against 0.40). Is there an alpha that gives both?
2. Do items with an ambiguous word meaning make the ridge probes worse, as for difference of means?
3. Does the mean direction of ridge probes (over categories and training prompts) grade typicality on held-out categories and on the evaluation prompts, as the difference-of-means mean direction does (Spearman about 0.5)? Is the result above the permutation null?

### Dataset

The same as `qwen2-5-1-5B-diff-means`: `data/processed/rosch/rosch_mean_contrast.csv` (565 items, 10 categories, 7 training prompt formats) with the column `is_ambiguous` (72 items), and the evaluation prompts `data/processed/rosch/rosch_eval.csv` (3 formats that are never used for training, raw activations). All activations come from the cache.

### Configuration

`config.yaml`. Layer 15, period token, baselines none and mean contrast, 5-fold CV per category, seed 0, as in the parent experiments. Only ridge probes.

All combinations of these settings (22 alphas × 2 ambiguous options × 2 baselines):

| Setting | Values | Meaning |
|:-|:-|:-|
| `ridge_alphas` | `cv`, 0.01 to 1e8 in half-decade steps (21 values) | A fixed alpha for all fits of the setting. `cv`: alpha chosen for each fit by inner 5-fold CV from 1 to 1e6 (`ridge_cv_alphas`), the method of the parent experiment. |
| `ambiguous` | include, exclude | exclude removes the ambiguous items from training **and** evaluation. The folds are made again on the remaining items. |

The activations are standardised with the mean and std of the training items (`ridge_standardise: true`, as in the parent), so alpha is on the scale of the standardised activations. A category has about 45 training items (CV fold) and 1536 features, so:
- small alpha (0.01): almost the minimum-norm solution that fits the training items exactly
- large alpha (1e8, much larger than the squared singular values): the weights are almost Z^T (y − ȳ), the covariance of each standardised feature with typicality. In activation space the direction is cov(x_j, y) / var(x_j). This is not the difference of means, which is close to cov(x_j, y) without the division.

The parent inner CV chose alphas mostly from 316 to 3160 (some at the edges, 1 and 1e6).

There is no weighting (that is only for difference of means). All other parts are the same as in `qwen2-5-1-5B-diff-means`: the evaluation prompts, the mean direction over a choice of categories and all training prompts (unit directions in activation space, weights / scale), and the permutation null. Null settings: alpha 1000 (the alpha the parent inner CV chose most often), include and exclude, 200 shuffles.

### Command

    uv run python scripts/run_ridge_rosch.py experiments/probes/rosch/qwen2-5-1-5B-ridge/config.yaml

Outputs in `outputs/probes/rosch/qwen2-5-1-5B-ridge/<run_id>/`: the same files as `qwen2-5-1-5B-diff-means` (`summary.csv`, `correlations.csv`, `cosine_similarity.csv`, `mean_direction_best.csv`, `probe_weights.npz`, `null_distribution.csv`, `null_summary.csv`), `ridge_alphas.csv` (the alpha chosen in each fit of the `cv` settings) and `ridge-explore.html` (selectors for alpha, ambiguous items and metric).

Expected time on the local machine: about 5 s for each setting and baseline (about 8 min in total, the `cv` settings are slower), and about 2 s for each shuffle of the null (about 30 min). Peak memory about 1 GB.

**Check.** `acv_include` must give the same values as the ridge probes of the parent run `20261103T135510Z`. In a test run with only `cv` and 1000 (2026-11-05), all 4760 ridge cells were the same (maximum difference 0).

### Result

Run `20261105T181749Z_probes-rosch-qwen2-5-1-5B-ridge` (2026-11-05), 36 min, status completed, seed 0. All activations from the cache. The git tree was dirty (the code of this experiment was not committed). `ridge_alphas.csv` of this run has a row for each fit of all settings; the code now writes only the `cv` settings.

**Check.** `acv_include` gives the same values as the ridge probes of the parent run `20261103T135510Z` for all 4760 cells (maximum difference 0).

**Warning: the pooled CV values are biased low, mostly for large alpha.** With `cv_aggregate: pooled`, the out-of-fold scores of the 5 fold probes are put together before the correlation. Each fold probe adds its own offset (for ridge, the mean of y in its training items; for all probes, the score of the mean activation, which changes between folds). These offsets are not related to the held-out items, and the fold mean of y is anti-correlated with the held-out items (the null mean of within-CV is −0.05). When the score range is small compared to the offsets, the offsets control the pooled correlation. With large alpha the weights are very small, so within-CV goes to 0.06 at α = 1e8. The direction is still good: with the scores z-scored inside each fold before pooling (no labels used), it gives 0.52 to 0.56. The statistics with one probe per cell (between categories, eval between categories, mean direction loco, cosines) are not affected. The statistics that are affected: within (CV), between prompts, eval within, and a little the mean direction "all" (the own category uses fold directions).

Check of the bias (scratch script, not in the repository; Spearman, include, mean over categories and prompts):

| Probe | Baseline | Within, pooled | Within, z-scored per fold | Within, mean over folds | Between prompts, pooled | Between prompts, z-scored | Eval within, pooled | Eval within, z-scored |
|:-|:-|-:|-:|-:|-:|-:|-:|-:|
| DoM thirds, c = 0 | None | 0.36 | 0.48 | 0.47 | 0.19 | 0.36 | 0.21 | 0.39 |
| DoM thirds, c = 0 | MC | 0.49 | 0.54 | 0.53 | 0.40 | 0.43 | 0.19 | 0.31 |
| Ridge α = 1000 | None | 0.62 | 0.64 | 0.63 | 0.27 | 0.40 | 0.27 | 0.40 |
| Ridge α = 1000 | MC | 0.64 | 0.66 | 0.66 | 0.33 | 0.40 | 0.19 | 0.28 |
| Ridge α = 1e4 | None | 0.56 | 0.58 | 0.57 | 0.29 | 0.42 | 0.29 | 0.42 |
| Ridge α = 1e4 | MC | 0.58 | 0.61 | 0.60 | 0.37 | 0.45 | 0.24 | 0.33 |
| Ridge α = 1e5 | MC | 0.53 | 0.57 | 0.56 | 0.33 | 0.45 | 0.24 | 0.34 |
| Ridge α = 1e8 | None | 0.06 | 0.52 | 0.51 | 0.04 | 0.40 | 0.04 | 0.41 |
| Ridge α = 1e8 | MC | 0.07 | 0.56 | 0.55 | 0.05 | 0.45 | 0.03 | 0.34 |

This bias also changes the results of `qwen2-5-1-5B-diff-means` and the parent `qwen2-5-1-5B` (see Interpretation 6).

Spearman, mean over cells, as written by the run (pooled CV; values marked * are biased low, see the table above):

| α | Baseline | Within (CV)* | Between categories | Between prompts* | Eval between categories | Mean dir. loco, train | Mean dir. loco, eval | Cosine, categories |
|-:|:-|-:|-:|-:|-:|-:|-:|-:|
| 0.01 to 3.16 | None | 0.61 | 0.34 | 0.19 | 0.17 | 0.52 | 0.54 | 0.06 |
| 0.01 to 3.16 | MC | 0.62 | 0.34 | 0.23 | 0.09 | 0.54 | 0.37 | 0.03 |
| 316 | MC | 0.64 | 0.40 | 0.29 | 0.14 | 0.54 | 0.42 | 0.07 |
| 1000 | None | 0.62 | 0.39 | 0.27 | 0.24 | 0.53 | 0.55 | 0.17 |
| 1000 | MC | 0.64 | 0.43 | 0.33 | 0.17 | 0.54 | 0.45 | 0.11 |
| 3160 | MC | 0.62 | 0.44 | 0.35 | 0.21 | 0.53 | 0.49 | 0.18 |
| 1e4 | MC | 0.58 | 0.45 | 0.37 | 0.25 | 0.52 | 0.52 | 0.26 |
| ≥ 1e6 | None | (0.06 to 0.23) | 0.31 | (0.04 to 0.09) | 0.24 | 0.49 | 0.53 | 0.25 |
| ≥ 1e6 | MC | (0.07 to 0.30) | 0.43 | (0.05 to 0.14) | 0.27 | 0.50 | 0.53 | 0.35 |
| cv | None | 0.60 | 0.37 | 0.22 | 0.22 | 0.52 | 0.54 | 0.13 |
| cv | MC | 0.62 | 0.42 | 0.28 | 0.15 | 0.54 | 0.44 | 0.09 |

Exclude (MC): α = 1000: within 0.64, between categories 0.47, mean dir. loco eval 0.52. α = 1e4: 0.61, 0.49, 0.56. α ≥ 1e6: between categories 0.48, mean dir. loco eval 0.56, eval between categories 0.29.

Difference of means for comparison (from `qwen2-5-1-5B-diff-means`, MC, pooled): thirds, c = 0, include: within 0.49*, between categories 0.39, between prompts 0.40*, mean dir. loco eval 0.53, cosine 0.31. Quarters, c = 1, exclude: between categories 0.48, mean dir. loco eval 0.56.

- **α ≤ 3.16 gives the same probe.** The squared singular values of the standardised training activations are much larger than α, so the solution is the minimum-norm fit of the training items.
- **Within a category, α from 100 to 1000 is best** (MC: pooled 0.64, z-scored 0.66). The `cv` setting is a little lower (0.62), because inner CV on 45 items is noisy (alphas from 1 to 1e6).
- **Transfer increases with α up to about 1e4, then stays the same.** Between categories (MC): 0.34 at small α, 0.43 at α = 1000, 0.45 at 1e4, 0.43 at α ≥ 1e6. Between prompts (z-scored): 0.40 at α = 1000, 0.45 at α ≥ 1e4. The cosine between the category directions increases from 0.03 to 0.35.
- **The large-α direction is close to the difference of means.** Mean cosine with the DoM direction (thirds, c = 0, include) of the same category, prompt and baseline: α = 1: 0.29 to 0.32; α = 1000: 0.52 to 0.55; α = 1e4: 0.71 to 0.73; α = 1e8: 0.75 to 0.79.
- **Mean direction.** Without a baseline, the mean ridge direction over 9 categories and 7 prompts gives 0.53 to 0.55 on the held-out category in the evaluation prompts, for all α. With mean contrast it depends on α: 0.37 at small α, 0.45 at α = 1000, 0.52 to 0.53 at α ≥ 1e4. Small-α directions trained with mean contrast do not work on raw activations of new prompts. The best values (MC, exclude, α ≥ 1e5: 0.56) are the same as for difference of means (0.56).
- **Ambiguous items.** Exclusion increases transfer between categories (MC, α = 1000: 0.43 → 0.47; α ≥ 1e6: 0.43 → 0.48) and the mean direction on the eval prompts (0.45 → 0.52 at α = 1000), but not within-category CV (0.635 → 0.643). With difference of means, exclusion also increased within-CV (0.49 → 0.54).
- **Permutation null** (α = 1000, include and exclude, both baselines, 200 shuffles): every statistic is above all 200 null values (p = 0.005). The null 95th percentiles are below 0.05. The cosine between prompts has a high null again (null mean 0.07 to 0.10, observed 0.10 to 0.14). The null mean of within-CV is −0.05 because of the pooling bias.

### Interpretation

1. **Ridge is best within a category, but small α overfits to the category and the prompt.** Small α gives the best fit of each (category, prompt) cell but almost orthogonal directions between categories (cosine 0.03 to 0.06) and the worst transfer. Increasing α trades a little within-category accuracy (0.66 → 0.57 z-scored from α = 1000 to 1e8) for more transfer and more similar directions.
2. **Answer to question 1: α about 1e4 is a good compromise.** With mean contrast, it gives within 0.61 (z-scored), between categories 0.45, between prompts 0.45 (z-scored), and the mean direction 0.52 on new prompts. This is better than difference of means within a category (0.54 z-scored) and in transfer between categories (0.39 to 0.43), and the same in transfer between prompts (0.43 z-scored) and the mean direction (0.53).
3. **Large α and difference of means find almost the same direction.** At large α, the direction is cov(x_j, y) / var(x_j). This is close to the difference of means (cosine about 0.8). Transfer and the mean direction then give the same values as difference of means. So the shared typicality direction from the diff-means experiment does not depend on the probe method.
4. **Answer to question 3: the mean ridge direction also transfers to held-out categories and new prompts** (0.53 to 0.56 without a baseline, any α). The averaging over categories and prompts removes the parts of each direction that are specific to the category or prompt, so even the overfitted small-α directions give a good mean direction without a baseline. With mean contrast, the small-α directions use features that only work on mean-contrast activations.
5. **Answer to question 2: ambiguous items mostly change transfer, not the fit within a category.** Ridge can fit the ambiguous items within a category, but they make the direction less general.
6. **The pooled CV aggregate is biased low, also in the earlier experiments.** The effect is large where the score range is small compared to the change of the offset between folds: ridge with large α, difference of means without a baseline, and probes evaluated on a different prompt. The corrected values for difference of means (thirds, c = 0) are: within 0.48 (no baseline; reported 0.36), between prompts 0.36 (no baseline; reported 0.19) and 0.43 (MC; reported 0.40), single probes on the evaluation prompts 0.39 (no baseline; reported 0.21) and 0.31 (MC; reported 0.19). Thus "single probes only about 0.2 on the evaluation prompts" in `qwen2-5-1-5B-diff-means` is too low; the correct values are 0.3 to 0.4, still lower than the mean direction (0.50 to 0.56). The conclusions about the mean direction and between-category transfer do not change. A CV aggregate that z-scores the scores in each fold before pooling (or "mean_over_folds", which is already in the code) does not have this bias.
7. **Cautions.** The between-prompt cosine is only a little above its null, as for difference of means. The values with alpha selected on the same data (best α) are optimistic. Confounds (word frequency, item–category association) are not tested.

**Possible next steps:** add a fold-centred CV aggregate and run this experiment and `qwen2-5-1-5B-diff-means` again; run with `ridge_standardise: false` (at large α this gives cov(x_j, y), which is closer to difference of means); a permutation null at α = 1e4.
