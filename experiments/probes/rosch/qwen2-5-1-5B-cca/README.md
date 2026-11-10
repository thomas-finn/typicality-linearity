# Contrastive activation addition (CAA) directions

### Question

1. Does a CAA direction per category, pooled over all 7 training prompt formats, grade typicality on new prompt formats (evaluation prompts)?
2. Does the mean of the category directions do better than each category direction alone, on the evaluation prompts?
3. How does the mean contrast baseline change the directions, their cosine similarity and their transfer, compared with no baseline?

### Dataset

`data/processed/rosch/rosch_mean_contrast.csv` (565 items, 10 categories, 7 training prompt formats), as in `qwen2-5-1-5B-diff-means`. Evaluation prompts: `data/processed/rosch/rosch_eval.csv` (`rosch_eval_1` to `rosch_eval_3`), never used for training, raw activations (no baseline). All activations come from `outputs/activation_cache`.

### Configuration

`config.yaml`. Qwen2.5-1.5B, layer 15, period token, seed 0, 5 folds per category, all items (`ambiguous: [include]`).

- **CAA direction of a category**: mean activation of the most typical third minus the least typical third of its items, with all 7 training prompt rows (no weighting). With the same items in each prompt, this is the difference of means of the item activations averaged over the prompts (`format_variance_experiment.fit_pooled_probes`). The vector is not normalised, so it can be used for steering.
- **Baselines**: none, and mean contrast (a(i, c) − mean over c' ≠ c of a(i, c')).
- **Mean direction, all categories**: mean of the unit directions of all 10 categories. For the items of category c, the term of c is the fold direction that did not see the item.
- **Mean direction, other categories** (loco): mean of the unit directions of the 9 other categories.
- **Scores**: training prompts use the activations of the baseline that the direction was trained with; evaluation prompts use raw activations. Correlation with typicality (−`rating_zscore`) inside each prompt and category; own-category cells are cross-validated (pooled over folds).

### Command

    uv run python scripts/run_cca_rosch.py experiments/probes/rosch/qwen2-5-1-5B-cca/config.yaml

Outputs in `outputs/probes/rosch/qwen2-5-1-5B-cca/<run_id>/`: `summary.csv`, `correlations.csv` (direction × eval category × prompt), `cosine_similarity.csv` (between categories, and between the no-baseline and mean-contrast direction of each category), `mean_all_scores.csv` (item scores with the mean direction), `directions.npz` and `cca.html`.

`cca.html` sections: a summary table with no baseline, mean contrast and Δ (mean contrast − no baseline); the mean direction on the evaluation prompts by category and by prompt; the per-category direction matrices (no baseline, mean contrast, Δ) for a selected prompt; the cosine matrices (no baseline, mean contrast, Δ) and the cosine between the two baselines for each direction.

### Result

Run `20261107T205501Z_probes-rosch-qwen2-5-1-5B-cca` (2026-11-07). Status completed, all activations from the cache. The git tree was dirty (the code of this experiment was not committed). The run was started from a Python wrapper that called `cca_experiment.run` with the model load blocked, not from the script; the computation is the same.

Spearman, mean over cells and prompts:

| Statistic | Prompts | No baseline | Mean contrast | Δ |
|:-|:-|-:|-:|-:|
| Own-category direction (CV) | training | 0.34 | 0.50 | +0.16 |
| Other-category directions | training | 0.27 | 0.40 | +0.13 |
| Mean direction, all categories | training | 0.47 | 0.50 | +0.03 |
| Mean direction, other categories | training | 0.45 | 0.49 | +0.04 |
| Own-category direction (CV) | evaluation | 0.33 | 0.31 | −0.01 |
| Other-category directions | evaluation | 0.27 | 0.35 | +0.08 |
| Mean direction, all categories | evaluation | 0.51 | 0.51 | −0.01 |
| Mean direction, other categories | evaluation | 0.51 | 0.52 | +0.02 |
| Cosine between categories | – | 0.23 | 0.37 | +0.13 |

Mean direction, all categories, on the evaluation prompts, by prompt (none / MC): `rosch_eval_1` 0.53 / 0.52, `rosch_eval_2` 0.66 / 0.66, `rosch_eval_3` 0.35 / 0.35.

By category on the evaluation prompts (none / MC):

| Category | Mean direction, all | Own-category direction (CV) |
|:-|-:|-:|
| bird | 0.30 / 0.35 | 0.27 / 0.16 |
| clothing | 0.75 / 0.72 | 0.59 / 0.58 |
| fruit | 0.56 / 0.51 | 0.27 / 0.26 |
| furniture | 0.55 / 0.56 | 0.34 / 0.38 |
| sport | 0.68 / 0.65 | 0.50 / 0.45 |
| tool | 0.14 / 0.13 | 0.01 / −0.06 |
| toy | 0.40 / 0.42 | 0.31 / 0.19 |
| vegetable | 0.49 / 0.42 | 0.11 / 0.12 |
| vehicle | 0.65 / 0.66 | 0.50 / 0.65 |
| weapon | 0.60 / 0.65 | 0.39 / 0.40 |

Cosine between the no-baseline and the mean-contrast direction: categories 0.49 (tool) to 0.85 (weapon); mean direction, all categories 0.92.

### Interpretation

1. **Pooling over prompts helps a single category direction transfer to new formats.** The own-category direction gives 0.33 (none) on the evaluation prompts, against about 0.2 for the per-prompt probes in `qwen2-5-1-5B-diff-means`.
2. **The mean direction is better than the category directions on new formats** (0.51 vs 0.33 own, 0.27 to 0.35 other), in 10 of 10 categories without a baseline. Tool (0.14) and bird (0.30) are weak, as before.
3. **Mean contrast changes the single directions, not the mean direction.** With mean contrast the category directions are more similar (cosine 0.23 → 0.37) and transfer better between categories (+0.13 on training prompts, +0.08 on evaluation prompts). The large gain within a category on the training prompts (+0.16) does not carry to the raw activations of the evaluation prompts (−0.01). The two mean directions are almost the same (cosine 0.92) and give the same values (0.51). Thus averaging over categories removes most of what mean contrast removes.
4. The mean direction over all categories is not better than the one over the other 9 (0.51 vs 0.51 to 0.52), so the own category adds nothing to the mean. The pooled CV correlation is biased low by fold offsets (see the ridge README), which possibly explains part of the small difference.
5. **Cautions.** One layer and one token; no permutation null in this experiment (the diff-means null for similar statistics was about 0 with 95% at ≤ 0.07); per-category values have a large error (about ±0.25 for ρ = 0.4 with 50 items). Confounds (frequency, association) are not tested.
