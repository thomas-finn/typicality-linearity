# PCA of Rosch prompts: Qwen2.5-1.5B, mean contrast baseline

## Hypothesis/question

The no-baselines PCA mixes typicality with everything else that changes the
activation of an item, for example the item's own features and the prompt
category. If we subtract the activation of the same item with the other
categories, is typicality more visible in the top principal components at the
period token?

For item $i$ rated in category $c$, prompt format $p$ and layer $\ell$, the
PCA is fitted on

$$
a(\ell,p,i,c) - \text{mean}_{c'\neq c}\left(a(\ell,p,i,c')\right).
$$

## Dataset

`data/processed/rosch/rosch_mean_contrast.csv`: the 565 item-category pairs
from Rosch (1975) with all 10 prompt categories, 5650 rows and seven prompt
formats. See `data/processed/rosch/README.md`. The rows with
`is_item_category` true are the same prompts as `rosch_prompts.csv`.

## Configuration

`config.yaml` in this directory. Same as no-baselines, except:

- `prompts_csv: data/processed/rosch/rosch_mean_contrast.csv`
- `baseline: mean_contrast`. Each (item, item_category) pair has 10 rows. The
  row of the item's own category is the target. The mean of the other 9 rows
  is subtracted. Categories in which the item is also rated (for example
  "tomato" in "vegetable" when $c$ is "fruit") are not excluded. 94 of the 5090
  baseline rows are like this. Code: `src/typicality_linearity/baselines.py`.
- `positions: [period]`. In all prompts the item comes before the category and
  the item text is the same for all 10 categories. The model is causal, so the
  item-token activation is the same for all $c$ and the contrast is 0. The HTML
  has one panel.
- The points are the 565 target rows, coloured by `rating_zscore`.
  Per-category PCA as in no-baselines.
- `compare_no_baseline: true`. A second PCA is fitted on the same 565 points
  without the baseline, that is $a(\ell,p,i,c)$. This is the same as the
  period panel of no-baselines (explained variance agrees to about 1e-8). The
  HTML shows the two PCAs side by side with the same selectors. Move the
  pointer over a point in one panel to see its prompt above both panels.

## Command

From the repository root:

    uv run python scripts/run_pca_rosch.py experiments/pca/rosch/qwen2-5-1-5B/mean-contrast/config.yaml

The first run loads the model and extracts activations for 5650 prompts in
each of seven formats (10 times the no-baselines run). The cache is about
1 GB (`outputs/activation_cache/`). Later runs read the cache.

Outputs go to `outputs/pca/rosch/qwen2-5-1-5B/mean-contrast/<run_id>/`:
`metadata.json`, `config.yaml`, `explained_variance.csv`,
`pca_projections.npz` and `qwen2-5-1-5B-mean_contrast-baseline.html`. The
formats are the same as no-baselines. The PCA without the baseline is in
`explained_variance_no_baseline.csv` and `pca_projections_no_baseline.npz`,
with the same formats.

## Result

Runs:

- `20261103T114821Z`: mean contrast only.
- `20261103T130220Z`: mean contrast and no baseline side by side. All
  activations from the cache.

Not analysed yet.

## Interpretation

Not analysed yet.
