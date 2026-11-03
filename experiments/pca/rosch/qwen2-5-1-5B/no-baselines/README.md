# PCA of Rosch prompts: Qwen2.5-1.5B, no baselines

## Hypothesis/question

Is typicality visible in the top principal components of the residual stream of
`Qwen/Qwen2.5-1.5B` at intermediate layers? Is it encoded at the item token
(before the model sees the category) or only at the period token (after the
item-category pair)?

## Dataset

`data/processed/rosch/rosch_prompts.csv`: 565 item-category pairs from
Rosch (1975), seven prompt formats. See `data/processed/rosch/README.md`.

## Configuration

`config.yaml` in this directory.

- Layers 10, 15, 20, 25. Layer L is the residual stream after block L
  (`hidden_states[L]` in Hugging Face). The model has 28 blocks.
- Positions: the last token of the item, and the final period token.
- PCA is fitted separately for each prompt format, layer and position, on all
  565 rows (all categories together). Top 3 components.
- `per_category_pca: true`: PCA is also fitted on the rows of each category
  only (50-60 rows each). In the HTML, "All categories" shows the pooled PCA.
  One selected category shows the PCA of that category, so the axes are the
  directions of most variance in that category. With 50-60 rows and 1536
  dimensions, these fits are noisier than the pooled fit. The sign of each
  component is arbitrary, so do not compare directions between categories.
- Prefix token: `<|endoftext|>` before every prompt (`prefix_token`). Qwen2.5
  adds no start token, so the first prompt token is an attention sink with a
  norm about 200 times larger than other tokens. Without the prefix, an item at
  the start of a prompt ("Doll: toy.", "Books are an example of...") gives this
  sink activation, and PC1 only separates these rows (99.9% of variance). The
  prefix is a token id, so the prompt text and spans do not change. A target
  token at index 0 is now an error.
- Colour: `rating_zscore`, red = low (typical), blue = high (atypical). The
  range is centred on 0 and is the same for all categories.
- float32, seed 0. PCA is deterministic. Each component has a fixed sign.

## Command

From the repository root:

    uv run python scripts/run_pca_rosch.py experiments/pca/rosch/qwen2-5-1-5B/no-baselines/config.yaml

Outputs go to `outputs/pca/rosch/qwen2-5-1-5B/no-baselines/<run_id>/`:
`metadata.json`, `config.yaml`, `explained_variance.csv` (one row for each
prompt, layer, scope and position; scope is `all` or a category),
`pca_projections.npz` (keys `<prompt>/<layer>/<scope>/<position>`, and
`rows/<scope>` for the row indices of each scope) and
`qwen2-5-1-5B-no-baselines.html`.

Activations are cached in `outputs/activation_cache/` (`activation_cache_dir`
in the config). A second run with the same model, dtype, layers, positions and
prompts reads the cache and does not load the model. This is useful to change
the PCA or the plots. `metadata.json` records the source of the activations
for each prompt format ("cache" or "model"). On Colab, set
`activation_cache_dir` to a persistent location (for example mounted Drive),
because the local disk is temporary. The cache is about 200 MB for this
experiment.

The HTML page loads plotly.js from a CDN, so it needs an internet connection.

## Result

- Run `20261103T110824Z` (no prefix token): not valid. At the item position
  for `rosch_example`, `rosch_neg`, `rosch_simple` and `rosch_fav`, PC1 had
  99.5-99.97% of the variance because of attention-sink rows (12 of 565 rows
  for `rosch_example`, 89 for `rosch_simple`). Do not use this run.
- Run `20261103T112033Z` (prefix token, pooled PCA only): completed. Replaced
  by the next run.
- Run `20261103T113646Z` (prefix token, `per_category_pca: true`, activations
  from the cache): completed. Not analysed yet.

## Interpretation

Not run yet.
