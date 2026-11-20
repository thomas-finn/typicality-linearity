# Activations for the global direction experiments: Qwen2.5-1.5B

## Question

This is a preliminary step, not an analysis. It extracts the residual stream activations that the global direction experiments use (see `../../README.md`), so that later runs read them from the cache.

## Dataset

All prompt files in `data/processed/rosch/` (see `data/processed/rosch/README.md`):

| File | Rows | Prompt columns |
|:-|:-|:-|
| `rosch_prompts.csv` | 565 | 15 |
| `rosch_mean_cat_contrast.csv` | 5650 | 15 |
| `rosch_mean_item_contrast.csv` | 5650 (seed 0) | 15 |
| `rosch_eval.csv` | 5650 | 3 |

48 cache entries, 194,925 prompts (94,700 different prompts). 33 entries (110,175 prompts) are extracted with the model. The 15 entries of `rosch_mean_item_contrast.csv` are derived from the cache, without the model (see below).

## Configuration

`config.yaml`:

- Model `Qwen/Qwen2.5-1.5B`, float32, prefix token `<|endoftext|>`.
- Layers 10, 15, 20, 25 (`hidden_states[L]`, the residual stream after block L).
- Position `period`: the token that contains the final period. A check with the tokenizer shows that for all 94,700 prompts this token is "." alone and is the last token of the prompt.
- Cache: `outputs/activation_cache/Qwen2.5-1.5B/` (one subdirectory for each model, from 2026-11-10; the manifests of the earlier runs show the old flat paths). One entry per (file, prompt column). The cache key is the model, dtype, layers, positions, prefix token, and the prompt and item texts (see `activation_cache.py`). Experiments with the same settings and prompts use these entries.
- Derived entries (`derive_from`): every prompt of `rosch_mean_item_contrast.csv` is also in `rosch_mean_cat_contrast.csv`, with the same item text. So each item contrast entry is made from the category contrast entry of the same column: each row takes the activations of the category contrast row with the same prompt and item text. All 15 columns match for all rows (checked before the run). The derived entry has its own cache key and is a normal entry for experiments. Its JSON file has `derived_from` with the key of the source entry. The prompts are the same, so only the batch and the padding are different from a direct extraction.
- Extended entries (`extend_from`, added 2026-11-18): `rosch_mean_cat_contrast_extended.csv` has the rows of `rosch_mean_cat_contrast.csv` and the rows of 10 new categories. The rows of the rated categories are copied from the category contrast entries; only the 5,650 new rows per column are extracted with the model (15 columns, 84,750 prompts). The manifest column `n_model_rows` gives the number of rows extracted with the model for each entry.
- Size: about 4.8 GB (n_prompts × 4 layers × 1536 × 4 bytes). The derived entries are copies, about 2.1 GB of this.

The old cache was deleted on 2026-11-08 before this step, so all entries come from this extraction.

## Command

From the repository root:

    uv run python scripts/extract_rosch_activations.py experiments/activations/qwen2-5-1-5B/config.yaml

The run writes `config.yaml`, `metadata.json` and `manifest.csv` (one row per entry, with the cache key and the `.npz` path) to `outputs/activations/global-direction/qwen2-5-1-5B/<run_id>/`. An entry that is already in the cache is not made again, so a run that stops can be started again. The manifest `source` column is `model`, `derived` or `cache`.

## Result

Run `20261118T165006Z` (2026-11-18, 16:50 to 17:57 UTC, MPS, batch size 16), status completed. Adds `rosch_mean_cat_contrast_extended.csv` (`extend_from`).

- 66 entries: 15 extended (11,300 rows each; 5,650 copied from the category contrast entries, 5,650 extracted with the model), 51 from the cache. Cache size 11 GB.
- All extended entries have shape (11300, 4, 1536), no NaN or inf values, and the token text "." for all rows.
- Copied rows: equal to the category contrast entry, except 312 rows (over the 15 entries) whose prompt is in the file twice (an item rated in two categories with the same grammar, for example "tomato"). These rows are copied from the first occurrence, which was in another batch: relative difference at most 1.2e-5. The `derive_from` entries have the same property.
- New rows: 16 rows of `rosch_hedge_3` extracted again directly with the model give the same values (relative difference 0). The mean norm of the new rows is the same as for the rated categories at each layer (for example `rosch_example`, L25: 178.8 against 176.7).

Run `20261118T152940Z` (2026-11-18, 15:29 to 15:57 UTC, MPS, batch size 16), status completed. Re-extraction after the plural agreement change of the prompts (see "Plural items" in `data/processed/rosch/README.md`).

- 51 entries: 11 extracted with the model (`rosch_typ_frame_1`, `rosch_typ_frame_3`, `rosch_hedge_1` to `rosch_hedge_3` of `rosch_prompts.csv` and `rosch_mean_cat_contrast.csv`, and `rosch_eval_3` of `rosch_eval.csv`), 6 derived (the same columns of the two item contrast files), 34 from the cache (prompts not changed). Cache size 7.4 GB; the entries of the old prompts are still in the cache, but no config uses them.
- All new entries have shape (n_prompts, 4, 1536), no NaN or inf values, and the token text "." for all rows.
- Old against new entry, for the rows whose prompt did not change: relative difference 0 (`rosch_hedge_3` category contrast, `rosch_eval_3`) and at most 4.2e-6 (derived `rosch_typ_frame_1` item contrast). The changed rows (224 or 235 per entry) have a median relative difference of 0.06 to 0.18.
- The nonce files are not in this config. Their changed columns are extracted by the nonce probe runs.
- Results made from the 1.5B activations before this run (PCA, probes) used the old prompts in these 5 training columns and `rosch_eval_3`.

Earlier run:

Run `20261108T162626Z` (2026-11-08, 16:26 to 17:50 UTC, MPS, batch size 16), status completed.

- 48 entries: 33 extracted with the model, 15 derived. Cache size 4.5 GB.
- All entries have shape (n_prompts, 4, 1536), no NaN or inf values, and the token text "." for all rows.
- Derived against direct extraction: the anchor rows of the item contrast entries and the separately extracted `rosch_prompts.csv` entries (the same 565 prompts) have a relative difference of at most 2.4e-6 and cosine 1.000000 (checked for `rosch_example`, `rosch_hedge_3`, `rosch_cat_first_2`).

## Interpretation

The cache has the period-token activations for all prompts in `data/processed/rosch/`. The batch and the padding change the activations only by float rounding, so derived entries can be used as extracted entries.
