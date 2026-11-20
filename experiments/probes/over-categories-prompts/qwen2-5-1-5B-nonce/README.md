# Probes pooled over categories and over prompts with a nonce baseline: Qwen2.5-1.5B

## Question

Does the nonce contrast (the raw activation minus the activation of the same prompt with a nonce word as the category) give better pooled probes than raw activations and the mean category contrast? In particular:

- Pooled over categories (page 1): does one probe per prompt on the nonce contrast grade typicality inside each category, and transfer between prompt formats and to the evaluation prompts?
- Pooled over prompts (page 2): does one probe per category transfer between categories?
- Pooled over categories and prompts (page 3): how well does one probe score the evaluation prompts?

The mean category contrast subtracts the mean over the 9 other real categories, so it can also remove some of the item–category relation. The nonce baseline has the same sentence and item but a category word with no meaning. See `../../per-category-prompt/qwen2-5-1-5B-nonce/README.md` for the same comparison with one probe per category and prompt.

## Dataset

- The same as `../qwen2-5-1-5B/`: the 565 rated pairs from `data/processed/rosch/rosch_mean_cat_contrast.csv`, the 15 training prompt formats and the 3 evaluation prompts (`rosch_eval_1` to `rosch_eval_3`, from `data/processed/rosch/rosch_eval.csv`).
- `data/processed/rosch/rosch_nonce.csv`: the nonce prompts in the 15 training formats (1,695 rows), as in the per-category nonce experiment.
- `data/processed/rosch/rosch_nonce_eval.csv` (new): the nonce prompts in the 3 evaluation formats (1,695 rows), for the nonce contrast of the evaluation prompts. Made by `write_nonce_eval_suite` in `rosch_prompts.py` (`scripts/make_rosch_prompts.py`).
- `data/processed/rosch/rosch_mean_item_contrast.csv` and `rosch_mean_item_contrast_eval.csv` are read for the point check only; the item contrast types are not used.
- Ambiguous items: 34 core, 41 extended.
- Nonce prompt change (2026-11-18): for the 28 plural items, the nonce prompts now say "Marbles are typical wugawrars." in place of "Marbles are a typical wugawrar." (84 rows in `rosch_typ_frame_1`, `rosch_typ_frame_3`, `rosch_hedge_1` to `rosch_hedge_3`, `rosch_eval_3`). See `data/processed/rosch/README.md`. Runs before this change used the old prompts. The cache key includes the prompt texts, so the next run extracts the changed nonce columns again.

## Configuration

`config.yaml`. The same as `../qwen2-5-1-5B/config.yaml`, with these changes:

- `activation_types: [raw, cat_contrast, nonce_contrast]`. Nonce contrast = a(i, c) − mean_w a(i, w), the mean over the 3 nonce words w of the same prompt format. The nonce rows are matched to the points by (item, rated category).
- `nonce_csv: data/processed/rosch/rosch_nonce.csv`. The evaluation prompts are scored on raw activations, so no nonce file for the evaluation prompts is necessary (`eval_nonce_csv` was removed on 2026-11-20).
- `experiment`, `title`, `output_dir` and `html_name` (`qwen2-5-1-5B-nonce.html`).

All other settings are the same: layers 10, 15, 20, 25, period token, float32, prefix token `<|endoftext|>`; ambiguity include / exclude core / exclude extended; halves / thirds / quarters; `over_categories_groups: per_category`; Spearman, Pearson, Wilcoxon AUC; 5 folds (seed 0); `cv_aggregate: fold_zscored`. 3 × 3 × 4 × 3 = 108 settings. See `../qwen2-5-1-5B/README.md` for the three pages.

- `diagonal: full` (added 2026-11-18): the diagonal cells of the transfer matrices use the probe trained on all items, evaluated on the same items (in-sample, not cross-validated). The off-diagonal cells do not change. The results below are from runs before this change, with a CV diagonal (`diagonal: cv`, the default when the key is not set).

Code changes for this experiment: `global_pooled_probe_experiment.py` now uses `activation_types` from the config (before, it always used the four default types), Since 2026-11-20 the evaluation prompts load only raw activations, and `eval_nonce_csv` is not used.


**Scoring change (2026-11-20).** Baselines are now used only to train the probes. All scores (transfer cells, held-out items, in-sample diagonals and evaluation prompts) are x_raw · direction. Before, a probe was scored on activations of its own type (for example a category contrast probe on category contrast activations). The results below are from runs before this change. Run the experiment again to get results with raw scoring. The raw rows do not change.

## Command

From the repository root:

    uv run python scripts/run_global_pooled_probes_rosch.py experiments/probes/over-categories-prompts/qwen2-5-1-5B-nonce/config.yaml

The raw and category contrast activations are in the cache. If the per-category nonce experiment was run first, the 15 training nonce entries are also in the cache. The evaluation prompts need only raw activations, which are in the cache. Without the training nonce entries, the first run loads `Qwen/Qwen2.5-1.5B` and extracts 25,425 prompts (about 625 MB) into `outputs/activation_cache/Qwen2.5-1.5B/`. Later runs use the cache.

Outputs in `outputs/probes/global-direction/over-categories-prompts/qwen2-5-1-5B-nonce/<run_id>/`: `config.yaml`, `metadata.json`, `summary.csv`, `matrices.npz`, `axes.json` and `qwen2-5-1-5B-nonce.html`. A copy of the page goes to `experiments/probes/over-categories-prompts/qwen2-5-1-5B-nonce.html` (in `.gitignore`).

## Result

Not run yet.

## Interpretation

Not run yet.
