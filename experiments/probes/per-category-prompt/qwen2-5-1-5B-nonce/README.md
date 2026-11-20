# Probes per category and prompt with a nonce baseline: Qwen2.5-1.5B

## Question

Does subtracting the activation of the same prompt with a nonce word as the category ("A doll is an example of a wugawrar.") remove the parts of the activation that do not depend on the category, as the mean category contrast does? How do difference-of-means probes on the nonce contrast compare with raw activations and the mean category contrast, within a category and in transfer between categories and between prompt formats?

The mean category contrast subtracts the mean over the 9 other real categories, so it can also remove some of the item–category relation. The nonce baseline has the same sentence and item but a category word with no meaning, so it removes the item and the prompt format but not the real category.

## Dataset

- `data/processed/rosch/rosch_mean_cat_contrast.csv` (raw activations from the 565 target rows, and the mean category contrast).
- `data/processed/rosch/rosch_nonce.csv`: the same 565 items and 15 prompt formats, with each of the 3 nonce words "wugawrar", "njonjobura" and "duplambi" as the category (1,695 rows; see `data/processed/rosch/README.md`). The nonce words are count nouns in all prompts, also for clothing and furniture items. The tokenizer splits them into 4, 4 and 3 tokens (` w ug aw rar`, ` n jon job ura`, ` dup l ambi`); the period is a separate, last token in all 25,425 nonce prompts (checked with the tokenizer).
- `data/processed/rosch/rosch_mean_item_contrast.csv` is read for the point check only; the item contrast types are not used.
- The 15 training prompt formats.
- Ambiguous items: 34 core, 41 extended.
- Nonce prompt change (2026-11-18): for the 28 plural items, the nonce prompts now say "Marbles are typical wugawrars." in place of "Marbles are a typical wugawrar." (84 rows in `rosch_typ_frame_1`, `rosch_typ_frame_3`, `rosch_hedge_1` to `rosch_hedge_3`, `rosch_eval_3`). See `data/processed/rosch/README.md`. Runs before this change used the old prompts. The cache key includes the prompt texts, so the next run extracts the changed nonce columns again.

## Configuration

`config.yaml`. The same as `../qwen2-5-1-5B/config.yaml`, with these changes:

- `activation_types: [raw, cat_contrast, nonce_contrast]`. Nonce contrast = a(i, c) − mean_w a(i, w): the raw activation minus the mean activation of the same prompt format over the 3 nonce words w. The nonce rows are matched to the point by (item, rated category).
- `nonce_csv: data/processed/rosch/rosch_nonce.csv`.
- `experiment`, `title`, `output_dir` and `html_name` (`qwen2-5-1-5B-nonce.html`).

All other settings are the same: layers 10, 15, 20, 25, period token, float32, prefix token `<|endoftext|>`; ambiguity include / exclude core / exclude extended; halves / thirds / quarters; Spearman, Pearson, Wilcoxon AUC; 5 folds (seed 0); `cv_aggregate: fold_zscored`. 3 × 3 × 4 × 3 = 108 settings. See `../qwen2-5-1-5B/README.md` for the probes, metrics and the four matrices.

- `diagonal: full` (added 2026-11-18): the diagonal cells of the transfer matrices use the probe trained on all items, evaluated on the same items (in-sample, not cross-validated). The off-diagonal cells do not change. The results below are from runs before this change, with a CV diagonal (`diagonal: cv`, the default when the key is not set).

The page has the same four tabs and selectors as `../qwen2-5-1-5B-probes-01.html`, with the three activation types side by side in one row (and the dendrograms under them on the cosine tabs).

## Command

From the repository root:

    uv run python scripts/run_global_probes_rosch.py experiments/probes/per-category-prompt/qwen2-5-1-5B-nonce/config.yaml

The raw and category contrast activations are in the cache. The nonce activations (15 entries, 25,425 prompts) are not, so the first run loads `Qwen/Qwen2.5-1.5B` and puts them in `outputs/activation_cache/Qwen2.5-1.5B/` (about 625 MB: 25,425 prompts × 4 layers × 1536 × 4 bytes). Later runs use the cache.

Outputs in `outputs/probes/global-direction/per-category-prompt/qwen2-5-1-5B-nonce/<run_id>/`: `config.yaml`, `metadata.json`, `summary.csv`, `matrices.npz`, `axes.json` and `qwen2-5-1-5B-nonce.html`. A copy of the page goes to `experiments/probes/per-category-prompt/qwen2-5-1-5B-nonce.html` (in `.gitignore`).

## Result

Run `20261118T162404Z` (2026-11-18, 8 min 45 s, MPS), status completed. The 5 nonce columns changed by the plural agreement fix were extracted with the model; all other entries came from the cache. `within_method` = `in_sample` (`diagonal: full`). Earlier runs (`20261113T093200Z`, `20261118T094025Z`, `20261118T113742Z`) used the old prompts; against `20261118T113742Z` the largest difference over all settings and summary columns is 0.003.

Thirds, include, Spearman. Category within is in-sample.

| Layer | Activation type | Category within | Between categories | Between prompts | Cosine between categories |
|:-|:-|:-|:-|:-|:-|
| 15 | raw | 0.673 | 0.277 | 0.405 | 0.198 |
| 15 | cat contrast | 0.698 | 0.392 | 0.443 | 0.292 |
| 15 | nonce contrast | 0.682 | 0.284 | 0.392 | 0.197 |

Between categories by layer (raw / cat contrast / nonce contrast): L10 0.19 / 0.37 / 0.22, L15 0.28 / 0.39 / 0.28, L20 0.25 / 0.35 / 0.27, L25 0.17 / 0.28 / 0.21. Between prompts: L10 0.40 / 0.47 / 0.41, L15 0.41 / 0.44 / 0.39, L20 0.33 / 0.35 / 0.29, L25 0.46 / 0.33 / 0.30.

Between categories without ambiguous items, L15: cat contrast 0.420 (core) / 0.435 (extended), nonce contrast 0.312 / 0.321.

## Interpretation

The nonce contrast gives almost the same values as the raw activations (between categories 0.28 against 0.28 at L15, cosine between categories 0.20 against 0.20), and it is below the mean category contrast at all layers (0.28 against 0.39 at L15). So subtracting the activation of the same prompt with a nonce category does not remove the part of the activation that blocks transfer between categories. The mean category contrast does more than remove a category-independent item component. A possible reason: the mean over the 9 real categories also removes how the item relates to real categories in general, which a nonce word cannot give. At L25 the raw activations transfer best between prompts (0.46); both contrasts remove some of this. The within-category values are in-sample, so do not compare them with the CV values of runs before 2026-11-18.
