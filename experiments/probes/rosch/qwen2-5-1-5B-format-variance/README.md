# Investigating format variance for qwen2.5-1.5B

In this experiment we replicate `qwen2-5-1-5B-diff-means` experiment with the goal to remove format variance between prompts.

We will compare the mean-contrast probes from `qwen2-5-1-5B-diff-means` with some attempted improvements to the mean-contrast approach:
 - Mean-center activations within each prompt format before pooling to strip the large format-specific offset.
 - Use INLP to erase a format probe's subspace and look for typicality in what remains.
 - Use LEACE to erase a format probe's subspace and look for typicality in what remains.
 - Use mean-center activations + INLP.
 - Use mean-center activations + LEACE.

The experiment should be at layer 15 again and have the option to include/exclude ambiguous items. We will not do the weighting applied in `qwen2-5-1-5B-diff-means` to save time as improvements where marginal and the approach was not principled. 

### Question

1. How much of the activation variance at layer 15 (period token) comes from the prompt format, and can centering, INLP or LEACE remove it?
2. When the format information is removed, do difference-of-means probes trained on all prompt formats together grade typicality better within a category, between categories, and on new prompt formats?

### Dataset

The same as `qwen2-5-1-5B-diff-means`: `data/processed/rosch/rosch_mean_contrast.csv` (565 items, 10 categories, 7 training prompt formats, `is_ambiguous` column) and the evaluation prompts `data/processed/rosch/rosch_eval.csv` (3 formats, never used for training or for fitting an eraser). The activations come from the cache.

### Method

**Probes pooled over prompts.** In the parent experiment each probe is trained on one category and one prompt, and each score is a correlation inside one prompt. An offset of a prompt format then has no effect. Here there is one difference-of-means probe for each category, trained on all 7 training prompts together: top third minus bottom third of the items, with all their prompt rows. The same items are in each prompt, so this is the difference of means of the item activations averaged over the prompts. There is no weighting (c = 0). The fold probes use the same item folds in all prompts.

**Preprocessing variants** (`variants` in the config). The format label of a row is its training prompt (7 classes). No typicality values are used to fit a variant. The variant is fitted on all items that are used (also the held-out items of a fold). The evaluation prompts get the eraser of the training prompts.

| Variant | Training prompts | Evaluation prompts |
|:-|:-|:-|
| `raw` | no change | no change |
| `center` | subtract the mean of each prompt | subtract the mean of each prompt (own mean) |
| `inlp` | INLP eraser for the format | the same eraser |
| `leace` | LEACE eraser for the format | the same eraser |
| `center_inlp` | center, then INLP | center (own mean), then the eraser |
| `center_leace` | center, then LEACE | center (own mean), then the eraser |

- **INLP** (Ravfogel et al. 2020): fit a logistic regression for the format (scikit-learn, on standardised activations, train/dev split 70/30), remove the row space of its weights (up to 7 directions) by an orthogonal projection, and do this again. At most `inlp_iterations` (20) steps. A step is not done when the dev accuracy of its classifier is below `inlp_stop_accuracy` (0.2; chance is 0.143). The accuracy of each step is in `inlp_history.csv`.
- **LEACE** (Belrose et al. 2023): closed form. After LEACE all format means are equal, and at most 6 directions change. Thus LEACE is close to centering, and **`center_leace` is the same as `center`** (after centering, the eraser has rank 0). It is kept as a check.

**Statistics.** Each one is computed two ways:
- *by prompt*: the correlation inside each prompt, then the mean over the prompts
- *pooled*: the correlation over the rows of all prompts of the category together. A different offset of each format makes this value lower. This is the value where format variance does harm, and where centering can help.

| Statistic | Probe | Items |
|:-|:-|:-|
| `within_{train,eval}_*` | fold probes of the category (CV) | own category |
| `between_{train,eval}_*` | full probe of another category | each other category |
| `loco_{train,eval}_*` | mean unit direction of the other 9 categories | the held-out category |
| `cosine_between_categories` | full probes | pairs of categories |

`train` is the 7 training prompts and `eval` the 3 evaluation prompts. Format diagnostics: `format_variance_train/eval` (fraction of the variance from the format means), `format_accuracy_train` (dev accuracy of a new logistic regression for the format after the variant; chance 0.143) and `eraser_rank`.

Note: with `cv_aggregate: pooled` the out-of-fold scores of the 5 fold probes are joined. Each fold probe gives the format offset a different constant, so centering can also change `within_*_by_prompt` (the fold-offset bias that the ridge experiment found). `between_*_by_prompt` and `loco_*_by_prompt` use one direction for each prompt, so centering does not change them.

### Configuration

`config.yaml`. Layer 15, period token, baselines none and mean contrast, thirds with no weighting, ambiguous include and exclude, 6 variants, 5-fold CV per category, seed 0. No permutation null in this run.

### Command

    uv run python scripts/run_format_variance_rosch.py experiments/probes/rosch/qwen2-5-1-5B-format-variance/config.yaml

Outputs in `outputs/probes/rosch/qwen2-5-1-5B-format-variance/<run_id>/`: `summary.csv` (one row per variant, baseline, ambiguous option and metric, with the diagnostics), `correlations.csv` (each cell, for each prompt and `pooled`), `format_diagnostics.csv`, `inlp_history.csv`, `probe_weights.npz` (keys `<variant>/<baseline>/<ambiguous>/<category>/<transformed|raw>`; `raw` gives the same scores on raw activations of a training prompt, for steering), `config.yaml`, `metadata.json` and `format-variance.html` (interactive page: summary of the variants, format diagnostics and INLP steps, transfer between categories for each prompt, score in each prompt, cosine between categories). To write the page again for a run without loading the model:

    uv run python scripts/plot_format_variance_rosch.py outputs/probes/rosch/qwen2-5-1-5B-format-variance/<run_id>

All activations are in the cache. The run does not load the model. Most of the time is the logistic regressions of INLP and the format check (about 3955 × 1536 rows for each fit).

### Comparison with the parent experiment

The parent values (per-prompt probes, thirds, c = 0) are in `summary.csv` of run `20261105T123218Z_probes-rosch-qwen2-5-1-5B-diff-means`, settings `g3_c0_include` and `g3_c0_exclude`. The `raw` variant here is the pooled-prompt reference with the same items.

### Result

Not run yet.

### Interpretation

Not run yet.
