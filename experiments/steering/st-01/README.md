# st-01: Steering along the general typicality direction

### Question

Is the general typicality direction from the difference-of-means experiment causal? When this direction is added to the residual stream at the item token, does the model prefer typical pole words ("typical", "standard", "representative") over atypical pole words ("poor", "rare", "odd")? Is the change larger than for random directions? Is it different for typical and atypical items, and between categories?

### Dataset

`data/processed/rosch/rosch_prompts.csv` (565 items, 10 categories), all items (ambiguous items included). One prompt for each item:

    As an example of {category}, {item} {be} fairly BLANK

- {category}: the category with its determiner, as in the training prompts ("a fruit", "clothing", "furniture").
- {item}: "the" + the item for count nouns and plurals ("the apple", "the Brussels sprouts"); the item alone for mass nouns and sports ("spinach", "swimming"). The forms come from `item_phrase` in `rosch_prompts.py`.
- {be}: "is" or "are".

Examples: "As an example of a fruit, the apple is fairly BLANK", "As an example of a sport, swimming is fairly BLANK".

| Typical poles | Atypical poles |
|:-|:-|
| typical, standard, representative | poor, rare, odd |

All pole words are one token (Qwen2.5 tokenizer, with a leading space). The logit of a pole word is read at the token before it ("fairly"). For each prompt, the context is checked to be a token prefix of each filled prompt, so one forward pass of the context gives the logits of all pole words.

Typicality thirds: the items of each category are ranked by typicality (−`rating_zscore`) and split into 3 groups (190 / 189 / 186 items).

### Direction

Source run: `outputs/probes/rosch/qwen2-5-1-5B-diff-means/20261105T123218Z_probes-rosch-qwen2-5-1-5B-diff-means` (the experiment folder `experiments/probes/rosch/qwen2-5-1-5B-diff-means/` and the code `diff_means_experiment.py` are removed; both are in git commit `88c73a3`). In that run, `exclude` removed the 72 items of the first ambiguity review (`AMBIGUOUS_ITEMS` in `rosch_prompts.py`), not the 34 core or 41 extended items used now. The steering code reads the saved `summary.csv` and `probe_weights.npz` of the run, so it does not need the removed code. `items.csv` now has the column `is_ambiguous_extended` (before: `is_ambiguous`).

The setting with the highest `mean_all_eval` (Spearman) is selected: the mean direction over all 10 categories and 7 training prompts, scored on the 3 evaluation prompts. This is **quarters, c = 0.5, ambiguous items excluded, mean contrast** (`g4_c0.5_exclude`, 0.550). The settings near the top are almost equal (`g4_c0.25_exclude` 0.549, `g4_c0_exclude` 0.549). Ranked by `mean_loco_eval`, the best is `g3_c0_exclude` (0.560).

The direction is the mean of the 70 unit directions (10 categories × 7 prompts) of the full-data probes, then normalised. The probes were trained at layer 15, period token.

### Method

At layer 15 (`hidden_states[15]`, the output of decoder block 14), at the **last token of the item** in each prompt (for example " apple"; "-box" for "jack-in-the-box"):

    h <- h + α · n · u

u is the unit direction. n is the norm of the unsteered residual stream of that prompt at the steered token, so α is a fraction of the usual activation norm. α is in [−1, 1] (17 values, 0 included). The prefix token `<|endoftext|>` is put before the prompt, as in the activation extraction.

Metric: logit difference = mean logit of the typical poles − mean logit of the atypical poles, for each item. The change is the difference from α = 0 for the same item.

Control: 10 random unit directions (seed 0), with the same α.

Note: the direction was trained at the period token; here it is applied at the item token.

### Configuration

`config.yaml`.

### Command

    uv run python scripts/run_steering.py experiments/steering/st-01/config.yaml

Outputs in `outputs/steering/st-01/<run_id>/`: `st-01.html`, `items.csv` (prompt, steered token, norm and third of each item), `results.csv` (one row per direction, α and item), `top_tokens.csv` (top 5 next tokens, typicality direction), `direction.npy`, `metadata.json` (also the selected setting).

`st-01.html` has a selector for the category (or all categories), the steered positions and α. For the selected items it shows: the mean change of the logit difference for each α (typicality direction and each random direction); the mean change by typicality third; the logit difference of each item against its typicality, before steering and at the selected α, with the Spearman correlation; the mean logit of each pole word; and a table of the items with the top next tokens.

About 6400 forward passes of 16 prompts (36 batches × (1 + 16 α × 11 directions)). Only the logits of the last token are computed, because the full logits do not fit in memory on a 16 GB machine.

### Result

Run `20261105T162342Z_steering-st-01` (2026-11-05), status completed, seed 0, MPS, 14 minutes. **All 10 categories (565 items), 3 random directions, α in {−1, −0.5, −0.2, −0.1, 0, 0.1, 0.2, 0.5, 1}** (reduced from 10 random directions and 17 values of α, because the full grid did not fit in the memory of a 16 GB machine in a short time; see the failed runs `20261105T151319Z` and `20261105T160827Z`). Selected setting `g4_c0.5_exclude`, mean contrast. The git tree was dirty.

**Mean change of the logit difference, all items** (typicality thirds for the typicality direction):

| α | Typicality | Random 0 | Random 1 | Random 2 | Least typical third | Middle | Most typical third |
|-:|-:|-:|-:|-:|-:|-:|-:|
| −1.0 | **−1.53** | +0.03 | +0.09 | −0.75 | −1.40 | −1.54 | −1.65 |
| −0.5 | **−0.85** | +0.05 | −0.02 | −0.30 | −0.82 | −0.87 | −0.87 |
| −0.2 | −0.27 | +0.06 | −0.02 | −0.05 | −0.27 | −0.28 | −0.26 |
| −0.1 | −0.12 | +0.04 | −0.01 | −0.01 | −0.12 | −0.13 | −0.11 |
| +0.1 | +0.09 | −0.06 | +0.01 | −0.02 | +0.10 | +0.10 | +0.08 |
| +0.2 | +0.15 | −0.13 | +0.01 | −0.06 | +0.17 | +0.16 | +0.13 |
| +0.5 | +0.20 | −0.34 | +0.06 | −0.22 | +0.25 | +0.22 | +0.13 |
| +1.0 | +0.07 | −0.38 | +0.38 | −0.24 | +0.18 | +0.11 | −0.07 |

**By category** (logit difference before steering; Spearman of typicality and logit difference before steering; change for the typicality direction; range of the 3 random directions at α = ±1):

| Category | n | Before | ρ before | α = −1 | Random, α = −1 | α = −0.5 | α = +0.5 | α = +1 | Random, α = +1 |
|:-|-:|-:|-:|-:|:-|-:|-:|-:|:-|
| bird | 54 | 2.78 | 0.53 | −1.46 | −0.73 to −0.02 | −0.94 | +0.25 | +0.05 | −0.34 to +0.59 |
| clothing | 55 | 3.53 | 0.30 | −1.54 | −0.79 to +0.05 | −0.69 | +0.07 | −0.07 | −0.43 to +0.10 |
| fruit | 51 | 1.95 | 0.53 | −1.59 | −0.54 to +0.47 | −1.03 | +0.35 | +0.24 | −0.16 to +0.45 |
| furniture | 60 | 3.74 | 0.58 | −1.67 | −0.82 to −0.01 | −0.87 | +0.12 | −0.05 | −0.57 to −0.02 |
| sport | 59 | 2.86 | 0.01 | −1.02 | −0.70 to +0.09 | −0.62 | +0.17 | +0.09 | −0.32 to +0.55 |
| tool | 60 | 3.12 | 0.16 | −1.55 | −0.91 to −0.06 | −0.84 | +0.19 | +0.02 | −0.56 to +0.23 |
| toy | 60 | 3.11 | 0.36 | −1.03 | −0.64 to −0.13 | −0.48 | −0.01 | −0.10 | −0.33 to +0.70 |
| vegetable | 56 | 1.34 | 0.14 | −1.87 | −0.55 to +0.34 | −1.16 | +0.32 | +0.16 | −0.30 to +0.62 |
| vehicle | 50 | 4.30 | 0.28 | −1.90 | −0.84 to +0.40 | −1.00 | +0.39 | +0.45 | −0.68 to +0.15 |
| weapon | 60 | 2.35 | 0.67 | −1.73 | −0.95 to +0.15 | −0.96 | +0.18 | +0.04 | −0.55 to +0.42 |

**Spearman of typicality and logit difference inside each category** (mean over the 10 categories):

| α | −1 | −0.5 | −0.2 | 0 | +0.2 | +0.5 | +1 |
|:-|-:|-:|-:|-:|-:|-:|-:|
| Typicality | 0.15 | 0.30 | 0.36 | 0.36 | 0.34 | 0.28 | 0.19 |
| Random (3 directions) | 0.16–0.27 | 0.23–0.32 | 0.30–0.34 | 0.36 | 0.35–0.41 | 0.29–0.41 | 0.16–0.38 |

The weapon values are the same as in the weapon-only run `20261105T161921Z` (same items, same α).

### Interpretation

1. **Subtracting the direction makes the model prefer the atypical poles, in all 10 categories.** At α = −1 the change is −1.02 to −1.90, more negative than all 3 random directions in every category (all items: −1.53 vs −0.75 to +0.09). The effect is almost linear in α for α ≤ 0 (−0.12, −0.27, −0.85, −1.53).
2. **Adding the direction has a small positive effect that saturates.** +0.09 at α = 0.1, +0.15 at 0.2, +0.20 at 0.5, then +0.07 at α = 1. Up to α = 0.5 it is larger than all random directions (at most +0.06); at α = 1 one random direction is larger (+0.38). Thus the effect is asymmetric: the direction can decrease the typicality judgement much more than it can increase it. Possibly the default reading of "the X is fairly ..." is already near the typical end.
3. **The direction moves all items of a category together; it does not make the grading sharper.** The change is almost the same for the three typicality thirds (at α = −0.5: −0.82, −0.87, −0.87). The correlation with typicality inside a category goes down in both directions of steering, about as much as for the random directions (α = −1: 0.15 vs 0.16 to 0.27). Thus the steering changes the general typical/atypical judgement, not the ranking of the items. The thirds show a small floor/ceiling effect: subtraction hits the most typical items a little more, addition helps the least typical items more.
4. **The behaviour without steering grades typicality only weakly** (mean ρ 0.36 inside a category; 0.01 for sport, 0.14 for vegetable, 0.16 for tool, 0.67 for weapon). The prompt is a weak read-out of typicality. The steering effect does not depend on how well the behaviour grades typicality: it is as large for sport (ρ 0.01, −1.02) and vegetable (ρ 0.14, −1.87) as for weapon (ρ 0.67, −1.73).
5. **Causal evidence, with limits.** The direction found by correlation (probes at the period token) has a specific causal effect at the item token on a typicality judgement in a new prompt format, larger than random directions of the same norm. But: only 3 random directions; one prompt template; the pole words can also carry other meaning (for example "poor" as quality, "rare" as frequency), so the direction can also be a frequency or familiarity direction (not tested). The asymmetry and the saturation at α = 1 are not explained.
6. **Next steps:** more random directions (10 or more, for a null distribution of the change); a direction from the probes with shuffled typicality (permutation null) as a control that has the same construction; other templates and pole words; steering at the period token of the training prompts.
