# Exploration of difference of means

In this experiment we explore different ways of refining the difference-of-means probes from the initial `qwen2-5-1-5B` experiment (`experiments/probes/rosch/qwen2-5-1-5B/`).

### Question

1. Does a different split (halves, thirds, quarters) or a weighting towards the ends of the typicality scale give a better difference-of-means direction, within a category and in transfer between categories and prompts?
2. Do items with an ambiguous word meaning (for example *bat*, *plane*) make the probes worse? Is this a cause of the bad result for tool and vegetable?
3. Is there a general typicality direction in the residual stream? Does a direction from the training prompts grade typicality on new prompt formats that were never used for training, with raw activations (no baseline)? Are the results above chance (permutation null)?

### Dataset

`data/processed/rosch/rosch_mean_contrast.csv` (565 items, 10 categories, 7 prompt formats), the same as the parent experiment. A new column `is_ambiguous` is added to `rosch_prompts.csv` and `rosch_mean_contrast.csv`.

An item is ambiguous when its word, as written in the prompt, has a common second meaning that is not the rated meaning: *bat* (bird) can be a baseball bat, *plane* (tool) can be an aircraft. Items that are weak members of the category with the same meaning are not ambiguous: *car* as a toy, *hammer* as a weapon. The list is `AMBIGUOUS_ITEMS` in `src/typicality_linearity/rosch_prompts.py`, with the second meaning of each item. 72 items are flagged (furniture 13, tool 11, bird 8, toy 7, clothing 7, sport 6, vehicle 6, weapon 6, fruit 5, vegetable 3). *wood* (tool) and *picture* (furniture) were added after the first run (2026-11-05).

The cache key of the activations uses only the prompt and item texts, so the new column does not change the activations.

Evaluation prompts: `data/processed/rosch/rosch_eval.csv`, 3 formats that are never used for training. Only the 565 rows of the item's own category are used (no baseline is applied to them).

| Name | Format |
|:-|:-|
| rosch_eval_1 | "A ITEM is a kind of CATEGORY." |
| rosch_eval_2 | "When you think about CATEGORIES, what comes to mind first for me is a ITEM." |
| rosch_eval_3 | "Slow down there, I do not think that a ITEM is a CATEGORY, do you think that." |

### Configuration

`config.yaml`. Layer 15, period token, baselines none and mean contrast, 5-fold CV per category, seed 0, as in the parent experiment. Only difference-of-means probes (no ridge).

All combinations of these settings (42 settings × 2 baselines):

| Setting | Values | Meaning |
|:-|:-|:-|
| `n_groups` | 2, 3, 4 | The probe uses the top and bottom 1/n of the items: halves, thirds (the parent method), quarters. |
| `weight_constants` | 0, 0.25, 0.5, 1, 2, 5, 10 | The group means are weighted. c = 0 is no weighting. c = 2, 5, 10 were added in the third run. |
| `ambiguous` | include, exclude | exclude removes the ambiguous items from training **and** evaluation. The folds are made again on the remaining items. |

Weighting, with typicality t = −`rating_zscore`:

    direction = Σ_top w_i x_i − Σ_bottom v_i x_i
    w_i ∝ exp(c · t_i)   (top group, the most typical items count more)
    v_i ∝ exp(−c · t_i)  (bottom group, the least typical items count more)

Evaluation prompts (`eval_prompts_csv`, `eval_prompt_columns`): all probes and mean directions are trained on the 7 training prompts, with or without the mean contrast baseline. They are evaluated on the **raw** activations of the evaluation prompts. The score is linear, so a direction trained with mean contrast can be applied to raw activations. For the item's own category, the fold probes that did not see the item are used.

Permutation null (`null_settings`: thirds, c = 0, include and quarters, c = 1, exclude; `n_permutations`: 200): the typicality values are shuffled inside each category (the same shuffle in all prompts), and all probes and statistics are computed again. The folds do not change. p is one-sided, (1 + number of null values ≥ observed) / 201, so the smallest possible p is 0.005.

Each set of weights is normalised to sum to 1. `rating_zscore` is in [−1.8, 3.5], so with c = 1 the largest weight ratio in the bottom group is about 20. `weight_groups: top` weights only the top group. It is not used.

Effective number of items in each group (Kish, 1 / Σw², mean over categories, all items):

| c | Thirds, top / bottom | Quarters, top / bottom |
|-:|-:|-:|
| 0 | 18.6 / 18.6 | 13.8 / 13.8 |
| 0.5 | 18.3 / 16.7 | 13.7 / 12.8 |
| 1 | 17.5 / 12.9 | 13.3 / 10.5 |
| 2 | 15.0 / 7.7 | 12.0 / 7.0 |
| 5 | 8.4 / 3.3 | 7.7 / 3.3 |
| 10 | 4.5 / 2.2 | 4.5 / 2.2 |

The bottom group is reduced faster because the z-scores of atypical items have a larger range. At c = 5 and 10, thirds and quarters give almost the same direction, because the same few items at the ends get almost all the weight.

### Command

    uv run python scripts/run_diff_means_rosch.py experiments/probes/rosch/qwen2-5-1-5B-diff-means/config.yaml

Outputs in `outputs/probes/rosch/qwen2-5-1-5B-diff-means/<run_id>/`: `summary.csv` (mean values for each setting), `correlations.csv`, `cosine_similarity.csv`, `mean_direction_best.csv` (the 10 best subsets of categories for each setting, baseline and metric), `probe_weights.npz` and `diff-means-explore.html`. The HTML page has one selector for each setting (split, weight constant, ambiguous items, metric) and a summary table of all settings (click a row to select it). The format of the selected prompt is shown above the category plots, and a key of all prompt formats above the prompt plots. The section "Transfer to the evaluation prompts" shows the probe of each training prompt and category evaluated on the evaluation prompts. The mean-direction section has the mean direction over a choice of categories and all training prompts (check boxes), evaluated on each training and evaluation prompt: its score in each prompt and evaluation category, the mean held-out score for each number of selected categories (training and evaluation prompts), and the 10 best subsets (ranked on the training prompts, with their held-out score on the evaluation prompts; a Select button loads a subset in the check boxes). The last section is the permutation null. Also `null_distribution.csv` and `null_summary.csv`.

The first run with the evaluation prompts loads the model to extract their activations (3 × 565 prompts, about 1 minute on MPS); later runs use the cache. The null takes most of the time (about 20 minutes in total).

### Result

Run `20261105T123218Z_probes-rosch-qwen2-5-1-5B-diff-means` (2026-11-05), with the evaluation prompts and the permutation null. Status completed, seed 0. Training activations from the cache; evaluation activations from the model (MPS). The git tree was dirty (the code of this experiment was not committed). All values of the previous run `20261105T110859Z` are the same (maximum difference 0).

Earlier runs: `20261105T110859Z` had the same settings without the evaluation prompts and the null. `20261105T103058Z` had 70 ambiguous items (without *wood* and *picture*). `20261105T103951Z` had 72 ambiguous items and c up to 1. `20261105T104138Z` had the same settings, without the mean-direction section. `20261105T105325Z` had a mean direction for each prompt alone. The include settings are the same in all runs, and the settings with 72 ambiguous items give the same values in all runs that have them.

**Check.** Thirds, c = 0, include gives the same values as the parent run `20261103T135510Z` for all 4760 difference-of-means cells (maximum difference 8e−15).

Spearman, mean over cells:

| Baseline | Split | c | Ambiguous | Within (CV) | Between categories | Between prompts | Cosine, categories |
|:-|:-|-:|:-|-:|-:|-:|-:|
| None | thirds | 0 | include | 0.36 | 0.25 | 0.19 | 0.18 |
| None | quarters | 1 | include | 0.36 | 0.29 | 0.19 | 0.27 |
| None | quarters | 1 | exclude | 0.35 | 0.33 | 0.18 | 0.28 |
| MC | thirds | 0 | include | 0.49 | 0.39 | 0.40 | 0.31 |
| MC | halves | 0 | include | 0.49 | 0.37 | 0.39 | 0.27 |
| MC | quarters | 1 | include | 0.49 | 0.43 | 0.40 | 0.42 |
| MC | thirds | 0 | exclude | 0.54 | 0.46 | 0.44 | 0.36 |
| MC | thirds | 1 | exclude | 0.54 | 0.48 | 0.44 | 0.42 |

Effect of the weight constant (MC, quarters; the values for thirds are almost the same at c ≥ 2):

| c | Include: within | Include: between categories | Include: between prompts | Include: cosine | Exclude: within | Exclude: between categories |
|-:|-:|-:|-:|-:|-:|-:|
| 0 | 0.495 | 0.414 | 0.401 | 0.363 | 0.525 | 0.464 |
| 1 | 0.492 | 0.429 | 0.404 | 0.423 | 0.528 | 0.476 |
| 2 | 0.488 | 0.433 | 0.401 | 0.430 | 0.526 | 0.478 |
| 5 | 0.484 | 0.427 | 0.390 | 0.380 | 0.497 | 0.469 |
| 10 | 0.475 | 0.415 | 0.380 | 0.323 | 0.472 | 0.453 |

- **Split and weighting do not change the within-category result for c ≤ 2.** The include settings with mean contrast and c ≤ 2 are in 0.49 to 0.50. With c = 5 and 10, all values decrease (MC include within 0.48, then 0.47; exclude within 0.50, then 0.47).
- **More extreme contrasts increase transfer between categories, up to c = 2.** With mean contrast and include, from 0.37 (halves, c = 0) to 0.43 (quarters, c = 1 or 2). At c = 10 it goes down to 0.42 and the cosine to 0.32. The cosine similarity between the category directions increases from 0.27 to 0.42. Transfer between prompts does not change (0.39 to 0.41).
- **Excluding ambiguous items increases all values with mean contrast:** within 0.49 → 0.54, between categories 0.39 → 0.46, between prompts 0.40 → 0.44 (thirds, c = 0). Without a baseline, only transfer between categories increases.
- By category (MC, thirds, c = 0, within CV, include → exclude): toy 0.50 → 0.64, vegetable 0.17 → 0.30, furniture 0.50 → 0.58, weapon 0.76 → 0.82, tool 0.05 → 0.07. Fruit decreases (0.54 → 0.44). Vehicle does not change.
- **Single items change the per-category values.** In the first run (without *wood* and *picture* flagged), the exclude values were tool 0.16 and furniture 0.64. When these two items are also removed, tool goes down to 0.07 and furniture to 0.58. The mean values change by less than 0.02.

**Training or evaluation effect of exclusion.** Exclusion changes the items in training and the items in evaluation. The full probes (off-diagonal cells, MC, thirds, c = 0) were evaluated on the non-ambiguous items only (script not saved in the repository):

| Train | Evaluate | Between categories |
|:-|:-|-:|
| include | all items | 0.39 |
| include | non-ambiguous items | 0.43 |
| exclude | non-ambiguous items | 0.46 |

About half of the increase comes from the easier evaluation set, and half from the cleaner training set. With c = 1 the training part is smaller (0.46 → 0.48). The values are the same in both runs to two decimals.

**Mean direction over a choice of categories and all prompt formats.** The direction is the mean of the unit directions of the probes of the selected categories in all 7 prompt formats (one direction). It is evaluated on the activations of each prompt format. For a selected category, its own directions are the cross-validated fold probes; the folds are the same in all prompts, so no item is scored with a direction that saw it in any prompt. (Run `20261105T105325Z` used one mean direction for each prompt, from the probes of that prompt only.)

Mean held-out Spearman over all subsets of each size (averaged over evaluation prompts and held-out categories):

| Setting | Baseline | 1 | 2 | 3 | 5 | 7 | 9 |
|:-|:-|-:|-:|-:|-:|-:|-:|
| thirds, c = 0, include | None | 0.28 | 0.35 | 0.39 | 0.43 | 0.45 | 0.46 |
| thirds, c = 0, include | MC | 0.40 | 0.46 | 0.48 | 0.49 | 0.49 | 0.49 |
| quarters, c = 1, include | None | 0.34 | 0.40 | 0.43 | 0.45 | 0.46 | 0.47 |
| quarters, c = 1, include | MC | 0.44 | 0.47 | 0.48 | 0.49 | 0.49 | 0.49 |
| quarters, c = 1, exclude | None | 0.38 | 0.44 | 0.47 | 0.49 | 0.50 | 0.51 |
| quarters, c = 1, exclude | MC | 0.50 | 0.52 | 0.53 | 0.53 | 0.54 | 0.54 |

Compared with the mean direction of each prompt alone (previous run), averaging over the prompts increases the values without a baseline (thirds, c = 0: 0.25 → 0.28 with 1 category, 0.44 → 0.46 with 9) and changes them little with mean contrast (0.39 → 0.40, 0.49 → 0.49).

All 10 categories selected (thirds, c = 0, include; cross-validated, mean over the categories): no baseline 0.49 (by prompt 0.31 for `rosch_fav` to 0.56 for `rosch_example`), mean contrast 0.51 (0.46 for `rosch_fav` to 0.54 for `rosch_story_2`). The probes of each category and prompt alone give 0.36 (no baseline) and 0.49 (mean contrast).

Best subsets (ranked by the advantage over subsets of the same size, with 3 or more held-out categories; see `mean_direction.py`):
- With mean contrast, the best subset is one category in 39 of 42 settings, mostly fruit, furniture or vehicle (thirds, c = 0, include: fruit +0.09, clothing +0.08, furniture +0.08). Bird, sport, tool, toy and weapon are never in a top-3 subset.
- Without a baseline, the best subset is one category in 26 settings and 2 or 3 categories in 16, mostly with furniture (in 46% of the top-3 subsets), vehicle (43%) or vegetable (36%), for example furniture + vehicle (+0.12).
- The advantage compares subsets of the same size only. Single categories vary most, so they are often at the top. The best subset is selected from about 1000 on the same data, so its value is optimistic.

**Transfer to the evaluation prompts** (Spearman; raw activations; loco = mean direction over the other 9 categories and all 7 training prompts, evaluated on the held-out category; all = mean direction over all 10 categories, cross-validated for the own category):

| Setting | Baseline (training) | Single probes, own category (CV) | Single probes, between categories | Mean dir. loco, train prompts | Mean dir. loco, eval prompts | Mean dir. all, eval prompts |
|:-|:-|-:|-:|-:|-:|-:|
| thirds, c = 0, include | None | 0.21 | 0.20 | 0.46 | 0.50 | 0.52 |
| thirds, c = 0, include | MC | 0.19 | 0.23 | 0.49 | 0.53 | 0.52 |
| quarters, c = 1, include | MC | 0.21 | 0.28 | 0.49 | 0.53 | 0.52 |
| thirds, c = 0, exclude | MC | 0.21 | 0.28 | 0.54 | 0.56 | 0.55 |
| quarters, c = 1, exclude | None | 0.20 | 0.25 | 0.51 | 0.54 | 0.53 |
| quarters, c = 1, exclude | MC | 0.21 | 0.30 | 0.54 | 0.56 | 0.55 |

By evaluation prompt (mean direction loco; thirds, c = 0, include, MC; quarters, c = 1, exclude, MC in brackets): `rosch_eval_1` 0.52 (0.58), `rosch_eval_2` 0.67 (0.68), `rosch_eval_3` 0.40 (0.41). The single probes are weakest on `rosch_eval_3` (0.08 with MC, 0.14 without a baseline).

By held-out category on the evaluation prompts (thirds, c = 0, include, MC): clothing 0.72, weapon 0.68, sport 0.63, vehicle 0.63, furniture 0.59, fruit 0.51, toy 0.46, vegetable 0.46, bird 0.36, tool 0.22.

**Permutation null** (200 shuffles, both null settings, both baselines, both metrics): every statistic is above all 200 null values (p = 0.005, the smallest possible). The null means are about 0 and the 95th percentiles are 0.01 to 0.07. The exception is the cosine between prompts: the null mean is 0.13 to 0.17 and the observed values are only 0.20 to 0.22. The same item groups are used in all prompts, so even random groups give similar directions between prompts. Thus the cosine between prompts is not evidence for a shared typicality direction.

| Statistic (Spearman, thirds, c = 0, include, MC) | Observed | Null mean | Null 95% | Null max |
|:-|-:|-:|-:|-:|
| Within (CV) | 0.49 | 0.00 | 0.05 | 0.08 |
| Between categories | 0.39 | 0.00 | 0.02 | 0.06 |
| Eval prompts, single probes, own category | 0.19 | 0.00 | 0.04 | 0.06 |
| Mean dir. loco, eval prompts | 0.53 | 0.00 | 0.06 | 0.10 |
| Cosine between categories | 0.31 | 0.00 | 0.03 | 0.06 |
| Cosine between prompts | 0.21 | 0.13 | 0.15 | 0.17 |

### Interpretation

1. **The split and the weighting are not important within a category, until the weights are too concentrated.** For c ≤ 2 the CV correlation does not change. At c = 5 and 10 the direction comes from about 2 to 8 items in each group, and all values decrease. The best range is c = 1 to 2: transfer between categories and the cosine are highest and the within-category value decreases by less than 0.01.
2. **More extreme contrasts make directions that are more similar between categories, and they transfer a little better.** Quarters or c > 0 increase the cosine similarity of the category directions by more (0.31 → 0.42) than transfer (0.39 → 0.43). Possibly the most typical and least typical items of all categories share a feature (for example familiarity or frequency), and an extreme contrast puts more weight on it.
3. **Ambiguous items add noise, mostly in toy and furniture.** But about half of the increase comes from removing these items from the test set: the probe grades them badly. Tool is still near 0 after exclusion (0.07), so ambiguity is not the cause of the bad tool result. The tool and furniture values change by 0.06 to 0.09 when one item is removed, so the per-category differences are not reliable. Fruit becomes worse, possibly because the 5 removed items (orange, apple, date, nut, squash) contain some of its most typical items.
4. **One mean direction over categories and prompts grades typicality well in all prompts and categories.** The held-out score increases with the number of categories (MC: 0.40 with one, 0.48 with three, 0.49 with nine; no baseline: 0.28 to 0.46), and is almost constant after five. Averaging over prompts helps most without a baseline, where the directions of each prompt are the most different. With all categories, the mean direction (cross-validated) gives 0.49 to 0.51, better than the probe of each category and prompt alone (0.36 without a baseline). Thus averaging removes the parts of each direction that are specific to the category or the prompt, and keeps a shared typicality direction. The choice of categories is less important than their number: no subset is much better than the mean of its size (advantage at most about 0.12), and with the selection effect this is possibly noise. `rosch_fav` is the prompt where the shared direction works least, mostly without a baseline (0.31).
5. **The mean direction transfers to new prompt formats; single probes do not.** On the evaluation prompts, which were never used for training, the mean direction from 9 other categories gives 0.50 to 0.56 on a held-out category, the same as or better than on the training prompts (0.46 to 0.54). A single probe of one category and one prompt gives only about 0.2 on the evaluation prompts, also in its own category. Averaging over categories and prompts is necessary: each single direction is mostly specific to its category and prompt, and the shared part is small in each one.
6. **No baseline is necessary at evaluation.** The directions are evaluated on raw activations. Directions trained with mean contrast still work (0.53), so mean contrast helps to find the direction but is not necessary to read it. Directions trained without a baseline give almost the same values (0.50 to 0.54).
7. **The prompt format changes the strength.** `rosch_eval_2` ("what comes to mind first … is ITEM") gives the best values of all prompts (0.67). This prompt asks almost directly for the most typical item. `rosch_eval_3` (a negation with a long, unusual frame) gives 0.40. Negation is not the cause alone: the training negations `rosch_neg` and `rosch_story_neg` give 0.49 to 0.50.
8. **The results are not chance.** All statistics are far above the permutation null. But the null tests only "no relation between activations and typicality". It does not test confounds such as word frequency, familiarity or item–category association, which correlate with typicality.
9. **Answer to question 3.** There is a shared direction at layer 15 (period token) that grades typicality in held-out categories and in new prompt formats, from raw activations, at Spearman about 0.5 (0.4 to 0.7 by prompt). It is weak for tool (0.2) and bird (0.4). It is not yet shown that this direction is typicality and not a correlate of typicality.
10. **Cautions.**
   - Exclusion removes 72 of 565 items. The per-category values have a large error (about ±0.25 at 95% for ρ = 0.4 with 50 items).
   - The ambiguity flags are a judgement. Edge cases are not flagged (see below).
   - Off-diagonal cells use probes trained on all items. Diagonal cells use 80%. With quarters, a CV fold has about 11 items in each group.

**Edge cases not flagged** (to decide): bird: crow, chicken, hawk, thrush; toy: rattle, cards, checkers; sport: boxing, football, baseball, checkers, cards, horseshoes; vegetable: corn, pickle, mushroom; fruit: lime, plum, peach, lemon, blackberry, prune, olive, cherry; tool: brace, drill, bench, brush, rasp, screw, wedge; clothing: dress, boot, purse, cane, pants, nylons, coat, belt; vehicle: trolley, rocket, ship, carriage, raft; furniture: table, bed, bench, chair, closet, drawer, davenport; weapon: rocket, bullet, lance, whip, rock.

**Possible next steps:** control for word frequency and item–category association (partial correlations), because the shared direction can be a correlate of typicality; evaluate the mean direction at layers 10, 20 and 25 (in the cache); a causal test (steering along the mean direction); decide the edge cases and run again; compare the mean direction with one probe trained on the items of all categories together, tested on held-out categories.
