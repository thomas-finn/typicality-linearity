# PCA by activation type: Qwen2.5-7B

## Question

The same question as `../qwen2-5-1-5B/README.md`, for Qwen2.5-7B: how is typicality visible in the first principal components of the residual stream at the period token, for the four activation types? How many components explain 95% of the variance, by category, prompt format and layer? How do the counts compare with Qwen2.5-1.5B?

## Dataset

The same as `../qwen2-5-1-5B/README.md` (565 target rows, 15 training prompt formats). Activations from the cache made by `experiments/activations/qwen2-5-7B/` (`outputs/activation_cache/Qwen2.5-7B/`, layers 10, 15, 20, 25, period token, float32).

## Configuration

`config.yaml`. The same as the 1.5B config, except `experiment`, `title`, `model` (`Qwen/Qwen2.5-7B`), `output_dir` and `html_name`. Hidden size 3584 (1.5B: 1536). The layers are at the same depth (both models have 28 blocks). See `../qwen2-5-1-5B/README.md` for the four activation types and the fits (2640 fits).

## Command

From the repository root:

    uv run python scripts/run_global_pca_rosch.py experiments/pca/qwen2-5-7B/config.yaml

Outputs in `outputs/pca/global-direction/qwen2-5-7B/<run_id>/`: `config.yaml`, `metadata.json`, `variance_components.csv` and `qwen2-5-7B-rosch-pca.html`. A copy of the page goes to `experiments/pca/qwen2-5-7B-rosch-pca.html` (in `.gitignore`). The page has the same format as the 1.5B page.

## Result

Run `20261110T154458Z` (2026-11-10, 2 min, local CPU, all activations from the cache), status completed. 2640 fits, all counts in [1, n − 1]. Page: 6.0 MB. Git `88c73a3` with uncommitted changes (`dirty: true`): the change that puts the activation cache in one subdirectory for each model. This change does not change the activations or the PCA.

Components for 95% of the variance, all categories (565 points), median over the 15 prompt formats [min–max]. In brackets after the median: the 1.5B median (`../qwen2-5-1-5B/README.md`).

| Type | Layer 10 | Layer 15 | Layer 20 | Layer 25 |
|:-|:-|:-|:-|:-|
| Raw | 154 [138–240] (109) | 172 [128–234] (114) | 170 [93–229] (92) | 210 [112–276] (135) |
| Mean category contrast | 112 [83–239] (69) | 149 [111–253] (100) | 167 [86–255] (87) | 188 [90–285] (91) |
| Mean item contrast | 230 [192–293] (155) | 237 [180–279] (149) | 217 [119–270] (124) | 250 [171–307] (172) |
| Both contrasts | 242 [192–327] (177) | 240 [191–311] (164) | 230 [135–300] (142) | 266 [166–339] (191) |

One category (50 to 60 points), median over 10 categories × 15 prompts: 36 to 41 for all types and layers (raw 38 / 37 / 36 / 39; 1.5B: 34 / 33 / 30 / 37).

PC1 explained ratio, all categories, layer 15, median over prompts: raw 0.20, category contrast 0.19, item contrast 0.17, both 0.17 (1.5B: 0.19 / 0.22 / 0.18 / 0.20).

The projection plots are not yet analysed.

## Interpretation

Preliminary, from the counts only:

- 7B needs about 1.4 to 1.9 times more components than 1.5B for all types and layers. The hidden size is 2.3 times larger (3584 against 1536), so the count increases less than the hidden size.
- The order of the types is the same as for 1.5B: category contrast < raw < item contrast < both contrasts, at all layers.
- The layer pattern is different. For 1.5B, layer 20 needs the fewest components for all types. For 7B, layer 10 needs the fewest for raw and category contrast, and the count increases with depth for the category contrast (112 → 188). Layer 25 needs the most for all types, the same as for 1.5B.
- The spread over prompt formats is large again (for example raw layer 20: 93 to 229).
- In one category the counts (36 to 41) are near the maximum (n − 1 = 49 to 59), so they do not separate the types or the models well.
