# Activations for the global direction experiments: Qwen2.5-7B

## Question

This is a preliminary step, not an analysis. It extracts the residual stream activations of Qwen2.5-7B for the same prompts as the 1.5B step (`../qwen2-5-1-5B/`), so that the global direction experiments can be run for the larger model from the cache.

## Dataset

The same files as `../qwen2-5-1-5B/README.md`: all prompt files in `data/processed/rosch/`. 33 entries (110,175 prompts) are extracted with the model. The 18 entries of the two item contrast files are derived from the cache, without the model.

## Configuration

`config.yaml`. Differences from the 1.5B step:

- Model `Qwen/Qwen2.5-7B` (28 blocks, hidden size 3584). Layers 10, 15, 20, 25 are at the same depth as for 1.5B (also 28 blocks).
- dtype float32, the same as the 1.5B step, so the comparison with 1.5B changes only the model. The weights are about 30 GB.
- Batch size 64 (1.5B: 16), for a GPU with 40 GB or more.
- Output: `outputs/activations/global-direction/qwen2-5-7B/`. The cache directory in the config is the same (`outputs/activation_cache`). Each model has its own subdirectory, so the entries are in `outputs/activation_cache/Qwen2.5-7B/` and the 1.5B entries are in `outputs/activation_cache/Qwen2.5-1.5B/`.

All other settings (dtype, positions, prefix token, derive_from, seed) are the same.

## Requirements

- Not for the local 16 GB Mac: the float32 weights are about 30 GB. Use a CUDA GPU with 40 GB or more (A100 40 GB or 80 GB, H100). An L4 (24 GB) or T4 (16 GB) is not sufficient.
- Disk: about 11 GB for the cache entries (n_prompts × 4 layers × 3584 × 4 bytes, 194,925 rows with the derived entries), and about 15 GB for the model download.
- Estimated time (scaled from the 1.5B run, 84 min on M1 MPS): about 30-60 min on an A100 (float32 is about 2 times slower than bfloat16).

## Command

From the repository root, on the GPU machine:

    uv run python scripts/extract_rosch_activations.py experiments/activations/qwen2-5-7B/config.yaml

An entry that is already in the cache is not made again, so a run that stops can be started again. After the run, copy `outputs/activation_cache/Qwen2.5-7B/` and `outputs/activations/global-direction/qwen2-5-7B/` to the machine that runs the experiments.

## Result

To do: the prompts changed on 2026-11-18 (plural agreement, see "Plural items" in `data/processed/rosch/README.md`). The 7B cache does not have the 11 changed entries (5 training columns of `rosch_prompts.csv` and `rosch_mean_cat_contrast.csv`, `rosch_eval_3`), so run this config again; the other entries come from the cache.

Run on Colab, 2026-11-10. Completed run: `20261110T145109Z` (git `88c73a3`, clean; torch 2.15.0, transformers 5.22.1, CUDA). The config was not changed.

- 51 entries, 211,875 rows, 12.2 GB. All entries have shape (n, 4, 3584), float32, and no NaN or inf values.
- Earlier runs on the same day: `20261110T143322Z` (stopped during the model download), `20261110T143440Z` (CUDA out of memory on a 22 GB GPU (L4), as expected for float32), `20261110T143743Z` (24 entries extracted, then stopped by hand). The completed run used these 24 entries from the cache, extracted 9 entries and derived 18 entries. Time of the completed run: 10 min.
- The cache entries were written to Google Drive from Colab and copied to the local `outputs/activation_cache/`. On 2026-11-10 the cache was split by model (`activation_cache.model_cache_dir`) and the entries were moved to `outputs/activation_cache/Qwen2.5-7B/`. The `npz_path` values in the manifests of these runs show the old flat paths.

## Interpretation

This is a preliminary step. The cache is complete, so the global direction experiments can be run for Qwen2.5-7B.
