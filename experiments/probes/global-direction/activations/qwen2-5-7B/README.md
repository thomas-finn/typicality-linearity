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
- Output: `outputs/activations/global-direction/qwen2-5-7B/`. The cache directory is the same (`outputs/activation_cache`); the cache key has the model and dtype, so the entries do not replace the 1.5B entries.

All other settings (dtype, positions, prefix token, derive_from, seed) are the same.

## Requirements

- Not for the local 16 GB Mac: the float32 weights are about 30 GB. Use a CUDA GPU with 40 GB or more (A100 40 GB or 80 GB, H100). An L4 (24 GB) or T4 (16 GB) is not sufficient.
- Disk: about 11 GB for the cache entries (n_prompts × 4 layers × 3584 × 4 bytes, 194,925 rows with the derived entries), and about 15 GB for the model download.
- Estimated time (scaled from the 1.5B run, 84 min on M1 MPS): about 30-60 min on an A100 (float32 is about 2 times slower than bfloat16).

## Command

From the repository root, on the GPU machine:

    uv run python scripts/extract_rosch_activations.py experiments/probes/global-direction/activations/qwen2-5-7B/config.yaml

An entry that is already in the cache is not made again, so a run that stops can be started again. After the run, copy `outputs/activation_cache/` (the new entries) and `outputs/activations/global-direction/qwen2-5-7B/` to the machine that runs the experiments.

## Result

Not yet run.

## Interpretation

Not yet run.
