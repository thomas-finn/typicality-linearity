# Experiments

| Experiment | Directory | Question | Status | Last run | Result |
|:-|:-|:-|:-|:-|:-|
| PCA Rosch, Qwen2.5-1.5B, no baselines | `experiments/pca/rosch/qwen2-5-1-5B/no-baselines/` | Is typicality visible in the top PCs at the item token and the period token? | Run, not analysed | 2026-11-03 (`20261103T113646Z`, prefix token, per-category PCA) | - |
| PCA Rosch, Qwen2.5-1.5B, mean contrast | `experiments/pca/rosch/qwen2-5-1-5B/mean-contrast/` | After subtracting the mean activation of the item with the other categories, is typicality more visible in the top PCs at the period token? | Run, not analysed | 2026-11-03 (`20261103T130220Z`, with no-baseline comparison) | - |
| Probes Rosch, Qwen2.5-1.5B, layer 15 | `experiments/probes/rosch/qwen2-5-1-5B/` | Do difference-of-means and ridge probes at the period token grade typicality, and do they transfer between categories and between prompt formats (no baseline and mean contrast)? | Not run | - | - |
