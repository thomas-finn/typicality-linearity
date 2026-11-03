"""Compute the cosine similarity of probe directions from a finished probe run.

Reads <run_dir>/probe_weights.npz and writes <run_dir>/cosine_similarity.csv
(or the path given with --out). The probes are not fitted again. The directions
are in activation space: weights / scale (see probes.activation_direction).

Run from the repository root:

    uv run python scripts/probe_cosine_similarity.py <run_dir>
"""

import argparse
from pathlib import Path

import yaml

from typicality_linearity.probe_experiment import (
    cosine_rows_from_directions,
    load_probe_directions,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    # Use the prompt order of the run config, if there is one.
    prompt_order = None
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            prompt_order = yaml.safe_load(f).get("prompt_columns")

    directions = load_probe_directions(run_dir / "probe_weights.npz")
    rows = cosine_rows_from_directions(directions, prompt_order)
    out = Path(args.out) if args.out else run_dir / "cosine_similarity.csv"
    write_csv(rows, out)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
