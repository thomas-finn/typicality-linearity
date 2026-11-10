"""Run the global direction probes pooled over categories and over prompts.

Run from the repository root:

    uv run python scripts/run_global_pooled_probes_rosch.py <path to config.yaml>
"""

import argparse

from typicality_linearity.global_pooled_probe_experiment import load_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = run(config, args.config)
    print(f"Wrote results to {run_dir}")


if __name__ == "__main__":
    main()
