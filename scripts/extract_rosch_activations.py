"""Extract activations for all Rosch prompts into the activation cache.

Run from the repository root:

    uv run python scripts/extract_rosch_activations.py <path to config.yaml>
"""

import argparse

from typicality_linearity.activation_extraction import load_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = run(config, args.config)
    print(f"Wrote the manifest and metadata to {run_dir}")


if __name__ == "__main__":
    main()
