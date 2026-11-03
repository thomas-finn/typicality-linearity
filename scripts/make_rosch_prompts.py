"""Create the Rosch suites of prompts as CSVs in data/processed/rosch/."""

import argparse

from typicality_linearity.rosch_prompts import write_mean_contrast_suite, write_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", default="data/raw/rosch/rosch1975_ratings.csv")
    parser.add_argument("--out", default="data/processed/rosch/rosch_prompts.csv")
    parser.add_argument(
        "--mean-contrast-out",
        default="data/processed/rosch/rosch_mean_contrast.csv",
    )
    args = parser.parse_args()

    path = write_suite(args.ratings, args.out)
    print(f"Wrote {path}")
    path = write_mean_contrast_suite(args.ratings, args.mean_contrast_out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
