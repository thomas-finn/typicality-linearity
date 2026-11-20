"""Create the Rosch suites of prompts as CSVs in data/processed/rosch/."""

import argparse

from typicality_linearity.rosch_prompts import (
    write_eval_suite,
    write_mean_cat_contrast_extended_suite,
    write_mean_cat_contrast_suite,
    write_mean_item_contrast_suite,
    write_nonce_suite,
    write_suite,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", default="data/raw/rosch/rosch1975_ratings.csv")
    parser.add_argument("--out", default="data/processed/rosch/rosch_prompts.csv")
    parser.add_argument(
        "--mean-cat-contrast-out",
        default="data/processed/rosch/rosch_mean_cat_contrast.csv",
    )
    parser.add_argument(
        "--mean-cat-contrast-extended-out",
        default="data/processed/rosch/rosch_mean_cat_contrast_extended.csv",
    )
    parser.add_argument(
        "--mean-item-contrast-out",
        default="data/processed/rosch/rosch_mean_item_contrast.csv",
    )
    parser.add_argument("--eval-out", default="data/processed/rosch/rosch_eval.csv")
    parser.add_argument("--nonce-out", default="data/processed/rosch/rosch_nonce.csv")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random contrast items in the mean item contrast file.",
    )
    args = parser.parse_args()

    path = write_suite(args.ratings, args.out)
    print(f"Wrote {path}")
    path = write_mean_cat_contrast_suite(args.ratings, args.mean_cat_contrast_out)
    print(f"Wrote {path}")
    path = write_mean_cat_contrast_extended_suite(
        args.ratings, args.mean_cat_contrast_extended_out
    )
    print(f"Wrote {path}")
    path = write_mean_item_contrast_suite(
        args.ratings, args.mean_item_contrast_out, args.seed
    )
    print(f"Wrote {path} (seed {args.seed})")
    path = write_eval_suite(args.ratings, args.eval_out)
    print(f"Wrote {path}")
    path = write_nonce_suite(args.ratings, args.nonce_out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
