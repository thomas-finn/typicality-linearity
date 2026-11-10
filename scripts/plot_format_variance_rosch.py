"""Write the HTML page of a format-variance run from its output files.

The model is not loaded. Run from the repository root:

    uv run python scripts/plot_format_variance_rosch.py <run directory>
"""

import argparse

from typicality_linearity.format_variance_plot import write_run_page
from typicality_linearity.rosch_prompts import EVAL_PROMPT_FORMATS, PROMPT_FORMATS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--html-name", default="format-variance.html")
    args = parser.parse_args()

    path = write_run_page(
        args.run_dir, args.html_name, PROMPT_FORMATS, EVAL_PROMPT_FORMATS
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
