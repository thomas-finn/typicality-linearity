"""Ridge probes on Rosch prompts over a grid of regularisation values.

The same experiment as diff_means_experiment.py, with ridge probes in place of
difference-of-means probes. The settings grid is:

- alpha: a fixed ridge alpha, or "cv". With "cv" the alpha is chosen for each
  fit by inner cross-validation on the training items only, from
  `ridge_cv_alphas` (the method of the parent probe experiment).
- ambiguous: "include" or "exclude", as in diff_means_experiment.py.

The activations are standardised with the mean and std of the training items
when `ridge_standardise` is true. Alpha is then on the scale of the
standardised activations.

All outputs are the same as in diff_means_experiment.py. probe_weights.npz has
the directions in activation space (weights / scale). ridge_alphas.csv has the
chosen alphas of the "cv" settings.
"""

from typicality_linearity.diff_means_experiment import (
    COMMON_KEYS,
    ProbeFamily,
    check_common,
    read_config,
)
from typicality_linearity.diff_means_experiment import (
    run as run_family,
)
from typicality_linearity.diff_means_plot import AMBIGUOUS_LABELS

PROBE_TYPE = "ridge"
CV_ALPHA = "cv"
RIDGE_KEYS = ("ridge_alphas", "ridge_standardise")
REQUIRED_KEYS = COMMON_KEYS + RIDGE_KEYS


def parse_alpha(value):
    """Return CV_ALPHA or the alpha as a float."""
    if value == CV_ALPHA:
        return CV_ALPHA
    return float(value)


def alpha_text(alpha):
    """Return a short text for an alpha: "cv", "0.01", "3.16e6"."""
    if alpha == CV_ALPHA:
        return CV_ALPHA
    return f"{alpha:g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def setting_key(alpha, ambiguous):
    return f"a{alpha_text(alpha)}_{ambiguous}"


def settings_grid(config):
    """Return a list of dicts, one per setting, in a fixed order.

    For each ambiguous option, the "cv" setting (if any) is first, then the
    fixed alphas from small to large.
    """
    alphas = [parse_alpha(a) for a in config["ridge_alphas"]]
    fixed = sorted(a for a in alphas if a != CV_ALPHA)
    ordered = [CV_ALPHA] * (CV_ALPHA in alphas) + fixed
    grid = []
    for ambiguous in config["ambiguous"]:
        for alpha in ordered:
            grid.append(
                {
                    "key": setting_key(alpha, ambiguous),
                    "alpha": alpha,
                    "ambiguous": ambiguous,
                }
            )
    return grid


def fit_settings(setting, config):
    settings = {"ridge_standardise": bool(config["ridge_standardise"])}
    if setting["alpha"] == CV_ALPHA:
        settings["ridge_alphas"] = [float(a) for a in config["ridge_cv_alphas"]]
        settings["ridge_inner_folds"] = int(config["ridge_inner_folds"])
    else:
        settings["ridge_alpha"] = setting["alpha"]
    return settings


def check_ridge(config):
    alphas = [parse_alpha(a) for a in config["ridge_alphas"]]
    if not alphas:
        raise ValueError("ridge_alphas must not be empty")
    for alpha in alphas:
        if alpha != CV_ALPHA and alpha <= 0:
            raise ValueError("Each ridge alpha must be more than 0")
    if CV_ALPHA in alphas:
        for key in ("ridge_cv_alphas", "ridge_inner_folds"):
            if key not in config:
                raise ValueError(f"The 'cv' alpha needs the key {key!r}")
        if int(config["ridge_inner_folds"]) < 2:
            raise ValueError("ridge_inner_folds must be 2 or more")
        if any(float(a) <= 0 for a in config["ridge_cv_alphas"]):
            raise ValueError("Each ridge_cv_alphas value must be more than 0")


def ridge_probe_note(config):
    note = "Ridge regression of typicality on the activations"
    if config["ridge_standardise"]:
        note += ", standardised with the mean and std of the training items"
    note += ". Alpha is fixed for each setting"
    alphas = [parse_alpha(a) for a in config["ridge_alphas"]]
    if CV_ALPHA in alphas:
        cv_alphas = [float(a) for a in config["ridge_cv_alphas"]]
        note += (
            f"; 'cv': chosen for each fit by inner {config['ridge_inner_folds']}-fold "
            f"CV on the training items, from {min(cv_alphas):g} to {max(cv_alphas):g}"
        )
    return note + "."


def ridge_page_parameters(config):
    return [
        {
            "field": "alpha",
            "id": "alpha",
            "label": "Ridge alpha",
            "column": "Alpha",
            "labels": {CV_ALPHA: "Inner CV"},
            "prefix": "alpha = ",
        },
        {
            "field": "ambiguous",
            "id": "ambiguous",
            "label": "Ambiguous items",
            "column": "Ambiguous",
            "labels": AMBIGUOUS_LABELS,
        },
    ]


RIDGE_FAMILY = ProbeFamily(
    probe_type=PROBE_TYPE,
    required_keys=REQUIRED_KEYS,
    settings_grid=settings_grid,
    fit_settings=fit_settings,
    check=check_ridge,
    probe_note=ridge_probe_note,
    parameters=ridge_page_parameters,
)


def check_config(config):
    check_common(config, RIDGE_FAMILY)


def load_config(path):
    config = read_config(path, REQUIRED_KEYS)
    check_config(config)
    return config


def run(config, config_path):
    return run_family(config, config_path, RIDGE_FAMILY)
