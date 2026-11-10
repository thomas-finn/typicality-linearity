"""Difference-of-means probe variants on Rosch prompts.

For each setting (split, weight constant, ambiguous items) and each baseline,
fit difference-of-means probes and measure transfer between categories and
between prompts, as in probe_experiment.py.

- split: n_groups. The probe uses the top and bottom 1/n_groups of the items
  (2: halves, 3: thirds, 4: quarters).
- weight constant c: the group means are weighted by exp(c * typicality). See
  probes.fit_diff_of_means. c = 0 is no weighting.
- ambiguous: "include" uses all items. "exclude" removes the items with
  `is_ambiguous_strict` true from training and evaluation. The folds are made again
  on the remaining items with the same seed.

Evaluation prompts (`eval_prompt_columns` in `eval_prompts_csv`) are never used
for training. All probes and mean directions are evaluated on their raw
activations, with no baseline (see eval_transfer.py).

Permutation null: for the settings in `null_settings`, the typicality values
are shuffled inside each category (the same shuffle for all prompts) and the
whole analysis is done again `n_permutations` times. The folds do not change.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml, metadata.json
- summary.csv: one row per setting, baseline and metric with the mean values
- correlations.csv: one row per matrix cell
- cosine_similarity.csv: one row per pair of probe directions
- probe_weights.npz: full-data directions, key
  "<setting>/<baseline>/<category>/<prompt>"
- mean_direction_best.csv: the subsets of categories whose mean direction
  transfers best, for each setting, baseline and metric (see mean_direction.py)
- null_distribution.csv: one row per permutation and statistic
- null_summary.csv: observed value, null mean, 95th percentile and p-value
- ridge_alphas.csv: only if a setting chooses the ridge alpha by inner CV
- <html_name>: interactive heatmaps, one selector for each setting

The probe type and its settings grid are given by a ProbeFamily. DOM_FAMILY
(difference of means) is the default. ridge_experiment.py gives the ridge
family and uses the same run function.
"""

import base64
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.baselines import is_true
from typicality_linearity.diff_means_plot import (
    build_plot_data,
    dom_parameters,
    write_html,
)
from typicality_linearity.eval_transfer import (
    eval_summary,
    eval_transfer,
    null_summary,
    permute_within,
)
from typicality_linearity.mean_direction import (
    all_subsets,
    held_out_by_size,
    item_projections,
    leave_one_out_scores,
    quantise,
    rank_subsets,
    subset_scores,
)
from typicality_linearity.pca_experiment import (
    LazyModel,
    get_activations,
    read_prompt_rows,
)
from typicality_linearity.probe_experiment import (
    BASELINE_LABELS,
    baseline_name,
    correlation_rows,
    cosine_rows,
    point_indices_for,
    probe_activations,
    write_csv,
)
from typicality_linearity.probe_transfer import AGGREGATES, transfer_matrices
from typicality_linearity.probes import (
    METRICS,
    WEIGHT_GROUPS,
    activation_direction,
    make_folds,
)
from typicality_linearity.rosch_prompts import EVAL_PROMPT_FORMATS, PROMPT_FORMATS
from typicality_linearity.run_metadata import (
    make_run_id,
    now_utc,
    set_seed,
    start_metadata,
    write_metadata,
)

PROBE_TYPE = "diff_of_means"
AMBIGUOUS_OPTIONS = ("include", "exclude")
SPLIT_LABELS = {2: "Halves", 3: "Thirds", 4: "Quarters"}

# Keys that all probe families need.
COMMON_KEYS = (
    "experiment",
    "model",
    "prompts_csv",
    "prompt_columns",
    "layers",
    "positions",
    "probe_layer",
    "probe_position",
    "baselines",
    "target",
    "target_sign",
    "metrics",
    "n_folds",
    "cv_aggregate",
    "ambiguous",
    "batch_size",
    "device",
    "dtype",
    "prefix_token",
    "seed",
    "activation_cache_dir",
    "output_dir",
    "html_name",
    "eval_prompts_csv",
    "eval_prompt_columns",
    "null_settings",
    "n_permutations",
)
DOM_KEYS = ("n_groups", "weight_constants", "weight_groups")
REQUIRED_KEYS = COMMON_KEYS + DOM_KEYS

# The order of the statistics in summary.csv and on the page.
STATISTICS = (
    "within_cv",
    "between_categories",
    "between_prompts",
    "eval_within",
    "eval_between_categories",
    "mean_loco_train",
    "mean_loco_eval",
    "mean_all_train",
    "mean_all_eval",
    "cosine_between_categories",
    "cosine_between_prompts",
)


@dataclass
class ProbeFamily:
    """The parts of the experiment that depend on the probe type.

    - settings_grid(config): list of setting dicts. Each has "key",
      "ambiguous" and the other parameter fields, in the column order.
    - fit_settings(setting, config): the settings for probes.fit_probe
    - check(config): raise ValueError for a bad config of this family
    - probe_note(config): one note for the page that describes the probe
    - parameters(config): the page selectors (see diff_means_plot.page_parameters)
    """

    probe_type: str
    required_keys: tuple
    settings_grid: Callable
    fit_settings: Callable
    check: Callable
    probe_note: Callable
    parameters: Callable


def read_config(path, required_keys):
    with open(path) as f:
        config = yaml.safe_load(f)
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")
    return config


def load_config(path):
    config = read_config(path, REQUIRED_KEYS)
    check_config(config)
    return config


def check_common(config, family):
    if config["probe_layer"] not in config["layers"]:
        raise ValueError("probe_layer must be one of layers")
    if config["probe_position"] not in config["positions"]:
        raise ValueError("probe_position must be one of positions")
    for metric in config["metrics"]:
        if metric not in METRICS:
            raise ValueError(f"Unknown metric {metric!r}")
    if config["cv_aggregate"] not in AGGREGATES:
        raise ValueError(f"cv_aggregate must be one of {AGGREGATES}")
    if config["target_sign"] not in (1, -1):
        raise ValueError("target_sign must be 1 or -1")
    for option in config["ambiguous"]:
        if option not in AMBIGUOUS_OPTIONS:
            raise ValueError(f"ambiguous options must be in {AMBIGUOUS_OPTIONS}")
    if set(config["eval_prompt_columns"]) & set(config["prompt_columns"]):
        raise ValueError("eval_prompt_columns must not be training prompts")
    family.check(config)
    keys = [setting["key"] for setting in family.settings_grid(config)]
    if len(set(keys)) != len(keys):
        raise ValueError("The settings grid has repeated settings")
    for key in config["null_settings"]:
        if key not in keys:
            raise ValueError(f"null setting {key!r} is not in the settings grid")
    if int(config["n_permutations"]) < 0:
        raise ValueError("n_permutations must be 0 or more")


def check_dom(config):
    for n_groups in config["n_groups"]:
        if int(n_groups) < 2:
            raise ValueError("Each n_groups must be 2 or more")
    for constant in config["weight_constants"]:
        if float(constant) < 0:
            raise ValueError("Each weight constant must be 0 or more")
    if config["weight_groups"] not in WEIGHT_GROUPS:
        raise ValueError(f"weight_groups must be one of {WEIGHT_GROUPS}")


def check_config(config):
    check_common(config, DOM_FAMILY)


def setting_key(n_groups, weight_constant, ambiguous):
    return f"g{int(n_groups)}_c{float(weight_constant):g}_{ambiguous}"


def settings_grid(config):
    """Return a list of dicts, one per setting, in a fixed order."""
    grid = []
    for ambiguous in config["ambiguous"]:
        for n_groups in config["n_groups"]:
            for constant in config["weight_constants"]:
                grid.append(
                    {
                        "key": setting_key(n_groups, constant, ambiguous),
                        "n_groups": int(n_groups),
                        "weight_constant": float(constant),
                        "ambiguous": ambiguous,
                    }
                )
    return grid


def keep_mask(point_rows, ambiguous):
    """Return a boolean mask of the probe items to use."""
    if ambiguous == "include":
        return np.ones(len(point_rows), dtype=bool)
    return np.array([not is_true(row["is_ambiguous_strict"]) for row in point_rows])


def fit_settings(setting, config):
    return {
        "dom_n_groups": setting["n_groups"],
        "dom_weight_constant": setting["weight_constant"],
        "dom_weight_groups": config["weight_groups"],
    }


def dom_probe_note(config):
    bottom_weights = "1"
    if config["weight_groups"] == "both":
        bottom_weights = "exp(-c x typicality)"
    return (
        "Difference of means: weighted mean of the top 1/n items minus weighted "
        "mean of the bottom 1/n items. Top weights exp(c x typicality); bottom "
        f"weights {bottom_weights}. c = 0 is no weighting."
    )


def dom_page_parameters(config):
    return dom_parameters(
        {n: SPLIT_LABELS.get(n, f"1/{n}") for n in config["n_groups"]}
    )


DOM_FAMILY = ProbeFamily(
    probe_type=PROBE_TYPE,
    required_keys=REQUIRED_KEYS,
    settings_grid=settings_grid,
    fit_settings=fit_settings,
    check=check_dom,
    probe_note=dom_probe_note,
    parameters=dom_page_parameters,
)


def setting_columns(setting):
    """Return the CSV columns of a setting: its key, then its parameters."""
    columns = {"setting": setting["key"]}
    for field, value in setting.items():
        if field != "key":
            columns[field] = value
    return columns


def off_diagonal_mean(matrix):
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.nanmean(matrix[mask]))


def diagonal_mean(matrix):
    return float(np.nanmean(np.diag(matrix)))


def mean_over(matrices, function):
    return float(np.mean([function(m) for m in matrices]))


def summary_values(result, metric, probe_type=PROBE_TYPE):
    """Return the mean values of the matrices of one result and metric."""
    category = list(result["category_transfer"][probe_type][metric].values())
    prompt = list(result["prompt_transfer"][probe_type][metric].values())
    category_cosine = list(result["category_cosine"][probe_type].values())
    prompt_cosine = list(result["prompt_cosine"][probe_type].values())
    return {
        "within_cv": mean_over(category, diagonal_mean),
        "between_categories": mean_over(category, off_diagonal_mean),
        "between_prompts": mean_over(prompt, off_diagonal_mean),
        "cosine_between_categories": mean_over(category_cosine, off_diagonal_mean),
        "cosine_between_prompts": mean_over(prompt_cosine, off_diagonal_mean),
    }


def evaluate(
    result,
    x_train,
    x_eval,
    y,
    categories,
    folds,
    metrics,
    aggregate,
    probe_type=PROBE_TYPE,
):
    """Return the projections, the eval transfer matrices and all statistics.

    x_train: activations of the training prompts (with the baseline of the
    probes). x_eval: raw activations of the evaluation prompts.
    """
    full = result["full_probes"][probe_type]
    by_fold = result["fold_probes"][probe_type]
    category_names = result["categories"]
    x_all = dict(x_train)
    x_all.update(x_eval)
    projections = item_projections(x_all, categories, folds, full, by_fold)
    matrices = {}
    statistics = {}
    for metric in metrics:
        matrix = eval_transfer(
            full,
            by_fold,
            x_eval,
            y,
            categories,
            folds,
            category_names,
            result["prompts"],
            metric,
            aggregate,
        )
        matrices[metric] = matrix
        values = summary_values(result, metric, probe_type)
        values.update(eval_summary(matrix))
        for name, prompts in (("train", x_train), ("eval", x_eval)):
            loco, all_categories = leave_one_out_scores(
                {p: projections[p] for p in prompts},
                y,
                categories,
                category_names,
                metric,
            )
            values[f"mean_loco_{name}"] = float(np.nanmean(loco))
            values[f"mean_all_{name}"] = float(np.nanmean(all_categories))
        statistics[metric] = {key: values[key] for key in STATISTICS}
    return {
        "projections": projections,
        "eval_transfer": matrices,
        "statistics": statistics,
    }


def eval_page(matrices):
    """Return the eval transfer matrices for the page, for each metric.

    "prompts": (n_train_prompts, n_eval_prompts), mean of the CV diagonal over
    the categories. "categories": (n_categories, n_categories), mean over the
    training and evaluation prompts.
    """
    page = {}
    for metric, matrix in matrices.items():
        n_categories = matrix.shape[-1]
        diagonal = matrix[:, :, np.arange(n_categories), np.arange(n_categories)]
        page[metric] = {
            "prompts": np.nanmean(diagonal, axis=2),
            "categories": np.nanmean(matrix, axis=(0, 1)),
        }
    return page


def summary_rows(setting, baseline, statistics, metrics, n_items):
    for metric in metrics:
        row = setting_columns(setting)
        row.update({"baseline": baseline, "metric": metric, "n_items": n_items})
        row.update(statistics[metric])
        yield row


def with_setting(setting, rows):
    for row in rows:
        out = setting_columns(setting)
        out.update(row)
        yield out


def direction_arrays(setting, baseline, result, probe_type=PROBE_TYPE):
    """Return the full-probe directions in activation space.

    For difference of means this is the weight vector (scale 1). For ridge on
    standardised activations it is weights / scale.
    """
    arrays = {}
    for category, by_prompt in result["full_probes"][probe_type].items():
        for prompt, probe in by_prompt.items():
            key = f"{setting['key']}/{baseline}/{category}/{prompt}"
            arrays[key] = activation_direction(probe).astype(np.float32)
    return arrays


def mean_direction_outputs(
    setting,
    baseline,
    category_names,
    projections,
    train_prompts,
    eval_prompts,
    y,
    categories,
    metrics,
):
    """Return (page data, ranking rows) for the mean-direction section.

    The subsets are ranked on the training prompts. The held-out mean on the
    evaluation prompts is a check of the ranking on new prompts.
    """
    prompts = list(train_prompts) + list(eval_prompts)
    n_train = len(train_prompts)
    quantised, scales = quantise(projections, prompts)
    page = {
        "values": base64.b64encode(quantised.tobytes()).decode("ascii"),
        "scales": scales.tolist(),
        "best": {},
        "by_size": {},
    }
    subsets = all_subsets(len(category_names))
    rows = []
    for metric in metrics:
        scores = subset_scores(
            {p: projections[p] for p in prompts},
            y,
            categories,
            category_names,
            subsets,
            metric,
        )
        train_scores = scores[:n_train]
        eval_scores = scores[n_train:]
        best = rank_subsets(train_scores, subsets, eval_scores=eval_scores)
        page["best"][metric] = best
        page["by_size"][metric] = {
            "train": held_out_by_size(train_scores, subsets),
            "eval": held_out_by_size(eval_scores, subsets),
        }
        for rank, record in enumerate(best, start=1):
            row = setting_columns(setting)
            row.update(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "rank": rank,
                    "subset": " ".join(category_names[k] for k in record["subset"]),
                    "advantage": record["advantage"],
                    "held_out_mean": record["held_out_mean"],
                    "in_subset_mean": record["in_subset_mean"],
                    "eval_held_out_mean": record["eval_held_out_mean"],
                }
            )
            rows.append(row)
    return page, rows


def mean_direction_items(point_rows, masks, all_categories, all_y):
    """Return {ambiguous option: {"category": [...], "y": [...]}} for the page.

    The order of the items is the order of the rows of the projections."""
    category_names = sorted(set(all_categories.tolist()))
    items = {}
    for option, mask in masks.items():
        items[option] = {
            "category": [category_names.index(c) for c in all_categories[mask]],
            "y": [round(float(value), 6) for value in all_y[mask]],
        }
    return items


def page_notes(config, n_points_by_option, family=None):
    if family is None:
        family = DOM_FAMILY
    counts = ", ".join(
        f"{option}: {n} items" for option, n in n_points_by_option.items()
    )
    model_note = (
        f"Model {config['model']}, layer {config['probe_layer']}, "
        f"{config['probe_position']} token. Ambiguous items {counts}."
    )
    target_note = (
        f"Target: typicality = {config['target_sign']} x {config['target']} "
        "(higher is more typical)."
    )
    probe_note = family.probe_note(config)
    cv_note = (
        f"Cross-validation: {config['n_folds']} folds per category, stratified by "
        "typicality, the same items in each fold for all prompts. "
        f"Aggregate: {config['cv_aggregate']}."
    )
    eval_note = (
        "Evaluation prompts (" + ", ".join(config["eval_prompt_columns"]) + ") are "
        "never used for training. Probes are evaluated on their raw activations "
        "(no baseline), also the probes trained with the mean contrast baseline."
    )
    null_note = (
        f"Permutation null: {config['n_permutations']} shuffles of the typicality "
        "values inside each category, for the settings "
        + ", ".join(config["null_settings"])
        + "."
    )
    return [model_note, target_note, probe_note, cv_note, eval_note, null_note]


def load_probe_inputs(config, lazy_model, rows, point_indices, metadata, run_dir):
    """Return {baseline name: {prompt: (n_points, d_model)}}."""
    item_texts = [row["item_text"] for row in rows]
    layer_index = config["layers"].index(config["probe_layer"])
    position = config["probe_position"]
    x = {baseline_name(b): {} for b in config["baselines"]}
    for prompt_name in config["prompt_columns"]:
        prompts = [row[prompt_name] for row in rows]
        extracted, source = get_activations(
            config, lazy_model, prompt_name, prompts, item_texts
        )
        print(f"Activations for {prompt_name}: from {source}")
        metadata["activation_sources"][prompt_name] = source
        write_metadata(metadata, run_dir / "metadata.json")
        for baseline in config["baselines"]:
            x[baseline_name(baseline)][prompt_name] = probe_activations(
                extracted, rows, point_indices, baseline, layer_index, position
            )
    return x


def row_key(row):
    if "item_category" in row:
        return (row["item"], row["item_category"])
    return (row["item"], row["category"])


def eval_point_rows(eval_rows, point_rows):
    """Return the rows of the evaluation file for the probe items, in order."""
    own = [
        row
        for row in eval_rows
        if "is_item_category" not in row or is_true(row["is_item_category"])
    ]
    if [row_key(row) for row in own] != [row_key(row) for row in point_rows]:
        raise ValueError("The evaluation rows do not match the probe items")
    return own


def load_eval_inputs(config, lazy_model, point_rows, metadata, run_dir):
    """Return {eval prompt: (n_points, d_model)}, raw activations."""
    eval_rows = read_prompt_rows(config["eval_prompts_csv"], config.get("max_rows"))
    rows = eval_point_rows(eval_rows, point_rows)
    item_texts = [row["item_text"] for row in rows]
    layer_index = config["layers"].index(config["probe_layer"])
    position = config["probe_position"]
    x = {}
    for prompt_name in config["eval_prompt_columns"]:
        prompts = [row[prompt_name] for row in rows]
        extracted, source = get_activations(
            config, lazy_model, prompt_name, prompts, item_texts
        )
        print(f"Activations for {prompt_name}: from {source}")
        metadata["activation_sources"][prompt_name] = source
        write_metadata(metadata, run_dir / "metadata.json")
        x[prompt_name] = extracted.activations[position][:, layer_index, :]
    return x


def null_distribution(
    config, setting, baseline, x_train, x_eval, y, categories, folds, family=None
):
    """Yield one row per permutation, metric and statistic."""
    if family is None:
        family = DOM_FAMILY
    rng = np.random.default_rng(config["seed"])
    for permutation in range(int(config["n_permutations"])):
        y_null = permute_within(y, categories, rng)
        result = transfer_matrices(
            x_train,
            y_null,
            categories,
            folds,
            [family.probe_type],
            config["metrics"],
            family.fit_settings(setting, config),
            config["cv_aggregate"],
            config["seed"],
        )
        statistics = evaluate(
            result,
            x_train,
            x_eval,
            y_null,
            categories,
            folds,
            config["metrics"],
            config["cv_aggregate"],
            family.probe_type,
        )["statistics"]
        for metric, values in statistics.items():
            for statistic, value in values.items():
                yield {
                    "setting": setting["key"],
                    "baseline": baseline,
                    "metric": metric,
                    "permutation": permutation,
                    "statistic": statistic,
                    "value": value,
                }


def null_summary_rows(null_rows, observed):
    """Return the rows of null_summary.csv.

    observed[setting][baseline][metric][statistic]: the value with the real
    typicality values.
    """
    groups = {}
    for row in null_rows:
        key = (row["setting"], row["baseline"], row["metric"], row["statistic"])
        groups.setdefault(key, []).append(row["value"])
    rows = []
    for (setting, baseline, metric, statistic), values in groups.items():
        out = {
            "setting": setting,
            "baseline": baseline,
            "metric": metric,
            "statistic": statistic,
        }
        value = observed[setting][baseline][metric][statistic]
        out.update(null_summary(value, values))
        rows.append(out)
    return rows


def run(config, config_path, family=None):
    if family is None:
        family = DOM_FAMILY
    check_common(config, family)
    probe_type = family.probe_type
    grid = family.settings_grid(config)
    run_id = make_run_id(config["experiment"])
    run_dir = Path(config["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = start_metadata(run_id, config, config_path)
    write_metadata(metadata, run_dir / "metadata.json")

    try:
        seed = config["seed"]
        set_seed(seed)
        rows = read_prompt_rows(config["prompts_csv"], config.get("max_rows"))
        point_indices = point_indices_for(rows, config["baselines"])
        point_rows = [rows[i] for i in point_indices]
        all_categories = np.array([row["category"] for row in point_rows])
        all_y = config["target_sign"] * np.array(
            [float(row[config["target"]]) for row in point_rows]
        )
        lazy_model = LazyModel(config, metadata)
        metadata["device"] = None
        metadata["activation_sources"] = {}
        x = load_probe_inputs(
            config, lazy_model, rows, point_indices, metadata, run_dir
        )
        x_eval_all = load_eval_inputs(config, lazy_model, point_rows, metadata, run_dir)

        masks = {}
        folds = {}
        n_items = {}
        for option in config["ambiguous"]:
            mask = keep_mask(point_rows, option)
            masks[option] = mask
            folds[option] = make_folds(
                all_categories[mask], all_y[mask], config["n_folds"], seed
            )
            n_items[option] = {
                c: int((all_categories[mask] == c).sum())
                for c in sorted(set(all_categories.tolist()))
            }

        results = {}
        summary = []
        correlations = []
        cosines = []
        directions = {}
        mean_page = {}
        mean_best_rows = []
        eval_pages = {}
        observed = {}
        alphas = []
        for setting in grid:
            mask = masks[setting["ambiguous"]]
            results[setting["key"]] = {}
            mean_page[setting["key"]] = {}
            eval_pages[setting["key"]] = {}
            observed[setting["key"]] = {}
            x_eval = {p: values[mask] for p, values in x_eval_all.items()}
            for baseline in config["baselines"]:
                name = baseline_name(baseline)
                print(f"Fitting probes: {setting['key']}, baseline {name}")
                x_by_prompt = {p: values[mask] for p, values in x[name].items()}
                result = transfer_matrices(
                    x_by_prompt,
                    all_y[mask],
                    all_categories[mask],
                    folds[setting["ambiguous"]],
                    [probe_type],
                    config["metrics"],
                    family.fit_settings(setting, config),
                    config["cv_aggregate"],
                    seed,
                )
                results[setting["key"]][name] = result
                # Record the alphas only when they are chosen by inner CV.
                if "ridge_alpha" not in family.fit_settings(setting, config):
                    alphas.extend(
                        with_setting(
                            setting,
                            (
                                {"baseline": name, **record}
                                for record in result["alphas"]
                            ),
                        )
                    )
                evaluated = evaluate(
                    result,
                    x_by_prompt,
                    x_eval,
                    all_y[mask],
                    all_categories[mask],
                    folds[setting["ambiguous"]],
                    config["metrics"],
                    config["cv_aggregate"],
                    probe_type,
                )
                observed[setting["key"]][name] = evaluated["statistics"]
                eval_pages[setting["key"]][name] = eval_page(evaluated["eval_transfer"])
                summary.extend(
                    summary_rows(
                        setting,
                        name,
                        evaluated["statistics"],
                        config["metrics"],
                        int(mask.sum()),
                    )
                )
                correlations.extend(
                    with_setting(setting, correlation_rows(name, result))
                )
                cosines.extend(
                    with_setting(
                        setting,
                        cosine_rows(
                            name,
                            result["category_cosine"],
                            result["prompt_cosine"],
                            result["categories"],
                            result["prompts"],
                        ),
                    )
                )
                directions.update(direction_arrays(setting, name, result, probe_type))
                page, best_rows = mean_direction_outputs(
                    setting,
                    name,
                    result["categories"],
                    evaluated["projections"],
                    result["prompts"],
                    list(x_eval),
                    all_y[mask],
                    all_categories[mask],
                    config["metrics"],
                )
                mean_page[setting["key"]][name] = page
                mean_best_rows.extend(best_rows)
                # The probes are not used after this point. They use much memory.
                del result["full_probes"]
                del result["fold_probes"]

        null_rows = []
        settings_by_key = {s["key"]: s for s in grid}
        for key in config["null_settings"]:
            setting = settings_by_key[key]
            mask = masks[setting["ambiguous"]]
            x_eval = {p: values[mask] for p, values in x_eval_all.items()}
            for baseline in config["baselines"]:
                name = baseline_name(baseline)
                print(f"Permutation null: {key}, baseline {name}")
                null_rows.extend(
                    null_distribution(
                        config,
                        setting,
                        name,
                        {p: values[mask] for p, values in x[name].items()},
                        x_eval,
                        all_y[mask],
                        all_categories[mask],
                        folds[setting["ambiguous"]],
                        family,
                    )
                )
        null_summary_list = null_summary_rows(null_rows, observed)

        write_csv(summary, run_dir / "summary.csv")
        if null_rows:
            write_csv(null_rows, run_dir / "null_distribution.csv")
            write_csv(null_summary_list, run_dir / "null_summary.csv")
        write_csv(correlations, run_dir / "correlations.csv")
        write_csv(cosines, run_dir / "cosine_similarity.csv")
        write_csv(mean_best_rows, run_dir / "mean_direction_best.csv")
        if alphas:
            write_csv(alphas, run_dir / "ridge_alphas.csv")
        np.savez_compressed(run_dir / "probe_weights.npz", **directions)

        baselines = {
            baseline_name(b): BASELINE_LABELS.get(b, str(b))
            for b in config["baselines"]
        }
        n_points = {option: int(masks[option].sum()) for option in masks}
        plot_data = build_plot_data(
            title=config.get("title", config["experiment"]),
            notes=page_notes(config, n_points, family),
            settings=grid,
            baselines=baselines,
            results=results,
            summary=summary,
            n_items=n_items,
            parameters=family.parameters(config),
            prompt_formats=PROMPT_FORMATS,
            eval_prompts=list(x_eval_all),
            eval_prompt_formats=EVAL_PROMPT_FORMATS,
            eval_transfer=eval_pages,
            null=null_summary_list,
            mean_direction=mean_page,
            mean_items=mean_direction_items(point_rows, masks, all_categories, all_y),
            probe_type=probe_type,
        )
        write_html(plot_data, run_dir / config["html_name"])
    except BaseException:
        metadata["status"] = "failed"
        metadata["error"] = traceback.format_exc()
        metadata["finished_at"] = now_utc()
        write_metadata(metadata, run_dir / "metadata.json")
        raise

    metadata["status"] = "completed"
    metadata["finished_at"] = now_utc()
    write_metadata(metadata, run_dir / "metadata.json")
    return run_dir
