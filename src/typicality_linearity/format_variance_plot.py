"""Write an interactive HTML page for a format-variance run.

The page is built from the files of a run directory (summary.csv,
correlations.csv, format_diagnostics.csv, inlp_history.csv,
probe_weights.npz), so it can also be written again for an old run.

Global selectors: preprocessing variant, ambiguous items, metric, and the
correlation scope (by prompt or pooled) of the summary. The baselines are
side by side.

Sections:
- Summary: a table of the statistics for all variants. Click a row to select
  that variant.
- Comparison of the variants: bars of the statistics.
- Format diagnostics: format variance, format classifier accuracy, eraser rank,
  and the INLP steps.
- Transfer between categories (prompt selector), with the mean direction of
  the other categories as the last row.
- Score in each prompt and category (probe selector).
- Cosine similarity of the probe directions between categories.
"""

import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.pca_plot import PLOTLY_URL
from typicality_linearity.probe_plot import METRIC_LABELS

POOLED = "pooled"
LOCO = "others"

VARIANT_LABELS = {
    "raw": "Raw",
    "center": "Center",
    "inlp": "INLP",
    "leace": "LEACE",
    "center_inlp": "Center + INLP",
    "center_leace": "Center + LEACE",
}

BASELINE_LABELS = {
    "none": "No baseline",
    "mean_contrast": "Mean contrast baseline",
}

AMBIGUOUS_LABELS = {
    "include": "Include",
    "exclude": "Exclude",
}

SCOPE_LABELS = {
    "by_prompt": "By prompt (mean over prompts)",
    "pooled": "Pooled (all prompt rows together)",
}

# Summary statistics without the scope suffix.
SUMMARY_COLUMNS = {
    "within_train": "Within (CV), train",
    "between_train": "Between cat., train",
    "loco_train": "Mean dir. others, train",
    "within_eval": "Within (CV), eval",
    "between_eval": "Between cat., eval",
    "loco_eval": "Mean dir. others, eval",
}

DIAGNOSTIC_COLUMNS = {
    "eraser_rank": "Eraser rank",
    "format_variance_train": "Format variance, train",
    "format_variance_eval": "Format variance, eval",
    "format_accuracy_train": "Format accuracy, train",
}

PROBE_LABELS = {
    "within": "Own category (CV)",
    "between": "Other categories (mean over the probes)",
    "loco": "Mean direction of the other categories",
}


def clean_number(value):
    if value is None or value == "":
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return round(value, 4)


def nan_mean(values):
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    return round(sum(kept) / len(kept), 4)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def setting_key(variant, baseline, ambiguous, metric=None):
    parts = [variant, baseline, ambiguous]
    if metric is not None:
        parts.append(metric)
    return "|".join(parts)


def prompt_rows(train_prompts, eval_prompts):
    """Return the row labels of the prompt plots, with their source.

    Each item is (label, prompt set, prompt or POOLED or None for the mean).
    """
    rows = [(p, "train", p) for p in train_prompts]
    rows.append(("mean, train prompts", "train", None))
    rows.append(("pooled, train prompts", "train", POOLED))
    rows += [(p + " (eval)", "eval", p) for p in eval_prompts]
    rows.append(("mean, eval prompts", "eval", None))
    rows.append(("pooled, eval prompts", "eval", POOLED))
    return rows


def cell_matrices(cells, categories, train_prompts, eval_prompts):
    """Return {setting key with metric: {"single": ..., "loco": ...}}.

    single: {prompt row label: [train category][eval category]}, with the
    mean direction of the other categories as an extra last row.
    """
    index = {}
    for cell in cells:
        key = setting_key(
            cell["variant"], cell["baseline"], cell["ambiguous"], cell["metric"]
        )
        source = (cell["prompt_set"], cell["prompt"])
        index[(key, source, cell["train_category"], cell["eval_category"])] = (
            clean_number(cell["value"])
        )
    keys = sorted({k[0] for k in index})
    rows = prompt_rows(train_prompts, eval_prompts)
    out = {}
    for key in keys:
        by_prompt = {}
        for label, prompt_set, prompt in rows:
            if prompt is None:
                continue
            source = (prompt_set, prompt)
            matrix = [
                [index.get((key, source, a, b)) for b in categories] for a in categories
            ]
            matrix.append([index.get((key, source, LOCO, b)) for b in categories])
            by_prompt[label] = matrix
        for label, prompt_set, prompt in rows:
            if prompt is not None:
                continue
            members = [
                by_prompt[other_label]
                for other_label, other_set, other in rows
                if other_set == prompt_set and other not in (None, POOLED)
            ]
            by_prompt[label] = [
                [nan_mean([m[i][j] for m in members]) for j in range(len(categories))]
                for i in range(len(categories) + 1)
            ]
        out[key] = {label: by_prompt[label] for label, _, _ in rows}
    return out


def cosine_matrices(weights, categories):
    """Return {setting key: [category][category] cosine} of the full probes."""
    groups = {}
    for name in weights:
        variant, baseline, ambiguous, category, space = name.split("/")
        if space != "transformed":
            continue
        key = setting_key(variant, baseline, ambiguous)
        groups.setdefault(key, {})[category] = np.asarray(
            weights[name], dtype=np.float64
        )
    out = {}
    for key, directions in groups.items():
        units = []
        for category in categories:
            direction = directions[category]
            norm = np.linalg.norm(direction)
            units.append(direction / norm if norm > 0 else direction)
        units = np.stack(units)
        out[key] = [[round(float(v), 4) for v in row] for row in units @ units.T]
    return out


def summary_table(summary):
    out = {}
    for row in summary:
        key = setting_key(
            row["variant"], row["baseline"], row["ambiguous"], row["metric"]
        )
        values = {}
        for column in SUMMARY_COLUMNS:
            for scope in ("by_prompt", POOLED):
                values[f"{column}_{scope}"] = clean_number(row[f"{column}_{scope}"])
        values["cosine_between_categories"] = clean_number(
            row["cosine_between_categories"]
        )
        values["n_items"] = int(row["n_items"])
        out[key] = values
    return out


def diagnostics_table(diagnostics):
    out = {}
    for row in diagnostics:
        key = setting_key(row["variant"], row["baseline"], row["ambiguous"])
        out[key] = {column: clean_number(row[column]) for column in DIAGNOSTIC_COLUMNS}
        out[key]["format_chance"] = clean_number(row["format_chance"])
    return out


def history_table(history):
    """Return {variant|baseline|ambiguous: {"step": [...], "accuracy": [...], ...}}."""
    out = {}
    for row in history:
        key = setting_key(row["variant"], row["baseline"], row["ambiguous"])
        record = out.setdefault(key, {"step": [], "accuracy": [], "rank": []})
        record["step"].append(int(row["step"]))
        record["accuracy"].append(clean_number(row["dev_accuracy"]))
        record["rank"].append(int(row["rank"]))
    return out


def unique(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def page_notes(config, n_items):
    counts = ", ".join(f"{option}: {n} items" for option, n in n_items.items())
    stop = config["inlp_stop_accuracy"]
    return [
        (
            f"Model {config['model']}, layer {config['probe_layer']}, "
            f"{config['probe_position']} token. Ambiguous items {counts}."
        ),
        (
            f"Target: typicality = {config['target_sign']} x {config['target']} "
            "(higher is more typical)."
        ),
        (
            f"One difference-of-means probe per category (top and bottom 1/"
            f"{config['n_groups']} of the items, no weighting), trained on all "
            f"{len(config['prompt_columns'])} training prompts together."
        ),
        (
            f"Cross-validation: {config['n_folds']} folds per category, the same items "
            f"in each fold for all prompts. Aggregate: {config['cv_aggregate']}."
        ),
        (
            "Variants: the format label of a row is its training prompt. The eraser "
            "(INLP, LEACE) is fitted on the training prompts only and applied to the "
            "evaluation prompts. Centering of the evaluation prompts uses their own "
            "means. Center + LEACE is the same as Center (the eraser has rank 0)."
        ),
        (
            f"INLP: at most {config['inlp_iterations']} steps; a step is not done when "
            f"the dev accuracy of its format classifier is below {stop}."
        ),
        "Evaluation prompts (" + ", ".join(config["eval_prompt_columns"]) + ") are "
        "never used for training or for fitting an eraser. Their activations have no "
        "baseline, also for the probes trained with the mean contrast baseline.",
        (
            "By prompt: the correlation inside each prompt, then the mean over the "
            "prompts. Pooled: one correlation over the rows of all prompts of the "
            "category. Only the pooled value is affected by an offset between formats."
        ),
    ]


def build_plot_data(
    config,
    summary,
    cells,
    diagnostics,
    history,
    weights,
    prompt_formats=None,
    eval_prompt_formats=None,
):
    """Return a JSON-ready dict for the page.

    summary, cells, diagnostics, history: rows of the CSV files of the run (as
    dicts of strings or numbers). weights: {key: direction} of probe_weights.npz.
    """
    prompt_formats = prompt_formats or {}
    eval_prompt_formats = eval_prompt_formats or {}
    train_prompts = list(config["prompt_columns"])
    eval_prompts = list(config["eval_prompt_columns"])
    categories = sorted({c["eval_category"] for c in cells})
    variants = unique(row["variant"] for row in summary)
    baselines = unique(row["baseline"] for row in summary)
    ambiguous = unique(row["ambiguous"] for row in summary)
    metrics = unique(row["metric"] for row in summary)
    n_items = {}
    for row in summary:
        n_items[row["ambiguous"]] = int(row["n_items"])
    rows = prompt_rows(train_prompts, eval_prompts)
    return {
        "title": config.get("title", config["experiment"]),
        "notes": page_notes(config, n_items),
        "variants": variants,
        "variant_labels": {v: VARIANT_LABELS.get(v, v) for v in variants},
        "baselines": baselines,
        "baseline_labels": {b: BASELINE_LABELS.get(b, b) for b in baselines},
        "ambiguous": ambiguous,
        "ambiguous_labels": {a: AMBIGUOUS_LABELS.get(a, a) for a in ambiguous},
        "metrics": metrics,
        "metric_labels": {m: METRIC_LABELS.get(m, m) for m in metrics},
        "scope_labels": SCOPE_LABELS,
        "categories": categories,
        "prompts": train_prompts,
        "eval_prompts": eval_prompts,
        "prompt_rows": [label for label, _, _ in rows],
        "prompt_formats": {
            p: prompt_formats[p] for p in train_prompts if p in prompt_formats
        },
        "eval_prompt_formats": {
            p: eval_prompt_formats[p] for p in eval_prompts if p in eval_prompt_formats
        },
        "summary_columns": SUMMARY_COLUMNS,
        "diagnostic_columns": DIAGNOSTIC_COLUMNS,
        "probe_labels": PROBE_LABELS,
        "summary": summary_table(summary),
        "diagnostics": diagnostics_table(diagnostics),
        "history": history_table(history),
        "inlp_stop_accuracy": config["inlp_stop_accuracy"],
        "matrices": cell_matrices(cells, categories, train_prompts, eval_prompts),
        "cosine": cosine_matrices(weights, categories),
    }


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY_URL__"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; color: #1a1a1a; background: #fff; }
  h1 { font-size: 20px; margin: 0 0 8px; }
  h2 { font-size: 17px; margin: 24px 0 8px; }
  h3 { font-size: 14px; margin: 0 0 4px; }
  .notes { font-size: 13px; color: #444; margin: 0 0 12px; padding-left: 18px; }
  .controls { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; }
  .controls label { display: flex; flex-direction: column; font-size: 13px; gap: 4px; }
  .sticky { position: sticky; top: 0; background: #fff; padding: 8px 0;
            border-bottom: 1px solid #ddd; z-index: 10; }
  select { font-size: 14px; padding: 4px; }
  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .plot { width: 100%; height: 540px; }
  .plot.short { height: 360px; }
  .caption { font-size: 12px; color: #555; margin-top: 4px; }
  .subhead { font-size: 15px; margin: 20px 0 4px; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 3px 8px; border-bottom: 1px solid #eee; text-align: right; white-space: nowrap; }
  th { background: #f4f4f4; position: sticky; top: 0; }
  td.label, th.label { text-align: left; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: #f0f6fb; }
  tbody tr.selected { background: #dbeaf6; font-weight: 600; }
  th.group { text-align: center; border-left: 2px solid #ccc; }
  td.first, th.first { border-left: 2px solid #ccc; }
  .format { font-size: 14px; margin: 8px 0; padding: 6px 10px; background: #f4f4f4;
            border-left: 3px solid #4393c3; }
  .format code, table.formats code { font-family: ui-monospace, monospace; }
  table.formats { margin: 8px 0; }
  table.formats td { text-align: left; white-space: normal; }
  table.formats tbody tr { cursor: default; }
  table.formats tbody tr:hover { background: none; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<ul class="notes" id="notes"></ul>
<div class="controls sticky">
  <label>Variant <select id="variant"></select></label>
  <label>Ambiguous items <select id="ambiguous"></select></label>
  <label>Metric <select id="metric"></select></label>
  <label>Summary scope <select id="scope"></select></label>
</div>

<h2>Summary of all variants</h2>
<div class="caption">Means over the cells for the selected metric and scope. Within (CV):
fold probes of the category on its own items. Between categories: the full probe of each
other category. Mean dir. others: the mean unit direction of the probes of the other
categories, on the held-out category. Train: the training prompts; eval: the evaluation
prompts. Cosine: mean cosine similarity of the directions of two categories. Click a row
to select it.</div>
<div class="table-wrap"><table id="summary"></table></div>

<h2>Comparison of the variants</h2>
<div class="caption">The summary statistics of each variant, for the selected ambiguous
option, metric and scope.</div>
<div class="panels" id="bar-panels"></div>

<h2>Format diagnostics</h2>
<div class="caption">Format variance: the fraction of the variance from the means of the
prompt formats (0 after centering). Format accuracy: the dev accuracy of a new logistic
regression for the format of the training prompts, after the variant (chance
<span id="chance"></span>). Eraser rank: the number of directions removed by INLP or
LEACE.</div>
<div class="table-wrap"><table id="diagnostics"></table></div>
<h3 class="subhead">INLP steps</h3>
<div class="caption">Dev accuracy of the format classifier at each INLP step, before its
directions are removed. The dashed line is the stop accuracy, the dotted line is chance.</div>
<div class="panels" id="history-panels"></div>

<h2>Transfer between categories</h2>
<div class="controls">
  <label>Prompt <select id="prompt"></select></label>
</div>
<div class="caption">Rows: category the probe was trained on (all training prompts). The
last row is the mean direction of the other categories. Columns: category it was evaluated
on. Outlined diagonal cells are cross-validated; other cells use the probe trained on all
items of the row category. The prompt selects the activations it is evaluated on.</div>
<div class="format" id="prompt-format"></div>
<div class="panels" id="category-panels"></div>

<h2>Score in each prompt</h2>
<div class="controls">
  <label>Probe <select id="probe"></select></label>
</div>
<div class="caption">Rows: prompt the probe is evaluated on: the training prompts, their
mean and the pooled value, then the same for the evaluation prompts. Columns: category it
is evaluated on, then the mean over the categories.</div>
<div class="table-wrap"><table class="formats" id="format-list"></table></div>
<div class="panels" id="prompt-panels"></div>

<h2>Cosine similarity of probe directions between categories</h2>
<div class="caption">Directions of the full probes, in the space of the preprocessed
activations.</div>
<div class="panels" id="cosine-panels"></div>

<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const variantSelect = document.getElementById("variant");
const ambiguousSelect = document.getElementById("ambiguous");
const metricSelect = document.getElementById("metric");
const scopeSelect = document.getElementById("scope");
const promptSelect = document.getElementById("prompt");
const probeSelect = document.getElementById("probe");

function addOptions(select, values, labels) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels ? labels[value] : value;
    select.appendChild(option);
  }
}
addOptions(variantSelect, data.variants, data.variant_labels);
addOptions(ambiguousSelect, data.ambiguous, data.ambiguous_labels);
addOptions(metricSelect, data.metrics, data.metric_labels);
addOptions(scopeSelect, Object.keys(data.scope_labels), data.scope_labels);
addOptions(promptSelect, data.prompt_rows);
addOptions(probeSelect, Object.keys(data.probe_labels), data.probe_labels);

for (const note of data.notes) {
  const li = document.createElement("li");
  li.textContent = note;
  document.getElementById("notes").appendChild(li);
}

function key(variant, baseline, ambiguous, metric) {
  const parts = [variant, baseline, ambiguous];
  if (metric !== undefined) {
    parts.push(metric);
  }
  return parts.join("|");
}

function formatNumber(v) {
  return v === null || v === undefined ? "" : v.toFixed(3);
}

function drawFormatList() {
  const list = document.getElementById("format-list");
  const rows = data.prompts.map(p => [p, data.prompt_formats[p]])
    .concat(data.eval_prompts.map(p => [p + " (eval)", data.eval_prompt_formats[p]]));
  list.innerHTML = "";
  const tbody = document.createElement("tbody");
  for (const [name, format] of rows) {
    if (!format) {
      continue;
    }
    const tr = document.createElement("tr");
    tr.innerHTML = "<td class='label'><b></b></td><td class='label'><code></code></td>";
    tr.querySelector("b").textContent = name;
    tr.querySelector("code").textContent = format;
    tbody.appendChild(tr);
  }
  list.appendChild(tbody);
}
drawFormatList();

function drawPromptFormat() {
  const box = document.getElementById("prompt-format");
  const label = promptSelect.value;
  const name = label.replace(" (eval)", "");
  const text = data.prompt_formats[name] || data.eval_prompt_formats[name];
  box.style.display = text ? "" : "none";
  box.innerHTML = "<b></b>: <code></code>";
  box.querySelector("b").textContent = label;
  box.querySelector("code").textContent = text || "";
}

function selectRow(variant, ambiguous) {
  variantSelect.value = variant;
  ambiguousSelect.value = ambiguous;
  render();
}

function drawTable(tableId, columns, valueOf, digits) {
  const table = document.getElementById(tableId);
  const names = Object.keys(columns);
  let head = "<thead><tr><th class='label' colspan='2'></th>";
  for (const b of data.baselines) {
    head += "<th class='group' colspan='" + names.length + "'></th>";
  }
  head += "</tr><tr><th class='label'>Variant</th><th class='label'>Ambiguous</th>";
  for (const b of data.baselines) {
    names.forEach((c, k) => {
      head += "<th" + (k === 0 ? " class='first'" : "") + ">" + columns[c] + "</th>";
    });
  }
  head += "</tr></thead>";
  let body = "<tbody>";
  for (const v of data.variants) {
    for (const a of data.ambiguous) {
      const selected = v === variantSelect.value && a === ambiguousSelect.value;
      body += "<tr data-variant='" + v + "' data-ambiguous='" + a + "'" +
              (selected ? " class='selected'" : "") + ">" +
              "<td class='label'>" + data.variant_labels[v] + "</td>" +
              "<td class='label'>" + data.ambiguous_labels[a] + "</td>";
      for (const b of data.baselines) {
        names.forEach((c, k) => {
          const value = valueOf(v, b, a, c);
          const text = value === null || value === undefined ? "" :
            (Number.isInteger(value) && digits[c] === 0 ? String(value) : value.toFixed(3));
          body += "<td" + (k === 0 ? " class='first'" : "") + ">" + text + "</td>";
        });
      }
      body += "</tr>";
    }
  }
  table.innerHTML = head + body + "</tbody>";
  const groupCells = table.querySelectorAll("th.group");
  data.baselines.forEach((b, k) => { groupCells[k].textContent = data.baseline_labels[b]; });
  for (const tr of table.querySelectorAll("tbody tr")) {
    tr.addEventListener("click", () => selectRow(tr.dataset.variant, tr.dataset.ambiguous));
  }
}

function drawSummary() {
  const metric = metricSelect.value;
  const scope = scopeSelect.value;
  const columns = Object.assign({}, data.summary_columns,
                                { cosine: "Cosine, categories" });
  drawTable("summary", columns, (v, b, a, c) => {
    const values = data.summary[key(v, b, a, metric)];
    if (!values) {
      return null;
    }
    return c === "cosine" ? values.cosine_between_categories : values[c + "_" + scope];
  }, {});
}

function drawDiagnostics() {
  drawTable("diagnostics", data.diagnostic_columns, (v, b, a, c) => {
    const values = data.diagnostics[key(v, b, a)];
    return values ? values[c] : null;
  }, { eraser_rank: 0 });
  const first = Object.values(data.diagnostics)[0];
  document.getElementById("chance").textContent = first ? formatNumber(first.format_chance) : "";
}

function makePanels(containerId, prefix, className) {
  const container = document.getElementById(containerId);
  for (const baseline of data.baselines) {
    const panel = document.createElement("div");
    panel.innerHTML = '<h3></h3><div class="plot ' + (className || "") + '" id="' +
      prefix + "-" + baseline + '"></div>';
    panel.querySelector("h3").textContent = data.baseline_labels[baseline];
    container.appendChild(panel);
  }
}
makePanels("bar-panels", "bar-plot", "short");
makePanels("history-panels", "history-plot", "short");
makePanels("category-panels", "category-plot");
makePanels("prompt-panels", "prompt-plot");
makePanels("cosine-panels", "cosine-plot");

// Red is -1, white is 0, blue is 1.
const COLORSCALE = [
  [0, "#67001f"], [0.25, "#d6604d"], [0.5, "#f7f7f7"],
  [0.75, "#4393c3"], [1, "#053061"],
];

// One hue for each kind of statistic: dark for train, light for eval.
const BAR_COLORS = {
  within_train: "#2166ac", within_eval: "#92c5de",
  between_train: "#4d4d4d", between_eval: "#bababa",
  loco_train: "#b35806", loco_eval: "#fdb863",
};

const AMBIGUOUS_DASH = ["solid", "dash", "dot"];

function outline(i, j) {
  return {
    type: "rect", xref: "x", yref: "y",
    x0: j - 0.5, x1: j + 0.5, y0: i - 0.5, y1: i + 0.5,
    line: { color: "#000000", width: 2 },
  };
}

function drawGrid(plotId, matrix, rowLabels, columnLabels, options) {
  const shapes = (options.outlined || []).map(([i, j]) => outline(i, j));
  for (const y of options.rowBreaks || []) {
    shapes.push({
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: y - 0.5, y1: y - 0.5,
      line: { color: "#444444", width: 3 },
    });
  }
  for (const x of options.columnBreaks || []) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: x - 0.5, x1: x - 0.5, y0: 0, y1: 1,
      line: { color: "#444444", width: 3 },
    });
  }
  const trace = {
    type: "heatmap",
    z: matrix,
    x: columnLabels.map((_, j) => j),
    y: rowLabels.map((_, i) => i),
    zmin: -1,
    zmax: 1,
    colorscale: COLORSCALE,
    texttemplate: "%{z:.2f}",
    hovertemplate: options.rowName + ": %{customdata[0]}<br>" +
      options.columnName + ": %{customdata[1]}<br>" +
      options.valueLabel + ": %{z:.3f}<extra></extra>",
    customdata: rowLabels.map(r => columnLabels.map(c => [r, c])),
    colorbar: { title: { text: options.valueLabel } },
  };
  const layout = {
    margin: { l: 160, r: 10, t: 10, b: 120 },
    xaxis: { tickmode: "array", tickvals: columnLabels.map((_, k) => k),
             ticktext: columnLabels, tickangle: -40, title: { text: options.columnTitle } },
    yaxis: { tickmode: "array", tickvals: rowLabels.map((_, k) => k),
             ticktext: rowLabels, autorange: "reversed", title: { text: options.rowTitle } },
    shapes: shapes,
  };
  Plotly.react(document.getElementById(plotId), [trace], layout, { responsive: true });
}

function drawBars() {
  const metric = metricSelect.value;
  const scope = scopeSelect.value;
  const a = ambiguousSelect.value;
  const x = data.variants.map(v => data.variant_labels[v]);
  for (const b of data.baselines) {
    const traces = Object.keys(data.summary_columns).map(c => ({
      type: "bar",
      name: data.summary_columns[c],
      x: x,
      y: data.variants.map(v => {
        const values = data.summary[key(v, b, a, metric)];
        return values ? values[c + "_" + scope] : null;
      }),
      marker: { color: BAR_COLORS[c] },
      hovertemplate: "%{x}<br>" + data.summary_columns[c] + ": %{y:.3f}<extra></extra>",
    }));
    const layout = {
      barmode: "group",
      margin: { l: 60, r: 10, t: 10, b: 60 },
      yaxis: { title: { text: data.metric_labels[metric] }, zeroline: true },
      legend: { orientation: "h", y: -0.2 },
    };
    Plotly.react(document.getElementById("bar-plot-" + b), traces, layout, { responsive: true });
  }
}

function drawHistory() {
  for (const b of data.baselines) {
    const traces = [];
    let maxStep = 1;
    data.variants.forEach(v => {
      data.ambiguous.forEach((a, k) => {
        const record = data.history[key(v, b, a)];
        if (!record) {
          return;
        }
        maxStep = Math.max(maxStep, ...record.step);
        traces.push({
          type: "scatter", mode: "lines+markers",
          name: data.variant_labels[v] + ", " + data.ambiguous_labels[a],
          x: record.step, y: record.accuracy,
          customdata: record.rank,
          line: { dash: AMBIGUOUS_DASH[k % AMBIGUOUS_DASH.length] },
          hovertemplate: "step %{x}<br>dev accuracy %{y:.3f}<br>rank after step %{customdata}" +
            "<extra>%{fullData.name}</extra>",
        });
      });
    });
    const shapes = [];
    const first = Object.values(data.diagnostics)[0];
    if (first && first.format_chance !== null) {
      shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: first.format_chance,
                    y1: first.format_chance, line: { color: "#777", dash: "dot", width: 1 } });
    }
    if (data.inlp_stop_accuracy !== null) {
      shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: data.inlp_stop_accuracy,
                    y1: data.inlp_stop_accuracy, line: { color: "#777", dash: "dash", width: 1 } });
    }
    const layout = {
      margin: { l: 60, r: 10, t: 10, b: 50 },
      xaxis: { title: { text: "INLP step" } },
      yaxis: { title: { text: "Dev accuracy" }, range: [0, 1.05] },
      shapes: shapes,
      legend: { orientation: "h", y: -0.25 },
    };
    Plotly.react(document.getElementById("history-plot-" + b), traces, layout, { responsive: true });
  }
}

function drawCategories() {
  const metric = metricSelect.value;
  const label = promptSelect.value;
  const rowLabels = data.categories.concat(["mean of others"]);
  const outlined = data.categories.map((_, i) => [i, i]);
  for (const b of data.baselines) {
    const byPrompt = data.matrices[key(variantSelect.value, b, ambiguousSelect.value, metric)];
    drawGrid("category-plot-" + b, byPrompt[label], rowLabels, data.categories, {
      valueLabel: data.metric_labels[metric],
      rowTitle: "Trained on category", columnTitle: "Evaluated on category",
      rowName: "train", columnName: "eval",
      outlined: outlined, rowBreaks: [data.categories.length],
    });
  }
}

function meanOf(values) {
  const kept = values.filter(v => v !== null && v !== undefined);
  return kept.length === 0 ? null : kept.reduce((s, v) => s + v, 0) / kept.length;
}

function probeRow(matrix, probe) {
  const n = data.categories.length;
  const row = [];
  for (let j = 0; j < n; j++) {
    if (probe === "within") {
      row.push(matrix[j][j]);
    } else if (probe === "loco") {
      row.push(matrix[n][j]);
    } else {
      const others = [];
      for (let i = 0; i < n; i++) {
        if (i !== j) {
          others.push(matrix[i][j]);
        }
      }
      row.push(meanOf(others));
    }
  }
  row.push(meanOf(row));
  return row.map(v => v === null ? null : Math.round(v * 10000) / 10000);
}

function drawPrompts() {
  const metric = metricSelect.value;
  const probe = probeSelect.value;
  const columnLabels = data.categories.concat(["mean"]);
  const nTrain = data.prompts.length + 2;
  const breaks = [data.prompts.length, nTrain, nTrain + data.eval_prompts.length];
  for (const b of data.baselines) {
    const byPrompt = data.matrices[key(variantSelect.value, b, ambiguousSelect.value, metric)];
    const matrix = data.prompt_rows.map(label => probeRow(byPrompt[label], probe));
    drawGrid("prompt-plot-" + b, matrix, data.prompt_rows, columnLabels, {
      valueLabel: data.metric_labels[metric],
      rowTitle: "Evaluated on prompt", columnTitle: "Evaluated on category",
      rowName: "prompt", columnName: "category",
      rowBreaks: breaks, columnBreaks: [data.categories.length],
    });
  }
}

function drawCosine() {
  for (const b of data.baselines) {
    const matrix = data.cosine[key(variantSelect.value, b, ambiguousSelect.value)];
    if (!matrix) {
      continue;
    }
    drawGrid("cosine-plot-" + b, matrix, data.categories, data.categories, {
      valueLabel: "Cosine", rowTitle: "Category", columnTitle: "Category",
      rowName: "category", columnName: "category",
    });
  }
}

function render() {
  drawSummary();
  drawDiagnostics();
  drawBars();
  drawPromptFormat();
  drawCategories();
  drawPrompts();
  drawCosine();
}

drawHistory();
for (const select of [variantSelect, ambiguousSelect, metricSelect, scopeSelect,
                      promptSelect, probeSelect]) {
  select.addEventListener("change", render);
}
render();
</script>
</body>
</html>
"""


def render_html(plot_data):
    data_json = json.dumps(plot_data, allow_nan=False).replace("</", "<\\/")
    page = PAGE_TEMPLATE.replace("__PLOTLY_URL__", PLOTLY_URL)
    page = page.replace("__TITLE__", html.escape(plot_data["title"]))
    return page.replace("__DATA__", data_json)


def write_html(plot_data, path):
    with open(path, "w") as f:
        f.write(render_html(plot_data))


def write_run_page(run_dir, html_name, prompt_formats=None, eval_prompt_formats=None):
    """Read the files of a run directory and write the page into it."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    history_path = run_dir / "inlp_history.csv"
    history = read_csv(history_path) if history_path.exists() else []
    with np.load(run_dir / "probe_weights.npz") as archive:
        weights = {name: archive[name] for name in archive.files}
    plot_data = build_plot_data(
        config,
        read_csv(run_dir / "summary.csv"),
        read_csv(run_dir / "correlations.csv"),
        read_csv(run_dir / "format_diagnostics.csv"),
        history,
        weights,
        prompt_formats,
        eval_prompt_formats,
    )
    path = run_dir / html_name
    write_html(plot_data, path)
    return path
