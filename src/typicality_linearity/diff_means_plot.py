"""Write an interactive HTML page for a grid of probe settings.

Used for the difference-of-means variants and for the ridge alphas. Global
selectors: one for each setting parameter (for difference of means: split,
weight constant, ambiguous items; for ridge: alpha, ambiguous items) and the
metric. The baselines are side by side.

Sections:
- Summary: a table of the mean values for all settings. Click a row to select
  that setting.
- Transfer between categories (prompt selector) and the cosine similarity of
  the directions between categories.
- Transfer between prompts (category selector) and the cosine similarity of the
  directions between prompts.
- Transfer to the evaluation prompts: the probes of each training prompt,
  evaluated on the raw activations of the evaluation prompts.
- Mean direction over a choice of categories and all training prompts (check
  boxes): the score of the mean unit direction in each prompt (training and
  evaluation) and evaluation category, and the subsets that transfer best (see
  mean_direction.py).
- Permutation null for the settings where it was computed.
"""

import html
import json
import math

from typicality_linearity.mean_direction import MIN_HELD_OUT
from typicality_linearity.pca_plot import PLOTLY_URL
from typicality_linearity.probe_plot import (
    METRIC_LABELS,
    matrices_to_lists,
    matrix_to_list,
)

PROBE_TYPE = "diff_of_means"


def parameter_value(value):
    """Return the text of a setting value, as used by the page selectors."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def page_parameters(settings, parameters):
    """Return the selector data for the page.

    parameters: list of dicts {"field", "id", "label", "column", "labels",
    "prefix"}. "field" is the key in each setting dict. "labels" ({value text:
    label}) and "prefix" (put before the value text when it has no label) are
    optional. The values are in the order they first occur in settings.
    """
    out = []
    for parameter in parameters:
        values = []
        for setting in settings:
            value = parameter_value(setting[parameter["field"]])
            if value not in values:
                values.append(value)
        labels = parameter.get("labels") or {}
        prefix = parameter.get("prefix", "")
        out.append(
            {
                "field": parameter["field"],
                "id": parameter["id"],
                "label": parameter["label"],
                "column": parameter["column"],
                "values": values,
                "labels": {v: labels.get(v, prefix + v) for v in values},
            }
        )
    return out


def dom_parameters(split_labels):
    """Return the selector parameters of the difference-of-means settings."""
    return [
        {
            "field": "n_groups",
            "id": "split",
            "label": "Split",
            "column": "Split",
            "labels": {str(n): label for n, label in split_labels.items()},
        },
        {
            "field": "weight_constant",
            "id": "constant",
            "label": "Weight constant c",
            "column": "c",
            "prefix": "c = ",
        },
        {
            "field": "ambiguous",
            "id": "ambiguous",
            "label": "Ambiguous items",
            "column": "Ambiguous",
            "labels": AMBIGUOUS_LABELS,
        },
    ]


AMBIGUOUS_LABELS = {
    "include": "Include",
    "exclude": "Exclude",
}

SUMMARY_COLUMNS = {
    "within_cv": "Within (CV)",
    "between_categories": "Between categories",
    "between_prompts": "Between prompts",
    "eval_within": "Eval prompts, within (CV)",
    "eval_between_categories": "Eval prompts, between categories",
    "mean_loco_train": "Mean dir. held-out cat., train prompts",
    "mean_loco_eval": "Mean dir. held-out cat., eval prompts",
    "mean_all_train": "Mean dir. all cat. (CV), train prompts",
    "mean_all_eval": "Mean dir. all cat. (CV), eval prompts",
    "cosine_between_categories": "Cosine, categories",
    "cosine_between_prompts": "Cosine, prompts",
}


def clean_number(value):
    value = float(value)
    if math.isnan(value):
        return None
    return round(value, 4)


def build_plot_data(
    title,
    notes,
    settings,
    baselines,
    results,
    summary,
    n_items,
    parameters,
    prompt_formats=None,
    eval_prompts=None,
    eval_prompt_formats=None,
    eval_transfer=None,
    null=None,
    mean_direction=None,
    mean_items=None,
    probe_type=PROBE_TYPE,
):
    """Return a JSON-ready dict for the page.

    settings: list of dicts with "key", "ambiguous" and the other parameter
        fields (for example n_groups, weight_constant)
    baselines: {baseline name: panel label}, in panel order
    results: {setting key: {baseline name: transfer_matrices result}}
    summary: rows from diff_means_experiment.summary_rows
    n_items: {ambiguous option: {category: number of items}}
    parameters: selector parameters, see page_parameters. For difference of
        means, use dom_parameters(split_labels)
    prompt_formats: {prompt name: format text}, shown above the plots
    eval_prompts: names of the evaluation prompts, in the order of the rows of
        the mean-direction projections after the training prompts
    eval_prompt_formats: {eval prompt name: format text}
    eval_transfer: {setting key: {baseline: diff_means_experiment.eval_page}}
    null: rows of null_summary.csv
    mean_direction: {setting key: {baseline: output of
        diff_means_experiment.mean_direction_outputs}}, or None
    mean_items: {ambiguous option: {"category": [...], "y": [...]}}, or None
    probe_type: the key of the probes in the results
    """
    if prompt_formats is None:
        prompt_formats = {}
    if eval_prompts is None:
        eval_prompts = []
    if eval_prompt_formats is None:
        eval_prompt_formats = {}
    eval_lists = {}
    for key, by_baseline in (eval_transfer or {}).items():
        eval_lists[key] = {}
        for baseline, by_metric in by_baseline.items():
            eval_lists[key][baseline] = {
                metric: {
                    name: matrix_to_list(matrix) for name, matrix in matrices.items()
                }
                for metric, matrices in by_metric.items()
            }
    null_rows = []
    for row in null or []:
        null_rows.append(
            {
                "setting": row["setting"],
                "baseline": row["baseline"],
                "metric": row["metric"],
                "statistic": row["statistic"],
                "observed": clean_number(row["observed"]),
                "null_mean": clean_number(row["null_mean"]),
                "null_q95": clean_number(row["null_q95"]),
                "null_max": clean_number(row["null_max"]),
                "p_value": clean_number(row["p_value"]),
                "n_permutations": int(row["n_permutations"]),
            }
        )
    first = results[settings[0]["key"]][next(iter(baselines))]
    metrics = list(first["category_transfer"][probe_type])
    category_transfer = {}
    prompt_transfer = {}
    category_cosine = {}
    prompt_cosine = {}
    for setting in settings:
        key = setting["key"]
        category_transfer[key] = {}
        prompt_transfer[key] = {}
        category_cosine[key] = {}
        prompt_cosine[key] = {}
        for baseline in baselines:
            result = results[key][baseline]
            if result["categories"] != first["categories"]:
                raise ValueError("Settings have different categories")
            if result["prompts"] != first["prompts"]:
                raise ValueError("Settings have different prompts")
            category_transfer[key][baseline] = {
                metric: matrices_to_lists(
                    result["category_transfer"][probe_type][metric]
                )
                for metric in metrics
            }
            prompt_transfer[key][baseline] = {
                metric: matrices_to_lists(result["prompt_transfer"][probe_type][metric])
                for metric in metrics
            }
            category_cosine[key][baseline] = matrices_to_lists(
                result["category_cosine"][probe_type]
            )
            prompt_cosine[key][baseline] = matrices_to_lists(
                result["prompt_cosine"][probe_type]
            )

    selectors = page_parameters(settings, parameters)
    page_settings = []
    for setting in settings:
        page_setting = dict(setting)
        page_setting["selector"] = {
            p["field"]: parameter_value(setting[p["field"]]) for p in selectors
        }
        page_settings.append(page_setting)

    summary_rows = []
    for row in summary:
        values = {}
        for column in SUMMARY_COLUMNS:
            values[column] = clean_number(row[column])
        summary_rows.append(
            {
                "setting": row["setting"],
                "baseline": row["baseline"],
                "metric": row["metric"],
                "values": values,
            }
        )

    return {
        "title": title,
        "notes": list(notes),
        "settings": page_settings,
        "parameters": selectors,
        "baselines": list(baselines),
        "baseline_labels": dict(baselines),
        "metrics": metrics,
        "metric_labels": {m: METRIC_LABELS.get(m, m) for m in metrics},
        "categories": list(first["categories"]),
        "prompts": list(first["prompts"]),
        "prompt_formats": {
            p: prompt_formats[p] for p in first["prompts"] if p in prompt_formats
        },
        "eval_prompts": list(eval_prompts),
        "eval_prompt_formats": {
            p: eval_prompt_formats[p] for p in eval_prompts if p in eval_prompt_formats
        },
        "mean_prompts": list(first["prompts"]) + list(eval_prompts),
        "eval_transfer": eval_lists,
        "null": null_rows,
        "n_items": n_items,
        "summary_columns": SUMMARY_COLUMNS,
        "summary": summary_rows,
        "category_transfer": category_transfer,
        "prompt_transfer": prompt_transfer,
        "category_cosine": category_cosine,
        "prompt_cosine": prompt_cosine,
        "mean_direction": mean_direction or {},
        "mean_items": mean_items or {},
        "mean_min_held_out": MIN_HELD_OUT,
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
  table.formats td { text-align: left; white-space: normal; cursor: default; }
  table.formats tbody tr { cursor: default; }
  table.formats tbody tr:hover { background: none; }
  .checks { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 8px 0; font-size: 13px; }
  .checks label { display: flex; align-items: center; gap: 4px; }
  button { font-size: 12px; padding: 3px 8px; cursor: pointer; }
  table.best tbody tr { cursor: default; }
  table.best td.subset { text-align: left; white-space: normal; max-width: 360px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<ul class="notes" id="notes"></ul>
<div class="controls sticky">
__SELECTORS__
  <label>Metric <select id="metric"></select></label>
</div>

<h2>Summary of all settings</h2>
<div class="caption">Means over the cells of the matrices for the selected metric.
Within (CV): diagonal of the category matrix. Between categories: off-diagonal of
the category matrix. Between prompts: off-diagonal of the prompt matrix. Cosine:
off-diagonal mean of the cosine similarity of the directions. Click a row to select it.</div>
<div class="table-wrap"><table id="summary"></table></div>

<h2>Transfer between categories</h2>
<div class="controls">
  <label>Prompt format <select id="prompt"></select></label>
</div>
<div class="caption">Rows: category the probe was trained on. Columns: category it was
evaluated on. Outlined diagonal cells are cross-validated; other cells use the probe
trained on all items of the row category.</div>
<div class="format" id="prompt-format"></div>
<div class="panels" id="category-panels"></div>
<h3 class="subhead">Cosine similarity of probe directions between categories</h3>
<div class="panels" id="category-cosine-panels"></div>

<h2>Transfer between prompts</h2>
<div class="controls">
  <label>Category <select id="category"></select></label>
</div>
<div class="caption">Rows: prompt format the probe was trained on. Columns: prompt format
it was evaluated on. All cells are cross-validated on held-out items.</div>
<div class="table-wrap"><table class="formats" id="prompt-format-list"></table></div>
<div class="panels" id="prompt-panels"></div>
<h3 class="subhead">Cosine similarity of probe directions between prompts</h3>
<div class="panels" id="prompt-cosine-panels"></div>

<h2>Transfer to the evaluation prompts</h2>
<div class="caption">The evaluation prompts are never used for training. The probes are
evaluated on their raw activations, with no baseline, also the probes trained with the
mean contrast baseline (the panel title is the baseline of the training activations).
For the category of the probe, the fold probes are used, so no item is scored with a
probe that saw it in any prompt.</div>
<div class="table-wrap"><table class="formats" id="eval-format-list"></table></div>
<h3 class="subhead">Probe of each training prompt, own category (CV), mean over the categories</h3>
<div class="panels" id="eval-prompt-panels"></div>
<h3 class="subhead">Between categories, mean over the training and evaluation prompts</h3>
<div class="caption">Rows: category the probe was trained on. Columns: category it was
evaluated on. Outlined diagonal cells are cross-validated.</div>
<div class="panels" id="eval-category-panels"></div>

<h2>Mean direction over a choice of categories and all training prompts</h2>
<div class="caption">One direction: the mean of the unit directions of the probes of the
selected categories in all training prompt formats. Rows: prompt format of the activations
that the direction is evaluated on: the training prompts, then the evaluation prompts
(marked "eval", raw activations, never used for training), then the means over the two
groups. Columns: evaluation category, then the mean over the held-out categories (not
selected) and over the selected categories. Outlined columns are selected categories.
For these, the directions of the category itself are the cross-validated fold probes (the
same folds in all prompts), so no item is scored with a direction that was trained on it
in any prompt.</div>
<div class="checks" id="mean-categories"></div>
<div class="controls">
  <button id="mean-all">Select all</button>
  <button id="mean-none">Select none</button>
</div>
<div class="panels" id="mean-panels"></div>
<h3 class="subhead">Subsets of categories that transfer best</h3>
<div class="caption">Mean score on the held-out categories, over all subsets with the
same number of selected categories (no selection of the best subset). Each category is
held out equally often, so the sizes can be compared.</div>
<div class="panels" id="mean-size-panels"></div>
<div class="caption" id="mean-best-caption"></div>
<div class="panels" id="mean-best-panels"></div>

<h2>Permutation null</h2>
<div class="caption">The typicality values are shuffled inside each category (the same
shuffle in all prompts), and all probes and statistics are computed again. Observed: the
value with the real typicality values. p: fraction of shuffles with a value at least the
observed value (one-sided, the observed value counted once). The smallest possible p is
1 / (number of shuffles + 1).</div>
<div class="caption" id="null-caption"></div>
<div class="panels" id="null-panels"></div>

<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const parameterSelects = data.parameters.map(p => document.getElementById(p.id));
const ambiguousSelect = document.getElementById("ambiguous");
const metricSelect = document.getElementById("metric");
const promptSelect = document.getElementById("prompt");
const categorySelect = document.getElementById("category");

function addOptions(select, values, labels) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels ? labels[value] : value;
    select.appendChild(option);
  }
}
data.parameters.forEach((p, k) => addOptions(parameterSelects[k], p.values, p.labels));
addOptions(metricSelect, data.metrics, data.metric_labels);
addOptions(promptSelect, data.prompts);

function drawFormatList(tableId, prompts, formats) {
  const list = document.getElementById(tableId);
  list.innerHTML = "";
  const tbody = document.createElement("tbody");
  for (const p of prompts) {
    if (!(p in formats)) {
      continue;
    }
    const tr = document.createElement("tr");
    tr.innerHTML = "<td class='label'><b></b></td><td class='label'><code></code></td>";
    tr.querySelector("b").textContent = p;
    tr.querySelector("code").textContent = formats[p];
    tbody.appendChild(tr);
  }
  list.appendChild(tbody);
}

function drawPromptFormats() {
  const box = document.getElementById("prompt-format");
  const text = data.prompt_formats[promptSelect.value];
  box.style.display = text ? "" : "none";
  box.innerHTML = "<b></b>: <code></code>";
  box.querySelector("b").textContent = promptSelect.value;
  box.querySelector("code").textContent = text || "";
  drawFormatList("prompt-format-list", data.prompts, data.prompt_formats);
  drawFormatList("eval-format-list", data.eval_prompts, data.eval_prompt_formats);
}

function updateCategoryOptions() {
  const counts = data.n_items[ambiguousSelect.value];
  const labels = {};
  for (const c of data.categories) {
    labels[c] = c + " (" + counts[c] + " items)";
  }
  const previous = categorySelect.value;
  addOptions(categorySelect, data.categories, labels);
  if (previous) {
    categorySelect.value = previous;
  }
}
updateCategoryOptions();

for (const note of data.notes) {
  const li = document.createElement("li");
  li.textContent = note;
  document.getElementById("notes").appendChild(li);
}

function settingLabel(s) {
  return data.parameters.map(p => p.labels[s.selector[p.field]]);
}

function selectedKey() {
  for (const s of data.settings) {
    if (data.parameters.every((p, k) => s.selector[p.field] === parameterSelects[k].value)) {
      return s.key;
    }
  }
  return null;
}

const summaryIndex = {};
for (const row of data.summary) {
  summaryIndex[row.setting + "|" + row.baseline + "|" + row.metric] = row.values;
}
const columns = Object.keys(data.summary_columns);

function drawSummary() {
  const table = document.getElementById("summary");
  const metric = metricSelect.value;
  const key = selectedKey();
  let head = "<thead><tr><th class='label' colspan='" + data.parameters.length + "'></th>";
  for (const b of data.baselines) {
    head += "<th class='group' colspan='" + columns.length + "'></th>";
  }
  head += "</tr><tr>";
  for (const p of data.parameters) {
    head += "<th class='label'>" + p.column + "</th>";
  }
  for (const b of data.baselines) {
    columns.forEach((c, k) => {
      head += "<th" + (k === 0 ? " class='first'" : "") + ">" + data.summary_columns[c] + "</th>";
    });
  }
  head += "</tr></thead>";
  let body = "<tbody>";
  for (const s of data.settings) {
    const label = settingLabel(s);
    body += "<tr data-key='" + s.key + "'" + (s.key === key ? " class='selected'" : "") + ">";
    for (const text of label) {
      body += "<td class='label'>" + text + "</td>";
    }
    for (const b of data.baselines) {
      const values = summaryIndex[s.key + "|" + b + "|" + metric];
      columns.forEach((c, k) => {
        const v = values ? values[c] : null;
        body += "<td" + (k === 0 ? " class='first'" : "") + ">" +
                (v === null || v === undefined ? "" : v.toFixed(3)) + "</td>";
      });
    }
    body += "</tr>";
  }
  body += "</tbody>";
  table.innerHTML = head + body;
  const groupCells = table.querySelectorAll("th.group");
  data.baselines.forEach((b, k) => { groupCells[k].textContent = data.baseline_labels[b]; });
  for (const tr of table.querySelectorAll("tbody tr")) {
    tr.addEventListener("click", () => {
      const s = data.settings.find(item => item.key === tr.dataset.key);
      data.parameters.forEach((p, k) => { parameterSelects[k].value = s.selector[p.field]; });
      updateCategoryOptions();
      render();
    });
  }
}

function makePanels(containerId, prefix) {
  const container = document.getElementById(containerId);
  for (const baseline of data.baselines) {
    const panel = document.createElement("div");
    panel.innerHTML = '<h3></h3><div class="plot" id="' + prefix + "-" + baseline + '"></div>';
    panel.querySelector("h3").textContent = data.baseline_labels[baseline];
    container.appendChild(panel);
  }
}
makePanels("category-panels", "category-plot");
makePanels("prompt-panels", "prompt-plot");
makePanels("category-cosine-panels", "category-cosine-plot");
makePanels("prompt-cosine-panels", "prompt-cosine-plot");
makePanels("eval-prompt-panels", "eval-prompt-plot");
makePanels("eval-category-panels", "eval-category-plot");

// Red is -1, white is 0, blue is 1.
const COLORSCALE = [
  [0, "#67001f"], [0.25, "#d6604d"], [0.5, "#f7f7f7"],
  [0.75, "#4393c3"], [1, "#053061"],
];

function outline(i, j) {
  return {
    type: "rect", xref: "x", yref: "y",
    x0: j - 0.5, x1: j + 0.5, y0: i - 0.5, y1: i + 0.5,
    line: { color: "#000000", width: 2 },
  };
}

function drawHeatmap(plotId, matrix, labels, options) {
  const shapes = [];
  if (options.outlineDiagonal) {
    for (let i = 0; i < labels.length; i++) {
      shapes.push(outline(i, i));
    }
  }
  const trace = {
    type: "heatmap",
    z: matrix,
    x: labels.map((_, j) => j),
    y: labels.map((_, i) => i),
    zmin: -1,
    zmax: 1,
    colorscale: COLORSCALE,
    texttemplate: "%{z:.2f}",
    hovertemplate: options.rowName + ": %{customdata[0]}<br>" +
      options.columnName + ": %{customdata[1]}<br>" +
      options.valueLabel + ": %{z:.3f}<extra></extra>",
    customdata: labels.map(row => labels.map(column => [row, column])),
    colorbar: { title: { text: options.valueLabel } },
  };
  const ticks = { tickmode: "array", tickvals: labels.map((_, k) => k), ticktext: labels };
  const layout = {
    margin: { l: 120, r: 10, t: 10, b: 120 },
    xaxis: Object.assign({ title: { text: options.columnTitle }, tickangle: -40 }, ticks),
    yaxis: Object.assign({ title: { text: options.rowTitle }, autorange: "reversed" }, ticks),
    shapes: shapes,
  };
  Plotly.react(document.getElementById(plotId), [trace], layout, { responsive: true });
}

function drawRectangle(plotId, matrix, rowLabels, columnLabels, options) {
  const shapes = [];
  if (options.outlineDiagonal) {
    for (let i = 0; i < rowLabels.length; i++) {
      shapes.push(outline(i, i));
    }
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
    hovertemplate: "train: %{customdata[0]}<br>eval: %{customdata[1]}<br>" +
      options.valueLabel + ": %{z:.3f}<extra></extra>",
    customdata: rowLabels.map(r => columnLabels.map(c => [r, c])),
    colorbar: { title: { text: options.valueLabel } },
  };
  const layout = {
    margin: { l: 120, r: 10, t: 10, b: 120 },
    xaxis: { tickmode: "array", tickvals: columnLabels.map((_, k) => k),
             ticktext: columnLabels, tickangle: -40, title: { text: options.columnTitle } },
    yaxis: { tickmode: "array", tickvals: rowLabels.map((_, k) => k),
             ticktext: rowLabels, autorange: "reversed", title: { text: options.rowTitle } },
    shapes: shapes,
  };
  Plotly.react(document.getElementById(plotId), [trace], layout, { responsive: true });
}

function drawEval(key, metric) {
  const byKey = data.eval_transfer[key];
  if (!byKey) {
    return;
  }
  const valueLabel = data.metric_labels[metric];
  for (const b of data.baselines) {
    drawRectangle("eval-prompt-plot-" + b, byKey[b][metric].prompts,
      data.prompts, data.eval_prompts,
      { valueLabel: valueLabel, rowTitle: "Trained on prompt",
        columnTitle: "Evaluated on prompt", outlineDiagonal: false });
    drawRectangle("eval-category-plot-" + b, byKey[b][metric].categories,
      data.categories, data.categories,
      { valueLabel: valueLabel, rowTitle: "Trained on category",
        columnTitle: "Evaluated on category", outlineDiagonal: true });
  }
}

function transferOptions(axisName, outlineDiagonal) {
  return {
    valueLabel: data.metric_labels[metricSelect.value],
    rowTitle: "Trained on " + axisName,
    columnTitle: "Evaluated on " + axisName,
    rowName: "train",
    columnName: "eval",
    outlineDiagonal: outlineDiagonal,
  };
}

function cosineOptions(axisName) {
  return {
    valueLabel: "Cosine",
    rowTitle: axisName,
    columnTitle: axisName,
    rowName: axisName,
    columnName: axisName,
    outlineDiagonal: false,
  };
}

function render() {
  const key = selectedKey();
  const metric = metricSelect.value;
  const prompt = promptSelect.value;
  const category = categorySelect.value;
  drawSummary();
  drawPromptFormats();
  if (key === null) {
    return;
  }
  renderMean(key);
  drawEval(key, metric);
  drawNull(key, metric);
  for (const b of data.baselines) {
    drawHeatmap("category-plot-" + b,
      data.category_transfer[key][b][metric][prompt],
      data.categories, transferOptions("category", true));
    drawHeatmap("prompt-plot-" + b,
      data.prompt_transfer[key][b][metric][category],
      data.prompts, transferOptions("prompt", false));
    drawHeatmap("category-cosine-plot-" + b,
      data.category_cosine[key][b][prompt],
      data.categories, cosineOptions("Category"));
    drawHeatmap("prompt-cosine-plot-" + b,
      data.prompt_cosine[key][b][category],
      data.prompts, cosineOptions("Prompt"));
  }
}

// Mean direction over a choice of categories

const meanChecks = {};
for (const c of data.categories) {
  const label = document.createElement("label");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = true;
  box.addEventListener("change", render);
  label.appendChild(box);
  label.appendChild(document.createTextNode(c));
  document.getElementById("mean-categories").appendChild(label);
  meanChecks[c] = box;
}
function setChecks(selected) {
  for (const c of data.categories) {
    meanChecks[c].checked = selected.includes(c);
  }
  render();
}
document.getElementById("mean-all").addEventListener("click", () => setChecks(data.categories));
document.getElementById("mean-none").addEventListener("click", () => setChecks([]));

const decoded = {};
function meanValues(key, baseline) {
  const id = key + "|" + baseline;
  if (!(id in decoded)) {
    const text = atob(data.mean_direction[key][baseline].values);
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) {
      bytes[i] = text.charCodeAt(i);
    }
    decoded[id] = new Int16Array(bytes.buffer);
  }
  return decoded[id];
}

function averageRanks(values) {
  const order = values.map((v, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Array(values.length);
  let start = 0;
  while (start < order.length) {
    let end = start;
    while (end + 1 < order.length && values[order[end + 1]] === values[order[start]]) {
      end++;
    }
    const rank = (start + end) / 2 + 1;
    for (let k = start; k <= end; k++) {
      ranks[order[k]] = rank;
    }
    start = end + 1;
  }
  return ranks;
}

function pearsonJs(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  if (saa === 0 || sbb === 0) {
    return null;
  }
  return sab / Math.sqrt(saa * sbb);
}

function correlation(a, b, metric) {
  if (metric === "spearman") {
    return pearsonJs(averageRanks(a), averageRanks(b));
  }
  return pearsonJs(a, b);
}

function meanOf(values) {
  const kept = values.filter(v => v !== null);
  if (kept.length === 0) {
    return null;
  }
  return kept.reduce((s, v) => s + v, 0) / kept.length;
}

function meanMatrix(key, baseline, selected, metric) {
  const items = data.mean_items[ambiguousSelect.value];
  const values = meanValues(key, baseline);
  const scales = data.mean_direction[key][baseline].scales;
  const nItems = items.category.length;
  const nCategories = data.categories.length;
  const chosen = data.categories.map((c, k) => k).filter(k => selected.includes(data.categories[k]));
  const matrix = [];
  data.mean_prompts.forEach((prompt, p) => {
    const scores = new Array(nItems);
    for (let i = 0; i < nItems; i++) {
      let total = 0;
      for (const k of chosen) {
        total += values[(p * nItems + i) * nCategories + k] * scales[p][k];
      }
      scores[i] = total / chosen.length;
    }
    const row = [];
    for (let k = 0; k < nCategories; k++) {
      const a = [], b = [];
      for (let i = 0; i < nItems; i++) {
        if (items.category[i] === k) {
          a.push(scores[i]);
          b.push(items.y[i]);
        }
      }
      row.push(correlation(a, b, metric));
    }
    const heldOut = row.filter((v, k) => !chosen.includes(k));
    const inSubset = row.filter((v, k) => chosen.includes(k));
    row.push(meanOf(heldOut));
    row.push(meanOf(inSubset));
    matrix.push(row);
  });
  // Means over the training prompts and over the evaluation prompts.
  const nTrain = data.prompts.length;
  for (const group of [matrix.slice(0, nTrain), matrix.slice(nTrain)]) {
    if (group.length === 0) {
      continue;
    }
    matrix.push(group[0].map((_, k) => meanOf(group.map(row => row[k]))));
  }
  const rounded = matrix.map(row => row.map(v => v === null ? null : Math.round(v * 10000) / 10000));
  return { matrix: rounded, chosen: chosen };
}

function drawGrid(plotId, matrix, rowLabels, columnLabels, outlined, valueLabel, rowBreaks) {
  const shapes = outlined.map(j => ({
    type: "rect", xref: "x", yref: "paper",
    x0: j - 0.5, x1: j + 0.5, y0: 0, y1: 1,
    line: { color: "#000000", width: 2 },
  }));
  // A vertical line before the two mean columns.
  const nCategories = columnLabels.length - 2;
  shapes.push({
    type: "line", xref: "x", yref: "paper",
    x0: nCategories - 0.5, x1: nCategories - 0.5, y0: 0, y1: 1,
    line: { color: "#444444", width: 3 },
  });
  // Horizontal lines between the training, evaluation and mean rows.
  for (const y of rowBreaks) {
    shapes.push({
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: y - 0.5, y1: y - 0.5,
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
    hovertemplate: "prompt: %{customdata[0]}<br>evaluated on: %{customdata[1]}<br>" +
      valueLabel + ": %{z:.3f}<extra></extra>",
    customdata: rowLabels.map(r => columnLabels.map(c => [r, c])),
    colorbar: { title: { text: valueLabel } },
  };
  const layout = {
    margin: { l: 120, r: 10, t: 10, b: 120 },
    xaxis: { tickmode: "array", tickvals: columnLabels.map((_, k) => k),
             ticktext: columnLabels, tickangle: -40, title: { text: "Evaluated on" } },
    yaxis: { tickmode: "array", tickvals: rowLabels.map((_, k) => k),
             ticktext: rowLabels, autorange: "reversed", title: { text: "Evaluated on prompt" } },
    shapes: shapes,
  };
  Plotly.react(document.getElementById(plotId), [trace], layout, { responsive: true });
}

makePanels("mean-panels", "mean-plot");
for (const baseline of data.baselines) {
  const panel = document.createElement("div");
  panel.innerHTML = '<h3></h3><div class="table-wrap"><table class="best" id="mean-best-' +
    baseline + '"></table></div>';
  panel.querySelector("h3").textContent = data.baseline_labels[baseline];
  document.getElementById("mean-best-panels").appendChild(panel);
}

for (const baseline of data.baselines) {
  const panel = document.createElement("div");
  panel.innerHTML = '<h3></h3><div class="table-wrap"><table class="best" id="mean-size-' +
    baseline + '"></table></div>';
  panel.querySelector("h3").textContent = data.baseline_labels[baseline];
  document.getElementById("mean-size-panels").appendChild(panel);
}

function drawSizes(key, baseline, metric) {
  const table = document.getElementById("mean-size-" + baseline);
  const bySize = data.mean_direction[key][baseline].by_size[metric];
  const sizes = Object.keys(bySize.train).map(Number).sort((a, b) => a - b);
  let text = "<thead><tr><th class='label'>Categories selected</th>";
  for (const size of sizes) {
    text += "<th>" + size + "</th>";
  }
  text += "</tr></thead><tbody>";
  const groups = [["train", "Held-out mean, train prompts"], ["eval", "Held-out mean, eval prompts"]];
  for (const [group, label] of groups) {
    if (!bySize[group]) {
      continue;
    }
    text += "<tr><td class='label'>" + label + "</td>";
    for (const size of sizes) {
      const v = bySize[group][size];
      text += "<td>" + (v === null || v === undefined ? "" : v.toFixed(3)) + "</td>";
    }
    text += "</tr>";
  }
  table.innerHTML = text + "</tbody>";
}

function drawBest(key, baseline, metric) {
  const table = document.getElementById("mean-best-" + baseline);
  const records = data.mean_direction[key][baseline].best[metric];
  let text = "<thead><tr><th>#</th><th class='label'>Subset</th>" +
    "<th class='label'>Held-out categories</th><th>Advantage</th>" +
    "<th>Held-out mean</th><th>Selected mean</th><th>Held-out mean, eval prompts</th>" +
    "<th></th></tr></thead><tbody>";
  records.forEach((record, rank) => {
    const names = record.subset.map(k => data.categories[k]);
    const held = data.categories.filter(c => !names.includes(c));
    text += "<tr><td>" + (rank + 1) + "</td><td class='subset'>" + names.join(", ") +
      "</td><td class='subset'>" + held.join(", ") + "</td><td>" +
      record.advantage.toFixed(3) + "</td><td>" + record.held_out_mean.toFixed(3) +
      "</td><td>" + record.in_subset_mean.toFixed(3) +
      "</td><td>" + (record.eval_held_out_mean === undefined ? "" :
                     record.eval_held_out_mean.toFixed(3)) +
      "</td><td><button data-rank='" + rank + "'>Select</button></td></tr>";
  });
  table.innerHTML = text + "</tbody>";
  for (const button of table.querySelectorAll("button")) {
    button.addEventListener("click", () => {
      const record = records[Number(button.dataset.rank)];
      setChecks(record.subset.map(k => data.categories[k]));
    });
  }
}

function renderMean(key) {
  if (!data.mean_direction[key]) {
    return;
  }
  const metric = metricSelect.value;
  const selected = data.categories.filter(c => meanChecks[c].checked);
  const columnLabels = data.categories.concat(["held-out mean", "selected mean"]);
  for (const b of data.baselines) {
    const plot = document.getElementById("mean-plot-" + b);
    if (selected.length === 0) {
      Plotly.purge(plot);
      plot.dataset.empty = "1";
      plot.textContent = "Select one or more categories.";
    } else {
      if (plot.dataset.empty) {
        plot.textContent = "";
        delete plot.dataset.empty;
      }
      const result = meanMatrix(key, b, selected, metric);
      const nTrain = data.prompts.length;
      const nAll = data.mean_prompts.length;
      const rowLabels = data.prompts.concat(data.eval_prompts.map(p => p + " (eval)"));
      const rowBreaks = [];
      if (nAll > nTrain) {
        rowLabels.push("mean, train prompts", "mean, eval prompts");
        rowBreaks.push(nTrain, nAll);
      } else {
        rowLabels.push("mean, train prompts");
        rowBreaks.push(nTrain);
      }
      drawGrid("mean-plot-" + b, result.matrix, rowLabels, columnLabels,
               result.chosen, data.metric_labels[metric], rowBreaks);
    }
    drawSizes(key, b, metric);
    drawBest(key, b, metric);
  }
  document.getElementById("mean-best-caption").textContent =
    "Below: for the selected setting and metric, the 10 subsets with the largest advantage " +
    "on the training prompts. The last column is the held-out mean of the same subset on the " +
    "evaluation prompts, which were not used to select it. " +
    "Advantage: for each held-out category, the score of the subset minus the mean score " +
    "of all subsets of the same size that do not contain that category, averaged over the " +
    "held-out categories and the prompts. This removes the effect of categories that are " +
    "easy for all directions. Only subsets with " + data.mean_min_held_out +
    " or more held-out categories are ranked. The advantage compares subsets of the " +
    "same size only: single categories vary most, so they are often at the top, but on " +
    "average more categories transfer better (table above). The best of about 1000 " +
    "subsets is selected on the same data, so its value is optimistic.";
}

// Permutation null

for (const baseline of data.baselines) {
  const panel = document.createElement("div");
  panel.innerHTML = '<h3></h3><div class="table-wrap"><table class="best" id="null-' +
    baseline + '"></table></div>';
  panel.querySelector("h3").textContent = data.baseline_labels[baseline];
  document.getElementById("null-panels").appendChild(panel);
}

function formatNumber(v) {
  return v === null || v === undefined ? "" : v.toFixed(3);
}

function drawNull(key, metric) {
  const nullSettings = [...new Set(data.null.map(row => row.setting))];
  const caption = document.getElementById("null-caption");
  const labels = nullSettings.map(k => {
    const s = data.settings.find(item => item.key === k);
    return s ? settingLabel(s).join(", ") : k;
  });
  caption.textContent = nullSettings.length === 0 ?
    "No permutation null in this run." :
    "Computed for: " + labels.join("; ") + ". " +
    (nullSettings.includes(key) ? "" : "Select one of these settings to see the null.");
  for (const b of data.baselines) {
    const table = document.getElementById("null-" + b);
    const rows = data.null.filter(row => row.setting === key && row.baseline === b &&
                                         row.metric === metric);
    if (rows.length === 0) {
      table.innerHTML = "";
      continue;
    }
    let text = "<thead><tr><th class='label'>Statistic</th><th>Observed</th>" +
      "<th>Null mean</th><th>Null 95%</th><th>Null max</th><th>p</th>" +
      "<th>Shuffles</th></tr></thead><tbody>";
    for (const column of Object.keys(data.summary_columns)) {
      const row = rows.find(item => item.statistic === column);
      if (!row) {
        continue;
      }
      text += "<tr><td class='label'>" + data.summary_columns[column] + "</td><td>" +
        formatNumber(row.observed) + "</td><td>" + formatNumber(row.null_mean) +
        "</td><td>" + formatNumber(row.null_q95) + "</td><td>" +
        formatNumber(row.null_max) + "</td><td>" + formatNumber(row.p_value) +
        "</td><td>" + row.n_permutations + "</td></tr>";
    }
    table.innerHTML = text + "</tbody>";
  }
}

ambiguousSelect.addEventListener("change", updateCategoryOptions);
for (const select of [...parameterSelects, metricSelect, promptSelect, categorySelect]) {
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
    selectors = []
    for parameter in plot_data["parameters"]:
        selectors.append(
            f"  <label>{html.escape(parameter['label'])} "
            f'<select id="{html.escape(parameter["id"])}"></select></label>'
        )
    page = page.replace("__SELECTORS__", "\n".join(selectors))
    return page.replace("__DATA__", data_json)


def write_html(plot_data, path):
    with open(path, "w") as f:
        f.write(render_html(plot_data))
