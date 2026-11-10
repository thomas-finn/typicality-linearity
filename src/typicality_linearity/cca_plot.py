"""Interactive HTML page for the CAA experiment (cca_experiment.py).

The page compares the no-baseline and mean-contrast directions in each section:
side-by-side values and their difference (mean contrast minus no baseline).

Sections:
1. Summary: each statistic for both baselines and the difference
2. Mean direction over all categories on the evaluation prompts, by category
3. Per-category directions: train category x eval category, on a selected prompt
4. Cosine similarity between the category directions, and between the
   no-baseline and mean-contrast direction of the same category
"""

import csv
import html
import json
import math
from pathlib import Path

import yaml

from typicality_linearity.pca_plot import PLOTLY_URL

MEAN_ALL = "mean_all"
MEAN_LOCO = "mean_loco"
MEAN_TRAIN = "mean_train"
MEAN_EVAL = "mean_eval"

BASELINE_LABELS = {"none": "No baseline", "mean_contrast": "Mean contrast"}
DIRECTION_LABELS = {
    MEAN_ALL: "Mean of all categories",
    MEAN_LOCO: "Mean of other categories",
}
STATISTIC_LABELS = {
    "within_cv": "Own-category direction (CV)",
    "between_categories": "Other-category directions",
    MEAN_ALL: "Mean direction, all categories",
    MEAN_LOCO: "Mean direction, other categories",
    "cosine_between_categories": "Cosine between categories",
}
PROMPT_SET_LABELS = {"train": "Training prompts", "eval": "Evaluation prompts"}


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


def unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def summary_data(summary):
    """Return {"<ambiguous>|<metric>|<prompt set>|<baseline>": {statistic: value}}."""
    result = {}
    for row in summary:
        key = "|".join(
            [row["ambiguous"], row["metric"], row["prompt_set"], row["baseline"]]
        )
        result[key] = {s: clean_number(row[s]) for s in STATISTIC_LABELS}
    return result


def matrix_data(cells, categories, train_prompts, eval_prompts):
    """Return {"<ambiguous>|<metric>|<baseline>|<prompt>": z}.

    z rows: the category directions, then MEAN_ALL and MEAN_LOCO. z columns:
    the eval categories. prompt is a prompt name, MEAN_TRAIN or MEAN_EVAL.
    """
    directions = list(categories) + [MEAN_ALL, MEAN_LOCO]
    values = {}
    for cell in cells:
        key = (cell["ambiguous"], cell["metric"], cell["baseline"])
        values.setdefault(key, {})[
            (cell["prompt"], cell["direction"], cell["eval_category"])
        ] = clean_number(cell["value"])
    result = {}
    for key, by_cell in values.items():
        prefix = "|".join(key)
        for prompt in list(train_prompts) + list(eval_prompts):
            result[f"{prefix}|{prompt}"] = [
                [by_cell.get((prompt, d, c)) for c in categories] for d in directions
            ]
        for name, prompts in ((MEAN_TRAIN, train_prompts), (MEAN_EVAL, eval_prompts)):
            result[f"{prefix}|{name}"] = [
                [
                    nan_mean([by_cell.get((p, d, c)) for p in prompts])
                    for c in categories
                ]
                for d in directions
            ]
    return result


def cosine_data(cosines, categories):
    """Return (between categories, between baselines).

    between categories: {"<ambiguous>|<baseline>": z over categories + MEAN_ALL}
    between baselines: {"<ambiguous>": {direction: cosine}}
    """
    names = list(categories) + [MEAN_ALL]
    values = {}
    between_baselines = {}
    for row in cosines:
        value = clean_number(row["cosine"])
        if row["kind"] == "between_baselines":
            between_baselines.setdefault(row["ambiguous"], {})[row["direction_a"]] = (
                value
            )
            continue
        key = f"{row['ambiguous']}|{row['baseline']}"
        values.setdefault(key, {})[(row["direction_a"], row["direction_b"])] = value
    between_categories = {
        key: [[pairs.get((a, b)) for b in names] for a in names]
        for key, pairs in values.items()
    }
    return between_categories, between_baselines


def page_notes(config, n_items):
    items = ", ".join(f"{option}: {n}" for option, n in n_items.items())
    return [
        (
            f"Model {config['model']}, layer {config['probe_layer']}, "
            f"{config['probe_position']} token. Items ({items})."
        ),
        (
            f"CAA direction of a category: mean activation of the top 1/{config['n_groups']}"
            f" most typical items minus the bottom 1/{config['n_groups']}, pooled over the "
            f"{len(config['prompt_columns'])} training prompts. Typicality = "
            f"{config['target_sign']} × {config['target']}."
        ),
        (
            "Mean direction, all categories: mean of the unit directions of all "
            "categories; for its own category, the fold direction that did not see the "
            "item is used. Mean direction, other categories: mean of the unit directions "
            "of the other categories only."
        ),
        (
            "Training prompts are scored with the activations of the baseline the "
            "direction was trained with. Evaluation prompts are never used for training "
            "and are scored with raw activations (no baseline)."
        ),
        (
            f"Own-category cells are cross-validated ({config['n_folds']} folds, "
            f"{config['cv_aggregate']} correlation); other cells use the direction "
            "trained on all items. The correlation is computed inside each prompt and "
            "category."
        ),
    ]


def build_plot_data(config, summary, cells, cosines, prompt_formats, eval_formats):
    categories = sorted(unique(cell["eval_category"] for cell in cells))
    train_prompts = list(config["prompt_columns"])
    eval_prompts = list(config["eval_prompt_columns"])
    baselines = unique(row["baseline"] for row in summary)
    n_items = {row["ambiguous"]: int(row["n_items"]) for row in summary}
    between_categories, between_baselines = cosine_data(cosines, categories)
    return {
        "title": config.get("title", config["experiment"]),
        "notes": page_notes(config, n_items),
        "ambiguous_options": unique(row["ambiguous"] for row in summary),
        "metrics": list(config["metrics"]),
        "baselines": baselines,
        "baseline_labels": {b: BASELINE_LABELS.get(b, b) for b in baselines},
        "categories": categories,
        "directions": categories + [MEAN_ALL, MEAN_LOCO],
        "direction_labels": {
            **{c: c for c in categories},
            **DIRECTION_LABELS,
        },
        "train_prompts": train_prompts,
        "eval_prompts": eval_prompts,
        "prompt_formats": {**prompt_formats, **eval_formats},
        "statistics": list(STATISTIC_LABELS),
        "statistic_labels": STATISTIC_LABELS,
        "prompt_set_labels": PROMPT_SET_LABELS,
        "summary": summary_data(summary),
        "matrices": matrix_data(cells, categories, train_prompts, eval_prompts),
        "cosine": between_categories,
        "cosine_between_baselines": between_baselines,
    }


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY_URL__"></script>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series-none: #2a78d6; --series-mc: #eb6834;
    --div-neg: #e34948; --div-mid: #f0efec; --div-pos: #2a78d6;
    --up: #006300; --down: #d03b3b; --accent-wash: #e8f1fb;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --series-none: #3987e5; --series-mc: #d95926;
      --div-neg: #e66767; --div-mid: #383835; --div-pos: #3987e5;
      --up: #0ca30c; --down: #e66767; --accent-wash: #1c2a3a;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-none: #3987e5; --series-mc: #d95926;
    --div-neg: #e66767; --div-mid: #383835; --div-pos: #3987e5;
    --up: #0ca30c; --down: #e66767; --accent-wash: #1c2a3a;
  }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0;
         padding: 16px; color: var(--ink); background: var(--page); }
  main { max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 8px; }
  h2 { font-size: 17px; margin: 32px 0 6px; }
  h3 { font-size: 14px; margin: 0 0 4px; }
  .notes { font-size: 13px; color: var(--ink-2); margin: 0 0 12px; padding-left: 18px; }
  .controls { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; }
  .controls label { display: flex; flex-direction: column; font-size: 13px; gap: 4px;
                    color: var(--ink-2); }
  .sticky { position: sticky; top: 0; background: var(--page); padding: 8px 0;
            border-bottom: 1px solid var(--border); z-index: 10; }
  select { font-size: 14px; padding: 4px; color: var(--ink); background: var(--surface);
           border: 1px solid var(--border); border-radius: 4px; }
  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr));
            gap: 16px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
          padding: 12px; min-width: 0; }
  .plot { width: 100%; height: 460px; }
  .plot.short { height: 340px; }
  .caption { font-size: 12px; color: var(--ink-2); margin: 0 0 10px; max-width: 900px; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { padding: 4px 10px; border-bottom: 1px solid var(--grid); text-align: right;
           white-space: nowrap; }
  th { color: var(--ink-2); font-weight: 600; }
  td.label, th.label { text-align: left; }
  td.delta { font-weight: 600; }
  .up { color: var(--up); } .down { color: var(--down); }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
            margin-right: 6px; vertical-align: -1px; }
  tr.section td { color: var(--muted); font-size: 12px; text-align: left; padding-top: 10px; }
  table.formats td { text-align: left; white-space: normal; }
  code { font-family: ui-monospace, monospace; font-size: 12px; }
  .callout { border-left: 3px solid var(--series-mc); background: var(--surface);
             padding: 8px 12px; font-size: 13px; margin: 8px 0 12px; border-radius: 0 6px 6px 0; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<ul class="notes" id="notes"></ul>
<div class="controls sticky">
  <label>Ambiguous items <select id="ambiguous"></select></label>
  <label>Metric <select id="metric"></select></label>
</div>

<h2>No baseline vs mean contrast</h2>
<p class="caption">Mean over cells and prompts. Δ = mean contrast − no baseline; green:
mean contrast is higher. Own-category cells are cross-validated.</p>
<div class="callout" id="headline"></div>
<div class="panels">
  <div class="card"><div class="table-wrap"><table id="summary"></table></div></div>
  <div class="card"><h3>Evaluation prompts</h3><div class="plot short" id="summary-bars"></div></div>
</div>

<h2>Mean direction over all categories, on the evaluation prompts</h2>
<p class="caption">One direction: the mean of the unit CAA directions of all 10 categories
(cross-validated for the item's own category). Scored on raw activations of prompt formats
that were never used for training.</p>
<div class="controls">
  <label>Prompt <select id="mean-prompt"></select></label>
</div>
<div class="panels">
  <div class="card"><h3>By category</h3><div class="plot short" id="mean-bars"></div></div>
  <div class="card"><h3>Δ by evaluation prompt (mean contrast − no baseline)</h3>
    <div class="plot short" id="mean-delta"></div></div>
</div>

<h2>Per-category directions</h2>
<p class="caption">Rows: direction (category it was trained on; the last two rows are the
mean directions). Columns: category it is evaluated on. Diagonal cells are cross-validated.
The prompt selects the activations it is evaluated on.</p>
<div class="controls">
  <label>Prompt <select id="matrix-prompt"></select></label>
</div>
<div class="table-wrap"><table class="formats"><tbody><tr><td id="matrix-format"></td></tr></tbody></table></div>
<div class="panels" id="matrix-panels"></div>
<div class="card" style="margin-top:16px"><h3>Own-category direction (CV) on the selected prompt</h3>
  <div class="plot short" id="diagonal-bars"></div></div>

<h2>Cosine similarity between directions</h2>
<p class="caption">Cosine between the unnormalised CAA directions of two categories, and
the mean direction over all categories (last row). Right: cosine between the no-baseline
and the mean-contrast direction of the same category.</p>
<div class="panels" id="cosine-panels"></div>
<div class="card" style="margin-top:16px"><h3>Cosine, no baseline vs mean contrast, same direction</h3>
  <div class="plot short" id="cosine-baselines"></div></div>

<h2>Prompt formats</h2>
<div class="table-wrap"><table class="formats" id="formats"></table></div>
</main>

<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, digits = 2) => (v === null || v === undefined) ? "–" : v.toFixed(digits);
const NONE = "none", MC = "mean_contrast";

function addOptions(select, values, labels) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels ? (labels[value] ?? value) : value;
    select.appendChild(option);
  }
}

for (const note of data.notes) {
  const li = document.createElement("li"); li.textContent = note; $("notes").appendChild(li);
}
addOptions($("ambiguous"), data.ambiguous_options);
addOptions($("metric"), data.metrics);
const promptLabels = {mean_eval: "Mean over evaluation prompts", mean_train: "Mean over training prompts"};
addOptions($("mean-prompt"), ["mean_eval", ...data.eval_prompts], promptLabels);
addOptions($("matrix-prompt"),
  ["mean_eval", ...data.eval_prompts, "mean_train", ...data.train_prompts], promptLabels);

function formatsTable() {
  let rows = "<thead><tr><th class='label'>Prompt</th><th class='label'>Set</th><th class='label'>Format</th></tr></thead><tbody>";
  for (const p of data.train_prompts) rows += `<tr><td><code>${p}</code></td><td>training</td><td>${data.prompt_formats[p] ?? ""}</td></tr>`;
  for (const p of data.eval_prompts) rows += `<tr><td><code>${p}</code></td><td>evaluation</td><td>${data.prompt_formats[p] ?? ""}</td></tr>`;
  $("formats").innerHTML = rows + "</tbody>";
}
formatsTable();

function seriesColor(baseline) { return baseline === MC ? css("--series-mc") : css("--series-none"); }

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: {family: "system-ui, sans-serif", size: 12, color: css("--ink-2")},
    margin: {l: 60, r: 16, t: 16, b: 80},
    xaxis: {gridcolor: css("--grid"), linecolor: css("--axis"), zerolinecolor: css("--axis")},
    yaxis: {gridcolor: css("--grid"), linecolor: css("--axis"), zerolinecolor: css("--axis")},
    legend: {orientation: "h", y: 1.12, x: 0},
    hoverlabel: {bgcolor: css("--surface"), font: {color: css("--ink")}, bordercolor: css("--axis")},
  }, extra);
}
const config = {displaylogo: false, responsive: true};

function divergingScale() {
  return [[0, css("--div-neg")], [0.5, css("--div-mid")], [1, css("--div-pos")]];
}

function heatmap(target, z, x, y, title, range) {
  const text = z.map(row => row.map(v => fmt(v)));
  Plotly.react(target, [{
    type: "heatmap", z, x, y, text, texttemplate: "%{text}", textfont: {size: 10},
    colorscale: divergingScale(), zmin: -range, zmax: range, zmid: 0, xgap: 2, ygap: 2,
    hovertemplate: "direction %{y}<br>evaluated on %{x}<br>value %{z:.3f}<extra>" + title + "</extra>",
    colorbar: {thickness: 10, outlinewidth: 0},
  }], baseLayout({
    margin: {l: 150, r: 16, t: 8, b: 90},
    yaxis: {autorange: "reversed", linecolor: css("--axis")},
    xaxis: {tickangle: -40, linecolor: css("--axis")},
  }), config);
}

function key(...parts) { return parts.join("|"); }
function current() { return {amb: $("ambiguous").value, metric: $("metric").value}; }
function subtract(a, b) {
  return a.map((row, i) => row.map((v, j) => (v === null || b[i][j] === null) ? null : v - b[i][j]));
}

function renderSummary() {
  const {amb, metric} = current();
  let rows = `<thead><tr><th class="label">Statistic</th>
    <th><span class="swatch" style="background:${seriesColor(NONE)}"></span>No baseline</th>
    <th><span class="swatch" style="background:${seriesColor(MC)}"></span>Mean contrast</th>
    <th>Δ</th></tr></thead><tbody>`;
  for (const set of ["eval", "train"]) {
    rows += `<tr class="section"><td colspan="4">${data.prompt_set_labels[set]}</td></tr>`;
    const none = data.summary[key(amb, metric, set, NONE)] || {};
    const mc = data.summary[key(amb, metric, set, MC)] || {};
    for (const s of data.statistics) {
      if (s === "cosine_between_categories" && set === "eval") continue;
      const d = (none[s] == null || mc[s] == null) ? null : mc[s] - none[s];
      const cls = d === null ? "" : (d > 0.005 ? "up" : (d < -0.005 ? "down" : ""));
      rows += `<tr><td class="label">${data.statistic_labels[s]}</td><td>${fmt(none[s])}</td>
        <td>${fmt(mc[s])}</td><td class="delta ${cls}">${d === null ? "–" : (d > 0 ? "+" : "") + d.toFixed(2)}</td></tr>`;
    }
  }
  $("summary").innerHTML = rows + "</tbody>";

  const none = data.summary[key(amb, metric, "eval", NONE)] || {};
  const mc = data.summary[key(amb, metric, "eval", MC)] || {};
  const stats = data.statistics.filter(s => s !== "cosine_between_categories");
  const labels = stats.map(s => data.statistic_labels[s]);
  Plotly.react("summary-bars", [NONE, MC].map(b => ({
    type: "bar", name: data.baseline_labels[b], x: labels,
    y: stats.map(s => (b === MC ? mc : none)[s]),
    marker: {color: seriesColor(b)},
    hovertemplate: "%{x}<br>%{y:.3f}<extra>" + data.baseline_labels[b] + "</extra>",
  })), baseLayout({barmode: "group", bargap: 0.3, bargroupgap: 0.08,
    yaxis: {title: metric, gridcolor: css("--grid"), zerolinecolor: css("--axis")},
    xaxis: {tickangle: 0, automargin: true}}), config);

  const cosNone = data.summary[key(amb, metric, "train", NONE)] || {};
  const cosMc = data.summary[key(amb, metric, "train", MC)] || {};
  const between = data.cosine_between_baselines[amb] || {};
  $("headline").innerHTML =
    `On the evaluation prompts, the mean direction over all categories gives <b>${fmt(none.mean_all)}</b> ` +
    `(no baseline) vs <b>${fmt(mc.mean_all)}</b> (mean contrast); the own-category direction gives ` +
    `<b>${fmt(none.within_cv)}</b> vs <b>${fmt(mc.within_cv)}</b>. Mean cosine between categories: ` +
    `<b>${fmt(cosNone.cosine_between_categories)}</b> vs <b>${fmt(cosMc.cosine_between_categories)}</b>. ` +
    `The two mean directions have cosine <b>${fmt(between.mean_all)}</b>.`;
}

function meanRow(amb, metric, baseline, prompt) {
  const z = data.matrices[key(amb, metric, baseline, prompt)];
  return z ? z[data.directions.indexOf("mean_all")] : null;
}

function renderMean() {
  const {amb, metric} = current();
  const prompt = $("mean-prompt").value;
  const x = [...data.categories, "mean"];
  const traces = [NONE, MC].map(b => {
    const row = meanRow(amb, metric, b, prompt) || [];
    const kept = row.filter(v => v !== null);
    const mean = kept.length ? kept.reduce((s, v) => s + v, 0) / kept.length : null;
    return {type: "bar", name: data.baseline_labels[b], x, y: [...row, mean],
      marker: {color: seriesColor(b)},
      hovertemplate: "%{x}<br>%{y:.3f}<extra>" + data.baseline_labels[b] + "</extra>"};
  });
  Plotly.react("mean-bars", traces, baseLayout({barmode: "group", bargap: 0.25,
    yaxis: {title: metric, gridcolor: css("--grid"), zerolinecolor: css("--axis")},
    xaxis: {tickangle: -40}}), config);

  const prompts = [...data.eval_prompts, "mean_eval"];
  const z = prompts.map(p => {
    const none = meanRow(amb, metric, NONE, p), mc = meanRow(amb, metric, MC, p);
    return (none && mc) ? mc.map((v, j) => (v === null || none[j] === null) ? null : v - none[j]) : data.categories.map(() => null);
  });
  heatmap("mean-delta", z, data.categories, prompts.map(p => promptLabels[p] ?? p), "Δ", 0.3);
}

function renderMatrices() {
  const {amb, metric} = current();
  const prompt = $("matrix-prompt").value;
  $("matrix-format").innerHTML = data.prompt_formats[prompt]
    ? `<code>${prompt}</code>: ${data.prompt_formats[prompt]}`
    : (promptLabels[prompt] ?? prompt);
  const panels = $("matrix-panels");
  panels.innerHTML = "";
  const none = data.matrices[key(amb, metric, NONE, prompt)];
  const mc = data.matrices[key(amb, metric, MC, prompt)];
  const ylabels = data.directions.map(d => data.direction_labels[d]);
  const items = [[NONE, none, 1], [MC, mc, 1], ["delta", none && mc ? subtract(mc, none) : null, 0.4]];
  for (const [name, z, range] of items) {
    if (!z) continue;
    const card = document.createElement("div"); card.className = "card";
    const title = name === "delta" ? "Δ (mean contrast − no baseline)" : data.baseline_labels[name];
    card.innerHTML = `<h3>${title}</h3><div class="plot"></div>`;
    panels.appendChild(card);
    heatmap(card.querySelector(".plot"), z, data.categories, ylabels, title, range);
  }
  Plotly.react("diagonal-bars", [NONE, MC].map(b => {
    const z = data.matrices[key(amb, metric, b, prompt)] || [];
    return {type: "bar", name: data.baseline_labels[b], x: data.categories,
      y: data.categories.map((c, i) => z[i] ? z[i][i] : null), marker: {color: seriesColor(b)},
      hovertemplate: "%{x}<br>%{y:.3f}<extra>" + data.baseline_labels[b] + "</extra>"};
  }), baseLayout({barmode: "group", bargap: 0.25,
    yaxis: {title: metric, gridcolor: css("--grid"), zerolinecolor: css("--axis")},
    xaxis: {tickangle: -40}}), config);
}

function renderCosine() {
  const {amb} = current();
  const names = [...data.categories, "mean_all"];
  const labels = names.map(n => data.direction_labels[n]);
  const panels = $("cosine-panels");
  panels.innerHTML = "";
  const none = data.cosine[key(amb, NONE)], mc = data.cosine[key(amb, MC)];
  for (const [name, z, range] of [[NONE, none, 1], [MC, mc, 1], ["delta", none && mc ? subtract(mc, none) : null, 0.5]]) {
    if (!z) continue;
    const card = document.createElement("div"); card.className = "card";
    const title = name === "delta" ? "Δ (mean contrast − no baseline)" : data.baseline_labels[name];
    card.innerHTML = `<h3>${title}</h3><div class="plot"></div>`;
    panels.appendChild(card);
    heatmap(card.querySelector(".plot"), z, labels, labels, title, range);
  }
  const between = data.cosine_between_baselines[amb] || {};
  Plotly.react("cosine-baselines", [{type: "bar", x: labels, y: names.map(n => between[n] ?? null),
    marker: {color: css("--muted")}, hovertemplate: "%{x}<br>cosine %{y:.3f}<extra></extra>"}],
    baseLayout({showlegend: false, yaxis: {title: "cosine", range: [0, 1], gridcolor: css("--grid")},
      xaxis: {tickangle: -40}}), config);
}

function render() { renderSummary(); renderMean(); renderMatrices(); renderCosine(); }
for (const id of ["ambiguous", "metric"]) $(id).addEventListener("change", render);
$("mean-prompt").addEventListener("change", renderMean);
$("matrix-prompt").addEventListener("change", renderMatrices);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
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


def write_run_page(run_dir, html_name, prompt_formats, eval_formats):
    """Read the files of a run directory and write the page into it."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    plot_data = build_plot_data(
        config,
        read_csv(run_dir / "summary.csv"),
        read_csv(run_dir / "correlations.csv"),
        read_csv(run_dir / "cosine_similarity.csv"),
        prompt_formats,
        eval_formats,
    )
    path = run_dir / html_name
    with open(path, "w") as f:
        f.write(render_html(plot_data))
    return path
