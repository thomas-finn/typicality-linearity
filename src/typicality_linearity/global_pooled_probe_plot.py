"""Write the three-page HTML for the pooled global direction probes.

One row of selectors (layer, ambiguous items, difference of means, metric) is
shared by the pages; each page shows the selectors it uses. Each page is a 2x2
grid, one panel per activation type.

- Page 1, "Pooled over categories": rows train prompt, columns eval prompt.
  One probe per prompt, trained on all categories. All cells are
  cross-validated; the value is the mean over the categories of the metric in
  each category. The evaluation prompts (never used for training) are extra
  columns on the right, after a line.
- Page 2, "Pooled over prompts": rows train category, columns eval category.
  One probe per category, trained on all prompts. The diagonal is
  cross-validated; the value is the mean over the prompts of the metric in
  each prompt.
- Page 3, "Pooled over categories and prompts" (only with evaluation
  prompts): one probe trained on all items of all categories in all training
  prompts. Scatter of typicality (-rating z-score) against the probe score of
  each item in the selected evaluation prompt. The score is raw (x . direction)
  or z-scored in each category. One category can be highlighted. The probe is
  not cross-validated: the items are the training items in a new format.

Colour: diverging around chance, with the same range for all settings of a
metric (see global_probe_plot.py).
"""

import html
import json

import numpy as np

from typicality_linearity.global_probe_plot import (
    PLOTLY_URL,
    STYLE,
    color_range,
    to_json_list,
)


def build_page_data(
    title,
    axes,
    arrays,
    n_items,
    labels,
    metric_chance,
    prompt_formats,
    cv_aggregate,
    n_folds,
    points=None,
    eval_texts=None,
):
    ranges = {}
    for mi, metric in enumerate(axes["metric"]):
        values = np.concatenate(
            [
                arrays["over_categories"][:, :, :, :, mi].ravel(),
                arrays["over_categories_eval"][:, :, :, :, mi].ravel(),
                arrays["over_prompts"][:, :, :, :, mi].ravel(),
            ]
        )
        ranges[metric] = color_range(values, metric_chance[metric])
    return {
        "title": title,
        "axes": {key: [str(v) for v in values] for key, values in axes.items()},
        "labels": labels,
        "chance": metric_chance,
        "range": ranges,
        "nItems": n_items,
        "promptFormats": prompt_formats,
        "cvAggregate": cv_aggregate,
        "nFolds": n_folds,
        "oc": to_json_list(arrays["over_categories"]),
        "oce": to_json_list(arrays["over_categories_eval"]),
        "op": to_json_list(arrays["over_prompts"]),
        "points": points or {},
        "evalTexts": eval_texts or {},
        "gr": to_json_list(arrays["global_scores_raw"]),
        "gz": to_json_list(arrays["global_scores_z"]),
        "gm": to_json_list(arrays["global_eval_metrics"]),
    }


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY_URL__"></script>
<style>
__STYLE__</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="subtitle">Difference-of-means probes pooled over categories or over prompt formats. Cross-validated cells use <span id="cv-text"></span>.</p>

<div class="tabs" role="tablist">
  <button class="tab" role="tab" aria-selected="true" data-page="oc">Pooled over categories</button>
  <button class="tab" role="tab" aria-selected="false" data-page="op">Pooled over prompts</button>
  <button class="tab" role="tab" aria-selected="false" data-page="gl" id="tab-gl">Pooled over categories and prompts</button>
</div>

<div class="filters">
  <label data-for="gl">Eval prompt <select id="sel-eval"></select></label>
  <label data-for="oc op gl">Layer <select id="sel-layer"></select></label>
  <label data-for="oc op gl">Ambiguous items <select id="sel-ambiguity"></select></label>
  <label data-for="oc op gl">Difference of means <select id="sel-groups"></select></label>
  <label data-for="oc op">Metric <select id="sel-metric"></select></label>
  <label data-for="gl">Score <select id="sel-scale"><option value="z">Z-scored in each category</option><option value="raw">Raw (x · direction)</option></select></label>
  <label data-for="gl">Highlight <select id="sel-highlight"></select></label>
</div>
<p class="context" id="context"></p>

<div class="grid" id="grid"></div>
<p class="note" id="note"></p>

<script id="page-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("page-data").textContent);
const AX = DATA.axes;
const state = {
  page: "oc",
  layer: AX.layer.includes("15") ? "15" : AX.layer[0],
  ambiguity: AX.ambiguity[0],
  groups: AX.n_groups.includes("3") ? "3" : AX.n_groups[0],
  metric: AX.metric[0],
  evalPrompt: AX.eval_prompt[0],
  scale: "z",
  highlight: "all",
};
const NOTES = {
  oc: "One probe per prompt format, trained on the items of all categories together. Rows: the train prompt. Columns: the eval prompt; the columns right of the line are the evaluation prompts, never used for training. All cells are cross-validated with the same folds, so a probe is never evaluated on its training items. The value is the metric in each category, averaged over the categories, so it measures the typicality order inside the categories.",
  gl: "One probe trained on all items of all categories in all training prompt formats (the direction is the mean over the prompts). Each point is one item in the selected evaluation prompt, never used for training. The probe is not cross-validated: the items are the training items in a new prompt format. The metrics are computed in each category, then averaged.",
  op: "One probe per category, trained on its items in all prompt formats together. Rows: the train category. Columns: the eval category. Off-diagonal cells use the probe trained on all items of the train category. Diagonal cells are cross-validated. The value is the metric in each prompt format, averaged over the prompt formats.",
};

function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function escapeHtml(text) { return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function fmt(v) { return v === null || v === undefined ? "n/a" : v.toFixed(2); }
function idx(axis, value) { return AX[axis].indexOf(value); }

function fillSelect(id, axis, labels, key) {
  const select = document.getElementById(id);
  select.innerHTML = "";
  AX[axis].forEach(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels ? labels[value] : value;
    if (value === state[key]) option.selected = true;
    select.appendChild(option);
  });
  select.onchange = e => { state[key] = e.target.value; render(); };
}

function meanAll(matrix) {
  let sum = 0, count = 0;
  matrix.forEach(row => row.forEach(v => { if (v !== null) { sum += v; count += 1; } }));
  return count ? sum / count : null;
}

function meanOf(matrix, diagonal) {
  let sum = 0, count = 0;
  matrix.forEach((row, i) => row.forEach((v, j) => {
    if (v === null || (i === j) !== diagonal) return;
    sum += v; count += 1;
  }));
  return count ? sum / count : null;
}

function buildPanels() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  AX.activation_type.forEach(t => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h2>${escapeHtml(DATA.labels.activation_type[t])}</h2>` +
      `<p class="formula">${escapeHtml(DATA.labels.formula[t])}</p>` +
      `<p class="stats" id="stats-${t}"></p><div class="heat" id="heat-${t}"></div>`;
    grid.appendChild(panel);
  });
}

function draw() {
  const byPrompt = state.page === "oc";
  const names = byPrompt ? AX.prompt : AX.category;
  const centre = DATA.chance[state.metric];
  const half = DATA.range[state.metric];
  const colorscale = [[0, token("--div-low")], [0.5, token("--div-mid")], [1, token("--div-high")]];
  const valueName = DATA.labels.metric[state.metric];
  const rowName = byPrompt ? "train prompt" : "train category";
  const colName = byPrompt ? "eval prompt" : "eval category";
  const a = idx("ambiguity", state.ambiguity), l = idx("layer", state.layer);
  const g = idx("n_groups", state.groups), m = idx("metric", state.metric);
  const evalNames = byPrompt ? AX.eval_prompt : [];
  const columns = names.concat(evalNames);
  AX.activation_type.forEach((t, ti) => {
    const z = DATA[state.page][a][ti][l][g][m];
    const zEval = byPrompt && evalNames.length ? DATA.oce[a][ti][l][g][m] : null;
    const zShown = zEval ? z.map((row, i) => row.concat(zEval[i])) : z;
    const trace = {
      type: "heatmap", x: columns, y: names, z: zShown, zmin: centre - half, zmax: centre + half, zmid: centre,
      colorscale: colorscale, xgap: 2, ygap: 2, hoverongaps: false,
      hovertemplate: `${rowName}: %{y}<br>${colName}: %{x}<br><b>${valueName} %{z:.3f}</b><extra></extra>`,
      showscale: ti === 1,
      colorbar: ti === 1 ? {
        title: { text: valueName, font: { color: token("--text-secondary") } },
        tickfont: { color: token("--text-muted") }, thickness: 10, len: 0.9, outlinewidth: 0,
      } : undefined,
    };
    const layout = {
      paper_bgcolor: token("--surface-1"), plot_bgcolor: token("--surface-1"),
      font: { family: "system-ui, -apple-system, sans-serif", size: 11, color: token("--text-secondary") },
      margin: { l: byPrompt ? 120 : 80, r: 16, t: 8, b: byPrompt ? 120 : 80 },
      xaxis: { tickangle: -45, showgrid: false, tickfont: { color: token("--text-muted") },
        title: { text: colName, font: { color: token("--text-secondary") } } },
      yaxis: { autorange: "reversed", showgrid: false, tickfont: { color: token("--text-muted") },
        title: { text: rowName, font: { color: token("--text-secondary") } } },
      hoverlabel: { bgcolor: token("--surface-1"), bordercolor: token("--border"), font: { color: token("--text-primary") } },
      shapes: zEval ? [{
        type: "line", xref: "x", yref: "paper", x0: names.length - 0.5, x1: names.length - 0.5, y0: 0, y1: 1,
        line: { color: token("--text-primary"), width: 2 },
      }] : [],
    };
    let stats = `Diagonal (CV) mean ${fmt(meanOf(z, true))} · off-diagonal mean ${fmt(meanOf(z, false))}`;
    if (zEval) stats += ` · eval prompts mean ${fmt(meanAll(zEval))}`;
    document.getElementById(`stats-${t}`).textContent = stats;
    Plotly.react(`heat-${t}`, [trace], layout, { responsive: true, displaylogo: false });
  });
}

function drawScatter() {
  const a = idx("ambiguity", state.ambiguity), l = idx("layer", state.layer);
  const g = idx("n_groups", state.groups), e = idx("eval_prompt", state.evalPrompt);
  const P = DATA.points;
  const texts = DATA.evalTexts[state.evalPrompt] || [];
  const yName = state.scale === "z" ? "probe score (z in category)" : "probe score (x · direction)";
  AX.activation_type.forEach((t, ti) => {
    const scores = (state.scale === "z" ? DATA.gz : DATA.gr)[a][ti][l][g][e];
    const traces = [];
    const groups = state.highlight === "all" ? [["all", true]] : [["other", false], [state.highlight, true]];
    groups.forEach(([name, strong]) => {
      const xs = [], ys = [], hover = [];
      scores.forEach((v, i) => {
        if (v === null) return;
        const inGroup = name === "all" || (name === "other" ? P.category[i] !== state.highlight : P.category[i] === name);
        if (!inGroup) return;
        xs.push(P.typicality[i]); ys.push(v);
        hover.push(`<b>${escapeHtml(P.item[i])}</b> (${escapeHtml(P.category[i])})<br>typicality ${P.typicality[i].toFixed(2)} · score ${v.toFixed(2)}<br>${escapeHtml(texts[i] || "")}`);
      });
      traces.push({
        type: "scattergl", mode: "markers", x: xs, y: ys, text: hover, hoverinfo: "text",
        marker: {
          size: strong ? 8 : 6,
          color: strong ? token("--series-1") : token("--text-muted"),
          opacity: strong ? 0.85 : 0.25,
          line: { color: token("--surface-1"), width: 1 },
        },
        showlegend: false,
      });
    });
    const layout = {
      paper_bgcolor: token("--surface-1"), plot_bgcolor: token("--surface-1"),
      font: { family: "system-ui, -apple-system, sans-serif", size: 11, color: token("--text-secondary") },
      margin: { l: 64, r: 16, t: 8, b: 52 },
      xaxis: { title: { text: "typicality (−rating z-score; higher is more typical)", font: { color: token("--text-secondary") } },
        gridcolor: token("--grid"), zerolinecolor: token("--border"), tickfont: { color: token("--text-muted") } },
      yaxis: { title: { text: yName, font: { color: token("--text-secondary") } },
        gridcolor: token("--grid"), zerolinecolor: token("--border"), tickfont: { color: token("--text-muted") } },
      hoverlabel: { bgcolor: token("--surface-1"), bordercolor: token("--border"), font: { color: token("--text-primary") } },
    };
    const metrics = DATA.gm[a][ti][l][g];
    const parts = AX.metric.map((m, mi) => `${DATA.labels.metric[m]} ${fmt(metrics[mi][e])}`);
    document.getElementById(`stats-${t}`).textContent = `In each category, mean: ${parts.join(" · ")}`;
    Plotly.react(`heat-${t}`, traces, layout, { responsive: true, displaylogo: false });
  });
}

function render() {
  document.querySelectorAll(".filters label").forEach(label => {
    label.hidden = !label.dataset.for.split(" ").includes(state.page);
  });
  const items = DATA.nItems[state.ambiguity];
  const total = Object.values(items).reduce((x, y) => x + y, 0);
  document.getElementById("context").innerHTML =
    `${total} items in ${AX.category.length} categories (${escapeHtml(DATA.labels.ambiguity[state.ambiguity])}) · ${AX.prompt.length} training prompt formats` +
    (state.page === "oc" && AX.eval_prompt.length ? ` · ${AX.eval_prompt.length} evaluation prompt formats: ` +
      AX.eval_prompt.map(p => `<code>${escapeHtml(DATA.promptFormats[p] || p)}</code>`).join(" ") : "");
  if (state.page === "gl") {
    document.getElementById("context").innerHTML =
      `${total} items (${escapeHtml(DATA.labels.ambiguity[state.ambiguity])}) · eval prompt: <code>${escapeHtml(DATA.promptFormats[state.evalPrompt] || state.evalPrompt)}</code>`;
  }
  document.getElementById("note").textContent = NOTES[state.page];
  if (state.page === "gl") drawScatter(); else draw();
}

function init() {
  const cvText = { fold_zscored: "scores z-scored in each held-out fold", pooled: "pooled out-of-fold scores", mean_over_folds: "the mean over held-out folds" };
  document.getElementById("cv-text").textContent = `${DATA.nFolds}-fold CV with ${cvText[DATA.cvAggregate] || DATA.cvAggregate}`;
  fillSelect("sel-layer", "layer", null, "layer");
  fillSelect("sel-ambiguity", "ambiguity", DATA.labels.ambiguity, "ambiguity");
  fillSelect("sel-groups", "n_groups", DATA.labels.n_groups, "groups");
  fillSelect("sel-metric", "metric", DATA.labels.metric, "metric");
  if (AX.eval_prompt.length) {
    fillSelect("sel-eval", "eval_prompt", null, "evalPrompt");
    const highlight = document.getElementById("sel-highlight");
    ["all"].concat(AX.category).forEach(c => {
      const option = document.createElement("option");
      option.value = c;
      option.textContent = c === "all" ? "All categories" : c;
      highlight.appendChild(option);
    });
    highlight.onchange = e => { state.highlight = e.target.value; render(); };
    document.getElementById("sel-scale").onchange = e => { state.scale = e.target.value; render(); };
  } else {
    document.getElementById("tab-gl").hidden = true;
  }
  buildPanels();
  document.querySelectorAll(".tab").forEach(tab => {
    tab.onclick = () => {
      state.page = tab.dataset.page;
      document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", String(t === tab)));
      render();
    };
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  render();
}
init();
</script>
</body>
</html>
"""


def render_html(page_data):
    data = json.dumps(page_data, separators=(",", ":")).replace("</", "<\\/")
    return (
        PAGE_TEMPLATE.replace("__STYLE__", STYLE)
        .replace("__PLOTLY_URL__", PLOTLY_URL)
        .replace("__TITLE__", html.escape(page_data["title"]))
        .replace("__DATA__", data)
    )


def write_html(path, **page_arguments):
    page_data = build_page_data(**page_arguments)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(page_data))
    return path
