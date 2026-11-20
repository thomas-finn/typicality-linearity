"""Write the four-page HTML for the global direction probes.

One row of selectors at the top is shared by all pages. Each page shows only
the selectors it uses. Each page is a 2x2 grid of heatmaps, one per
activation type.

- Page 1, "Transfer between categories": rows train category, columns eval
  category, for the selected prompt format. The diagonal is cross-validated,
  or with diagonal "full" the probe on all items of the category (in-sample).
- Page 2, "Cosine between categories": cosine of the category directions, for
  the selected prompt format.
- Page 3, "Transfer between prompts": rows train prompt, columns eval prompt,
  for the selected category. Off-diagonal cells are cross-validated. The
  diagonal is cross-validated, or in-sample with diagonal "full".
- Page 4, "Cosine between prompts": cosine of the prompt directions, for the
  selected category.

Pages 2 and 4 also have a 2x2 grid of dendrograms under the heatmaps, one
per activation type. Each is average linkage (UPGMA) on the distance
1 - cosine of the matrix above it. The heatmap above each dendrogram has its
rows and columns in the same leaf order, so the order changes with the
selectors.

On pages 3 and 4, the selected prompt format has an outline (its row and its
column). Click a cell on any page to select its prompt or category.

Colour: diverging, grey at chance (0 for correlations and cosines, 0.5 for the
AUC), blue above, red below. The range is the same for all settings of a page
and metric, so the colours can be compared when a selector changes.
"""

import html
import json
import math

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform

PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def to_json_list(array, digits=3):
    """Return nested lists, rounded. NaN becomes None (null in JSON)."""
    rounded = np.round(np.asarray(array, dtype=np.float64), digits)
    flat = [None if math.isnan(v) else float(v) for v in rounded.ravel()]
    return np.array(flat, dtype=object).reshape(rounded.shape).tolist()


def off_diagonal(array):
    n = array.shape[-1]
    return array[..., ~np.eye(n, dtype=bool)]


def color_range(values, centre):
    """Half width of the colour scale: the largest distance from the centre,
    rounded up to 0.05."""
    distance = np.nanmax(np.abs(np.asarray(values) - centre))
    if not np.isfinite(distance) or distance == 0:
        return 1.0
    return float(math.ceil(distance / 0.05) * 0.05)


def cosine_linkage(cosine):
    """Average linkage on the distance 1 - cosine.

    Return (merges, cophenetic correlation). Each merge is [left, right,
    height] with scipy numbering: leaves 0..n-1, merge k makes node n+k.
    Return (None, None) if the matrix has a NaN.
    """
    cosine = np.asarray(cosine, dtype=np.float64)
    if np.isnan(cosine).any():
        return None, None
    distance = 1.0 - (cosine + cosine.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(np.clip(distance, 0.0, None), checks=False)
    tree = linkage(condensed, method="average")
    if np.ptp(condensed) == 0:
        correlation = None
    else:
        correlation = round(float(cophenet(tree, condensed)[0]), 3)
    merges = [[int(a), int(b), round(float(h), 4)] for a, b, h, _ in tree]
    return merges, correlation


def linkage_lists(cosines):
    """Apply cosine_linkage to each matrix in the last two axes.

    Return nested lists of merges and of cophenetic correlations, and the
    largest merge height.
    """
    cosines = np.asarray(cosines)
    lead_shape = cosines.shape[:-2]
    merges = np.empty(lead_shape, dtype=object)
    correlations = np.empty(lead_shape, dtype=object)
    largest = 0.0
    for index in np.ndindex(*lead_shape):
        tree, correlation = cosine_linkage(cosines[index])
        merges[index] = tree
        correlations[index] = correlation
        if tree is not None:
            largest = max(largest, max(h for _, _, h in tree))
    return merges.tolist(), correlations.tolist(), largest


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
    diagonal="cv",
):
    metrics = axes["metric"]
    transfer_ranges = {}
    for mi, metric in enumerate(metrics):
        centre = metric_chance[metric]
        values = np.concatenate(
            [
                arrays["category_transfer"][:, :, :, :, mi].ravel(),
                arrays["prompt_transfer"][:, :, :, :, mi].ravel(),
            ]
        )
        transfer_ranges[metric] = color_range(values, centre)
    cc_tree, cc_cophenetic, cc_largest = linkage_lists(arrays["category_cosine"])
    pc_tree, pc_cophenetic, pc_largest = linkage_lists(arrays["prompt_cosine"])
    cosine_values = np.concatenate(
        [
            off_diagonal(arrays["category_cosine"]).ravel(),
            off_diagonal(arrays["prompt_cosine"]).ravel(),
        ]
    )
    return {
        "title": title,
        "axes": {key: [str(v) for v in values] for key, values in axes.items()},
        "labels": labels,
        "chance": metric_chance,
        "transferRange": transfer_ranges,
        "cosineRange": color_range(cosine_values, 0.0),
        "nItems": n_items,
        "promptFormats": prompt_formats,
        "cvAggregate": cv_aggregate,
        "nFolds": n_folds,
        "diagonal": diagonal,
        "ct": to_json_list(arrays["category_transfer"]),
        "cc": to_json_list(arrays["category_cosine"]),
        "pt": to_json_list(arrays["prompt_transfer"]),
        "pc": to_json_list(arrays["prompt_cosine"]),
        "ccTree": cc_tree,
        "ccCophenetic": cc_cophenetic,
        "pcTree": pc_tree,
        "pcCophenetic": pc_cophenetic,
        "treeMax": {"cc": cc_largest, "pc": pc_largest},
    }


STYLE = """:root {
  color-scheme: light;
  --surface-0: #f6f5f2;
  --surface-1: #fcfcfb;
  --border: #e2e1dc;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #7a7974;
  --div-low: #e34948;
  --div-mid: #f0efec;
  --div-high: #2a78d6;
  --series-1: #2a78d6;
  --grid: #ebeae6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --border: #33332f;
    --text-primary: #f0efec;
    --text-secondary: #c3c2b7;
    --text-muted: #8f8e86;
    --div-low: #e66767;
    --div-mid: #383835;
    --div-high: #3987e5;
    --series-1: #3987e5;
    --grid: #2a2a27;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --border: #33332f;
  --text-primary: #f0efec;
  --text-secondary: #c3c2b7;
  --text-muted: #8f8e86;
  --div-low: #e66767;
  --div-mid: #383835;
  --div-high: #3987e5;
  --series-1: #3987e5;
  --grid: #2a2a27;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 16px; background: var(--surface-0); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14px;
}
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.subtitle { color: var(--text-secondary); margin: 0 0 16px; max-width: 900px; }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
.tab {
  background: none; border: none; border-bottom: 2px solid transparent; color: var(--text-secondary);
  font: inherit; padding: 8px 12px; cursor: pointer;
}
.tab[aria-selected="true"] { color: var(--text-primary); border-bottom-color: var(--text-primary); font-weight: 600; }
.filters { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 8px; }
.filters label { display: flex; flex-direction: column; gap: 4px; color: var(--text-secondary); font-size: 12px; }
.filters select {
  font: inherit; font-size: 14px; color: var(--text-primary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; min-width: 110px;
}
.context { color: var(--text-secondary); margin: 0 0 12px; }
.context code { color: var(--text-primary); background: var(--surface-1); padding: 2px 6px; border-radius: 4px; }
.grid { display: grid; grid-template-columns: repeat(var(--columns, 2), minmax(0, 1fr)); gap: 12px; }
@media (max-width: 900px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0; }
.panel h2 { font-size: 15px; font-weight: 600; margin: 0; }
.panel .formula { color: var(--text-secondary); font-size: 12px; margin: 2px 0 0; }
.panel .stats { color: var(--text-muted); font-size: 12px; margin: 2px 0 4px; }
.heat { width: 100%; height: 440px; }
.tree { width: 100%; height: 340px; }
.section { font-size: 16px; font-weight: 600; margin: 20px 0 4px; }
.section-note { color: var(--text-secondary); margin: 0 0 12px; max-width: 900px; }
.note { color: var(--text-muted); font-size: 12px; margin: 12px 0 0; max-width: 900px; }
[hidden] { display: none !important; }
"""


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
<p class="subtitle">Difference-of-means probes: the mean activation of the most typical items minus the mean of the least typical items. Cross-validated cells use <span id="cv-text"></span>. Each probe is trained on the activation type of its panel; all scores are computed on the raw activations.</p>

<div class="tabs" role="tablist">
  <button class="tab" role="tab" aria-selected="true" data-page="ct">Transfer between categories</button>
  <button class="tab" role="tab" aria-selected="false" data-page="cc">Cosine between categories</button>
  <button class="tab" role="tab" aria-selected="false" data-page="pt">Transfer between prompts</button>
  <button class="tab" role="tab" aria-selected="false" data-page="pc">Cosine between prompts</button>
</div>

<div class="filters">
  <label data-for="pt pc">Category <select id="sel-category"></select></label>
  <label data-for="ct cc pt pc">Layer <select id="sel-layer"></select></label>
  <label data-for="ct cc pt pc">Ambiguous items <select id="sel-ambiguity"></select></label>
  <label data-for="ct cc pt pc">Difference of means <select id="sel-groups"></select></label>
  <label data-for="ct cc pt pc">Prompt format <select id="sel-prompt"></select></label>
  <label data-for="ct pt">Metric <select id="sel-metric"></select></label>
</div>
<p class="context" id="context"></p>

<div class="grid" id="grid"></div>
<div id="trees" hidden>
  <h2 class="section">Dendrograms of the probe directions</h2>
  <p class="section-note">Average linkage (UPGMA) on the distance 1 &minus; cosine of the heatmap above. The heatmap above each tree uses its leaf order. Merge height = 1 &minus; mean cosine between the two clusters; the height axis is the same for all settings of this page. Cophenetic correlation: how well the tree heights match the pair distances (1 = exact). Hover a merge to see its members. Click a leaf to select it.</p>
  <div class="grid" id="tree-grid"></div>
</div>
<p class="note" id="note"></p>

<script id="page-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("page-data").textContent);
const AX = DATA.axes;
const state = {
  page: "ct",
  category: AX.category[0],
  layer: AX.layer.includes("15") ? "15" : AX.layer[0],
  ambiguity: AX.ambiguity[0],
  groups: AX.n_groups.includes("3") ? "3" : AX.n_groups[0],
  prompt: AX.prompt[0],
  metric: AX.metric[0],
};
const DIAG_FULL = DATA.diagonal === "full";
const DIAG_LABEL = DIAG_FULL ? "in-sample" : "CV";
const NOTES = {
  ct: "Rows: the category the probe is trained on. Columns: the category it is evaluated on, in the selected prompt format. Off-diagonal cells use the probe trained on all items of the train category. " + (DIAG_FULL ? "Diagonal cells use the probe trained on all items of the category, evaluated on the same items (in-sample, not cross-validated)." : "Diagonal cells are cross-validated in the category.") + " Click a cell to select its train category.",
  cc: "Cosine similarity of the probe directions (trained on all items of each category) in the selected prompt format. The diagonal is 1. Rows and columns are in the leaf order of the dendrogram below, so the order changes with the selectors. Click a cell to select its row category.",
  pt: "Rows: the prompt format the probe is trained on. Columns: the prompt format it is evaluated on, for the selected category. " + (DIAG_FULL ? "Off-diagonal cells are cross-validated with the same folds, so a probe is never evaluated on its training items. Diagonal cells use the probe trained on all items, evaluated on the same items (in-sample, not cross-validated)." : "All cells are cross-validated with the same folds, so a probe is never evaluated on its training items.") + " The outline marks the selected prompt format. Click a cell to select its train prompt.",
  pc: "Cosine similarity of the probe directions (trained on all items of the selected category) between prompt formats. The diagonal is 1. Rows and columns are in the leaf order of the dendrogram below, so the order changes with the selectors. The outline marks the selected prompt format. Click a cell to select its row prompt.",
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

function meanOf(matrix, diagonal) {
  let sum = 0, count = 0;
  matrix.forEach((row, i) => row.forEach((v, j) => {
    if (v === null || (i === j) !== diagonal) return;
    sum += v; count += 1;
  }));
  return count ? sum / count : null;
}

function matrixFor(typeIndex) {
  const a = idx("ambiguity", state.ambiguity), l = idx("layer", state.layer), g = idx("n_groups", state.groups);
  const base = DATA[state.page][a][typeIndex][l][g];
  const m = idx("metric", state.metric), p = idx("prompt", state.prompt), c = idx("category", state.category);
  if (state.page === "ct") return base[m][p];
  if (state.page === "cc") return base[p];
  if (state.page === "pt") return base[m][c];
  return base[c];
}

// Four types: a 2x2 grid. Two or three types: one row, side by side.
function gridColumns() {
  const n = AX.activation_type.length;
  return n === 4 ? 2 : Math.min(n, 3);
}

function buildPanels() {
  document.documentElement.style.setProperty("--columns", gridColumns());
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
  const treeGrid = document.getElementById("tree-grid");
  treeGrid.innerHTML = "";
  AX.activation_type.forEach(t => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h2>${escapeHtml(DATA.labels.activation_type[t])}</h2>` +
      `<p class="stats" id="tree-stats-${t}"></p><div class="tree" id="tree-${t}"></div>`;
    treeGrid.appendChild(panel);
  });
}

function treeFor(typeIndex) {
  const a = idx("ambiguity", state.ambiguity), l = idx("layer", state.layer), g = idx("n_groups", state.groups);
  const key = state.page === "cc" ? "cc" : "pc";
  const k = state.page === "cc" ? idx("prompt", state.prompt) : idx("category", state.category);
  return {
    merges: DATA[key + "Tree"][a][typeIndex][l][g][k],
    cophenetic: DATA[key + "Cophenetic"][a][typeIndex][l][g][k],
  };
}

// Leaf order and line segments of a dendrogram from scipy linkage merges.
function dendrogram(merges, n) {
  const children = {};
  merges.forEach((m, k) => { children[n + k] = m; });
  const order = [];
  function visit(node) {
    if (node < n) { order.push(node); return; }
    visit(children[node][0]);
    visit(children[node][1]);
  }
  visit(n + merges.length - 1);
  const x = {}, y = {}, members = {};
  order.forEach((leaf, i) => { x[leaf] = i; y[leaf] = 0; members[leaf] = [leaf]; });
  const lineX = [], lineY = [], tops = [];
  merges.forEach(([left, right, height], k) => {
    const node = n + k;
    x[node] = (x[left] + x[right]) / 2;
    y[node] = height;
    members[node] = members[left].concat(members[right]);
    lineX.push(x[left], x[left], x[right], x[right], null);
    lineY.push(y[left], height, height, y[right], null);
    tops.push({ x: x[node], y: height, members: members[node] });
  });
  return { order, lineX, lineY, tops };
}

function drawTrees() {
  const names = state.page === "cc" ? AX.category : AX.prompt;
  const selected = state.page === "cc" ? state.category : state.prompt;
  const yMax = Math.max(DATA.treeMax[state.page] * 1.08, 0.05);
  AX.activation_type.forEach((t, ti) => {
    const { merges, cophenetic } = treeFor(ti);
    const div = document.getElementById(`tree-${t}`);
    const stats = document.getElementById(`tree-stats-${t}`);
    if (!merges) {
      stats.textContent = "No tree: a direction is zero (cosine undefined).";
      Plotly.purge(div);
      return;
    }
    const tree = dendrogram(merges, names.length);
    const leafNames = tree.order.map(i => names[i]);
    const root = merges[merges.length - 1][2];
    stats.textContent = `Cophenetic correlation ${fmt(cophenetic)} · root height ${fmt(root)} (mean cosine ${fmt(1 - root)})`;
    const lines = {
      type: "scatter", mode: "lines", x: tree.lineX, y: tree.lineY,
      line: { color: token("--text-secondary"), width: 1.5 }, hoverinfo: "skip",
    };
    const tops = {
      type: "scatter", mode: "markers", x: tree.tops.map(p => p.x), y: tree.tops.map(p => p.y),
      marker: { size: 8, color: token("--series-1"), line: { color: token("--surface-1"), width: 2 } },
      text: tree.tops.map(p => p.members.map(i => names[i]).join(", ")),
      customdata: tree.tops.map(p => [1 - p.y, p.members.length]),
      hovertemplate: "<b>%{customdata[1]} members</b>: %{text}<br>1 − cosine %{y:.3f} (mean cosine %{customdata[0]:.3f})<extra></extra>",
    };
    const leaves = {
      type: "scatter", mode: "markers", x: leafNames.map((_, i) => i), y: leafNames.map(() => 0),
      marker: {
        size: leafNames.map(name => name === selected ? 12 : 8),
        color: leafNames.map(name => name === selected ? token("--text-primary") : token("--surface-1")),
        line: { color: token("--text-primary"), width: 1.5 },
      },
      text: leafNames, hovertemplate: "%{text}<extra></extra>",
    };
    const longLabels = state.page === "pc";
    const layout = {
      paper_bgcolor: token("--surface-1"), plot_bgcolor: token("--surface-1"),
      font: { family: "system-ui, -apple-system, sans-serif", size: 11, color: token("--text-secondary") },
      margin: { l: 56, r: 16, t: 8, b: longLabels ? 120 : 80 },
      showlegend: false,
      xaxis: { tickvals: leafNames.map((_, i) => i), ticktext: leafNames, tickangle: -45,
        range: [-0.6, leafNames.length - 0.4], showgrid: false, zeroline: false,
        tickfont: { color: token("--text-muted") } },
      yaxis: { range: [-0.03 * yMax, yMax], gridcolor: token("--grid"), zeroline: false,
        tickfont: { color: token("--text-muted") },
        title: { text: "1 − cosine", font: { color: token("--text-secondary") } } },
      hoverlabel: { bgcolor: token("--surface-1"), bordercolor: token("--border"), font: { color: token("--text-primary") } },
    };
    Plotly.react(div, [lines, tops, leaves], layout, { responsive: true, displaylogo: false });
    if (!div.dataset.clickBound) {
      div.on("plotly_click", event => {
        const point = event.points[0];
        if (point.curveNumber !== 2) return;
        if (state.page === "cc") state.category = point.text; else state.prompt = point.text;
        syncSelects();
        render();
      });
      div.dataset.clickBound = "1";
    }
  });
}

function draw() {
  const isTransfer = state.page === "ct" || state.page === "pt";
  const byCategory = state.page === "ct" || state.page === "cc";
  const names = byCategory ? AX.category : AX.prompt;
  const centre = isTransfer ? DATA.chance[state.metric] : 0;
  const half = isTransfer ? DATA.transferRange[state.metric] : DATA.cosineRange;
  const colorscale = [[0, token("--div-low")], [0.5, token("--div-mid")], [1, token("--div-high")]];
  const valueName = isTransfer ? DATA.labels.metric[state.metric] : "cosine";
  const rowName = byCategory ? "train category" : "train prompt";
  const colName = byCategory ? "eval category" : "eval prompt";
  AX.activation_type.forEach((t, ti) => {
    const z = matrixFor(ti);
    // Cosine pages: rows and columns in the leaf order of the dendrogram below.
    let order = names.map((_, i) => i);
    if (!isTransfer) {
      const { merges } = treeFor(ti);
      if (merges) order = dendrogram(merges, names.length).order;
    }
    const labels = order.map(i => names[i]);
    const zOrdered = order.map(i => order.map(j => z[i][j]));
    const trace = {
      type: "heatmap", x: labels, y: labels, z: zOrdered, zmin: centre - half, zmax: centre + half, zmid: centre,
      // The value of each cell as text; Plotly picks a text colour with contrast to the cell.
      texttemplate: "%{z:.2f}", textfont: { size: 8 },
      colorscale: colorscale, xgap: 2, ygap: 2, hoverongaps: false,
      hovertemplate: `${rowName}: %{y}<br>${colName}: %{x}<br><b>${valueName} %{z:.3f}</b><extra></extra>`,
      showscale: ti === gridColumns() - 1,
      colorbar: ti === gridColumns() - 1 ? {
        title: { text: valueName, font: { color: token("--text-secondary") } },
        tickfont: { color: token("--text-muted") }, thickness: 10, len: 0.9, outlinewidth: 0,
      } : undefined,
    };
    const longLabels = !byCategory;
    const layout = {
      paper_bgcolor: token("--surface-1"), plot_bgcolor: token("--surface-1"),
      font: { family: "system-ui, -apple-system, sans-serif", size: 11, color: token("--text-secondary") },
      margin: { l: longLabels ? 120 : 80, r: 16, t: 8, b: longLabels ? 120 : 80 },
      xaxis: { tickangle: -45, showgrid: false, tickfont: { color: token("--text-muted") },
        title: { text: colName, font: { color: token("--text-secondary") } } },
      yaxis: { autorange: "reversed", showgrid: false, tickfont: { color: token("--text-muted") },
        title: { text: rowName, font: { color: token("--text-secondary") } } },
      hoverlabel: { bgcolor: token("--surface-1"), bordercolor: token("--border"), font: { color: token("--text-primary") } },
      shapes: [],
    };
    if (!byCategory) {
      const k = labels.indexOf(state.prompt);
      const n = labels.length;
      const line = { color: token("--text-primary"), width: 2 };
      layout.shapes.push({ type: "rect", x0: -0.5, x1: n - 0.5, y0: k - 0.5, y1: k + 0.5, line: line });
      layout.shapes.push({ type: "rect", x0: k - 0.5, x1: k + 0.5, y0: -0.5, y1: n - 0.5, line: line });
    }
    let stats;
    if (isTransfer) {
      stats = `Diagonal (${DIAG_LABEL}) mean ${fmt(meanOf(z, true))} · off-diagonal mean ${fmt(meanOf(z, false))}`;
    } else {
      stats = `Off-diagonal mean cosine ${fmt(meanOf(z, false))}`;
    }
    document.getElementById(`stats-${t}`).textContent = stats;
    const div = document.getElementById(`heat-${t}`);
    Plotly.react(div, [trace], layout, { responsive: true, displaylogo: false });
    if (!div.dataset.clickBound) {
      div.on("plotly_click", event => {
        const row = event.points[0].y;
        if (state.page === "ct" || state.page === "cc") state.category = row; else state.prompt = row;
        syncSelects();
        render();
      });
      div.dataset.clickBound = "1";
    }
  });
}

function syncSelects() {
  document.getElementById("sel-category").value = state.category;
  document.getElementById("sel-prompt").value = state.prompt;
}

function render() {
  document.querySelectorAll(".filters label").forEach(label => {
    label.hidden = !label.dataset.for.split(" ").includes(state.page);
  });
  const items = DATA.nItems[state.ambiguity];
  const total = Object.values(items).reduce((a, b) => a + b, 0);
  let context;
  if (state.page === "ct" || state.page === "cc") {
    context = `Prompt format: <code>${escapeHtml(DATA.promptFormats[state.prompt] || state.prompt)}</code> · ${total} items (${escapeHtml(DATA.labels.ambiguity[state.ambiguity])})`;
  } else {
    context = `Category: <code>${escapeHtml(state.category)}</code> · ${items[state.category]} items (${escapeHtml(DATA.labels.ambiguity[state.ambiguity])}) · selected prompt: <code>${escapeHtml(DATA.promptFormats[state.prompt] || state.prompt)}</code>`;
  }
  document.getElementById("context").innerHTML = context;
  document.getElementById("note").textContent = NOTES[state.page];
  draw();
  const showTrees = state.page === "cc" || state.page === "pc";
  document.getElementById("trees").hidden = !showTrees;
  if (showTrees) drawTrees();
}

function init() {
  const cvText = { fold_zscored: "scores z-scored in each held-out fold", pooled: "pooled out-of-fold scores", mean_over_folds: "the mean over held-out folds" };
  document.getElementById("cv-text").textContent = `${DATA.nFolds}-fold CV with ${cvText[DATA.cvAggregate] || DATA.cvAggregate}` + (DIAG_FULL ? ". Diagonal cells are in-sample (the probe trained on all items), not cross-validated" : "");
  fillSelect("sel-category", "category", null, "category");
  fillSelect("sel-layer", "layer", null, "layer");
  fillSelect("sel-ambiguity", "ambiguity", DATA.labels.ambiguity, "ambiguity");
  fillSelect("sel-groups", "n_groups", DATA.labels.n_groups, "groups");
  fillSelect("sel-prompt", "prompt", null, "prompt");
  fillSelect("sel-metric", "metric", DATA.labels.metric, "metric");
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
