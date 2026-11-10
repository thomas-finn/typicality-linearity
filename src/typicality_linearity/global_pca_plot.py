"""Write the two-page HTML for the global direction PCA.

One row of selectors at the top (layer, prompt format, category, number of
components) is shared by the two pages.

- Page 1, "PCA projections": a 2x2 grid, one panel per activation type. Each
  point is one rated pair. Colour is the rating z-score on a diverging scale:
  red is low (typical), grey is 0, blue is high (atypical). "All categories"
  shows the PCA fitted on all points. One category shows the PCA fitted only
  on that category.
- Page 2, "Dimensions for 95% variance": a 2x2 grid of heatmaps, one per
  activation type, for the selected layer. Rows are the scopes (all and each
  category), columns are the prompt formats. The cell is the number of
  components for the variance threshold. The colour is on a log scale with
  the same range for all panels and layers. The selected category and prompt
  have an outline. Click a cell to select it. Below, a line chart shows the
  number for the selected category and prompt over the layers.

Colours follow the light or dark setting of the browser.
"""

import html
import json

PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def round_list(values, digits):
    return [round(float(value), digits) for value in values]


def fits_to_json(fits, activation_types, layers, prompts, scopes):
    """Return {type: {layer: {prompt: {scope: {...}}}}} with short keys.

    p: projections, one list per component. r: explained ratios.
    k: components for the threshold. n: number of points.
    """
    result = {}
    for activation_type in activation_types:
        result[activation_type] = {}
        for layer in layers:
            by_prompt = {}
            for prompt in prompts:
                by_scope = {}
                for scope in scopes:
                    fit = fits[prompt][activation_type][layer][scope]
                    projections = fit["projections"]
                    by_scope[scope] = {
                        "p": [
                            round_list(projections[:, c], 3)
                            for c in range(projections.shape[1])
                        ],
                        "r": round_list(fit["explained_ratio"], 4),
                        "k": int(fit["n_components_threshold"]),
                        "n": int(fit["n_points"]),
                    }
                by_prompt[prompt] = by_scope
            result[activation_type][str(layer)] = by_prompt
    return result


def build_page_data(
    title,
    layers,
    prompts,
    prompt_formats,
    activation_types,
    activation_labels,
    activation_formulas,
    scopes,
    points,
    prompt_texts,
    fits,
    variance_threshold,
):
    return {
        "title": title,
        "layers": [str(layer) for layer in layers],
        "prompts": list(prompts),
        "promptFormats": prompt_formats,
        "types": [
            {
                "key": t,
                "label": activation_labels[t],
                "formula": activation_formulas[t],
            }
            for t in activation_types
        ],
        "scopes": list(scopes.keys()),
        "scopeRows": scopes,
        "points": points,
        "promptTexts": prompt_texts,
        "fits": fits_to_json(fits, activation_types, layers, prompts, scopes),
        "threshold": variance_threshold,
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
  --surface-0: #f6f5f2;
  --surface-1: #fcfcfb;
  --border: #e2e1dc;
  --grid: #ebeae6;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #7a7974;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --div-low: #e34948;
  --div-mid: #f0efec;
  --div-high: #2a78d6;
  --seq-0: #cde2fb;
  --seq-1: #86b6ef;
  --seq-2: #3987e5;
  --seq-3: #1c5cab;
  --seq-4: #0d366b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --border: #33332f;
    --grid: #2a2a27;
    --text-primary: #f0efec;
    --text-secondary: #c3c2b7;
    --text-muted: #8f8e86;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --div-low: #e66767;
    --div-mid: #383835;
    --div-high: #3987e5;
    --seq-0: #104281;
    --seq-1: #1c5cab;
    --seq-2: #3987e5;
    --seq-3: #86b6ef;
    --seq-4: #cde2fb;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --border: #33332f;
  --grid: #2a2a27;
  --text-primary: #f0efec;
  --text-secondary: #c3c2b7;
  --text-muted: #8f8e86;
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --series-4: #c98500;
  --div-low: #e66767;
  --div-mid: #383835;
  --div-high: #3987e5;
  --seq-0: #104281;
  --seq-1: #1c5cab;
  --seq-2: #3987e5;
  --seq-3: #86b6ef;
  --seq-4: #cde2fb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 16px;
  background: var(--surface-0);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px;
}
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.subtitle { color: var(--text-secondary); margin: 0 0 16px; }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
.tab {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: var(--text-secondary); font: inherit; padding: 8px 12px; cursor: pointer;
}
.tab[aria-selected="true"] { color: var(--text-primary); border-bottom-color: var(--text-primary); font-weight: 600; }
.filters { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 8px; }
.filters label { display: flex; flex-direction: column; gap: 4px; color: var(--text-secondary); font-size: 12px; }
.filters select {
  font: inherit; font-size: 14px; color: var(--text-primary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; min-width: 120px;
}
.format { color: var(--text-secondary); margin: 0 0 12px; }
.format code { color: var(--text-primary); background: var(--surface-1); padding: 2px 6px; border-radius: 4px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 800px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0; }
.panel h2 { font-size: 15px; font-weight: 600; margin: 0; }
.panel .formula { color: var(--text-secondary); font-size: 12px; margin: 2px 0 0; }
.panel .stats { color: var(--text-muted); font-size: 12px; margin: 2px 0 4px; }
.plot { width: 100%; height: 420px; }
.heat { width: 100%; height: 380px; }
.wide { margin-top: 12px; }
.line { width: 100%; height: 320px; }
.note { color: var(--text-muted); font-size: 12px; margin: 12px 0 0; max-width: 900px; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="subtitle">Each point is one rated item-category pair. Colour: rating z-score (red = typical, blue = atypical).</p>

<div class="tabs" role="tablist">
  <button class="tab" role="tab" id="tab-proj" aria-selected="true" data-page="page-proj">PCA projections</button>
  <button class="tab" role="tab" id="tab-dim" aria-selected="false" data-page="page-dim">Dimensions for <span id="thr-label"></span> variance</button>
</div>

<div class="filters">
  <label>Layer <select id="sel-layer"></select></label>
  <label>Prompt format <select id="sel-prompt"></select></label>
  <label>Category <select id="sel-scope"></select></label>
  <label id="lbl-comp">Components <select id="sel-comp"><option value="2">2</option><option value="3">3</option></select></label>
</div>
<p class="format" id="format"></p>

<section id="page-proj" role="tabpanel">
  <div class="grid" id="grid-proj"></div>
  <p class="note">"All categories" shows the PCA fitted on all points. One category shows the PCA fitted only on that category. Axis labels give the explained variance of each component. Move the pointer over a point to see its prompt.</p>
</section>

<section id="page-dim" role="tabpanel" hidden>
  <div class="grid" id="grid-dim"></div>
  <div class="panel wide">
    <h2 id="line-title"></h2>
    <p class="formula">Number of components for the threshold over layers, for the selected category and prompt format.</p>
    <div class="line" id="line"></div>
  </div>
  <p class="note">Cell: the smallest number of principal components that explain the threshold of the variance, for the selected layer. The colour scale is logarithmic and the same for all panels and layers. A scope with n points has at most n − 1 components, so a category (about 30 to 80 points) has a smaller maximum than all categories (565 points). The outline marks the selected category and prompt format. Click a cell to select it.</p>
</section>

<script id="page-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("page-data").textContent);
const SYMBOLS = ["circle", "square", "diamond", "triangle-up"];
const state = { layer: DATA.layers[1] || DATA.layers[0], prompt: DATA.prompts[0], scope: "all", comp: 2, page: "page-proj" };

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function scopeLabel(scope) { return scope === "all" ? "All categories" : scope; }
function pct(x) { return (100 * x).toFixed(1) + "%"; }
function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function fillSelect(id, values, labels, selected) {
  const select = document.getElementById(id);
  select.innerHTML = "";
  values.forEach((value, i) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels ? labels[i] : value;
    if (value === selected) option.selected = true;
    select.appendChild(option);
  });
}

function baseLayout() {
  return {
    paper_bgcolor: token("--surface-1"),
    plot_bgcolor: token("--surface-1"),
    font: { family: "system-ui, -apple-system, sans-serif", size: 12, color: token("--text-secondary") },
    margin: { l: 56, r: 16, t: 8, b: 48 },
    hoverlabel: { bgcolor: token("--surface-1"), bordercolor: token("--border"), font: { color: token("--text-primary") } },
  };
}
function axis(title) {
  return {
    title: { text: title, font: { color: token("--text-secondary") } },
    gridcolor: token("--grid"), zerolinecolor: token("--border"), linecolor: token("--border"),
    tickfont: { color: token("--text-muted") },
  };
}

const zMax = Math.max(...DATA.points.rating_zscore.map(Math.abs));

function buildPanels(gridId, kind) {
  const grid = document.getElementById(gridId);
  grid.innerHTML = "";
  DATA.types.forEach(t => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h2>${escapeHtml(t.label)}</h2><p class="formula">${escapeHtml(t.formula)}</p>` +
      `<p class="stats" id="${kind}-stats-${t.key}"></p><div class="${kind === "proj" ? "plot" : "heat"}" id="${kind}-${t.key}"></div>`;
    grid.appendChild(panel);
  });
}

function drawProjections() {
  const rows = DATA.scopeRows[state.scope];
  const texts = DATA.promptTexts[state.prompt];
  const hover = rows.map(i =>
    `<b>${escapeHtml(DATA.points.item[i])}</b> (${escapeHtml(DATA.points.category[i])})<br>` +
    `z-score ${DATA.points.rating_zscore[i].toFixed(2)}<br>${escapeHtml(texts[i])}`);
  const colors = rows.map(i => DATA.points.rating_zscore[i]);
  const colorscale = [[0, token("--div-low")], [0.5, token("--div-mid")], [1, token("--div-high")]];
  DATA.types.forEach((t, index) => {
    const fit = DATA.fits[t.key][state.layer][state.prompt][state.scope];
    const showBar = index === 1;
    const marker = {
      color: colors, colorscale: colorscale, cmin: -zMax, cmax: zMax, cmid: 0,
      line: { color: token("--surface-1"), width: 1 },
      showscale: showBar,
      colorbar: showBar ? {
        title: { text: "z-score", font: { color: token("--text-secondary") } },
        tickfont: { color: token("--text-muted") }, thickness: 10, len: 0.8, outlinewidth: 0,
      } : undefined,
    };
    const ratio = fit.r;
    let trace, layout = baseLayout();
    if (state.comp === 3) {
      marker.size = 3.5;
      trace = { type: "scatter3d", mode: "markers", x: fit.p[0], y: fit.p[1], z: fit.p[2],
        text: hover, hoverinfo: "text", marker: marker };
      layout.margin = { l: 0, r: 0, t: 0, b: 0 };
      layout.scene = {
        xaxis: Object.assign(axis(`PC1 (${pct(ratio[0])})`), { backgroundcolor: token("--surface-1") }),
        yaxis: Object.assign(axis(`PC2 (${pct(ratio[1])})`), { backgroundcolor: token("--surface-1") }),
        zaxis: Object.assign(axis(`PC3 (${pct(ratio[2])})`), { backgroundcolor: token("--surface-1") }),
      };
    } else {
      marker.size = 8;
      trace = { type: "scattergl", mode: "markers", x: fit.p[0], y: fit.p[1],
        text: hover, hoverinfo: "text", marker: marker };
      layout.xaxis = axis(`PC1 (${pct(ratio[0])})`);
      layout.yaxis = axis(`PC2 (${pct(ratio[1])})`);
    }
    document.getElementById(`proj-stats-${t.key}`).textContent =
      `${fit.n} points · ${fit.k} components for ${pct(DATA.threshold)} variance`;
    Plotly.react(`proj-${t.key}`, [trace], layout, { responsive: true, displaylogo: false });
  });
}

// Range of the log colour scale, the same for all panels and layers.
let kMax = 1;
DATA.types.forEach(t => DATA.layers.forEach(l => DATA.prompts.forEach(p => DATA.scopes.forEach(s => {
  kMax = Math.max(kMax, DATA.fits[t.key][l][p][s].k);
}))));
const LOG_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter(v => v <= kMax * 1.5);

function drawHeatmaps() {
  const colorscale = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4"].map((n, i) => [i / 4, token(n)]);
  const yLabels = DATA.scopes.map(scopeLabel);
  DATA.types.forEach((t, index) => {
    const k = DATA.scopes.map(s => DATA.prompts.map(p => DATA.fits[t.key][state.layer][p][s].k));
    const n = DATA.scopes.map(s => DATA.prompts.map(p => DATA.fits[t.key][state.layer][p][s].n));
    const custom = DATA.scopes.map((s, i) => DATA.prompts.map((p, j) => [k[i][j], n[i][j], s, p]));
    const showBar = index === 1;
    const trace = {
      type: "heatmap", x: DATA.prompts, y: yLabels, z: k.map(r => r.map(v => Math.log10(v))),
      customdata: custom, zmin: 0, zmax: Math.log10(kMax), colorscale: colorscale, xgap: 2, ygap: 2,
      hovertemplate: "%{y} · %{x}<br><b>%{customdata[0]}</b> components (%{customdata[1]} points)<extra></extra>",
      showscale: showBar,
      colorbar: showBar ? {
        title: { text: "components", font: { color: token("--text-secondary") } },
        tickvals: LOG_TICKS.map(v => Math.log10(v)), ticktext: LOG_TICKS.map(String),
        tickfont: { color: token("--text-muted") }, thickness: 10, len: 0.9, outlinewidth: 0,
      } : undefined,
    };
    const layout = baseLayout();
    layout.margin = { l: 96, r: 16, t: 8, b: 110 };
    layout.xaxis = Object.assign(axis(""), { tickangle: -45, showgrid: false });
    layout.yaxis = Object.assign(axis(""), { autorange: "reversed", showgrid: false });
    const col = DATA.prompts.indexOf(state.prompt);
    const row = DATA.scopes.indexOf(state.scope);
    layout.shapes = [{
      type: "rect", xref: "x", yref: "y", x0: col - 0.5, x1: col + 0.5, y0: row - 0.5, y1: row + 0.5,
      line: { color: token("--text-primary"), width: 2 },
    }];
    const selected = DATA.fits[t.key][state.layer][state.prompt][state.scope];
    document.getElementById(`dim-stats-${t.key}`).textContent =
      `Layer ${state.layer} · selected: ${scopeLabel(state.scope)}, ${state.prompt} = ${selected.k} components`;
    const div = document.getElementById(`dim-${t.key}`);
    Plotly.react(div, [trace], layout, { responsive: true, displaylogo: false });
    if (!div.dataset.clickBound) {
      div.on("plotly_click", event => {
        const cd = event.points[0].customdata;
        state.scope = cd[2];
        state.prompt = cd[3];
        syncSelects();
        render();
      });
      div.dataset.clickBound = "1";
    }
  });
}

function drawLine() {
  const series = ["--series-1", "--series-2", "--series-3", "--series-4"];
  const traces = DATA.types.map((t, i) => ({
    type: "scatter", mode: "lines+markers", name: t.label,
    x: DATA.layers.map(Number),
    y: DATA.layers.map(l => DATA.fits[t.key][l][state.prompt][state.scope].k),
    line: { color: token(series[i]), width: 2 },
    marker: { symbol: SYMBOLS[i], size: 9, color: token(series[i]), line: { color: token("--surface-1"), width: 2 } },
    hovertemplate: `${t.label}<br>layer %{x}: <b>%{y}</b> components<extra></extra>`,
  }));
  const layout = baseLayout();
  layout.margin = { l: 56, r: 16, t: 8, b: 48 };
  layout.xaxis = Object.assign(axis("Layer"), { tickvals: DATA.layers.map(Number) });
  layout.yaxis = Object.assign(axis(`Components for ${pct(DATA.threshold)}`), { rangemode: "tozero" });
  layout.legend = { orientation: "h", y: 1.12, font: { color: token("--text-secondary") } };
  layout.hovermode = "x unified";
  document.getElementById("line-title").textContent =
    `${scopeLabel(state.scope)} · ${state.prompt}`;
  Plotly.react("line", traces, layout, { responsive: true, displaylogo: false });
}

function syncSelects() {
  document.getElementById("sel-layer").value = state.layer;
  document.getElementById("sel-prompt").value = state.prompt;
  document.getElementById("sel-scope").value = state.scope;
  document.getElementById("sel-comp").value = String(state.comp);
}

function render() {
  const format = DATA.promptFormats[state.prompt];
  const example = DATA.promptTexts[state.prompt][0];
  document.getElementById("format").innerHTML =
    `Format: <code>${escapeHtml(format || state.prompt)}</code> &nbsp; Example: <code>${escapeHtml(example)}</code>`;
  document.getElementById("lbl-comp").hidden = state.page !== "page-proj";
  if (state.page === "page-proj") {
    drawProjections();
  } else {
    drawHeatmaps();
    drawLine();
  }
}

function init() {
  document.getElementById("thr-label").textContent = pct(DATA.threshold);
  fillSelect("sel-layer", DATA.layers, null, state.layer);
  fillSelect("sel-prompt", DATA.prompts, null, state.prompt);
  fillSelect("sel-scope", DATA.scopes, DATA.scopes.map(scopeLabel), state.scope);
  buildPanels("grid-proj", "proj");
  buildPanels("grid-dim", "dim");
  document.getElementById("sel-layer").onchange = e => { state.layer = e.target.value; render(); };
  document.getElementById("sel-prompt").onchange = e => { state.prompt = e.target.value; render(); };
  document.getElementById("sel-scope").onchange = e => { state.scope = e.target.value; render(); };
  document.getElementById("sel-comp").onchange = e => { state.comp = Number(e.target.value); render(); };
  document.querySelectorAll(".tab").forEach(tab => {
    tab.onclick = () => {
      state.page = tab.dataset.page;
      document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", String(t === tab)));
      document.querySelectorAll("section[role=tabpanel]").forEach(s => { s.hidden = s.id !== state.page; });
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
    title = html.escape(page_data["title"])
    return (
        PAGE_TEMPLATE.replace("__PLOTLY_URL__", PLOTLY_URL)
        .replace("__TITLE__", title)
        .replace("__DATA__", data)
    )


def write_html(path, **page_arguments):
    page_data = build_page_data(**page_arguments)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(page_data))
    return path
