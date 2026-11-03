"""Write an interactive HTML page for PCA results.

The page has four selectors (layer, prompt format, 2 or 3 components,
category) and one plot for each token position, side by side. Point colour
shows typicality: red is a low value (typical), blue is a high value. Above each plot, the page shows a prompt with the token that
gives the activation in bold. Move the pointer over a point to see its prompt.
"All categories" shows the PCA fitted on all rows. When one category is
selected, the page shows the PCA fitted only on that category, if the data has
it. If not, the page shows the points of that category in the pooled PCA.

If there are two or more variants (for example the PCA with and without a
baseline), the page has one plot for each variant and position, side by side,
with the same selectors. Move the pointer over a point to see its prompt in all
panels.
"""

import html
import json

PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

# Scope name for the PCA fitted on all rows.
ALL_SCOPE = "all"

# Variant name when there is only one variant.
MAIN_VARIANT = "main"

POSITION_LABELS = {
    "item": "Last token of the item",
    "period": "Final period token",
}


def round_list(values, digits=4):
    return [round(float(value), digits) for value in values]


def build_plot_data(
    title,
    layers,
    prompt_names,
    positions,
    points,
    prompts,
    token_spans,
    pca_results,
    color_label,
    scopes=None,
    variants=None,
):
    """Return a JSON-ready dict for the page.

    points: {"item": [...], "category": [...], "color": [...], ...}, one
        value per row. The rows are the same for each prompt format.
    prompts: {prompt_name: [prompt text for each row]}
    token_spans: {prompt_name: {position: [(start, end) for each row]}}
    pca_results: {prompt_name: {layer: {scope: {position: PcaResult}}}}
    scopes: {scope: row indices}. Row j of the projections of a scope is row
        scopes[scope][j] of points. None means only the "all" scope.
    variants: {variant: panel label}. If given, pca_results is
        {variant: {prompt_name: ...}} and the page has panels for each variant.
        None means one variant.
    """
    if scopes is None:
        scopes = {ALL_SCOPE: list(range(len(points["item"])))}
    if variants is None:
        variants = {MAIN_VARIANT: None}
        pca_results = {MAIN_VARIANT: pca_results}
    pca = {}
    for variant in variants:
        pca[variant] = pca_for_variant(
            pca_results[variant], prompt_names, layers, scopes, positions
        )

    spans = {}
    for prompt_name in prompt_names:
        spans[prompt_name] = {}
        for position in positions:
            spans[prompt_name][position] = [
                [int(start), int(end)]
                for start, end in token_spans[prompt_name][position]
            ]

    return {
        "title": title,
        "layers": [int(layer) for layer in layers],
        "prompt_names": list(prompt_names),
        "positions": list(positions),
        "position_labels": {p: POSITION_LABELS.get(p, p) for p in positions},
        "variants": list(variants),
        "variant_labels": dict(variants),
        "color_label": color_label,
        "points": points,
        "prompts": prompts,
        "spans": spans,
        "pca": pca,
        "all_scope": ALL_SCOPE,
        "scope_rows": {
            scope: [int(row) for row in rows] for scope, rows in scopes.items()
        },
    }


def pca_for_variant(pca_results, prompt_names, layers, scopes, positions):
    """Return {prompt: {layer: {scope: {position: coords and explained}}}}."""
    pca = {}
    for prompt_name in prompt_names:
        pca[prompt_name] = {}
        for layer in layers:
            pca[prompt_name][str(layer)] = {}
            for scope in scopes:
                pca[prompt_name][str(layer)][scope] = {}
                for position in positions:
                    result = pca_results[prompt_name][layer][scope][position]
                    pca[prompt_name][str(layer)][scope][position] = {
                        "coords": [
                            round_list(result.projections[:, k])
                            for k in range(result.projections.shape[1])
                        ],
                        "explained": round_list(result.explained_variance_ratio),
                    }
    return pca


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY_URL__"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; color: #1a1a1a; background: #fff; }
  h1 { font-size: 20px; margin: 0 0 12px; }
  .controls { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; }
  .controls label { display: flex; flex-direction: column; font-size: 13px; gap: 4px; }
  select { font-size: 14px; padding: 4px; }
  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .panel h2 { font-size: 15px; margin: 0 0 6px; }
  .prompt { font-family: ui-monospace, monospace; font-size: 13px; background: #f4f4f4;
            padding: 8px; border-radius: 4px; min-height: 2.6em; }
  .prompt b { background: #ffd54f; padding: 0 2px; border-radius: 2px; }
  .meta { font-size: 12px; color: #555; margin-top: 4px; }
  .plot { width: 100%; height: 520px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="controls">
  <label>Layer <select id="layer"></select></label>
  <label>Prompt format <select id="prompt"></select></label>
  <label>Components <select id="dims">
    <option value="2">Top 2 (PC1, PC2)</option>
    <option value="3">Top 3 (PC1, PC2, PC3)</option>
  </select></label>
  <label>Category <select id="category"></select></label>
</div>
<div class="panels" id="panels"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const layerSelect = document.getElementById("layer");
const promptSelect = document.getElementById("prompt");
const dimsSelect = document.getElementById("dims");
const categorySelect = document.getElementById("category");
const ALL_CATEGORIES = data.all_scope;
const panels = document.getElementById("panels");
const hoverAttached = {};

function addOptions(select, values) {
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}
addOptions(layerSelect, data.layers);
addOptions(promptSelect, data.prompt_names);

const allOption = document.createElement("option");
allOption.value = ALL_CATEGORIES;
allOption.textContent = "All categories";
categorySelect.appendChild(allOption);
addOptions(categorySelect, [...new Set(data.points.category)].sort());

// Fixed colour range, centred on 0, so that colours agree between categories.
// Red is a low value (typical), blue is a high value (atypical).
// The limit is the 90th percentile of |value|, so that outliers do not make most
// points grey. Values outside the limit get the end colour.
const absColors = data.points.color.map(Math.abs).sort((a, b) => a - b);
const colorLimit = absColors[Math.floor(0.9 * (absColors.length - 1))] || 1;
const colorMin = -colorLimit;
const colorMax = colorLimit;

// One panel for each variant and position.
const panelList = [];
for (const variant of data.variants) {
  for (const position of data.positions) {
    const id = variant + "-" + position;
    panelList.push({ variant: variant, position: position, id: id });
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML =
      '<h2></h2><div class="prompt" id="prompt-' + id + '"></div>' +
      '<div class="meta" id="meta-' + id + '"></div>' +
      '<div class="plot" id="plot-' + id + '"></div>';
    let heading = data.position_labels[position];
    if (data.variant_labels[variant] !== null) {
      heading = data.variant_labels[variant] + ": " + heading;
    }
    panel.querySelector("h2").textContent = heading;
    panels.appendChild(panel);
  }
}

// Show the prompt of the row above all panels.
function showPromptInAllPanels(row) {
  for (const panel of panelList) {
    showPrompt(panel.id, panel.position, row);
  }
}

function showPrompt(id, position, row) {
  const promptName = promptSelect.value;
  const text = data.prompts[promptName][row];
  const span = data.spans[promptName][position][row];
  const box = document.getElementById("prompt-" + id);
  box.textContent = "";
  box.appendChild(document.createTextNode(text.slice(0, span[0])));
  const bold = document.createElement("b");
  bold.textContent = text.slice(span[0], span[1]);
  box.appendChild(bold);
  box.appendChild(document.createTextNode(text.slice(span[1])));
  const p = data.points;
  document.getElementById("meta-" + id).textContent =
    "item: " + p.item[row] + " | category: " + p.category[row] +
    " | " + data.color_label + ": " + p.color[row];
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Break text into lines of at most `width` characters, at spaces.
function wrapLines(text, width) {
  const lines = [];
  let line = "";
  for (const word of text.split(" ")) {
    if (line.length > 0 && line.length + 1 + word.length > width) {
      lines.push(line);
      line = word;
    } else {
      line = line.length > 0 ? line + " " + word : word;
    }
  }
  lines.push(line);
  return lines;
}

// Hover label: the prompt (activation token in bold), then item, category and colour value.
function hoverText(position, row) {
  const promptName = promptSelect.value;
  const text = data.prompts[promptName][row];
  const span = data.spans[promptName][position][row];
  // Put markers around the token, wrap, then change the markers to HTML.
  const marked = text.slice(0, span[0]) + "\\u0001" + text.slice(span[0], span[1]) +
    "\\u0002" + text.slice(span[1]);
  const wrapped = wrapLines(marked, 60).map(escapeHtml).join("<br>")
    .replace("\\u0001", "<b>").replace("\\u0002", "</b>");
  const p = data.points;
  return wrapped + "<br><br>" +
    escapeHtml(p.item[row] + " (" + p.category[row] + ")") + "<br>" +
    escapeHtml(data.color_label) + ": " + p.color[row];
}

function render() {
  const promptName = promptSelect.value;
  const layer = layerSelect.value;
  const dims = Number(dimsSelect.value);
  const category = categorySelect.value;
  const firstScopes = data.pca[data.variants[0]][promptName][layer];
  // Use the PCA of the category if there is one. If not, show the points of
  // the category in the pooled PCA.
  const scope = category in firstScopes ? category : ALL_CATEGORIES;
  const scopeRows = data.scope_rows[scope];
  // j is the index in the projections of the scope, row is the index in points.
  const shown = scopeRows
    .map((row, j) => [row, j])
    .filter(([row, _]) => category === ALL_CATEGORIES || data.points.category[row] === category);
  const rows = shown.map(([row, _]) => row);
  const js = shown.map(([_, j]) => j);
  for (const panel of panelList) {
    const position = panel.position;
    const result = data.pca[panel.variant][promptName][layer][scope][position];
    const c = result.coords;
    const pct = result.explained.map(v => (100 * v).toFixed(1) + "%");
    const marker = {
      color: rows.map(i => data.points.color[i]),
      cmin: colorMin,
      cmax: colorMax,
      colorscale: [
        [0, "#67001f"], [0.2, "#d6604d"], [0.5, "#eeeeee"],
        [0.8, "#4393c3"], [1, "#053061"],
      ],
      line: { width: 0.5, color: "#444444" },
      size: dims === 3 ? 5 : 9,
      opacity: 1,
      colorbar: { title: { text: data.color_label } },
    };
    const trace = {
      x: js.map(j => c[0][j]),
      y: js.map(j => c[1][j]),
      mode: "markers",
      type: dims === 3 ? "scatter3d" : "scatter",
      marker: marker,
      customdata: rows,
      text: rows.map(row => hoverText(position, row)),
      hoverinfo: "text",
    };
    const layout = {
      margin: { l: 50, r: 10, t: 10, b: 50 },
      showlegend: false,
      hoverlabel: { align: "left" },
    };
    if (dims === 3) {
      trace.z = js.map(j => c[2][j]);
      layout.scene = {
        xaxis: { title: { text: "PC1 (" + pct[0] + ")" } },
        yaxis: { title: { text: "PC2 (" + pct[1] + ")" } },
        zaxis: { title: { text: "PC3 (" + pct[2] + ")" } },
      };
    } else {
      layout.xaxis = { title: { text: "PC1 (" + pct[0] + ")" } };
      layout.yaxis = { title: { text: "PC2 (" + pct[1] + ")" } };
    }
    const plot = document.getElementById("plot-" + panel.id);
    Plotly.react(plot, [trace], layout, { responsive: true });
    if (!hoverAttached[panel.id]) {
      plot.on("plotly_hover", event => showPromptInAllPanels(event.points[0].customdata));
      hoverAttached[panel.id] = true;
    }
  }
  if (rows.length > 0) {
    showPromptInAllPanels(rows[0]);
  }
}

layerSelect.addEventListener("change", render);
promptSelect.addEventListener("change", render);
dimsSelect.addEventListener("change", render);
categorySelect.addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def render_html(plot_data):
    data_json = json.dumps(plot_data).replace("</", "<\\/")
    page = PAGE_TEMPLATE.replace("__PLOTLY_URL__", PLOTLY_URL)
    page = page.replace("__TITLE__", html.escape(plot_data["title"]))
    return page.replace("__DATA__", data_json)


def write_html(plot_data, path):
    with open(path, "w") as f:
        f.write(render_html(plot_data))
