"""Write an interactive HTML page for probe transfer results.

The page has two global selectors (probe type, metric) and two sections:
- Transfer between categories: a prompt selector, and one category x category
  heatmap for each variant (baseline), side by side.
- Transfer between prompts: a category selector, and one prompt x prompt
  heatmap for each variant, side by side.

Rows are what the probe was trained on, columns what it was evaluated on.
Each section also has the cosine similarity of the probe directions (full
probes, in activation space), one heatmap for each variant. The metric selector
does not change these.

The colour scale is fixed from -1 (red) to 1 (blue). In the category matrix,
the outlined diagonal cells are cross-validated. In the prompt matrix, all
cells are cross-validated.
"""

import html
import json
import math

from typicality_linearity.pca_plot import PLOTLY_URL

PROBE_LABELS = {
    "diff_of_means": "Difference of means",
    "ridge": "Ridge regression",
}

METRIC_LABELS = {
    "spearman": "Spearman",
    "pearson": "Pearson",
}


def matrix_to_list(matrix, digits=4):
    """Return a list of lists. NaN becomes None (null in JSON)."""
    rows = []
    for row in matrix:
        values = []
        for value in row:
            value = float(value)
            if math.isnan(value):
                values.append(None)
            else:
                values.append(round(value, digits))
        rows.append(values)
    return rows


def matrices_to_lists(by_name):
    return {name: matrix_to_list(matrix) for name, matrix in by_name.items()}


def build_plot_data(title, notes, variants, results, n_items):
    """Return a JSON-ready dict for the page.

    variants: {variant: panel label}, in panel order
    results: {variant: result of probe_transfer.transfer_matrices}. All
        variants must have the same categories, prompts, probes and metrics.
    n_items: {category: number of items}
    """
    first = results[next(iter(variants))]
    probe_types = list(first["category_transfer"])
    metrics = list(first["category_transfer"][probe_types[0]])
    category_transfer = {}
    prompt_transfer = {}
    category_cosine = {}
    prompt_cosine = {}
    for variant in variants:
        result = results[variant]
        if result["categories"] != first["categories"]:
            raise ValueError("Variants have different categories")
        if result["prompts"] != first["prompts"]:
            raise ValueError("Variants have different prompts")
        category_transfer[variant] = {}
        prompt_transfer[variant] = {}
        category_cosine[variant] = {}
        prompt_cosine[variant] = {}
        for probe_type in probe_types:
            category_cosine[variant][probe_type] = matrices_to_lists(
                result["category_cosine"][probe_type]
            )
            prompt_cosine[variant][probe_type] = matrices_to_lists(
                result["prompt_cosine"][probe_type]
            )
            category_transfer[variant][probe_type] = {}
            prompt_transfer[variant][probe_type] = {}
            for metric in metrics:
                category_transfer[variant][probe_type][metric] = {
                    prompt: matrix_to_list(matrix)
                    for prompt, matrix in result["category_transfer"][probe_type][
                        metric
                    ].items()
                }
                prompt_transfer[variant][probe_type][metric] = {
                    category: matrix_to_list(matrix)
                    for category, matrix in result["prompt_transfer"][probe_type][
                        metric
                    ].items()
                }
    return {
        "title": title,
        "notes": list(notes),
        "variants": list(variants),
        "variant_labels": dict(variants),
        "probe_types": probe_types,
        "probe_labels": {p: PROBE_LABELS.get(p, p) for p in probe_types},
        "metrics": metrics,
        "metric_labels": {m: METRIC_LABELS.get(m, m) for m in metrics},
        "categories": list(first["categories"]),
        "prompts": list(first["prompts"]),
        "n_items": {c: int(n) for c, n in n_items.items()},
        "category_transfer": category_transfer,
        "prompt_transfer": prompt_transfer,
        "category_cosine": category_cosine,
        "prompt_cosine": prompt_cosine,
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
  select { font-size: 14px; padding: 4px; }
  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .plot { width: 100%; height: 540px; }
  .caption { font-size: 12px; color: #555; margin-top: 4px; }
  .subhead { font-size: 15px; margin: 20px 0 4px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<ul class="notes" id="notes"></ul>
<div class="controls">
  <label>Probe <select id="probe"></select></label>
  <label>Metric <select id="metric"></select></label>
</div>

<h2>Transfer between categories</h2>
<div class="controls">
  <label>Prompt format <select id="prompt"></select></label>
</div>
<div class="caption">Rows: category the probe was trained on. Columns: category it was
evaluated on. Outlined diagonal cells are cross-validated; other cells use the probe
trained on all items of the row category.</div>
<div class="panels" id="category-panels"></div>
<h3 class="subhead">Cosine similarity of probe directions between categories</h3>
<div class="caption">Probes trained on all items of each category, for the selected
prompt format. Ridge directions are weights / feature std, so all directions are in
activation space.</div>
<div class="panels" id="category-cosine-panels"></div>

<h2>Transfer between prompts</h2>
<div class="controls">
  <label>Category <select id="category"></select></label>
</div>
<div class="caption">Rows: prompt format the probe was trained on. Columns: prompt format
it was evaluated on. All cells are cross-validated on held-out items, so a probe is never
evaluated on items it was trained on.</div>
<div class="panels" id="prompt-panels"></div>
<h3 class="subhead">Cosine similarity of probe directions between prompts</h3>
<div class="caption">Probes trained on all items of the selected category, for each
prompt format.</div>
<div class="panels" id="prompt-cosine-panels"></div>

<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const probeSelect = document.getElementById("probe");
const metricSelect = document.getElementById("metric");
const promptSelect = document.getElementById("prompt");
const categorySelect = document.getElementById("category");

function addOptions(select, values, labels) {
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels ? labels[value] : value;
    select.appendChild(option);
  }
}
addOptions(probeSelect, data.probe_types, data.probe_labels);
addOptions(metricSelect, data.metrics, data.metric_labels);
addOptions(promptSelect, data.prompts);
const categoryLabels = {};
for (const c of data.categories) {
  categoryLabels[c] = c + " (" + data.n_items[c] + " items)";
}
addOptions(categorySelect, data.categories, categoryLabels);

for (const note of data.notes) {
  const li = document.createElement("li");
  li.textContent = note;
  document.getElementById("notes").appendChild(li);
}

function makePanels(containerId, prefix) {
  const container = document.getElementById(containerId);
  for (const variant of data.variants) {
    const panel = document.createElement("div");
    panel.innerHTML = '<h3></h3><div class="plot" id="' + prefix + "-" + variant + '"></div>';
    panel.querySelector("h3").textContent = data.variant_labels[variant];
    container.appendChild(panel);
  }
}
makePanels("category-panels", "category-plot");
makePanels("prompt-panels", "prompt-plot");
makePanels("category-cosine-panels", "category-cosine-plot");
makePanels("prompt-cosine-panels", "prompt-cosine-plot");

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

// options: valueLabel, rowTitle, columnTitle, rowName, columnName, outlineDiagonal
function drawHeatmap(plotId, matrix, labels, options) {
  const shapes = [];
  const outlineDiagonal = options.outlineDiagonal;
  if (outlineDiagonal) {
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
  const probe = probeSelect.value;
  const metric = metricSelect.value;
  const prompt = promptSelect.value;
  const category = categorySelect.value;
  for (const variant of data.variants) {
    drawHeatmap(
      "category-plot-" + variant,
      data.category_transfer[variant][probe][metric][prompt],
      data.categories, transferOptions("category", true));
    drawHeatmap(
      "prompt-plot-" + variant,
      data.prompt_transfer[variant][probe][metric][category],
      data.prompts, transferOptions("prompt", false));
    drawHeatmap(
      "category-cosine-plot-" + variant,
      data.category_cosine[variant][probe][prompt],
      data.categories, cosineOptions("Category"));
    drawHeatmap(
      "prompt-cosine-plot-" + variant,
      data.prompt_cosine[variant][probe][category],
      data.prompts, cosineOptions("Prompt"));
  }
}

for (const select of [probeSelect, metricSelect, promptSelect, categorySelect]) {
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
