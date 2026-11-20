"""Write the HTML page of the steering experiment.

Selectors: category (or all categories), steered positions, and the steering
constant for the item plot and table.

Sections, for the items of the selected category:
- Mean change of the logit difference (mean typical pole - mean atypical pole)
  from before steering, for each steering constant: the typicality direction
  and each random direction.
- The same for the typicality direction, by typicality third.
- Logit difference against typicality, for each item, before steering and at
  the selected constant (with the Spearman correlation).
- The mean logit of each pole word.
- A table of the items at the selected constant, with the top next tokens.
"""

import html
import json
from collections import defaultdict

from typicality_linearity.global_pca_plot import PLOTLY_URL

N_TABLE_TOKENS = 3

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Typicality Steering</title>
<script src="__PLOTLY_URL__"></script>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --text: #0b0b0b; --muted: #52514e; --grid: #e4e3df;
    --typical: #2a78d6; --atypical: #eb6834; --random: #b5b4ae; --zero: #52514e;
    --third-0: #a9c8ef; --third-1: #5a9be3; --third-2: #1c4f8f;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --grid: #34332f;
      --typical: #3987e5; --atypical: #d95926; --random: #5f5e59; --zero: #c3c2b7;
      --third-0: #1f4a80; --third-1: #3987e5; --third-2: #a9c8ef;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --grid: #34332f;
    --typical: #3987e5; --atypical: #d95926; --random: #5f5e59; --zero: #c3c2b7;
    --third-0: #1f4a80; --third-1: #3987e5; --third-2: #a9c8ef;
  }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 16px;
         color: var(--text); background: var(--surface); }
  main { max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  h2 { font-size: 1.1rem; margin: 28px 0 4px; }
  p, li { line-height: 1.45; color: var(--muted); }
  code { font-family: ui-monospace, monospace; }
  .controls { position: sticky; top: 0; z-index: 2; background: var(--surface);
              display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
              padding: 10px 0; border-bottom: 1px solid var(--grid); margin: 12px 0; }
  select { font: inherit; padding: 2px 4px; }
  .plot { width: 100%; height: 400px; }
  .table-wrap { overflow-x: auto; max-height: 520px; overflow-y: auto; }
  table { border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
  th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
  th { position: sticky; top: 0; background: var(--surface); }
  th:first-child, td:first-child, td.left { text-align: left; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; font-size: 0.9rem; }
  dt { color: var(--muted); } dd { margin: 0; }
</style>
</head>
<body>
<main>
<h1>Steering along the general typicality direction</h1>
<p>Run <code>__RUN_ID__</code>. A positive steering constant α adds the typicality direction to the
residual stream. The logit difference is the mean logit of the typical poles minus the mean logit
of the atypical poles at the blank. If the direction is causal, a positive α increases it.</p>
<dl id="info"></dl>

<div class="controls">
  <label>Category <select id="category"></select></label>
  <label>Steered positions <select id="mode"></select></label>
  <label>α for item plot and table <select id="alpha"></select></label>
</div>

<h2>Mean change of the logit difference</h2>
<p>Mean over the items of the category. Gray lines: random unit directions with the same α (control).</p>
<div id="change" class="plot"></div>

<h2>Mean change by typicality third (typicality direction)</h2>
<p>Thirds of the items of each category, ranked by typicality. Dashed gray: mean of the random directions.</p>
<div id="thirds" class="plot"></div>

<h2>Logit difference of each item against its typicality</h2>
<p id="spearman"></p>
<div id="scatter" class="plot"></div>

<h2>Mean logit of each pole word (typicality direction)</h2>
<div id="words" class="plot"></div>

<h2>Items</h2>
<div class="table-wrap"><table id="table"></table></div>
</main>
<script>
const DATA = __DATA__;
const THIRD_LABELS = ["least typical third", "middle third", "most typical third"];

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function layout(xTitle, yTitle) {
  return {
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: {family: "system-ui, sans-serif", color: css("--text"), size: 12},
    margin: {l: 60, r: 110, t: 10, b: 50},
    xaxis: {title: {text: xTitle}, gridcolor: css("--grid"), zerolinecolor: css("--zero")},
    yaxis: {title: {text: yTitle}, gridcolor: css("--grid"), zerolinecolor: css("--zero")},
    hovermode: "closest",
    legend: {orientation: "h", y: -0.2},
  };
}
const ALPHA_TITLE = "Steering constant α (fraction of the residual norm at the steered token)";
const CONFIG = {responsive: true, displaylogo: false};

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function ranks(values) {
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(values.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
    for (let k = i; k <= j; k++) r[order[k][1]] = (i + j) / 2;
    i = j + 1;
  }
  return r;
}

function spearman(a, b) {
  const ra = ranks(a), rb = ranks(b);
  const ma = mean(ra), mb = mean(rb);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < ra.length; i++) {
    num += (ra[i] - ma) * (rb[i] - mb);
    da += (ra[i] - ma) ** 2;
    db += (rb[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db);
}

function selected() {
  const category = document.getElementById("category").value;
  const mode = document.getElementById("mode").value;
  const alpha = Number(document.getElementById("alpha").value);
  const rows = DATA.items.map((item, i) => i)
    .filter(i => category === "all" || DATA.items[i].category === category);
  return {category, mode, alpha, a: DATA.alphas.indexOf(alpha), rows, m: DATA.modes[mode]};
}

function meanOver(rows, matrix) {
  return DATA.alphas.map((_, a) => mean(rows.map(i => matrix[i][a])));
}

function change(m, i, a) {
  return m.typicality[i][a] - m.typicality[i][DATA.zero];
}

function drawChange(s) {
  const traces = [];
  const m = s.m;
  m.random.forEach((matrix, k) => traces.push({
    x: DATA.alphas, y: meanOver(s.rows, matrix), name: "random", legendgroup: "random",
    showlegend: k === 0, mode: "lines", line: {color: css("--random"), width: 1.5},
    hovertemplate: "random " + k + ": %{y:.3f}<extra></extra>",
  }));
  traces.push({
    x: DATA.alphas, y: DATA.alphas.map((_, a) => mean(s.rows.map(i => change(m, i, a)))),
    name: "typicality direction", mode: "lines+markers",
    line: {color: css("--typical"), width: 2.5}, marker: {size: 8},
    hovertemplate: "typicality: %{y:.3f}<extra></extra>",
  });
  Plotly.react("change", traces, layout(ALPHA_TITLE, "Mean change in logit difference"), CONFIG);
}

function drawThirds(s) {
  const m = s.m;
  const randomMean = DATA.alphas.map((_, a) =>
    mean(m.random.map(matrix => mean(s.rows.map(i => matrix[i][a])))));
  const traces = [{
    x: DATA.alphas, y: randomMean, name: "random (mean)", mode: "lines",
    line: {color: css("--random"), width: 1.5, dash: "dash"},
    hovertemplate: "random mean: %{y:.3f}<extra></extra>",
  }];
  [0, 1, 2].forEach(t => {
    const rows = s.rows.filter(i => DATA.items[i].third === t);
    if (!rows.length) return;
    traces.push({
      x: DATA.alphas, y: DATA.alphas.map((_, a) => mean(rows.map(i => change(m, i, a)))),
      name: THIRD_LABELS[t] + " (" + rows.length + ")", mode: "lines+markers",
      line: {color: css("--third-" + t), width: 2}, marker: {size: 8},
      hovertemplate: THIRD_LABELS[t] + ": %{y:.3f}<extra></extra>",
    });
  });
  Plotly.react("thirds", traces, layout(ALPHA_TITLE, "Mean change in logit difference"), CONFIG);
}

function drawScatter(s) {
  const m = s.m;
  const x = s.rows.map(i => DATA.items[i].typicality);
  const before = s.rows.map(i => m.typicality[i][DATA.zero]);
  const after = s.rows.map(i => m.typicality[i][s.a]);
  const names = s.rows.map(i => DATA.items[i].item + " (" + DATA.items[i].category + ")");
  const traces = [{
    x, y: before, text: names, name: "before steering", mode: "markers",
    marker: {size: 8, color: css("--random"), line: {width: 1, color: css("--surface")}},
    hovertemplate: "%{text}<br>typicality %{x:.2f}<br>before %{y:.2f}<extra></extra>",
  }];
  if (s.alpha !== 0) traces.push({
    x, y: after, text: names, name: "α = " + s.alpha, mode: "markers",
    marker: {size: 8, color: css("--typical"), line: {width: 1, color: css("--surface")}},
    hovertemplate: "%{text}<br>typicality %{x:.2f}<br>α = " + s.alpha + ": %{y:.2f}<extra></extra>",
  });
  Plotly.react("scatter", traces,
    layout("Typicality (−rating z-score, higher is more typical)", "Logit difference"), CONFIG);
  let text = "Spearman (typicality, logit difference): before steering " +
    spearman(x, before).toFixed(2);
  if (s.alpha !== 0) text += ", at α = " + s.alpha + ": " + spearman(x, after).toFixed(2);
  document.getElementById("spearman").textContent = text + " (n = " + x.length + ").";
}

function drawWords(s) {
  const dashes = ["solid", "dash", "dot"];
  const traces = [];
  [["typical", DATA.typical_poles, "--typical"], ["atypical", DATA.atypical_poles, "--atypical"]]
    .forEach(([group, words, color]) => words.forEach((word, k) => {
      traces.push({
        x: DATA.alphas, y: meanOver(s.rows, s.m.words[word]), name: word + " (" + group + ")",
        mode: "lines", line: {color: css(color), width: 2, dash: dashes[k % 3]},
        hovertemplate: word + ": %{y:.2f}<extra></extra>",
      });
    }));
  const l = layout(ALPHA_TITLE, "Mean logit");
  l.annotations = traces.map(t => ({x: t.x[t.x.length - 1], y: t.y[t.y.length - 1],
    xanchor: "left", text: " " + t.name.split(" ")[0], showarrow: false,
    font: {color: css("--muted"), size: 11}}));
  Plotly.react("words", traces, l, CONFIG);
}

function escapeHtml(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;");
}

function drawTable(s) {
  const m = s.m;
  const rows = s.rows.slice().sort((i, j) => DATA.items[j].typicality - DATA.items[i].typicality);
  let h = "<thead><tr><th>Item</th><th>Category</th><th>Typicality</th><th>Third</th>" +
    "<th>Before</th><th>α = " + s.alpha + "</th><th>Change</th>" +
    "<th>Top tokens before</th><th>Top tokens at α</th><th>Prompt</th></tr></thead><tbody>";
  rows.forEach(i => {
    const item = DATA.items[i];
    const before = m.typicality[i][DATA.zero], after = m.typicality[i][s.a];
    const tops = a => m.tops[i][a].map(t => escapeHtml(JSON.stringify(t))).join(" ");
    h += "<tr><td>" + escapeHtml(item.item) + "</td><td class='left'>" + item.category +
      "</td><td>" + item.typicality.toFixed(2) + "</td><td>" + (item.third + 1) +
      "</td><td>" + before.toFixed(2) + "</td><td>" + after.toFixed(2) + "</td><td>" +
      (after - before >= 0 ? "+" : "") + (after - before).toFixed(2) + "</td><td class='left'>" +
      tops(DATA.zero) + "</td><td class='left'>" + tops(s.a) + "</td><td class='left'>" +
      escapeHtml(item.context) + " ___</td></tr>";
  });
  document.getElementById("table").innerHTML = h + "</tbody>";
}

function drawInfo(s) {
  const sel = DATA.selection;
  const info = [
    ["Model", DATA.model + ", layer " + DATA.layer],
    ["Template", DATA.template],
    ["Typical poles", DATA.typical_poles.join(", ")],
    ["Atypical poles", DATA.atypical_poles.join(", ")],
    ["Direction", sel.setting + ", baseline " + sel.baseline + ": mean of " +
      sel.n_directions_averaged + " unit directions (all categories × training prompts)"],
    ["Selected by", sel.statistic + " (" + sel.metric + ") = " + sel.value.toFixed(3)],
    ["Items", s.rows.length + (s.category === "all" ? " (all categories)" : " (" + s.category + ")")],
  ];
  document.getElementById("info").innerHTML = info.map(([k, v]) =>
    "<dt>" + k + "</dt><dd>" + escapeHtml(v) + "</dd>").join("");
}

function draw() {
  const s = selected();
  drawInfo(s); drawChange(s); drawThirds(s); drawScatter(s); drawWords(s); drawTable(s);
}

function addOptions(id, values, labels, chosen) {
  const select = document.getElementById(id);
  values.forEach((v, k) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = labels[k];
    if (v === chosen) o.selected = true;
    select.appendChild(o);
  });
  select.addEventListener("change", draw);
}

const MODE_LABELS = {item: "last token of the item", all: "all tokens (not the prefix)",
                     last: "token before the blank"};
addOptions("category", ["all"].concat(DATA.categories),
           ["All categories"].concat(DATA.categories), "all");
addOptions("mode", Object.keys(DATA.modes), Object.keys(DATA.modes).map(m => MODE_LABELS[m] || m));
addOptions("alpha", DATA.alphas.map(String), DATA.alphas.map(String),
           String(DATA.alphas[DATA.alphas.length - 1]));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
draw();
</script>
</body>
</html>
"""


def rounded(value):
    return round(value, 3)


def build_plot_data(config, selection, items, rows, top_rows):
    """Return the data of the page.

    modes[mode]: typicality[item][alpha] (logit difference), random[k][item][alpha]
    (change from before steering), words[word][item][alpha] (logit),
    tops[item][alpha] (top next tokens), all for the typicality direction
    except random.
    """
    alphas = [float(c) for c in config["steering_constants"]]
    alpha_index = {a: k for k, a in enumerate(alphas)}
    item_index = {(it["category"], it["item"]): k for k, it in enumerate(items)}
    words = list(config["typical_poles"]) + list(config["atypical_poles"])
    random_names = sorted({r["direction"] for r in rows} - {"typicality"})
    random_index = {name: k for k, name in enumerate(random_names)}

    def empty():
        return [[None] * len(alphas) for _ in items]

    modes = {}
    for mode in config["steer_modes"]:
        modes[mode] = {
            "typicality": empty(),
            "random": [empty() for _ in random_names],
            "words": {word: empty() for word in words},
            "tops": [[[] for _ in alphas] for _ in items],
        }
    for row in rows:
        m = modes[row["steer_mode"]]
        i = item_index[(row["category"], row["item"])]
        a = alpha_index[float(row["constant"])]
        if row["direction"] == "typicality":
            m["typicality"][i][a] = rounded(row["logit_difference"])
            for word in words:
                m["words"][word][i][a] = rounded(row[f"logit_{word}"])
        else:
            m["random"][random_index[row["direction"]]][i][a] = rounded(row["change"])

    by_key = defaultdict(list)
    for row in top_rows:
        if row["rank"] <= N_TABLE_TOKENS:
            key = (
                row["steer_mode"],
                row["category"],
                row["item"],
                float(row["constant"]),
            )
            by_key[key].append((row["rank"], row["token"]))
    for (mode, category, item, constant), tokens in by_key.items():
        tops = modes[mode]["tops"][item_index[(category, item)]]
        tops[alpha_index[constant]] = [token for _, token in sorted(tokens)]

    return {
        "model": config["model"],
        "layer": config["layer"],
        "template": config["template"],
        "typical_poles": list(config["typical_poles"]),
        "atypical_poles": list(config["atypical_poles"]),
        "selection": selection,
        "alphas": alphas,
        "zero": alpha_index[0.0],
        "categories": sorted({it["category"] for it in items}),
        "items": [
            {
                "item": it["item"],
                "category": it["category"],
                "typicality": rounded(it["typicality"]),
                "third": it["third"],
                "context": it["context"],
            }
            for it in items
        ],
        "modes": modes,
    }


def render_html(plot_data, run_id):
    # "</" in the JSON would close the script tag.
    data_json = json.dumps(plot_data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__PLOTLY_URL__", PLOTLY_URL)
    page = page.replace("__RUN_ID__", html.escape(run_id))
    return page.replace("__DATA__", data_json)


def write_html(config, selection, items, rows, top_rows, run_id, path):
    plot_data = build_plot_data(config, selection, items, rows, top_rows)
    with open(path, "w") as f:
        f.write(render_html(plot_data, run_id))
