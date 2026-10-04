# encoding: utf-8
"""Shared building blocks of the QA reports.

Theme (colors and fonts, with their dark-theme counterparts), plotly figure
styling, HTML snippets (tables, tiles, badges) and the self-contained HTML
dashboard page used by the QA modules in :mod:`lvmdrp.qa`.
"""

import os
import re
import json
from html import escape
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs_version


# report colors (light theme); the report swaps them for DARK_COLORS in the
# browser when the viewer uses a dark theme
THEME = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DARK_COLORS = {"#fcfcfb": "#1a1a19", "#0b0b0b": "#ffffff", "#52514e": "#c3c2b7", "#e1e0d9": "#2c2c2a", "#c3c2b7": "#383835",
               "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70"}
SEQUENTIAL = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
SEQUENTIAL_DARK = [[0.0, "#104281"], [0.25, "#1c5cab"], [0.5, "#3987e5"], [0.75, "#86b6ef"], [1.0, "#cde2fb"]]
PLOT_FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"
MONO_FONT = "IBM Plex Mono, ui-monospace, Menlo, monospace"

# status colors, light theme; DARK_MAP swaps them in the browser for the dark theme
STATUS_COLORS = {"ok": "#1baf7a", "warn": "#eb6834", "bad": "#c8372d"}
DARK_MAP = {**DARK_COLORS, "#c8372d": "#e5534b"}


def base_layout(height: int, **kwargs) -> Dict:
    layout = dict(
        template="none", height=height, margin=dict(l=64, r=16, t=40, b=52),
        paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
        font=dict(family=PLOT_FONT, size=13, color=THEME["ink2"]),
        hoverlabel=dict(bgcolor=THEME["surface"], bordercolor=THEME["axis"], font=dict(family=PLOT_FONT, color=THEME["ink"])),
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom", font=dict(color=THEME["ink2"]), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
    )
    layout.update(kwargs)
    return layout


def style_axes(fig):
    axis = dict(showgrid=True, gridcolor=THEME["grid"], gridwidth=1, linecolor=THEME["axis"], linewidth=1,
                zerolinecolor=THEME["axis"], zerolinewidth=1, tickcolor=THEME["axis"],
                tickfont=dict(color=THEME["muted"]), title_font=dict(color=THEME["ink2"]), ticks="outside", ticklen=4)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def restyle(fig, height=None):
    """Apply the dashboard theme to a figure made elsewhere, keeping its traces.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to restyle, in place unless it has WebGL traces.
    height : int, optional
        New height in pixels. The width always follows the page.

    Returns
    -------
    plotly.graph_objects.Figure
        The restyled figure, with WebGL traces replaced by SVG ones.
    """
    # WebGL traces don't draw without a GPU context (e.g., some remote desktops and headless browsers)
    if any(trace.type == "scattergl" for trace in fig.data):
        fig = go.Figure(data=[go.Scatter(**{key: value for key, value in trace.to_plotly_json().items() if key != "type"})
                              if trace.type == "scattergl" else trace for trace in fig.data], layout=fig.layout)
    fig.update_layout(
        template="none", width=None, paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
        font=dict(family=PLOT_FONT, size=12, color=THEME["ink2"]),
        hoverlabel=dict(bgcolor=THEME["surface"], bordercolor=THEME["axis"], font=dict(family=PLOT_FONT, color=THEME["ink"])),
        legend=dict(font=dict(color=THEME["ink2"]), bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(color=THEME["ink"], size=15)),
    )
    if height is not None:
        fig.update_layout(height=height)
    # recolor the annotations without an explicit color (e.g., subplot titles)
    for annotation in fig.layout.annotations:
        if annotation.font is None or annotation.font.color is None:
            annotation.update(font=dict(color=THEME["ink2"]))
    return style_axes(fig)


def figures_json(figures):
    """Serialize figures for the page, ASCII-safe and safe inside a script tag."""
    text = "{" + ",".join(f"{json.dumps(key)}: {pio.to_json(fig, validate=False)}" for key, fig in figures.items()) + "}"
    text = text.replace("</", "<\\/")
    return re.sub(r"[^\x00-\x7f]", lambda match: f"\\u{ord(match.group()):04x}", text)


def tiles_html(tiles):
    return "".join(f'<div class="tile"><div class="tile-label">{escape(label)}</div>'
                   f'<div class="tile-value">{escape(value)}</div><div class="tile-note">{escape(note)}</div></div>'
                   for label, value, note in tiles)


def meta_html(meta):
    return "".join(f"<div><dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd></div>" for key, value in meta)


def badge(status):
    return f'<span class="badge {status}">{status}</span>'


def issues_html(issues):
    if not issues:
        return '<span class="range">none</span>'
    return '<ul class="issues">' + "".join(f"<li>{escape(issue)}</li>" for issue in issues) + "</ul>"


def html_table(columns: List[Tuple[str, str]], rows: List[List[str]], numeric: List[bool] = None, table_id: str = None) -> str:
    numeric = numeric or [False] * len(columns)
    head = "".join(f'<th scope="col" class="{"num" if num else ""}" title="{escape(desc)}">{escape(name)}</th>'
                   for (name, desc), num in zip(columns, numeric))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if num else ""}">{cell}</td>' for cell, num in zip(row, numeric)) + "</tr>"
                   for row in rows)
    id_attr = f' id="{table_id}"' if table_id else ""
    return f'<div class="table-wrap"><table{id_attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def picker(chart_id, prefix, dimensions, height):
    """HTML of a chart whose figure is chosen with one select per dimension.

    Parameters
    ----------
    chart_id : str
        Id of the chart element.
    prefix : str
        Prefix of the figure keys, which are ``prefix|value1|value2...``.
    dimensions : list[tuple[str, list[tuple[str, str]]]]
        Label of each select and its (value, text) options; the first option
        is shown initially.
    height : int
        Minimum height of the chart in pixels, to keep the page from jumping.

    Returns
    -------
    str
        HTML of the selects and the chart.
    """
    selects = "".join(
        f'<label>{escape(label)} <select>' + "".join(f'<option value="{escape(value)}">{escape(text)}</option>' for value, text in options) + "</select></label>"
        for label, options in dimensions
    )
    first = "|".join([prefix] + [options[0][0] for _, options in dimensions])
    return (f'<div class="picker" data-chart="{chart_id}" data-prefix="{prefix}">{selects}</div>'
            f'<div class="chart" id="{chart_id}" data-fig="{escape(first)}" style="min-height: {height}px"></div>')


def write_dashboard(out_html, title, eyebrow, intro, meta, tiles, body, figures, generator):
    """Write a dashboard page.

    Parameters
    ----------
    out_html : str
        Path of the HTML file to write.
    title, eyebrow : str
        Page title and the small line above it.
    intro : str
        HTML of the introduction paragraphs.
    meta : list[tuple[str, str]]
        Run metadata shown under the introduction.
    tiles : list[tuple[str, str, str]]
        Summary tiles as (label, value, note).
    body : str
        HTML of the sections. Charts are ``<div class="chart" data-fig="key">``
        elements, drawn from ``figures[key]``.
    figures : dict[str, plotly.graph_objects.Figure]
        Figures of the page.
    generator : str
        Name of the function that made the page, shown at the bottom.

    Returns
    -------
    str
        Path of the written file.
    """
    html = PAGE_TEMPLATE
    for token, value in {
        "TITLE": escape(title), "EYEBROW": escape(eyebrow), "INTRO": intro, "META": meta_html(meta),
        "TILES": tiles_html(tiles), "BODY": body, "GENERATOR": escape(generator),
        "PLOTLY_VERSION": get_plotlyjs_version(), "DARK_MAP": json.dumps(DARK_MAP), "SEQ_DARK": json.dumps(SEQUENTIAL_DARK),
        "FIGURES": figures_json(figures),
    }.items():
        html = html.replace(f"@@{token}@@", value)
    # keep the page pure ASCII so it renders correctly however the file is served
    html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


# HTML head and stylesheet shared by all report pages, see page_head
PAGE_HEAD = """<meta charset="utf-8">
<title>@@TITLE@@</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --plane: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --accent: #2a78d6;
  --ok: #1baf7a; --warn: #eb6834; --bad: #c8372d;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --accent: #3987e5;
    --ok: #199e70; --warn: #d95926; --bad: #e5534b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --accent: #3987e5;
  --ok: #199e70; --warn: #d95926; --bad: #e5534b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--plane); color: var(--ink); font: 15px/1.55 var(--sans); }
.page { max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 32px 64px; display: grid; gap: 40px; }
header { display: grid; gap: 16px; }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
h1 { margin: 0; font-size: clamp(28px, 4vw, 38px); line-height: 1.15; font-weight: 600; text-wrap: balance; }
h2 { margin: 0; font-size: 20px; font-weight: 600; text-wrap: balance; }
p { margin: 0; max-width: 68ch; color: var(--ink-2); }
a { color: var(--accent); }
dl.meta { margin: 0; display: flex; flex-wrap: wrap; gap: 8px 28px; font-size: 13px; }
dl.meta div { display: grid; gap: 2px; }
dl.meta dt { color: var(--muted); }
dl.meta dd { margin: 0; font-family: var(--mono); color: var(--ink); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; display: grid; gap: 4px; align-content: start; }
.tile-label { font-size: 13px; color: var(--ink-2); }
.tile-value { font-size: 28px; font-weight: 600; line-height: 1.2; font-variant-numeric: tabular-nums; }
.tile-note { font-size: 12px; color: var(--muted); }
section { display: grid; gap: 14px; min-width: 0; }
.chart { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 4px 4px; min-height: 120px; min-width: 0; }
details { font-size: 14px; display: grid; gap: 8px; }
summary { cursor: pointer; color: var(--ink-2); padding-block: 4px; }
summary:focus-visible, a:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
details[open] summary { margin-bottom: 8px; }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; max-height: 560px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 7px 12px; text-align: left; border-bottom: 1px solid var(--grid); white-space: nowrap; vertical-align: top; }
th { position: sticky; top: 0; background: var(--surface); color: var(--ink-2); font-weight: 500; z-index: 1; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); }
tbody tr:last-child td { border-bottom: 0; }
.mono { font-family: var(--mono); }
.range { color: var(--muted); }
.note { font-size: 13px; color: var(--muted); }
.badge { display: inline-block; padding: 0 8px; border-radius: 999px; font: 500 12px/1.7 var(--mono); border: 1px solid currentColor; }
.badge.ok { color: var(--ok); }
.badge.warn { color: var(--warn); }
.badge.bad { color: var(--bad); }
ul.issues { margin: 0; padding-left: 16px; color: var(--ink-2); white-space: normal; min-width: 260px; max-width: 460px; }
.picker { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; font-size: 13px; color: var(--ink-2); }
.picker label { display: inline-flex; gap: 8px; align-items: center; }
.picker select { font: 13px var(--sans); color: var(--ink); background: var(--surface); border: 1px solid var(--axis); border-radius: 6px; padding: 4px 8px; }
#definitions { scroll-margin-top: 16px; }
.defs { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 12px; }
.def { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; display: grid; gap: 8px; align-content: start; min-width: 0; }
.def:last-child:nth-child(odd) { grid-column: 1 / -1; }
.def h3 { margin: 0; font-size: 15px; font-weight: 600; }
.def p { font-size: 14px; }
.eq { overflow-x: auto; overflow-y: hidden; padding-block: 2px; color: var(--ink); }
mjx-container { color: inherit; }
mjx-container[display="true"] { margin: 4px 0 !important; }
.plotly-missing { padding: 16px; color: var(--ink-2); font-size: 14px; }
</style>
"""

PAGE_BODY = """
<div class="page">
  <header>
    <div class="eyebrow">@@EYEBROW@@</div>
    <h1>@@TITLE@@</h1>
    @@INTRO@@
    <dl class="meta">@@META@@</dl>
  </header>

  <div class="tiles">@@TILES@@</div>

  @@BODY@@

  <p class="note">Generated by @@GENERATOR@@. Charts use plotly.js @@PLOTLY_VERSION@@.</p>
</div>

<script>
window.MathJax = { tex: { inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] }, svg: { fontCache: "global" } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js" async></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@@@PLOTLY_VERSION@@/plotly.min.js"></script>
<script>
(function () {
  const FIGURES = @@FIGURES@@;
  const DARK = @@DARK_MAP@@;
  const SEQ_DARK = @@SEQ_DARK@@;
  const pattern = new RegExp(Object.keys(DARK).join("|"), "gi");
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  function isDark() {
    const theme = document.documentElement.getAttribute("data-theme");
    return theme ? theme === "dark" : media.matches;
  }

  function draw(el) {
    const figure = FIGURES[el.dataset.fig];
    if (!figure) {
      if (window.Plotly) Plotly.purge(el);
      el.innerHTML = '<div class="plotly-missing">Not available for this selection.</div>';
      return;
    }
    const missing = el.querySelector(".plotly-missing");
    if (missing) missing.remove();
    const dark = isDark();
    let text = JSON.stringify(figure);
    if (dark) text = text.replace(pattern, (m) => DARK[m.toLowerCase()]);
    const fig = JSON.parse(text);
    if (dark) fig.data.forEach((trace) => { if (trace.meta === "seq") trace.colorscale = SEQ_DARK; });
    Plotly.react(el, fig.data, fig.layout, config);
  }

  function charts() { return document.querySelectorAll(".chart[data-fig]"); }

  function render() {
    if (!window.Plotly) {
      charts().forEach((el) => {
        el.innerHTML = '<div class="plotly-missing">This chart needs plotly.js from cdn.jsdelivr.net, which did not load. The tables hold the same numbers.</div>';
      });
      return;
    }
    // charts inside closed <details> are drawn when opened, so they get the right width
    charts().forEach((el) => { if (!el.closest("details:not([open])")) draw(el); });
  }

  document.querySelectorAll("details").forEach((details) => {
    details.addEventListener("toggle", () => {
      if (details.open && window.Plotly) details.querySelectorAll(".chart[data-fig]").forEach(draw);
    });
  });

  document.querySelectorAll(".picker").forEach((picker) => {
    const chart = document.getElementById(picker.dataset.chart);
    const selects = Array.from(picker.querySelectorAll("select"));
    selects.forEach((select) => select.addEventListener("change", () => {
      chart.dataset.fig = [picker.dataset.prefix, ...selects.map((s) => s.value)].join("|");
      if (window.Plotly) draw(chart);
    }));
  });

  render();
  media.addEventListener("change", render);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
</script>
"""

# dashboard page, see write_dashboard
PAGE_TEMPLATE = PAGE_HEAD + PAGE_BODY


def page_head(title: str) -> str:
    """HTML head of a report page (metadata, fonts and the shared stylesheet) with the given title."""
    return PAGE_HEAD.replace("@@TITLE@@", escape(title))
