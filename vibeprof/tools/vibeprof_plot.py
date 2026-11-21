#!/usr/bin/env python3
"""vibeprof visualization: a self-contained interactive HTML report, plus optional static PNGs.

    vibeprof_plot.py PROFILE_DIR [-o OUT_DIR] [--png]

The HTML report has no external dependencies (compute nodes are often offline): inline SVG charts
with hover tooltips, light/dark themes and a table view for every chart. Colors keep one meaning
across the report: blue = computation, orange = time inside MPI operations (their own cost),
aqua = waiting inside MPI for other ranks. Heatmaps use blue<->red around 1.0 (rank vs. average) and
a single-hue blue ramp for traffic volume.
"""
import argparse
import html
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vibeprof_analyze as LA  # noqa: E402

MAX_RANK_AXIS = 128   # beyond this, rank-axis charts aggregate per node


# ------------------------------------------------------------------------------------------------
def chart_data(A, M, F):
    n, wall = A["n_ranks"], A["wall"]
    nodes, node_of = A["nodes"], A["node_of"]
    by_node = n > MAX_RANK_AXIS
    g = lambda k: M.get(k)["value"] if M.get(k) else None

    tiles = [
        {"label": "Wall time", "value": LA.fmt(g("run.wall"), "s"), "id": M.get("run.wall")["id"]},
        {"label": "Ranks / nodes", "value": "%d / %d" % (n, len(nodes)), "id": M.get("run.ranks")["id"]},
        {"label": "Rank time inside MPI", "value": LA.fmt(g("run.mpi_frac"), "%"), "id": M.get("run.mpi_frac")["id"]},
        {"label": "Time in named phases", "value": LA.fmt(g("run.marker_coverage"), "%"), "id": M.get("run.marker_coverage")["id"]},
    ]
    G = A.get("gpu")
    if G:
        tiles.insert(3, {"label": "GPU busy (kernels)", "value": LA.fmt(g("gpu.util_mean"), "%"), "id": M.get("gpu.util_mean")["id"]})
    top = next((f for f in F if not f.get("info") and not f.get("symptom")), None)
    if top:
        tiles.append({"label": "Biggest opportunity", "value": "≈" + LA.fmt(top["impact_pct"], "%") + " of run", "id": top["id"],
                      "note": top["title"]})

    # phases: exclusive mean time split into computation / MPI intrinsic (fastest rank) / waiting
    phases = []
    for p in A["phases"]:
        comp = statistics.mean(p["comp"]) + (statistics.mean(p["cuda"]) if G else 0.0)
        intr = min(p["mpi"])
        wait = statistics.mean(p["mpi"]) - intr
        if comp + intr + wait < 0.005 * wall:
            continue
        phases.append({"label": p["path"], "comp": comp, "mpi": intr, "wait": wait,
                       "imbalance": max(p["comp"]) / comp if comp > 0 else 1.0, "slowest": p["slowest_rank"]})
    phases.sort(key=lambda r: -(r["comp"] + r["mpi"] + r["wait"]))
    phases = phases[:15]

    # per-rank decomposition: intrinsic MPI = sum over phases of the fastest rank's MPI time
    all_paths = A["phases"]
    intr_total = sum(min(p["mpi"]) for p in all_paths)
    ranks = [{"label": str(i), "node": nodes[node_of[i]], "comp": A["rank_comp"][i] + (G["ranks"][i]["api_total"] if G else 0.0),
              "mpi": min(intr_total, A["rank_mpi"][i]), "wait": max(0.0, A["rank_mpi"][i] - intr_total)} for i in range(n)]
    if by_node:
        agg = []
        for k, name in enumerate(nodes):
            mem = [r for i, r in enumerate(ranks) if node_of[i] == k]
            agg.append({"label": name, "node": name, **{f: statistics.mean(r[f] for r in mem) for f in ("comp", "mpi", "wait")}})
        ranks = agg

    # imbalance heatmap: computation relative to the phase mean
    sig = [p for p in A["phases"] if statistics.mean(p["comp"]) >= 0.01 * wall][:10]
    cols = [str(i) for i in range(n)] if not by_node else nodes
    heat = []
    for p in sig:
        mean = statistics.mean(p["comp"])
        vals = p["comp"] if not by_node else [statistics.mean(p["comp"][i] for i in range(n) if node_of[i] == k) for k in range(len(nodes))]
        heat.append({"label": p["path"], "values": [v / mean if mean > 0 else 1.0 for v in vals], "abs": vals})

    # communication matrix (bytes), per node if large
    mat = A["matrix"]
    if by_node:
        nn = len(nodes)
        agg = [[0] * nn for _ in range(nn)]
        for i in range(n):
            for j in range(n):
                agg[node_of[i]][node_of[j]] += mat[i][j]
        mat = agg

    # message-size histograms for the top byte-carrying operations
    hists = []
    for gop in A["ops"]:
        if sum(gop["bytes"]) and gop["hist"] and len(hists) < 3:
            hb = dict(gop["hist"])
            lo, hi = min(hb), max(hb)
            hists.append({"op": "MPI_" + gop["op"], "buckets": [{"b": b, "count": hb.get(b, 0)} for b in range(lo, hi + 1)]})

    trends = []
    for p in A["phases"]:
        if "series" in p and len(trends) < 4:
            trends.append({"label": p["path"], "max": p["series"]["max"], "mean": p["series"]["mean"]})

    sites = []
    for a in A["sites"][:10]:
        t = a["time"]
        sites.append({"label": "MPI_%s · %s" % (a["op"], a["site"]), "phase": a["phase"], "mpi": min(t),
                      "wait": statistics.mean(t) - min(t), "calls": sum(a["count"]), "bytes": sum(a["bytes"])})

    flagged = [f.get("node") for f in F if f["category"] == "slow_node"]
    node_bars = [{"label": nm, "comp": A["node_comp"][k], "flagged": nm in flagged} for k, nm in enumerate(nodes)] if len(nodes) > 1 else []

    finds = [{k: f.get(k) for k in ("id", "title", "text", "suggestion", "impact_pct", "info", "symptom", "category")} for f in F]
    metrics = [{"id": m["id"], "label": m["label"], "value": LA.fmt(m["value"], m["unit"]),
                "scope": m.get("phase") or m.get("op") or m.get("site") or m.get("node") or (("rank %s" % m["rank"]) if "rank" in m else "run")}
               for m in M.items]
    return {"tiles": tiles, "phases": phases, "ranks": ranks, "rank_axis": "node" if by_node else "rank", "gpu": gpu_chart_data(A, M, by_node),
            "heat": {"rows": heat, "cols": cols, "axis": "node" if by_node else "rank"},
            "matrix": {"labels": cols, "values": mat, "axis": "node" if by_node else "rank"},
            "hists": hists, "trends": trends, "sites": sites, "nodes": node_bars, "findings": finds, "metrics": metrics,
            "program": A["run"]["cmdline"] or A["run"]["exe"], "mpi_library": A["run"]["mpi_library"], "warnings": A["warnings"]}


def gpu_chart_data(A, M, by_node):
    G = A.get("gpu")
    if not G:
        return None
    n, wall, nodes, node_of = A["n_ranks"], A["wall"], A["nodes"], A["node_of"]
    R = G["ranks"]
    mean = statistics.mean
    phases = []
    for p in A["phases"]:
        if "gpu" not in p:
            continue
        gp = p["gpu"]
        busy, idle = mean(gp["busy"]), mean(gp["idle"])
        if busy + idle < 0.005 * wall:
            continue
        phases.append({"label": p["path"], "busy": busy, "idle": idle, "kernel": mean(gp["kernel"]), "cuda": mean(p["cuda"]),
                       "cats": {c: mean(gp["api"][c]) for c in LA.API_CATS}, "kernels": mean(gp["kcount"])})
    phases.sort(key=lambda r: -(r["busy"] + r["idle"]))
    cuda = sorted(({"label": r["label"], "cuda": r["cuda"], "cats": r["cats"]} for r in phases if r["cuda"] > 0), key=lambda r: -r["cuda"])[:12]
    ranks = [{"label": str(i), "node": nodes[node_of[i]], "util": R[i]["util"], "kernel": R[i]["kernel_time"],
              "devices": ", ".join(u[0] for u in R[i]["used"])} for i in range(n)]
    if by_node:
        ranks = [{"label": nm, "node": nm, "util": mean(r["util"] for i, r in enumerate(ranks) if node_of[i] == k),
                  "kernel": mean(r["kernel"] for i, r in enumerate(ranks) if node_of[i] == k), "devices": ""} for k, nm in enumerate(nodes)]
    tl = [d["timeline"] for d in R if d.get("timeline") and d["timeline"]["busy"]]
    timeline = None
    if tl:
        k = min(len(t["busy"]) for t in tl)
        cols = [[t["busy"][j] for t in tl] for j in range(k)]
        timeline = {"step": tl[0]["step"], "mean": [100 * mean(c) for c in cols], "min": [100 * min(c) for c in cols]}
    kernels = [{"label": a["display"], "phase": a["phase"], "time": mean(a["time"]), "count": a["count"],
                "dur": sum(a["time"]) / a["count"] if a["count"] else 0.0} for a in G["kernels"][:10]]
    copies = [{"label": "%s%s%s" % (e["kind"], " · " + e["host"] if e["host"] else "", " · by MPI" if e["by_mpi"] else ""),
               "time": e["time"], "bytes": e["bytes"], "count": e["count"]} for e in G["copy_table"][:8]]
    devs = {}
    for i in range(n):
        for host, pci, name in G["dev_of"][i]:
            devs.setdefault((host, pci, name), []).append(i)
    devices = [{"host": h_, "pci": p_, "name": nm, "ranks": rs} for (h_, p_, nm), rs in sorted(devs.items())]
    return {"phases": phases[:15], "cuda": cuda, "ranks": ranks, "timeline": timeline, "kernels": kernels, "copies": copies,
            "devices": devices}


# ------------------------------------------------------------------------------------------------
def md_to_html(md, known_ids=()):
    """Minimal, safe Markdown (headings, lists, bold/italic/code, paragraphs); [M12]/[F3] become links."""
    known = set(known_ids)

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", t)

        def cite(m):
            ids = [x.strip() for x in m.group(1).split(",")]
            parts = []
            for x in ids:
                if x in known:
                    parts.append('<a class="cite" href="#%s">%s</a>' % (x, x))
                else:
                    parts.append('<span class="cite bad" title="unknown id">%s</span>' % x)
            return "[" + ", ".join(parts) + "]"
        return re.sub(r"\[((?:[MF]\d+)(?:\s*,\s*[MF]\d+)*)\]", cite, t)

    out, para, list_tag = [], [], None

    def flush_para():
        if para:
            out.append("<p>%s</p>" % inline(" ".join(para)))
            para.clear()

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append("</li></%s>" % list_tag)
            list_tag = None

    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_para()
            continue
        h = re.match(r"^(#{1,4})\s+(.*)", line)
        ol = re.match(r"^\s*\d+[.)]\s+(.*)", line)
        ul = re.match(r"^\s*[-*]\s+(.*)", line)
        if h:
            flush_para(); close_list()
            lvl = min(len(h.group(1)) + 1, 4)
            out.append("<h%d>%s</h%d>" % (lvl, inline(h.group(2)), lvl))
        elif ol or ul:
            flush_para()
            tag = "ol" if ol else "ul"
            if list_tag != tag:
                close_list()
                out.append("<%s><li>" % tag)
                list_tag = tag
            else:
                out.append("</li><li>")
            out.append(inline((ol or ul).group(1)))
        elif list_tag and raw.startswith(("   ", "\t")):
            out.append("<br>" + inline(line.strip()))
        else:
            close_list()
            para.append(line.strip())
    flush_para(); close_list()
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------
CSS = r"""
.viz-root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --div-low: #2a78d6; --div-mid: #f0efec; --div-high: #e34948;
  --seq-lo: #cde2fb; --seq-hi: #0d366b;
  --deemph: #c3c2b7;
  --critical: #d03b3b; --warning: #fab219; --good: #0ca30c;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --div-low: #3987e5; --div-mid: #383835; --div-high: #e66767;
    --seq-lo: #184f95; --seq-hi: #cde2fb;
    --deemph: #52514e;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --div-low: #3987e5; --div-mid: #383835; --div-high: #e66767;
  --seq-lo: #184f95; --seq-hi: #cde2fb;
  --deemph: #52514e;
}
html, body { margin: 0; }
body { background: #f9f9f7; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) body { background: #0d0d0d; } }
:root[data-theme="dark"] body { background: #0d0d0d; }
.viz-root { background: var(--page); color: var(--text-primary); min-height: 100vh;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 24px 16px 64px; }
header h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
header .sub { color: var(--text-secondary); font-size: 13px; word-break: break-all; }
.toolbar { display: flex; gap: 8px; align-items: center; margin: 12px 0 20px; flex-wrap: wrap; }
.toolbar button { font: inherit; font-size: 13px; color: var(--text-primary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 20px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 22px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--text-secondary); font-size: 12px; margin-top: 2px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin-bottom: 16px; }
.card h2 { font-size: 16px; font-weight: 600; margin: 0 0 2px; }
.card .desc { color: var(--text-secondary); font-size: 13px; margin: 0 0 10px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr)); gap: 16px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--text-secondary); margin: 4px 0 6px; }
.legend .key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.legend .lkey { display: inline-block; width: 14px; height: 2px; margin-right: 6px; vertical-align: 3px; border-radius: 1px; }
.chart { width: 100%; position: relative; }
.chart svg { display: block; width: 100%; overflow: visible; }
.chart text { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart text.lbl { fill: var(--text-secondary); font-variant-numeric: normal; }
.chart text.val { fill: var(--text-secondary); }
.mark { cursor: default; }
.mark:hover, .mark:focus { opacity: 0.82; outline: none; }
details { margin-top: 8px; font-size: 13px; }
details summary { color: var(--text-secondary); cursor: pointer; }
table { border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 12.5px; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.tablewrap { overflow-x: auto; }
.finding { border-top: 1px solid var(--grid); padding: 10px 0; }
.finding:first-of-type { border-top: 0; }
.finding .t { font-weight: 600; }
.finding .meta { color: var(--text-secondary); font-size: 12.5px; }
.finding .sugg { margin-top: 4px; }
.badge { display: inline-block; font-size: 11.5px; padding: 1px 7px; border-radius: 9px; border: 1px solid var(--border);
  color: var(--text-secondary); margin-right: 6px; }
.prose h2 { font-size: 15px; margin: 14px 0 4px; } .prose h3 { font-size: 14px; margin: 12px 0 4px; }
.prose p, .prose li { margin: 4px 0; } .prose ol, .prose ul { padding-left: 22px; margin: 6px 0; }
.prose code, td code { font-size: 12.5px; background: var(--page); padding: 0 4px; border-radius: 4px; }
a.cite { color: var(--series-1); text-decoration: none; font-size: 12px; }
.cite.bad { color: var(--critical); font-size: 12px; }
.check { font-size: 13px; color: var(--text-secondary); margin-top: 10px; border-top: 1px solid var(--grid); padding-top: 8px; }
.check .ok::before { content: "✓ "; color: var(--good); font-weight: 700; }
.check .warn::before { content: "! "; color: var(--critical); font-weight: 700; }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 7px 9px; font-size: 12px; line-height: 1.45;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; max-width: 340px; }
#tip .row { display: flex; gap: 8px; align-items: center; }
#tip .v { font-weight: 600; font-variant-numeric: tabular-nums; }
#tip .n { color: var(--text-secondary); }
#tip .hd { color: var(--text-secondary); margin-bottom: 3px; }
.empty { color: var(--text-muted); font-size: 13px; }
"""

JS = r"""
(function () {
const D = JSON.parse(document.getElementById('data').textContent);
const root = document.querySelector('.viz-root');
const NS = 'http://www.w3.org/2000/svg';
const css = n => getComputedStyle(root).getPropertyValue(n).trim();
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const h = (tag, cls, text, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; if (parent) parent.appendChild(e); return e; };

// ---------- number formatting ----------
function fs(v) { if (v == null || !isFinite(v)) return 'n/a'; const a = Math.abs(v);
  if (a === 0) return '0 s'; if (a >= 1) return v.toPrecision(3) + ' s'; if (a >= 1e-3) return (v * 1e3).toPrecision(3) + ' ms'; return (v * 1e6).toPrecision(3) + ' µs'; }
function fb(v) { const u = [['GiB', 2 ** 30], ['MiB', 2 ** 20], ['KiB', 1024]]; for (const [n, d] of u) if (v >= d) return (v / d).toPrecision(3) + ' ' + n; return Math.round(v) + ' B'; }
function fx(v) { return v.toFixed(2) + '×'; }
function fn(v) { return Math.round(v).toLocaleString(); }
function niceScale(v, n) { // clean tick step (1/2/2.5/5 x 10^k), axis max = first tick >= v
  if (!(v > 0)) v = 1; const raw = v / n, p = Math.pow(10, Math.floor(Math.log10(raw))); let step = 10 * p;
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) { step = m * p; break; }
  const k = Math.max(1, Math.ceil(v / step - 1e-9)), t = []; for (let i = 0; i <= k; i++) t.push(+(i * step).toPrecision(12)); return { max: k * step, ticks: t }; }
function tk(str) { return str.replace(/(\.\d*?)0+(?=\s|$)/, '$1').replace(/\.(?=\s|$)/, ''); } // trim trailing zeros on axis ticks
function bucketLabel(b) { if (b === 0) return '0 B'; const lo = 2 ** (b - 1); return fb(lo).replace('.00', ''); }

// ---------- colors: OKLab interpolation for the continuous scales ----------
function hex2rgb(x) { x = x.replace('#', ''); return [0, 2, 4].map(i => parseInt(x.substr(i, 2), 16) / 255); }
const lin = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const unlin = c => c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
function toLab(hx) { const [r, g, b] = hex2rgb(hx).map(lin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b), m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b),
        s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]; }
function fromLab([L, a, b]) { const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
  return '#' + rgb.map(c => Math.round(Math.max(0, Math.min(1, unlin(c))) * 255).toString(16).padStart(2, '0')).join(''); }
function mix(a, b, t) { const A = toLab(a), B = toLab(b); return fromLab(A.map((v, i) => v + (B[i] - v) * t)); }
function diverging(v, d) { // v = log2(ratio), d = domain half-width
  const t = Math.max(-1, Math.min(1, v / d)); return t < 0 ? mix(css('--div-mid'), css('--div-low'), -t) : mix(css('--div-mid'), css('--div-high'), t); }
function sequential(t) { return mix(css('--seq-lo'), css('--seq-hi'), Math.max(0, Math.min(1, t))); }

// ---------- tooltip ----------
const tip = document.getElementById('tip');
function showTip(ev, head, rows) { tip.replaceChildren(); if (head) h('div', 'hd', head, tip);
  for (const r of rows) { const row = h('div', 'row', null, tip);
    if (r.color) { const k = h('span', null, null, row); k.style.cssText = 'width:12px;height:2px;border-radius:1px;background:' + r.color; }
    h('span', 'v', r.value, row); h('span', 'n', r.name, row); }
  tip.style.display = 'block'; moveTip(ev); }
function moveTip(ev) { const pad = 14; let x = ev.clientX + pad, y = ev.clientY + pad; const w = tip.offsetWidth, hh = tip.offsetHeight;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad; if (y + hh > innerHeight - 8) y = ev.clientY - hh - pad; tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
function hideTip() { tip.style.display = 'none'; }
function hover(node, fn) { node.setAttribute('tabindex', '0'); node.classList.add('mark');
  node.addEventListener('pointermove', ev => fn(ev)); node.addEventListener('pointerleave', hideTip);
  node.addEventListener('focus', () => { const r = node.getBoundingClientRect(); fn({ clientX: r.right, clientY: r.top }); }); node.addEventListener('blur', hideTip); }

// rounded data-end, square at the baseline
function barPath(x, y, w, hgt, horizontal, roundEnd) { const r = roundEnd ? Math.min(4, horizontal ? w / 2 : hgt / 2, horizontal ? hgt / 2 : w / 2) : 0;
  if (w <= 0 || hgt <= 0) return '';
  if (horizontal) return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt - r}Q${x + w},${y + hgt} ${x + w - r},${y + hgt}H${x}Z`;
  return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`; }

const SEG = [['comp', 'Computation', '--series-1'], ['mpi', 'MPI (operation cost)', '--series-2'], ['wait', 'Waiting in MPI', '--series-3']];
function legend(parent, items, line) { const L = h('div', 'legend', null, parent);
  for (const [name, color] of items) { const s = h('span', null, null, L); const k = h('span', line ? 'lkey' : 'key', null, s); k.style.background = css(color); s.appendChild(document.createTextNode(name)); } }
function table(parent, cols, rows) { const d = h('details', null, null, parent); h('summary', null, 'Table view', d); const w = h('div', 'tablewrap', null, d);
  const t = h('table', null, null, w); const tr = h('tr', null, null, h('thead', null, null, t)); cols.forEach(c => h('th', c.num ? 'num' : '', c.name, tr));
  const tb = h('tbody', null, null, t); rows.forEach(r => { const row = h('tr', null, null, tb); r.forEach((v, i) => h('td', cols[i].num ? 'num' : '', v, row)); }); }

// ---------- horizontal stacked bars (phases, call sites) ----------
function hbars(box, rows, segs, valueFmt) {
  const W = Math.max(320, box.clientWidth), labelW = Math.min(300, Math.max(120, W * 0.34)), rowH = 30, barH = 18, padR = 70;
  const H = rows.length * rowH + 26; const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box);
  const tot = r => segs.reduce((s, k) => s + r[k[0]], 0); const sc = niceScale(Math.max(...rows.map(tot)), 4), max = sc.max;
  const x0 = labelW, x1 = W - padR, sx = v => (v / max) * (x1 - x0);
  for (const t of sc.ticks) { const X = x0 + sx(t); el('line', { x1: X, x2: X, y1: 0, y2: H - 22, stroke: css('--grid'), 'stroke-width': 1 }, svg);
    el('text', { x: X, y: H - 8, 'text-anchor': 'middle' }, svg).textContent = tk(valueFmt(t)); }
  rows.forEach((r, i) => { const y = i * rowH + (rowH - barH) / 2;
    const lab = el('text', { x: x0 - 8, y: y + barH / 2 + 4, 'text-anchor': 'end', class: 'lbl' }, svg);
    lab.textContent = r.label.length > 46 ? r.label.slice(0, 45) + '…' : r.label;
    let x = x0; const present = segs.filter(k => r[k[0]] > 0); const g = el('g', {}, svg);
    present.forEach((k, j) => { const w = sx(r[k[0]]); const last = j === present.length - 1; const ww = Math.max(0, w - (last ? 0 : 2));
      el('path', { d: barPath(x, y, ww, barH, true, last), fill: css(k[2]) }, g); x += w; });
    const hit = el('rect', { x: x0, y: i * rowH, width: Math.max(24, x - x0), height: rowH, fill: 'transparent' }, svg);
    hover(hit, ev => showTip(ev, r.label + (r.phase ? '  (phase ' + r.phase + ')' : ''),
      segs.map(k => ({ color: css(k[2]), value: valueFmt(r[k[0]]), name: k[1] })).concat(r.extra || [])));
    el('text', { x: x + 6, y: y + barH / 2 + 4, class: 'val' }, svg).textContent = valueFmt(tot(r)); });
}

// ---------- vertical stacked columns (per rank / node) ----------
function cols(box, rows, segs, nodeOf) {
  const W = Math.max(320, box.clientWidth), H = 240, padL = 52, padB = nodeOf ? 44 : 28, top = 10;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box);
  const tot = r => segs.reduce((s, k) => s + r[k[0]], 0); const sc = niceScale(Math.max(...rows.map(tot)), 4), max = sc.max;
  const plotW = W - padL - 8, band = plotW / rows.length, bw = Math.min(24, Math.max(1, band - 2)), y0 = H - padB, sy = v => (v / max) * (y0 - top);
  for (const t of sc.ticks) { const Y = y0 - sy(t); el('line', { x1: padL, x2: W - 8, y1: Y, y2: Y, stroke: css(t === 0 ? '--axis' : '--grid'), 'stroke-width': 1 }, svg);
    el('text', { x: padL - 6, y: Y + 4, 'text-anchor': 'end' }, svg).textContent = tk(fs(t)); }
  const every = Math.ceil(rows.length / Math.max(1, Math.floor(plotW / 28)));
  rows.forEach((r, i) => { const cx = padL + band * i + band / 2, x = cx - bw / 2; let y = y0; const present = segs.filter(k => r[k[0]] > 0);
    present.forEach((k, j) => { const hh = sy(r[k[0]]); const last = j === present.length - 1; const hv = Math.max(0, hh - (last ? 0 : (bw > 6 ? 2 : 0)));
      el('path', { d: barPath(x, y - hv, bw, hv, false, last), fill: css(k[2]) }, svg); y -= hh; });
    if (i % every === 0) el('text', { x: cx, y: y0 + 14, 'text-anchor': 'middle' }, svg).textContent = r.label;
    const hit = el('rect', { x: cx - Math.max(12, band / 2), y: top, width: Math.max(24, band), height: y0 - top, fill: 'transparent' }, svg);
    hover(hit, ev => showTip(ev, (D.rank_axis === 'node' ? 'Node ' : 'Rank ') + r.label + (r.node && D.rank_axis !== 'node' ? ' · ' + r.node : ''),
      segs.map(k => ({ color: css(k[2]), value: fs(r[k[0]]), name: k[1] })))); });
  if (nodeOf) { let start = 0; for (let i = 1; i <= rows.length; i++) if (i === rows.length || rows[i].node !== rows[start].node) {
      const xa = padL + band * start + 2, xb = padL + band * i - 2; el('line', { x1: xa, x2: xb, y1: y0 + 22, y2: y0 + 22, stroke: css('--axis'), 'stroke-width': 1 }, svg);
      el('text', { x: (xa + xb) / 2, y: y0 + 36, 'text-anchor': 'middle', class: 'lbl' }, svg).textContent = rows[start].node; start = i; } }
}

// ---------- heatmap ----------
function heatmap(box, rows, colLabels, colorOf, tipOf, axisName) {
  const W = Math.max(320, box.clientWidth), labelW = Math.min(240, Math.max(110, W * 0.26)), cell = Math.max(3, Math.min(34, (W - labelW - 10) / colLabels.length));
  const H = rows.length * Math.min(30, Math.max(cell, 16)) + 24, rowH = (H - 24) / rows.length;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box); const gap = cell > 8 ? 2 : 0;
  const every = Math.ceil(colLabels.length / Math.max(1, Math.floor((W - labelW) / 26)));
  rows.forEach((r, i) => { const lab = el('text', { x: labelW - 8, y: i * rowH + rowH / 2 + 4, 'text-anchor': 'end', class: 'lbl' }, svg);
    lab.textContent = r.label.length > 34 ? r.label.slice(0, 33) + '…' : r.label;
    r.values.forEach((v, j) => { const c = colorOf(v, i, j); if (!c) return;
      const rc = el('rect', { x: labelW + j * cell, y: i * rowH, width: Math.max(1, cell - gap), height: Math.max(1, rowH - gap), rx: cell > 10 ? 3 : 0, fill: c }, svg);
      hover(rc, ev => showTip(ev, tipOf.head(r, j), tipOf.rows(r, j))); }); });
  colLabels.forEach((c, j) => { if (j % every === 0) el('text', { x: labelW + j * cell + cell / 2, y: H - 6, 'text-anchor': 'middle' }, svg).textContent = c; });
  el('text', { x: W - 4, y: H - 6, 'text-anchor': 'end' }, svg).textContent = axisName + ' →';
}
function scaleLegend(parent, stops, labels) { const L = h('div', 'legend', null, parent); const bar = h('span', null, null, L);
  bar.style.cssText = `display:inline-block;width:160px;height:10px;border-radius:3px;background:linear-gradient(90deg,${stops.join(',')})`;
  h('span', null, labels, L); }

// ---------- lines with crosshair ----------
function lines(box, series, yFmt, xName, xFmt) {
  xFmt = xFmt || (i => String(i + 1));
  const W = Math.max(280, box.clientWidth), H = 190, padL = 56, padB = 26, top = 10, n = Math.max(...series.map(s => s.values.length));
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box); const sc = niceScale(Math.max(...series.flatMap(s => s.values)), 3), max = sc.max;
  const x0 = padL, x1 = W - 10, y0 = H - padB, sx = i => x0 + (n > 1 ? (i / (n - 1)) * (x1 - x0) : 0), sy = v => y0 - (v / max) * (y0 - top);
  for (const t of sc.ticks) { const Y = sy(t); el('line', { x1: x0, x2: x1, y1: Y, y2: Y, stroke: css(t === 0 ? '--axis' : '--grid'), 'stroke-width': 1 }, svg);
    el('text', { x: x0 - 6, y: Y + 4, 'text-anchor': 'end' }, svg).textContent = tk(yFmt(t)); }
  [0, Math.floor((n - 1) / 2), n - 1].forEach(i => el('text', { x: sx(i), y: H - 8, 'text-anchor': 'middle' }, svg).textContent = xFmt(i));
  for (const s of series) el('path', { d: s.values.map((v, i) => (i ? 'L' : 'M') + sx(i) + ',' + sy(v)).join(''), fill: 'none', stroke: css(s.color), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
  const cross = el('line', { y1: top, y2: y0, stroke: css('--axis'), 'stroke-width': 1, visibility: 'hidden' }, svg);
  const dots = series.map(s => el('circle', { r: 4, fill: css(s.color), stroke: css('--surface-1'), 'stroke-width': 2, visibility: 'hidden' }, svg));
  const hit = el('rect', { x: x0, y: top, width: x1 - x0, height: y0 - top, fill: 'transparent' }, svg);
  const at = ev => { const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * (W / r.width); return Math.max(0, Math.min(n - 1, Math.round((px - x0) / ((x1 - x0) / Math.max(1, n - 1))))); };
  const show = ev => { const i = at(ev); cross.setAttribute('x1', sx(i)); cross.setAttribute('x2', sx(i)); cross.setAttribute('visibility', 'visible');
    series.forEach((s, k) => { dots[k].setAttribute('cx', sx(i)); dots[k].setAttribute('cy', sy(s.values[i])); dots[k].setAttribute('visibility', 'visible'); });
    showTip(ev, xName + ' ' + xFmt(i), series.map(s => ({ color: css(s.color), value: yFmt(s.values[i]), name: s.name }))); };
  hit.setAttribute('tabindex', '0'); hit.addEventListener('pointermove', show);
  hit.addEventListener('pointerleave', () => { hideTip(); cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); });
}

// ---------- single-series columns (message sizes, nodes) ----------
function columns(box, items, colorOf, yFmt, tipHead) {
  const W = Math.max(240, box.clientWidth), H = 170, padL = 48, padB = 26, top = 16;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box); const sc = niceScale(Math.max(...items.map(i => i.value)), 3), max = sc.max;
  const band = (W - padL - 6) / items.length, bw = Math.min(24, Math.max(2, band - 2)), y0 = H - padB, sy = v => (v / max) * (y0 - top);
  for (const t of sc.ticks) { const Y = y0 - sy(t); el('line', { x1: padL, x2: W - 6, y1: Y, y2: Y, stroke: css(t === 0 ? '--axis' : '--grid'), 'stroke-width': 1 }, svg);
    el('text', { x: padL - 6, y: Y + 4, 'text-anchor': 'end' }, svg).textContent = tk(yFmt(t)); }
  const every = Math.ceil(items.length / Math.max(1, Math.floor((W - padL) / 44)));
  items.forEach((it, i) => { const cx = padL + band * i + band / 2, hh = sy(it.value);
    if (hh > 0) el('path', { d: barPath(cx - bw / 2, y0 - hh, bw, hh, false, true), fill: colorOf(it) }, svg);
    if (i % every === 0) el('text', { x: cx, y: y0 + 14, 'text-anchor': 'middle' }, svg).textContent = it.label;
    if (it.note) el('text', { x: cx, y: y0 - hh - 5, 'text-anchor': 'middle', class: 'val' }, svg).textContent = it.note;
    const hit = el('rect', { x: cx - Math.max(12, band / 2), y: top, width: Math.max(24, band), height: y0 - top, fill: 'transparent' }, svg);
    hover(hit, ev => showTip(ev, tipHead(it), [{ value: yFmt(it.value), name: it.what || '' }])); });
}

// ---------- page ----------
function card(parent, title, desc) { const c = h('section', 'card', null, parent); h('h2', null, title, c); if (desc) h('p', 'desc', desc, c); return c; }

const fpct = v => (v >= 10 || v === 0 ? v.toFixed(0) : v.toFixed(1)) + '%';
function render() {
  hideTip(); const Q = []; const draw = (f, ...a) => Q.push(() => f(...a));
  SEG[0][1] = D.gpu ? 'Computation (CPU work + CUDA calls)' : 'Computation';
  const main = document.getElementById('charts'); main.replaceChildren();

  // phases
  let c = card(main, 'Where the time goes, by phase', 'Mean time per rank in each phase (exclusive of nested phases). Orange is the cost the fastest rank pays inside MPI; aqua is extra time ranks spend waiting for slower ones.');
  if (D.phases.length) { legend(c, SEG.map(s => [s[1], s[2]])); const b = h('div', 'chart', null, c);
    draw(hbars, b, D.phases.map(p => Object.assign({ extra: [{ value: fx(p.imbalance), name: 'computation imbalance (max/mean), slowest rank ' + p.slowest }] }, p)), SEG, fs);
    table(c, [{ name: 'phase' }, { name: 'computation', num: 1 }, { name: 'MPI cost', num: 1 }, { name: 'waiting', num: 1 }, { name: 'imbalance', num: 1 }],
      D.phases.map(p => [p.label, fs(p.comp), fs(p.mpi), fs(p.wait), fx(p.imbalance)])); } else h('p', 'empty', 'No phase markers recorded.', c);

  // per rank
  c = card(main, D.rank_axis === 'node' ? 'Time per node (mean of its ranks)' : 'Time per rank', 'Computation, MPI cost and waiting for each ' + D.rank_axis + '. Uneven blue bars mean uneven work; tall aqua bars mean a rank waits for the others.');
  legend(c, SEG.map(s => [s[1], s[2]])); let b = h('div', 'chart', null, c); draw(cols, b, D.ranks, SEG, D.rank_axis === 'rank' && new Set(D.ranks.map(r => r.node)).size > 1);
  table(c, [{ name: D.rank_axis }, { name: 'node' }, { name: 'computation', num: 1 }, { name: 'MPI cost', num: 1 }, { name: 'waiting', num: 1 }], D.ranks.map(r => [r.label, r.node, fs(r.comp), fs(r.mpi), fs(r.wait)]));

  if (D.gpu) renderGpu(main, draw);

  const two = h('div', 'grid2', null, main);
  // imbalance heatmap
  c = card(two, 'Computation imbalance', 'Each cell: a ' + D.heat.axis + "'s computation in a phase relative to that phase's mean. Red = slower than average, blue = faster.");
  if (D.heat.rows.length) { const all = D.heat.rows.flatMap(r => r.values.map(v => Math.abs(Math.log2(Math.max(1e-9, v)))));
    const d = Math.min(1, Math.max(Math.log2(1.25), ...all)); const lo = Math.pow(2, -d), hi = Math.pow(2, d);
    scaleLegend(c, [css('--div-low'), css('--div-mid'), css('--div-high')], fx(lo) + ' · 1× · ' + fx(hi)); b = h('div', 'chart', null, c);
    draw(heatmap, b, D.heat.rows, D.heat.cols, v => diverging(Math.log2(Math.max(1e-9, v)), d),
      { head: (r, j) => r.label + ' · ' + D.heat.axis + ' ' + D.heat.cols[j], rows: (r, j) => [{ value: fx(r.values[j]), name: 'of the phase mean' }, { value: fs(r.abs[j]), name: 'computation' }] }, D.heat.axis);
    table(c, [{ name: 'phase' }].concat(D.heat.cols.map(x => ({ name: x, num: 1 }))), D.heat.rows.map(r => [r.label].concat(r.values.map(v => v.toFixed(2)))));
  } else h('p', 'empty', 'No phase with significant computation.', c);

  // communication matrix
  c = card(two, 'Point-to-point traffic', 'Bytes sent from each ' + D.matrix.axis + ' (row) to each ' + D.matrix.axis + ' (column), log scale.');
  const flat = D.matrix.values.flat().filter(v => v > 0);
  if (flat.length) { const lmin = Math.log10(Math.min(...flat)), lmax = Math.log10(Math.max(...flat)), span = Math.max(1e-9, lmax - lmin);
    scaleLegend(c, [sequential(0), sequential(0.5), sequential(1)], (span > 1e-6 ? fb(Math.pow(10, lmin)) + ' → ' + fb(Math.pow(10, lmax)) : 'all pairs ' + fb(Math.pow(10, lmax)))); b = h('div', 'chart', null, c);
    draw(heatmap, b, D.matrix.values.map((row, i) => ({ label: 'from ' + D.matrix.labels[i], values: row })), D.matrix.labels,
      v => v > 0 ? sequential(span > 1e-6 ? (Math.log10(v) - lmin) / span : 1) : null,
      { head: (r, j) => r.label + ' to ' + D.matrix.labels[j], rows: (r, j) => [{ value: fb(r.values[j]), name: 'sent' }] }, 'to ' + D.matrix.axis);
    table(c, [{ name: 'from \\ to' }].concat(D.matrix.labels.map(x => ({ name: x, num: 1 }))), D.matrix.values.map((row, i) => [D.matrix.labels[i]].concat(row.map(v => v ? fb(v) : ''))));
  } else h('p', 'empty', 'No point-to-point messages recorded.', c);

  // call sites
  c = card(main, 'Top MPI call sites', 'Mean time per rank at each call site: the operation cost (fastest rank) and the extra waiting of the average rank.');
  if (D.sites.length) { legend(c, SEG.slice(1).map(s => [s[1], s[2]])); b = h('div', 'chart', null, c);
    draw(hbars, b, D.sites.map(s => Object.assign({ extra: [{ value: fn(s.calls), name: 'calls (all ranks)' }].concat(s.bytes ? [{ value: fb(s.bytes), name: 'payload' }] : []) }, s)), SEG.slice(1), fs);
    table(c, [{ name: 'call site' }, { name: 'phase' }, { name: 'MPI cost', num: 1 }, { name: 'waiting', num: 1 }, { name: 'calls', num: 1 }, { name: 'bytes', num: 1 }],
      D.sites.map(s => [s.label, s.phase, fs(s.mpi), fs(s.wait), fn(s.calls), s.bytes ? fb(s.bytes) : ''])); }

  // trends
  if (D.trends.length) { c = card(main, 'Phase duration over the run', 'Duration of each instance of a phase: the slowest rank (blue) and the average rank (gray). A rising line means the phase gets slower.');
    legend(c, [['Slowest rank', '--series-1'], ['Average rank', '--deemph']], true); const g2 = h('div', 'grid2', null, c);
    for (const t of D.trends) { const w = h('div', null, null, g2); h('div', 'legend', t.label, w); const bb = h('div', 'chart', null, w);
      draw(lines, bb, [{ name: 'slowest rank', values: t.max, color: '--series-1' }, { name: 'average rank', values: t.mean, color: '--deemph' }], fs, 'Instance'); }
    table(c, [{ name: 'phase' }, { name: 'instance', num: 1 }, { name: 'slowest rank', num: 1 }, { name: 'average rank', num: 1 }],
      D.trends.flatMap(t => t.max.map((v, i) => [t.label, String(i + 1), fs(v), fs(t.mean[i])]))); }

  const two2 = h('div', 'grid2', null, main);
  // message sizes
  if (D.hists.length) { c = card(two2, 'Message sizes', 'Number of messages per size bucket (bucket = lower bound, powers of two).');
    for (const hs of D.hists) { h('div', 'legend', hs.op, c); const bb = h('div', 'chart', null, c);
      draw(columns, bb, hs.buckets.map(x => ({ label: bucketLabel(x.b), value: x.count, what: 'messages' })), () => css('--series-2'), fn, it => hs.op + ' · ≥ ' + it.label); }
    table(c, [{ name: 'operation' }, { name: 'size ≥', num: 1 }, { name: 'messages', num: 1 }], D.hists.flatMap(hs => hs.buckets.filter(x => x.count).map(x => [hs.op, bucketLabel(x.b), fn(x.count)]))); }
  // nodes
  if (D.nodes.length) { c = card(two2, 'Computation per node', 'Mean computation per rank on each node' + (D.nodes.some(x => x.flagged) ? '; the node flagged as slow is highlighted.' : '.'));
    const bb = h('div', 'chart', null, c);
    draw(columns, bb, D.nodes.map(x => ({ label: x.label, value: x.comp, note: x.flagged ? 'flagged' : '', flagged: x.flagged, what: 'mean computation per rank' })),
      it => (D.nodes.some(x => x.flagged) && !it.flagged) ? css('--deemph') : css('--series-1'), fs, it => 'Node ' + it.label);
    table(c, [{ name: 'node' }, { name: 'computation per rank', num: 1 }, { name: 'flagged' }], D.nodes.map(x => [x.label, fs(x.comp), x.flagged ? 'slow node' : ''])); }
  for (const q of Q) q();
}

// ---------- GPU section ----------
function renderGpu(main, draw) {
  const Gd = D.gpu;
  const GSEG = [['busy', 'GPU busy (kernels)', '--series-1'], ['idle', 'GPU idle', '--deemph']];
  let c = card(main, 'GPU busy and idle, by phase', 'Mean time per rank while the host is in each phase: how much of it the GPU spends running kernels, and how much it sits idle. Long gray bars are where the GPU waits for the host, MPI or copies.');
  if (Gd.phases.length) { legend(c, GSEG.map(s => [s[1], s[2]])); const b = h('div', 'chart', null, c);
    draw(hbars, b, Gd.phases.map(p => Object.assign({ extra: [{ value: fs(p.kernel), name: 'kernel time launched here' }, { value: fs(p.cuda), name: 'host time in CUDA calls' }, { value: fn(p.kernels), name: 'kernel launches per rank' }] }, p)), GSEG, fs);
    table(c, [{ name: 'phase' }, { name: 'GPU busy', num: 1 }, { name: 'GPU idle', num: 1 }, { name: 'kernels launched here', num: 1 }, { name: 'CUDA calls (host)', num: 1 }],
      Gd.phases.map(p => [p.label, fs(p.busy), fs(p.idle), fs(p.kernel), fs(p.cuda)])); } else h('p', 'empty', 'No GPU activity inside phases.', c);

  let two = h('div', 'grid2', null, main);
  c = card(two, D.rank_axis === 'node' ? 'GPU busy per node' : 'GPU busy per rank', 'Share of the run the ' + D.rank_axis + "'s GPU runs kernels.");
  let b = h('div', 'chart', null, c);
  draw(columns, b, Gd.ranks.map(r => ({ label: r.label, value: r.util, what: 'of the run busy' })), () => css('--series-1'), fpct,
       it => (D.rank_axis === 'node' ? 'Node ' : 'Rank ') + it.label);
  table(c, [{ name: D.rank_axis }, { name: 'node' }, { name: 'GPU busy', num: 1 }, { name: 'kernel time', num: 1 }, { name: 'GPU' }],
    Gd.ranks.map(r => [r.label, r.node, fpct(r.util), fs(r.kernel), r.devices]));

  c = card(two, 'Host time in CUDA calls, by phase', 'Mean time per rank spent inside CUDA API calls: waiting for the GPU, blocking copies, launches and allocation.');
  if (Gd.cuda.length) { b = h('div', 'chart', null, c);
    draw(hbars, b, Gd.cuda.map(p => Object.assign({ extra: [['sync', 'synchronizing calls'], ['copy', 'copies/memsets'], ['launch', 'kernel launches'], ['alloc', 'allocation'], ['other', 'other']]
      .filter(([k]) => p.cats[k] > 0).map(([k, nm]) => ({ value: fs(p.cats[k]), name: nm })) }, p)), [['cuda', 'CUDA calls', '--series-1']], fs);
    table(c, [{ name: 'phase' }, { name: 'total', num: 1 }, { name: 'sync', num: 1 }, { name: 'copies', num: 1 }, { name: 'launch', num: 1 }, { name: 'alloc', num: 1 }],
      Gd.cuda.map(p => [p.label, fs(p.cuda), fs(p.cats.sync), fs(p.cats.copy), fs(p.cats.launch), fs(p.cats.alloc)])); } else h('p', 'empty', 'No CUDA calls recorded.', c);

  if (Gd.timeline && Gd.timeline.mean.length > 2) {
    c = card(main, 'GPU busy over the run', 'Share of each time interval in which the GPUs run kernels: mean over ranks (blue) and the least busy rank (gray).');
    legend(c, [['Mean over ranks', '--series-1'], ['Least busy rank', '--deemph']], true); b = h('div', 'chart', null, c);
    const st = Gd.timeline.step, tf = i => fs(i * st);
    draw(lines, b, [{ name: 'mean over ranks', values: Gd.timeline.mean, color: '--series-1' }, { name: 'least busy rank', values: Gd.timeline.min, color: '--deemph' }], fpct, 't =', tf);
    table(c, [{ name: 'time', num: 1 }, { name: 'mean busy', num: 1 }, { name: 'least busy rank', num: 1 }], Gd.timeline.mean.map((v, i) => [tf(i), fpct(v), fpct(Gd.timeline.min[i])]));
  }

  two = h('div', 'grid2', null, main);
  c = card(two, 'Top GPU kernels', 'Mean GPU time per rank by kernel (Kokkos label when available).');
  if (Gd.kernels.length) { b = h('div', 'chart', null, c);
    draw(hbars, b, Gd.kernels.map(k => ({ label: k.label, phase: k.phase, time: k.time, extra: [{ value: fn(k.count), name: 'launches (all ranks)' }, { value: fs(k.dur), name: 'mean duration' }] })), [['time', 'GPU time', '--series-1']], fs);
    table(c, [{ name: 'kernel' }, { name: 'phase' }, { name: 'GPU time per rank', num: 1 }, { name: 'launches', num: 1 }, { name: 'mean duration', num: 1 }],
      Gd.kernels.map(k => [k.label, k.phase, fs(k.time), fn(k.count), fs(k.dur)])); } else h('p', 'empty', 'No kernels recorded.', c);
  c = card(two, 'Memory copies', 'GPU time per rank for copies by direction and host memory kind (pageable memory is slow; "by MPI" = issued inside MPI calls).');
  if (Gd.copies.length) { b = h('div', 'chart', null, c);
    draw(hbars, b, Gd.copies.map(k => ({ label: k.label, time: k.time, extra: [{ value: fb(k.bytes), name: 'bytes (all ranks)' }, { value: fn(k.count), name: 'copies (all ranks)' }] })), [['time', 'copy time', '--series-1']], fs);
    table(c, [{ name: 'copy' }, { name: 'time per rank', num: 1 }, { name: 'bytes', num: 1 }, { name: 'copies', num: 1 }], Gd.copies.map(k => [k.label, fs(k.time), fb(k.bytes), fn(k.count)]));
  } else h('p', 'empty', 'No copies recorded.', c);
  if (Gd.devices.length) {
    c = card(main, 'GPUs used', 'Which GPU each rank ran its kernels on (host and PCI bus id).');
    table(c, [{ name: 'host' }, { name: 'PCI bus' }, { name: 'GPU' }, { name: 'ranks' }], Gd.devices.map(d => [d.host, d.pci, d.name, d.ranks.join(', ')]));
    c.querySelector('details').open = Gd.devices.length <= 16;
  }
}

// findings + metrics (static, rendered once)
function renderStatic() {
  const tl = document.getElementById('tiles');
  for (const t of D.tiles) { const d = h('div', 'tile', null, tl); h('div', 'label', t.label, d); h('div', 'value', t.value, d); if (t.note) h('div', 'note', t.note, d); }
  const fl = document.getElementById('findings');
  if (!D.findings.length) h('p', 'empty', 'No findings.', fl);
  for (const f of D.findings) { const d = h('div', 'finding', null, fl); d.id = f.id;
    const t = h('div', 't', null, d); h('span', 'badge', f.id, t); t.appendChild(document.createTextNode(f.title));
    h('div', 'meta', f.info ? 'Informational' : (f.symptom ? 'Consequence · ' : '') + '≈' + f.impact_pct.toFixed(1) + '% of the run', d);
    h('div', null, f.text, d); if (f.suggestion) { const s = h('div', 'sugg', null, d); h('strong', null, 'Suggestion: ', s); s.appendChild(document.createTextNode(f.suggestion)); } }
  const mt = document.getElementById('metrics');
  const t = h('table', null, null, mt); const tr = h('tr', null, null, h('thead', null, null, t)); ['id', 'scope', 'metric', 'value'].forEach((x, i) => h('th', i === 3 ? 'num' : '', x, tr));
  const tb = h('tbody', null, null, t); for (const m of D.metrics) { const r = h('tr', null, null, tb); r.id = m.id; h('td', null, m.id, r); h('td', null, m.scope, r); h('td', null, m.label, r); h('td', 'num', m.value, r); }
}

document.getElementById('theme').addEventListener('click', () => {
  const dark = getComputedStyle(root).colorScheme === 'dark'; document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark'); render(); });
renderStatic(); render();
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(render, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
})();
"""


def render_html(A, M, F, out_path, brief_md, llm=None):
    """llm: optional dict {"text": markdown, "check": validation dict, "model": name}"""
    data = chart_data(A, M, F)
    ids = [m["id"] for m in M.items] + [f["id"] for f in F]
    blob = json.dumps(data).replace("</", "<\\/")
    title = "vibeprof report"
    prog = html.escape(data["program"])
    if llm and llm.get("text"):
        chk = llm.get("check", {})
        rows = []
        if chk:
            cls = "ok" if chk.get("ok") else "warn"
            rows.append('<div class="%s">%s%d citations; %d unknown id(s)%s; %d number(s) not found in the data%s</div>' % (
                cls, "" if chk.get("citations") else "No citations: this text is not grounded in the measured data. ", chk.get("citations", 0), len(chk.get("unknown_ids", [])),
                (" (" + html.escape(", ".join(chk["unknown_ids"][:8])) + ")") if chk.get("unknown_ids") else "",
                len(chk.get("unsupported", [])),
                (": " + html.escape(", ".join(chk["unsupported"][:8]))) if chk.get("unsupported") else ""))
        report_html = ('<section class="card"><h2>Interpretation and suggestions</h2><p class="desc">Written by %s from the measured '
                       'facts; every number should cite a metric id below. Check marks come from an automatic validation.</p>'
                       '<div class="prose">%s</div><div class="check">%s</div></section>'
                       % (html.escape(llm.get("model", "the LLM")), md_to_html(llm["text"], ids), "".join(rows)))
    else:
        report_html = ('<section class="card"><h2>Interpretation and suggestions</h2><p class="desc">Rule-based brief (no LLM).</p>'
                       '<div class="prose">%s</div></section>' % md_to_html(brief_md, ids))
    warn = ""
    if data["warnings"]:
        warn = '<section class="card"><h2>Collector warnings</h2><ul>%s</ul></section>' % "".join("<li>%s</li>" % html.escape(w) for w in data["warnings"])
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title><style>%s</style></head>
<body><div class="viz-root"><div class="wrap">
<header><h1>MPI performance report</h1><div class="sub">%s</div><div class="sub">%s</div></header>
<div class="toolbar"><button id="theme" type="button">Toggle light/dark</button></div>
<div class="tiles" id="tiles"></div>
%s
<section class="card"><h2>Findings</h2><p class="desc">Rule-based findings ordered by estimated impact: causes first, then their consequences, then notes.</p><div id="findings"></div></section>
<div id="charts"></div>
%s
<section class="card"><h2>All metrics</h2><p class="desc">Every number in this report, with the id that the findings and the interpretation cite.</p><details><summary>Show %d metrics</summary><div class="tablewrap" id="metrics"></div></details></section>
</div></div>
<div id="tip" role="tooltip"></div>
<script type="application/json" id="data">%s</script>
<script>%s</script>
</body></html>
""" % (title, CSS, prog, html.escape(data["mpi_library"]), report_html, warn, len(data["metrics"]), blob, JS)
    with open(out_path, "w") as f:
        f.write(page)
    return out_path


# ------------------------------------------------------------------------------------------------
def render_png(A, M, F, out_dir):
    """Static figures for slides and papers (light theme, same color roles as the HTML report)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
    except ImportError:
        print("vibeprof: matplotlib not available, skipping PNG figures")
        return []
    data = chart_data(A, M, F)
    C = {"comp": "#2a78d6", "mpi": "#eb6834", "wait": "#1baf7a", "deemph": "#c3c2b7", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "text2": "#52514e", "muted": "#898781", "surface": "#fcfcfb"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": C["axis"], "axes.labelcolor": C["text2"],
                         "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.grid": True, "grid.color": C["grid"],
                         "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True, "figure.facecolor": C["surface"],
                         "axes.facecolor": C["surface"], "axes.spines.top": False, "axes.spines.right": False})
    names = [("comp", "Computation (CPU work + CUDA calls)" if data.get("gpu") else "Computation"), ("mpi", "MPI (operation cost)"),
             ("wait", "Waiting in MPI")]
    files = []

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    if data["phases"]:
        rows = data["phases"][::-1]
        fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(rows) + 1.0))
        left = [0.0] * len(rows)
        for k, lab in names:
            vals = [r[k] for r in rows]
            ax.barh([r["label"] for r in rows], vals, left=left, height=(0.55 if len(rows) >= 3 else 0.28), color=C[k], label=lab, edgecolor=C["surface"], linewidth=1.5)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_xlabel("mean time per rank (s)")
        ax.grid(axis="y", visible=False)
        if len(rows) < 3:   # keep few bars thin instead of filling the axis
            ax.set_ylim(-1.2, len(rows) + 0.2)
        ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.3, handlelength=1.0, fontsize=9)
        ax.set_title("Where the time goes, by phase", loc="left", fontsize=11, pad=24)
        save(fig, "phases.png")

    rows = data["ranks"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    bottom = [0.0] * len(rows)
    width = min(0.7, 0.07 * len(rows)) if len(rows) <= 64 else 0.9  # few ranks: keep columns thin, not slabs
    for k, lab in names:
        vals = [r[k] for r in rows]
        ax.bar(range(len(rows)), vals, bottom=bottom, width=width, color=C[k], label=lab, edgecolor=C["surface"], linewidth=1 if len(rows) <= 64 else 0)
        bottom = [a + b for a, b in zip(bottom, vals)]
    step = max(1, len(rows) // 24)
    ax.set_xticks(range(0, len(rows), step))
    ax.set_xticklabels([rows[i]["label"] for i in range(0, len(rows), step)])
    ax.set_xlabel(data["rank_axis"])
    ax.set_ylabel("time (s)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.3, handlelength=1.0, fontsize=9)
    ax.set_title("Time per " + data["rank_axis"], loc="left", fontsize=11, pad=24)
    save(fig, "ranks.png")

    if data["heat"]["rows"]:
        hr = data["heat"]["rows"]
        vals = [[v for v in r["values"]] for r in hr]
        flatv = [v for r in vals for v in r]
        d = max(1.25, max(max(flatv), 1 / max(1e-9, min(flatv))))
        cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(hr) + 1.2))
        im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vcenter=1.0, vmin=1 / d, vmax=d), interpolation="nearest")
        ax.set_yticks(range(len(hr)))
        ax.set_yticklabels([r["label"] for r in hr])
        step = max(1, len(data["heat"]["cols"]) // 24)
        ax.set_xticks(range(0, len(data["heat"]["cols"]), step))
        ax.set_xticklabels(data["heat"]["cols"][::step])
        ax.set_xlabel(data["heat"]["axis"])
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="computation / phase mean", fraction=0.04)
        ax.set_title("Computation imbalance", loc="left", fontsize=11)
        save(fig, "imbalance.png")

    mat = data["matrix"]["values"]
    if any(v > 0 for row in mat for v in row):
        cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#0d366b"])
        cmap.set_bad(C["surface"])
        import numpy as np
        arr = np.array(mat, dtype=float)
        arr[arr <= 0] = np.nan
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi <= lo * 1.0001:
            lo, hi = lo / 2, hi * 2
        im = ax.imshow(arr, cmap=cmap, norm=LogNorm(vmin=lo, vmax=hi), interpolation="nearest")
        labs = data["matrix"]["labels"]; st = max(1, len(labs) // 16)
        ax.set_xticks(range(0, len(labs), st)); ax.set_xticklabels(labs[::st])
        ax.set_yticks(range(0, len(labs), st)); ax.set_yticklabels(labs[::st])
        ax.set_xlabel("to " + data["matrix"]["axis"])
        ax.set_ylabel("from " + data["matrix"]["axis"])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, label="bytes sent", fraction=0.046)
        from matplotlib.ticker import FuncFormatter
        def _fb(v, _pos=None):
            for n, dd in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 1024)):
                if v >= dd:
                    return "%.3g %s" % (v / dd, n)
            return "%d B" % v
        cb.set_ticks([lo, (lo * hi) ** 0.5, hi]); cb.ax.yaxis.set_major_formatter(FuncFormatter(_fb)); cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
        ax.set_title("Point-to-point traffic", loc="left", fontsize=11)
        save(fig, "traffic.png")

    if data["sites"]:
        rows = data["sites"][::-1]
        fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(rows) + 1.0))
        left = [0.0] * len(rows)
        for k, lab in names[1:]:
            vals = [r[k] for r in rows]
            ax.barh([r["label"][:60] for r in rows], vals, left=left, height=(0.55 if len(rows) >= 3 else 0.28), color=C[k], label=lab, edgecolor=C["surface"], linewidth=1.5)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_xlabel("mean time per rank (s)")
        ax.grid(axis="y", visible=False)
        if len(rows) < 3:   # keep few bars thin instead of filling the axis
            ax.set_ylim(-1.2, len(rows) + 0.2)
        ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.3, handlelength=1.0, fontsize=9)
        ax.set_title("Top MPI call sites", loc="left", fontsize=11, pad=24)
        save(fig, "callsites.png")

    if data["trends"]:
        tr = data["trends"]
        fig, axes = plt.subplots(1, len(tr), figsize=(3.2 * len(tr) + 0.6, 2.4), squeeze=False)
        for ax, t in zip(axes[0], tr):
            xs = range(1, len(t["max"]) + 1)
            ax.plot(xs, t["max"], color=C["comp"], lw=2, label="slowest rank")
            ax.plot(xs, t["mean"], color=C["deemph"], lw=2, label="average rank")
            ax.set_title(t["label"], loc="left", fontsize=9.5)
            ax.set_xlabel("instance")
            ax.set_ylim(bottom=0)
        axes[0][0].set_ylabel("duration (s)")
        axes[0][-1].legend(frameon=False, fontsize=8)
        save(fig, "trends.png")

    Gd = data.get("gpu")
    if Gd:
        if Gd["phases"]:
            rows = Gd["phases"][::-1]
            fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(rows) + 1.0))
            ax.barh([r["label"] for r in rows], [r["busy"] for r in rows], height=(0.55 if len(rows) >= 3 else 0.28), color=C["comp"], label="GPU busy (kernels)",
                    edgecolor=C["surface"], linewidth=1.5)
            ax.barh([r["label"] for r in rows], [r["idle"] for r in rows], left=[r["busy"] for r in rows], height=(0.55 if len(rows) >= 3 else 0.28), color=C["deemph"],
                    label="GPU idle", edgecolor=C["surface"], linewidth=1.5)
            ax.set_xlabel("mean time per rank while in the phase (s)")
            ax.grid(axis="y", visible=False)
            if len(rows) < 3:   # keep few bars thin instead of filling the axis
                ax.set_ylim(-1.2, len(rows) + 0.2)
            ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.3, handlelength=1.0, fontsize=9)
            ax.set_title("GPU busy and idle, by phase", loc="left", fontsize=11, pad=24)
            save(fig, "gpu_phases.png")
        if Gd["kernels"]:
            rows = Gd["kernels"][::-1]
            fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(rows) + 1.0))
            ax.barh([r["label"][:60] for r in rows], [r["time"] for r in rows], height=(0.55 if len(rows) >= 3 else 0.28), color=C["comp"])
            ax.set_xlabel("mean GPU time per rank (s)")
            ax.grid(axis="y", visible=False)
            if len(rows) < 3:   # keep few bars thin instead of filling the axis
                ax.set_ylim(-1.2, len(rows) + 0.2)
            ax.set_title("Top GPU kernels", loc="left", fontsize=11)
            save(fig, "gpu_kernels.png")
        if Gd["timeline"] and len(Gd["timeline"]["mean"]) > 2:
            tl = Gd["timeline"]
            xs = [i * tl["step"] for i in range(len(tl["mean"]))]
            fig, ax = plt.subplots(figsize=(7.5, 2.6))
            ax.plot(xs, tl["min"], color=C["deemph"], lw=2, label="least busy rank")
            ax.plot(xs, tl["mean"], color=C["comp"], lw=2, label="mean over ranks")
            ax.set_ylim(0, 105)
            ax.set_xlabel("time since MPI_Init (s)")
            ax.set_ylabel("GPU busy (%)")
            ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.3, handlelength=1.0, fontsize=9)
            ax.set_title("GPU busy over the run", loc="left", fontsize=11, pad=24)
            save(fig, "gpu_timeline.png")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--png", action="store_true", help="also write static PNG figures")
    a = ap.parse_args()
    out = a.out or os.path.join(a.profile_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    run, ranks = LA.load(a.profile_dir)
    A, M = LA.analyze(run, ranks)
    F = LA.findings(A, M)
    p = render_html(A, M, F, os.path.join(out, "report.html"), LA.brief_report(A, M, F))
    print("vibeprof: wrote", p)
    if a.png:
        for f in render_png(A, M, F, out):
            print("vibeprof: wrote", f)


if __name__ == "__main__":
    main()
