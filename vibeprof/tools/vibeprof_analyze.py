#!/usr/bin/env python3
"""vibeprof analysis: per-rank profiles -> metrics, findings, deterministic report, LLM prompts.

usage: vibeprof_analyze.py PROFILE_DIR [-o OUT_DIR] [--map-reduce]

Writes to OUT_DIR (default PROFILE_DIR/analysis):
  metrics.json      every computed quantity, each with a stable id (M1, M2, ...)
  findings.json     rule-based findings (F1, ...) with severity and evidence ids
  summary.md        deterministic text report (no LLM)
  prompts.jsonl     LLM prompt(s) in the vibeprof_llm --prompts format
  facts.txt         the facts sheet given to the LLM (for validation)

Only the Python standard library is needed.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

# ------------------------------------------------------------------------------------------------
# thresholds for findings (fractions of wall time unless noted)
TH = {
    "imbalance_factor": 1.10,     # max/mean computation in a phase
    "imbalance_loss": 0.03,       # (max - mean) computation / wall
    "slow_node_factor": 1.10,     # node mean computation / other nodes' mean
    "collective_intrinsic": 0.05, # min-over-ranks time in a collective / wall
    "wait": 0.05,                 # mean waiting (excess over fastest rank) in a phase / wall
    "small_msg_fraction": 0.80,   # messages below 1 KiB
    "small_msg_time": 0.05,
    "barrier_time": 0.05,
    "send_wait": 0.05,
    "growth": 0.25,               # last quarter vs first quarter mean instance duration (halves must agree)
    "growth_impact": 0.03,        # ... and the extra time must matter (fraction of wall)
    "mpi_bound": 0.30,
    "compute_bound": 0.10,
    "coverage": 0.50,
    # GPU (fractions of wall unless noted)
    "gpu_used": 0.01,             # mean kernel time per rank: below this the run is treated as CPU-only
    "gpu_idle": 0.05,             # mean GPU idle time while the host is in a phase
    "gpu_imbalance_factor": 1.10, # max/mean kernel time in a phase
    "gpu_imbalance_loss": 0.03,
    "gpu_staging": 0.02,          # host<->device copy time around MPI in a phase
    "gpu_pageable": 0.02,         # copy time involving pageable host memory
    "gpu_alloc": 0.02,            # host time in CUDA allocation calls
    "gpu_launch": 0.05,           # host time launching kernels
    "gpu_sync_per_kernel": 0.5,   # synchronizing calls per kernel launch (frequent synchronization)
    "gpu_util_low": 50.0,         # % GPU busy over the run (informational summary)
}

P2P_KINDS = {"send", "recv", "sendrecv"}
COLL_KINDS = {"collective", "nb-collective"}


# ------------------------------------------------------------------------------------------------
class Metrics:
    """Registry of reported quantities. Every number shown to a reader or to the LLM has an id."""

    def __init__(self):
        self.items = []
        self.by_key = {}

    def add(self, key, value, unit, label, **ctx):
        mid = "M%d" % (len(self.items) + 1)
        item = {"id": mid, "key": key, "value": value, "unit": unit, "label": label}
        item.update(ctx)
        self.items.append(item)
        self.by_key[key] = item
        return mid

    def get(self, key):
        return self.by_key.get(key)


def fmt(v, unit=""):
    if v is None:
        return "n/a"
    if unit == "%":
        return "%.1f%%" % v
    if unit == "s":
        return "%.4g s" % v if abs(v) >= 0.01 else "%.3g ms" % (v * 1e3)
    if unit == "x":
        return "%.2fx" % v
    if unit == "B":
        for u, d in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
            if abs(v) >= d:
                return "%.3g %s" % (v / d, u)
        return "%d B" % v
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


# ------------------------------------------------------------------------------------------------
def load(profile_dir):
    files = sorted(glob.glob(os.path.join(profile_dir, "rank-*.json")))
    if not files:
        sys.exit("no rank-*.json files in %s" % profile_dir)
    ranks = []
    for f in files:
        with open(f) as fh:
            ranks.append(json.load(fh))
    ranks.sort(key=lambda r: r["rank"])
    run = {}
    rp = os.path.join(profile_dir, "run.json")
    if os.path.exists(rp):
        with open(rp) as fh:
            run = json.load(fh)
    return run, ranks


def short_sym(fn):
    """Readable C++ name: drop the argument list and collapse template arguments.
    'void ns::f<ns::A<int> >(ns::A<int>&)' -> 'ns::f<…>'"""
    s = fn.strip()
    # drop a trailing balanced "(...)" argument list (and cv/ref qualifiers after it)
    s = re.sub(r"\)\s*(const|volatile|&|&&|\s)*$", ")", s)
    if s.endswith(")"):
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(s[i], 0)
            if depth == 0:
                if not s[:i].rstrip().endswith("operator"):
                    s = s[:i]
                break
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            if depth == 0:
                out.append("<…>")
            depth += 1
        elif ch == ">" and depth > 0:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    s = "".join(out).strip()
    # drop a leading return type ("void ns::f" -> "ns::f")
    if " " in s and not s.startswith("operator"):
        s = s.rsplit(" ", 1)[1]
    return s


def symbolize(ranks):
    """(obj, offset) -> 'function file:line' via addr2line, batched per object."""
    want = defaultdict(set)
    for r in ranks:
        for s in r["sites"].values():
            if s["obj"]:
                want[s["obj"]].add(s["offset"])
    out = {}
    for obj, offs in want.items():
        offs = sorted(offs)
        labels = {}
        try:
            res = subprocess.run(["addr2line", "-f", "-C", "-e", obj] + offs, capture_output=True, text=True, timeout=60)
            lines = res.stdout.splitlines()
            for i, off in enumerate(offs):
                fn = lines[2 * i] if 2 * i < len(lines) else "??"
                loc = lines[2 * i + 1] if 2 * i + 1 < len(lines) else "??:0"
                loc = loc.split(" (discriminator")[0]
                file_line = os.path.basename(loc) if not loc.startswith("??") else ""
                labels[off] = (fn if fn != "??" else "", file_line)
        except (OSError, subprocess.SubprocessError):
            pass
        for off in offs:
            fn, fl = labels.get(off, ("", ""))
            name = fl or ("%s+%s" % (os.path.basename(obj), off))
            if fn:
                name = "%s (%s)" % (name, short_sym(fn))
            out[(obj, off)] = name
    return out


# ------------------------------------------------------------------------------------------------
API_CATS = ("sync", "alloc", "launch", "copy", "other")


def api_cat(name):
    """CUDA call -> sync (explicit synchronization), alloc, launch, copy (incl. blocking cudaMemcpy), other."""
    if "Synchronize" in name:
        return "sync"
    if re.match(r"(cudaMalloc|cudaFree|cudaHostAlloc|cudaHostRegister|cudaHostUnregister|cuMemAlloc|cuMemFree|cuMemHostAlloc|"
                r"cuMemHostRegister|cuMemHostUnregister|cuMemCreate|cuMemRelease)", name):
        return "alloc"
    if re.match(r"(cudaLaunch|cuLaunch|cudaGraphLaunch|cuGraphLaunch)", name):
        return "launch"
    if re.match(r"(cudaMemcpy|cuMemcpy|cudaMemset|cuMemset|cudaMemPrefetch|cuMemPrefetch)", name):
        return "copy"
    return "other"


def gpu_extract(ranks):
    """Per-rank GPU data keyed by phase path, or None when no rank has a GPU backend."""
    if not any((r.get("gpu") or {}).get("backend") for r in ranks):
        return None
    G = {"ranks": []}
    for r in ranks:
        g = r.get("gpu") or {}
        pid2path = {p["id"]: p["path"] for p in r["phases"]}
        path = lambda pid: pid2path.get(pid, "<main>")
        d = {"backend": g.get("backend", ""), "status": g.get("status", "absent" if not g else ""),
             "visible": g.get("visible_devices", 0), "devices": g.get("devices", []),
             "kernel_ph": defaultdict(float), "kcount_ph": defaultdict(int), "busy_ph": defaultdict(float),
             "api_ph": defaultdict(lambda: defaultdict(float)), "api_n_ph": defaultdict(lambda: defaultdict(int)),
             "copies": [], "kernels": [], "memset_time": 0.0,
             "dropped": g.get("dropped", 0), "unattributed": g.get("unattributed", 0), "records": g.get("records", 0)}
        for k in g.get("kernels", []):
            ph = path(k["phase"])
            d["kernel_ph"][ph] += k["time"]
            d["kcount_ph"][ph] += k["count"]
            d["kernels"].append(dict(k, phase=ph))
        for c in g.get("copies", []):
            d["copies"].append(dict(c, phase=path(c["phase"])))
        for m in g.get("memsets", []):
            d["memset_time"] += m["time"]
        for a in g.get("api", []):
            cat = api_cat(a["name"])
            d["api_ph"][path(a["phase"])][cat] += a["time"]
            d["api_n_ph"][path(a["phase"])][cat] += a["count"]
        for pid, b in g.get("phase_busy", []):
            d["busy_ph"][path(pid)] += b
        d["kernel_busy"] = sum(x.get("kernel_busy", 0.0) for x in d["devices"])
        d["kernel_time"] = sum(d["kernel_ph"].values())
        d["kernel_count"] = sum(d["kcount_ph"].values())
        d["copy_time"] = sum(c["time"] for c in d["copies"])
        d["api"] = {c: sum(d["api_ph"][p][c] for p in d["api_ph"]) for c in API_CATS}
        d["api_n"] = {c: sum(d["api_n_ph"][p][c] for p in d["api_n_ph"]) for c in API_CATS}
        d["api_total"] = sum(d["api"].values())
        d["used"] = [(x.get("pci") or str(x.get("id")), x.get("name", "")) for x in d["devices"] if x.get("kernel_time", 0) > 0]
        ndev = max(1, len(d["used"]))
        d["util"] = 100.0 * d["kernel_busy"] / (r["wall"] * ndev) if r["wall"] > 0 else 0.0
        d["mpi_dev_bytes"] = sum(o.get("dev_bytes", 0) for o in r["ops"])
        d["mpi_bytes"] = sum(o["bytes"] for o in r["ops"])
        d["timeline"] = None
        tl = [x for x in d["devices"] if x.get("timeline")]
        if tl:
            d["timeline"] = {"step": tl[0].get("timeline_step", 0.01), "busy": [statistics.mean(v) for v in zip(*[x["timeline"] for x in tl])]}
        G["ranks"].append(d)
    return G


def analyze(run, ranks):
    M = Metrics()
    n = len(ranks)
    hosts = [r["host"] for r in ranks]
    nodes = sorted(set(hosts))
    node_of = [nodes.index(h) for h in hosts]
    wall = max(r["wall"] for r in ranks)
    comp_tot = [r["comp_total"] for r in ranks]
    mpi_tot = [r["mpi_total"] for r in ranks]
    rank_secs = sum(r["wall"] for r in ranks)

    G = gpu_extract(ranks)
    gpu_on = bool(G) and statistics.mean(d["kernel_time"] for d in G["ranks"]) >= TH["gpu_used"] * wall
    if gpu_on:
        # computation as seen by the collector includes host time inside CUDA calls (launching, waiting for the GPU,
        # allocating); split it out so "computation" means CPU work
        comp_tot = [max(0.0, c - d["api_total"]) for c, d in zip(comp_tot, G["ranks"])]
    A = {"n_ranks": n, "nodes": nodes, "node_of": node_of, "wall": wall, "gpu": G if gpu_on else None, "gpu_profile": G}
    A["run"] = {
        "exe": run.get("exe") or ranks[0].get("exe", ""),
        "cmdline": run.get("cmdline") or ranks[0].get("cmdline", ""),
        "mpi_library": ranks[0].get("mpi_library", ""),
    }
    M.add("run.ranks", n, "", "MPI ranks")
    M.add("run.nodes", len(nodes), "", "nodes")
    M.add("run.wall", wall, "s", "wall time (slowest rank, MPI_Init..MPI_Finalize)")
    M.add("run.comp_frac", 100.0 * sum(comp_tot) / rank_secs, "%", "share of rank time in computation" + (" on the CPU" if gpu_on else ""))
    if gpu_on:
        M.add("run.cuda_frac", 100.0 * sum(d["api_total"] for d in G["ranks"]) / rank_secs, "%",
              "share of rank time inside CUDA calls (launching, waiting for the GPU, copies, allocation)")
    M.add("run.mpi_frac", 100.0 * sum(mpi_tot) / rank_secs, "%", "share of rank time inside MPI")
    calls = sum(r["calls_total"] for r in ranks)
    M.add("run.calls_per_rank_s", calls / rank_secs if rank_secs else 0.0, "", "MPI calls per rank-second")

    # ---------------- phases ----------------
    paths = []
    for r in ranks:
        for p in r["phases"]:
            if p["path"] not in paths:
                paths.append(p["path"])
    per = {}   # path -> per-rank arrays
    for path in paths:
        comp, mpi, cnt, inst = [], [], [], []
        for r in ranks:
            p = next((q for q in r["phases"] if q["path"] == path), None)
            comp.append(p["comp"] if p else 0.0)
            mpi.append(p["mpi"] if p else 0.0)
            cnt.append(p["n"] if p else 0)
            inst.append(p["inst"] if p else [])
        if gpu_on:
            cuda = [sum(G["ranks"][i]["api_ph"][path].values()) for i in range(n)]
            comp = [max(0.0, c - a) for c, a in zip(comp, cuda)]
        else:
            cuda = [0.0] * n
        per[path] = {"comp": comp, "mpi": mpi, "n": cnt, "inst": inst, "cuda": cuda}

    def incl(path, field):
        """inclusive per-rank time: this phase plus all descendants"""
        out = [0.0] * n
        for q in paths:
            if q == path or (path != "<main>" and q.startswith(path + "/")):
                for i, v in enumerate(per[q][field]):
                    out[i] += v
        return out

    named = sum(sum(per[p]["comp"]) + sum(per[p]["mpi"]) + sum(per[p]["cuda"]) for p in paths if p != "<main>")
    M.add("run.marker_coverage", 100.0 * named / rank_secs if rank_secs else 0.0, "%", "share of rank time inside named phases")

    phases = []
    for path in paths:
        if path == "<main>":
            continue
        comp_ex, mpi_ex = per[path]["comp"], per[path]["mpi"]
        comp_in, mpi_in, cuda_in = incl(path, "comp"), incl(path, "mpi"), incl(path, "cuda")
        tot_in = [c + m + g for c, m, g in zip(comp_in, mpi_in, cuda_in)]
        mean_c = statistics.mean(comp_ex)
        max_c = max(comp_ex)
        imb = max_c / mean_c if mean_c > 0 else 1.0
        min_m = min(mpi_ex)
        wait = [m - min_m for m in mpi_ex]
        ph = {
            "path": path, "depth": path.count("/"), "instances": max(per[path]["n"]),
            "comp": comp_ex, "mpi": mpi_ex, "comp_incl": comp_in, "mpi_incl": mpi_in, "cuda": per[path]["cuda"], "cuda_incl": cuda_in,
            "slowest_rank": max(range(n), key=lambda i: comp_ex[i]),
        }
        key = "phase[%s]" % path
        ph["ids"] = {
            "time": M.add(key + ".time_incl_mean", statistics.mean(tot_in), "s", "mean time per rank (inclusive)", phase=path),
            "share": M.add(key + ".share", 100.0 * statistics.mean(tot_in) / wall if wall else 0.0, "%", "share of wall time (inclusive)", phase=path),
            "comp_mean": M.add(key + ".comp_mean", mean_c, "s", "mean computation per rank (exclusive)", phase=path),
            "comp_max": M.add(key + ".comp_max", max_c, "s", "max computation over ranks (exclusive)", phase=path,
                              rank=ph["slowest_rank"]),
            "imbalance": M.add(key + ".imbalance", imb, "x", "computation imbalance (max/mean)", phase=path),
            "imb_loss": M.add(key + ".imbalance_loss", max_c - mean_c, "s", "time lost to computation imbalance (max - mean)", phase=path),
            "mpi_mean": M.add(key + ".mpi_mean", statistics.mean(mpi_ex), "s", "mean MPI time per rank (exclusive)", phase=path),
            "mpi_min": M.add(key + ".mpi_min", min_m, "s", "MPI time of the fastest rank (intrinsic cost estimate)", phase=path),
            "wait_mean": M.add(key + ".wait_mean", statistics.mean(wait), "s", "mean MPI waiting (excess over the fastest rank)", phase=path),
            "instances": M.add(key + ".instances", ph["instances"], "", "instances (begin/end pairs) per rank", phase=path),
        }
        # iteration trend: per instance index, max over ranks of instance duration
        insts = per[path]["inst"]
        k = min((len(x) for x in insts), default=0)
        if k >= 8:
            dur_max = [max(insts[r][i][1] - insts[r][i][0] for r in range(n)) for i in range(k)]
            dur_mean = [statistics.mean(insts[r][i][1] - insts[r][i][0] for r in range(n)) for i in range(k)]
            q = max(1, k // 4)
            first, last = statistics.mean(dur_max[:q]), statistics.mean(dur_max[-q:])
            h = k // 2
            ph["growth_half"] = statistics.mean(dur_max[h:]) / statistics.mean(dur_max[:h]) - 1.0 if statistics.mean(dur_max[:h]) > 0 else 0.0
            ph["series"] = {"max": dur_max, "mean": dur_mean}
            ph["ids"]["growth"] = M.add(key + ".growth", 100.0 * (last / first - 1.0) if first > 0 else 0.0, "%",
                                        "change of instance duration, last vs first quarter", phase=path)
            cv = statistics.pstdev(dur_max) / statistics.mean(dur_max) if statistics.mean(dur_max) > 0 else 0.0
            ph["ids"]["cv"] = M.add(key + ".cv", 100.0 * cv, "%", "variability of instance duration (coefficient of variation)", phase=path)
        phases.append(ph)
    if gpu_on:
        for ph in phases:
            path, key = ph["path"], "phase[%s]" % ph["path"]
            R = G["ranks"]
            kern = [d["kernel_ph"][path] for d in R]
            busy = [d["busy_ph"][path] for d in R]
            host = [ph["comp"][i] + ph["mpi"][i] + ph["cuda"][i] for i in range(n)]
            idle = [max(0.0, h - b) for h, b in zip(host, busy)]
            ph["gpu"] = {"kernel": kern, "busy": busy, "idle": idle,
                         "api": {c: [d["api_ph"][path][c] for d in R] for c in API_CATS},
                         "api_n": {c: [d["api_n_ph"][path][c] for d in R] for c in API_CATS},
                         "kcount": [d["kcount_ph"][path] for d in R]}
            if max(host) <= 0:
                continue
            km = statistics.mean(kern)
            ph["gpu"]["slowest"] = max(range(n), key=lambda i: kern[i])
            ph["ids"]["cuda_mean"] = M.add(key + ".cuda_mean", statistics.mean(ph["cuda"]), "s", "mean host time inside CUDA calls per rank", phase=path)
            ph["ids"]["cuda_sync_mean"] = M.add(key + ".cuda_sync_mean", statistics.mean(ph["gpu"]["api"]["sync"]), "s",
                                                "mean host time blocked in synchronizing CUDA calls (waiting for the GPU)", phase=path)
            ph["ids"]["gpu_busy_mean"] = M.add(key + ".gpu_busy_mean", statistics.mean(busy), "s", "mean GPU busy time while ranks are in this phase", phase=path)
            ph["ids"]["gpu_idle_mean"] = M.add(key + ".gpu_idle_mean", statistics.mean(idle), "s", "mean GPU idle time while ranks are in this phase", phase=path)
            ph["ids"]["gpu_idle_frac"] = M.add(key + ".gpu_idle_frac", 100.0 * statistics.mean(idle) / statistics.mean(host) if statistics.mean(host) else 0.0,
                                               "%", "share of the phase during which the GPU is idle", phase=path)
            if km > 0:
                ph["ids"]["gpu_kernel_mean"] = M.add(key + ".gpu_kernel_mean", km, "s", "mean GPU kernel time per rank, kernels launched in this phase", phase=path)
                ph["ids"]["gpu_kernel_max"] = M.add(key + ".gpu_kernel_max", max(kern), "s", "max GPU kernel time over ranks, kernels launched in this phase",
                                                    phase=path, rank=ph["gpu"]["slowest"])
                ph["ids"]["gpu_imbalance"] = M.add(key + ".gpu_imbalance", max(kern) / km, "x", "GPU kernel time imbalance (max/mean)", phase=path)
                ph["ids"]["gpu_imb_loss"] = M.add(key + ".gpu_imbalance_loss", max(kern) - km, "s", "time lost to GPU imbalance (max - mean)", phase=path)
                ph["ids"]["kernels"] = M.add(key + ".kernels", statistics.mean(ph["gpu"]["kcount"]), "", "kernel launches per rank", phase=path)
    phases.sort(key=lambda p: -statistics.mean([c + m + g for c, m, g in zip(p["comp_incl"], p["mpi_incl"], p["cuda_incl"])]))
    A["phases"] = phases

    # ---------------- operations and call sites ----------------
    labels = symbolize(ranks)
    sites = {}   # (path, op, label) -> aggregate
    ops = {}     # op -> aggregate
    for i, r in enumerate(ranks):
        pid2path = {p["id"]: p["path"] for p in r["phases"]}
        for o in r["ops"]:
            s = r["sites"][o["site"]]
            lab = labels.get((s["obj"], s["offset"]), s["offset"])
            k = (pid2path[o["phase"]], o["op"], lab)
            a = sites.setdefault(k, {"phase": k[0], "op": k[1], "site": lab, "kind": o["kind"],
                                     "time": [0.0] * n, "count": [0] * n, "bytes": [0] * n, "hist": defaultdict(int), "tmax": 0.0})
            a["time"][i] += o["time"]
            a["count"][i] += o["count"]
            a["bytes"][i] += o["bytes"]
            a["tmax"] = max(a["tmax"], o["tmax"])
            for b, c in o["hist"].items():
                a["hist"][int(b)] += c
            g = ops.setdefault(o["op"], {"op": o["op"], "kind": o["kind"], "time": [0.0] * n, "count": [0] * n,
                                         "bytes": [0] * n, "hist": defaultdict(int)})
            g["time"][i] += o["time"]
            g["count"][i] += o["count"]
            g["bytes"][i] += o["bytes"]
            for b, c in o["hist"].items():
                g["hist"][int(b)] += c

    def small_frac(hist):
        tot = sum(hist.values())
        return sum(c for b, c in hist.items() if b <= 10) / tot if tot else 0.0   # bucket <= 10: < 1 KiB

    op_list = []
    for g in ops.values():
        tot_time = sum(g["time"])
        if tot_time <= 0 and sum(g["count"]) == 0:
            continue
        key = "op[%s]" % g["op"]
        g["ids"] = {
            "time_mean": M.add(key + ".time_mean", statistics.mean(g["time"]), "s", "mean time per rank", op=g["op"]),
            "share": M.add(key + ".share", 100.0 * statistics.mean(g["time"]) / wall if wall else 0.0, "%", "share of wall time", op=g["op"]),
            "calls": M.add(key + ".calls", sum(g["count"]), "", "calls (all ranks)", op=g["op"]),
            "time_min": M.add(key + ".time_min", min(g["time"]), "s", "time of the fastest rank", op=g["op"]),
            "time_max": M.add(key + ".time_max", max(g["time"]), "s", "time of the slowest rank", op=g["op"]),
        }
        if sum(g["count"]):
            g["ids"]["per_call"] = M.add(key + ".per_call", sum(g["time"]) / sum(g["count"]), "s", "mean time per call", op=g["op"])
        if sum(g["bytes"]):
            g["ids"]["bytes"] = M.add(key + ".bytes", sum(g["bytes"]), "B", "payload bytes (all ranks)", op=g["op"])
            msgs = sum(g["hist"].values())
            if msgs:
                g["ids"]["msg_mean"] = M.add(key + ".msg_mean", sum(g["bytes"]) / msgs, "B", "mean message size", op=g["op"])
                g["ids"]["small"] = M.add(key + ".small_frac", 100.0 * small_frac(g["hist"]), "%", "messages below 1 KiB", op=g["op"])
        op_list.append(g)
    op_list.sort(key=lambda g: -sum(g["time"]))
    A["ops"] = op_list

    site_list = sorted(sites.values(), key=lambda a: -sum(a["time"]))
    for j, a in enumerate(site_list[:12]):
        key = "site[%s|%s|%s]" % (a["phase"], a["op"], a["site"])
        a["ids"] = {
            "time_mean": M.add(key + ".time_mean", statistics.mean(a["time"]), "s", "mean time per rank", site=a["site"], op=a["op"], phase=a["phase"]),
            "share": M.add(key + ".share", 100.0 * statistics.mean(a["time"]) / wall if wall else 0.0, "%", "share of wall time",
                           site=a["site"], op=a["op"], phase=a["phase"]),
            "wait": M.add(key + ".wait", statistics.mean(a["time"]) - min(a["time"]), "s", "mean excess over the fastest rank (waiting)",
                          site=a["site"], op=a["op"], phase=a["phase"]),
            "calls": M.add(key + ".calls", sum(a["count"]), "", "calls (all ranks)", site=a["site"], op=a["op"], phase=a["phase"]),
        }
        if sum(a["bytes"]):
            a["ids"]["bytes"] = M.add(key + ".bytes", sum(a["bytes"]), "B", "payload bytes (all ranks)", site=a["site"], op=a["op"], phase=a["phase"])
    A["sites"] = site_list

    # ---------------- nodes and ranks ----------------
    comp_rank = comp_tot
    A["rank_comp"] = comp_rank
    A["rank_mpi"] = mpi_tot
    node_comp = []
    for k in range(len(nodes)):
        members = [comp_rank[i] for i in range(n) if node_of[i] == k]
        node_comp.append(statistics.mean(members))
    A["node_comp"] = node_comp
    if len(nodes) > 1:
        for k, name in enumerate(nodes):
            others = [node_comp[j] for j in range(len(nodes)) if j != k]
            rel = node_comp[k] / statistics.mean(others) if statistics.mean(others) > 0 else 1.0
            M.add("node[%s].comp_mean" % name, node_comp[k], "s", "mean computation per rank on node", node=name)
            M.add("node[%s].comp_rel" % name, rel, "x", "node computation relative to the other nodes", node=name)
    med = statistics.median(comp_rank)
    A["rank_rel"] = [c / med if med > 0 else 1.0 for c in comp_rank]
    for i in sorted(range(n), key=lambda i: -comp_rank[i])[:3]:
        M.add("rank[%d].comp_rel" % i, A["rank_rel"][i], "x", "rank computation relative to the median rank", rank=i, node=nodes[node_of[i]])

    # ---------------- communication matrix ----------------
    mat = [[0] * n for _ in range(n)]
    for i, r in enumerate(ranks):
        for dst, b, c in r["peers"]:
            if 0 <= dst < n:
                mat[i][dst] += b
    A["matrix"] = mat
    tot_p2p = sum(map(sum, mat))
    if tot_p2p:
        off = sum(mat[i][j] for i in range(n) for j in range(n) if node_of[i] != node_of[j])
        deg = [sum(1 for j in range(n) if mat[i][j]) for i in range(n)]
        M.add("comm.p2p_bytes", tot_p2p, "B", "point-to-point bytes sent (all ranks)")
        M.add("comm.offnode_frac", 100.0 * off / tot_p2p, "%", "point-to-point bytes crossing nodes")
        M.add("comm.partners_mean", statistics.mean(deg), "", "communication partners per rank (mean)")
        M.add("comm.partners_max", max(deg), "", "communication partners per rank (max)")

    A["marks"] = defaultdict(int)
    for r in ranks:
        for m in r.get("marks", []):
            A["marks"][m["name"]] += m["n"]
    A["warnings"] = sorted({w for r in ranks for w in r.get("warnings", [])})
    if G:
        st = sorted({d["status"] for d in G["ranks"] if d["status"] not in ("ok", "")})
        A["warnings"] += ["GPU backend: " + x for x in st]
        drop = sum(d["dropped"] for d in G["ranks"])
        if drop:
            A["warnings"].append("GPU backend dropped %d activity records; GPU totals are lower bounds" % drop)
        un = sum(d["unattributed"] for d in G["ranks"])
        recs = sum(d["records"] for d in G["ranks"])
        if recs and un > 0.05 * recs:
            A["warnings"].append("%d of %d GPU records could not be attributed to a phase (counted under <main>)" % (un, recs))
    if gpu_on:
        analyze_gpu(A, M, ranks)
    return A, M


def analyze_gpu(A, M, ranks):
    G, n, wall = A["gpu"], A["n_ranks"], A["wall"]
    R = G["ranks"]
    mean = lambda xs: statistics.mean(xs) if xs else 0.0
    ids = G["ids"] = {}
    util = [d["util"] for d in R]
    ids["util"] = M.add("gpu.util_mean", mean(util), "%", "GPU busy with kernels, share of the run (mean over ranks)")
    lo = min(range(n), key=lambda i: util[i])
    ids["util_min"] = M.add("gpu.util_min", util[lo], "%", "GPU busy share of the least busy rank", rank=lo)
    ids["kernel"] = M.add("gpu.kernel_time_mean", mean([d["kernel_time"] for d in R]), "s", "GPU kernel time per rank (mean)")
    kc = mean([d["kernel_count"] for d in R])
    ids["kernels"] = M.add("gpu.kernels_mean", kc, "", "kernel launches per rank (mean)")
    if kc:
        ids["kdur"] = M.add("gpu.kernel_duration_mean", sum(d["kernel_time"] for d in R) / sum(d["kernel_count"] for d in R), "s", "mean kernel duration")
    ids["copy"] = M.add("gpu.copy_time_mean", mean([d["copy_time"] for d in R]), "s", "GPU memory copy time per rank (mean)")
    ids["cuda"] = M.add("gpu.cuda_api_mean", mean([d["api_total"] for d in R]), "s", "host time inside CUDA calls per rank (mean)")
    for c, lab in (("sync", "waiting for the GPU (synchronizing calls)"), ("alloc", "in CUDA allocation/free calls"),
                   ("launch", "launching kernels"), ("copy", "in copy/memset calls (blocking ones wait for the GPU)")):
        ids["api_" + c] = M.add("gpu.api_%s_mean" % c, mean([d["api"][c] for d in R]), "s", "host time %s per rank (mean)" % lab)
        ids["api_n_" + c] = M.add("gpu.api_%s_calls" % c, mean([d["api_n"][c] for d in R]), "", "%s calls per rank (mean)" % c)
    mb = sum(d["mpi_bytes"] for d in R)
    if mb:
        ids["mpi_dev"] = M.add("gpu.mpi_device_frac", 100.0 * sum(d["mpi_dev_bytes"] for d in R) / mb, "%", "MPI payload bytes in GPU memory (GPU-aware MPI)")
    # copies by direction and host memory kind
    agg = defaultdict(lambda: {"count": 0, "bytes": 0, "time": 0.0})
    for d in R:
        for c in d["copies"]:
            host = c["src"] if c["kind"] in ("HtoD", "HtoA") else c["dst"] if c["kind"] in ("DtoH", "AtoH") else ""
            k = (c["kind"], host, c.get("by_mpi", False))
            agg[k]["count"] += c["count"]; agg[k]["bytes"] += c["bytes"]; agg[k]["time"] += c["time"]
    G["copy_table"] = []
    for (kind, host, by_mpi), v in sorted(agg.items(), key=lambda kv: -kv[1]["time"]):
        key = "copy[%s|%s|%s]" % (kind, host or "-", "mpi" if by_mpi else "app")
        e = {"kind": kind, "host": host, "by_mpi": by_mpi, "count": v["count"], "bytes": v["bytes"], "time": v["time"] / n}
        e["ids"] = {"time": M.add(key + ".time_mean", v["time"] / n, "s", "copy time per rank (mean)", copy=kind, host_memory=host or "", by_mpi=by_mpi),
                    "bytes": M.add(key + ".bytes", v["bytes"], "B", "bytes copied (all ranks)", copy=kind, host_memory=host or "", by_mpi=by_mpi)}
        G["copy_table"].append(e)
    pg = mean([sum(c["time"] for c in d["copies"] if "pageable" in (c["src"], c["dst"])) for d in R])
    ids["pageable"] = M.add("gpu.pageable_copy_mean", pg, "s", "copy time involving pageable host memory per rank (mean)")
    # top kernels
    kag = {}
    for i, d in enumerate(R):
        for k in d["kernels"]:
            name = k["label"] or k["name"]
            a = kag.setdefault(name, {"name": name, "cuda_name": k["name"], "time": [0.0] * n, "count": 0, "phases": defaultdict(float)})
            a["time"][i] += k["time"]; a["count"] += k["count"]; a["phases"][k["phase"]] += k["time"]
    names = demangle([a["cuda_name"] for a in kag.values()])
    G["kernels"] = sorted(kag.values(), key=lambda a: -sum(a["time"]))
    for a in G["kernels"]:
        a["display"] = short_sym(names.get(a["cuda_name"], a["cuda_name"])) if a["name"] == a["cuda_name"] else a["name"]
        a["phase"] = max(a["phases"], key=a["phases"].get)
    for a in G["kernels"][:10]:
        key = "kernel[%s]" % a["display"]
        a["ids"] = {"time": M.add(key + ".time_mean", mean(a["time"]), "s", "GPU time per rank (mean)", kernel=a["display"], phase=a["phase"]),
                    "share": M.add(key + ".share", 100.0 * mean(a["time"]) / wall if wall else 0.0, "%", "share of wall time", kernel=a["display"]),
                    "count": M.add(key + ".count", a["count"], "", "launches (all ranks)", kernel=a["display"]),
                    "dur": M.add(key + ".duration", sum(a["time"]) / a["count"] if a["count"] else 0.0, "s", "mean duration", kernel=a["display"])}
    # devices: which GPU each rank drives
    G["dev_of"] = [[(ranks[i]["host"],) + u for u in R[i]["used"]] for i in range(n)]


def demangle(names):
    """C++ kernel names -> readable (c++filt, batched); missing tool -> names unchanged."""
    out = {x: x for x in names}
    mangled = [x for x in names if x.startswith("_Z")]
    if not mangled:
        return out
    try:
        res = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True, timeout=60)
        for a, b in zip(mangled, res.stdout.splitlines()):
            out[a] = b
    except (OSError, subprocess.SubprocessError):
        pass
    return out


# ------------------------------------------------------------------------------------------------
def ranks_on(members):
    return ("The only rank (%d)" % members[0]) if len(members) == 1 else ("All %d ranks" % len(members))


def findings(A, M):
    F = []
    wall = A["wall"]
    n = A["n_ranks"]

    def add(cat, sev, title, text, evidence, **kw):
        f = {"id": "F%d" % (len(F) + 1), "category": cat, "impact_s": sev, "impact_pct": 100.0 * sev / wall if wall else 0.0,
             "title": title, "text": text, "evidence": evidence}
        f.update(kw)
        F.append(f)

    nodes, node_of = A["nodes"], A["node_of"]
    # slow node: computation of every rank on one node is high
    slow_node = None
    if len(nodes) > 1:
        for k, name in enumerate(nodes):
            m = M.get("node[%s].comp_rel" % name)
            if m and m["value"] >= TH["slow_node_factor"]:
                members = [i for i in range(n) if node_of[i] == k]
                others = [A["rank_comp"][i] for i in range(n) if node_of[i] != k]
                # every rank on the node must be slow, not just one rank dragging the node mean up
                if min(A["rank_comp"][i] for i in members) >= TH["slow_node_factor"] * statistics.median(others):
                    # A slow machine slows every phase on its ranks; uneven work only slows some. Check the
                    # phases with significant computation.
                    sig = [p for p in A["phases"] if statistics.mean(p["comp"]) >= 0.02 * wall]
                    slow_in = []
                    for p in sig:
                        mine = statistics.mean(p["comp"][i] for i in members)
                        rest = statistics.median(p["comp"][i] for i in range(n) if node_of[i] != k)
                        if rest > 0 and mine / rest >= TH["slow_node_factor"]:
                            slow_in.append(p["path"])
                    loss = (A["node_comp"][k] - statistics.mean([A["node_comp"][j] for j in range(len(nodes)) if j != k]))
                    ev = [m["id"], M.get("node[%s].comp_mean" % name)["id"]]
                    if loss < TH["imbalance_loss"] * wall:
                        continue
                    if len(sig) >= 2 and len(slow_in) >= max(2, math.ceil(0.75 * len(sig))):
                        slow_node = name
                        add("slow_node", max(loss, 0.0), "Node %s computes slower than the others" % name,
                            "%s on node %s compute%s %s longer than ranks on the other nodes, in %d of %d phases with "
                            "significant computation (%s). A slowdown across all phases points to the node itself (CPU "
                            "frequency, thermal throttling, another job, memory) rather than the decomposition."
                            % (ranks_on(members), name, "s" if len(members) == 1 else "", fmt(m["value"], "x"), len(slow_in), len(sig), ", ".join(slow_in)),
                            ev, node=name, ranks=members, confidence="high")
                    elif len(sig) < 2:
                        add("slow_node", max(loss, 0.0), "Ranks on node %s compute more (slow node or uneven work)" % name,
                            "%s on node %s compute%s %s longer than the other nodes' ranks, but only one phase has "
                            "significant computation, so the data cannot tell a slow node from more work assigned to those "
                            "ranks." % (ranks_on(members), name, "s" if len(members) == 1 else "", fmt(m["value"], "x")),
                            ev, node=name, ranks=members, confidence="low")

    # load imbalance per phase (exclusive computation)
    for ph in A["phases"]:
        imb = M.get("phase[%s].imbalance" % ph["path"])
        loss = M.get("phase[%s].imbalance_loss" % ph["path"])
        if imb["value"] >= TH["imbalance_factor"] and loss["value"] >= TH["imbalance_loss"] * wall:
            sr = ph["slowest_rank"]
            on_slow_node = slow_node is not None and nodes[node_of[sr]] == slow_node
            add("load_imbalance", loss["value"], "Computation imbalance in phase '%s'" % ph["path"],
                "In '%s' the slowest rank (%d, node %s) computes %s the mean, losing about %s per rank while the others wait%s."
                % (ph["path"], sr, nodes[node_of[sr]], fmt(imb["value"], "x"), fmt(loss["value"], "s"),
                   "; the slowest rank is on the slow node, so this is likely the same problem" if on_slow_node else ""),
                [imb["id"], loss["id"], ph["ids"]["comp_max"], ph["ids"]["comp_mean"]], phase=ph["path"], rank=sr)

    # waiting inside MPI (consequence of imbalance elsewhere)
    for ph in A["phases"]:
        w = M.get("phase[%s].wait_mean" % ph["path"])
        if w["value"] >= TH["wait"] * wall:
            add("mpi_wait", w["value"], "Ranks wait inside MPI in phase '%s'" % ph["path"],
                "Ranks spend on average %s more inside MPI in '%s' than the fastest rank. This is waiting for late "
                "partners, typically caused by imbalance in the computation before it." % (fmt(w["value"], "s"), ph["path"]),
                [w["id"], ph["ids"]["mpi_min"], ph["ids"]["mpi_mean"]], phase=ph["path"])

    for g in A["ops"]:
        share = M.get("op[%s].share" % g["op"])
        tmin = M.get("op[%s].time_min" % g["op"])
        # costly collective even for the fastest rank: the operation itself, not waiting
        if g["kind"] in COLL_KINDS and g["op"] != "Barrier" and tmin["value"] >= TH["collective_intrinsic"] * wall:
            ev = [tmin["id"], share["id"]] + ([g["ids"]["per_call"]] if "per_call" in g["ids"] else [])
            add("collective_cost", tmin["value"], "MPI_%s is expensive even for the fastest rank" % g["op"],
                "Even the rank that arrives last (and so waits least) spends %s in MPI_%s, %s per call. That is the cost of "
                "the operation itself (network latency/bandwidth, message size, algorithm), not imbalance."
                % (fmt(tmin["value"], "s"), g["op"], fmt(M.get("op[%s].per_call" % g["op"])["value"], "s") if "per_call" in g["ids"] else "n/a"),
                ev, op=g["op"])
        # small messages
        if g["kind"] in P2P_KINDS and "small" in g["ids"]:
            sm = M.get("op[%s].small_frac" % g["op"])
            if sm["value"] >= 100 * TH["small_msg_fraction"] and share["value"] >= 100 * TH["small_msg_time"]:
                add("small_messages", statistics.mean(g["time"]), "MPI_%s sends mostly small messages" % g["op"],
                    "%s of MPI_%s messages are below 1 KiB and the call takes %s of the run: latency-bound communication; "
                    "aggregating messages would help." % (fmt(sm["value"], "%"), g["op"], fmt(share["value"], "%")),
                    [sm["id"], share["id"], g["ids"].get("msg_mean", sm["id"])], op=g["op"])
        # barriers
        if g["op"] == "Barrier" and share["value"] >= 100 * TH["barrier_time"]:
            add("synchronization", statistics.mean(g["time"]), "MPI_Barrier takes %s of the run" % fmt(share["value"], "%"),
                "Barriers cost %s per rank (%s calls). Barriers are rarely needed for correctness in SPMD codes; check whether "
                "they can be removed or replaced by the existing data dependencies."
                % (fmt(statistics.mean(g["time"]), "s"), M.get("op[Barrier].calls")["value"]),
                [share["id"], g["ids"]["calls"], g["ids"]["time_mean"]], op="Barrier")
        if g["op"] == "Barrier" and share["value"] < 100 * TH["barrier_time"]:
            iters = max((p["instances"] for p in A["phases"]), default=0)
            if iters >= 8 and sum(g["count"]) / n >= iters:
                add("synchronization", 0.0, "MPI_Barrier is called every iteration but is cheap here",
                    "Barriers run at least once per iteration (%s calls) but cost only %s of the run, because the ranks are "
                    "already synchronized when they reach them. They are probably unnecessary and would become costly if "
                    "the ranks drift apart." % (M.get("op[Barrier].calls")["value"], fmt(share["value"], "%")),
                    [share["id"], g["ids"]["calls"]], op="Barrier", info=True)
        # blocking sends that wait
        if g["op"] in ("Send", "Ssend") and share["value"] >= 100 * TH["send_wait"]:
            add("blocking_send", statistics.mean(g["time"]), "Blocking MPI_%s calls wait" % g["op"],
                "MPI_%s takes %s of the run. Blocking sends wait when the receiver is late or the message exceeds the eager "
                "limit (rendezvous); non-blocking sends or earlier receives would hide this." % (g["op"], fmt(share["value"], "%")),
                [share["id"], g["ids"]["time_mean"]], op=g["op"])

    # growth over iterations
    for ph in A["phases"]:
        gid = ph["ids"].get("growth")
        if gid:
            g = M.get("phase[%s].growth" % ph["path"])
            impact = max(0.0, (g["value"] / 100.0) * M.get("phase[%s].time_incl_mean" % ph["path"])["value"] / 2)
            if g["value"] >= 100 * TH["growth"] and ph.get("growth_half", 0.0) >= TH["growth"] / 2 and impact >= TH["growth_impact"] * wall:
                add("growth", impact,
                    "Phase '%s' gets slower over the run" % ph["path"],
                    "Instances of '%s' are %s slower at the end of the run than at the start (slowest rank per instance)."
                    % (ph["path"], fmt(g["value"], "%")), [gid, ph["ids"]["instances"]], phase=ph["path"])

    if A.get("gpu"):
        gpu_findings(A, M, add)

    mf = M.get("run.mpi_frac")
    if mf["value"] >= 100 * TH["mpi_bound"]:
        add("communication_bound", 0.0, "Communication-bound run",
            "%s of rank time is spent inside MPI." % fmt(mf["value"], "%"), [mf["id"]], info=True)
    elif mf["value"] <= 100 * TH["compute_bound"]:
        add("compute_bound", 0.0, "Communication is not a bottleneck",
            "Only %s of rank time is inside MPI; optimization effort belongs in the computation." % fmt(mf["value"], "%"),
            [mf["id"], M.get("run.comp_frac")["id"]], info=True)
    cov = M.get("run.marker_coverage")
    if cov["value"] < 100 * TH["coverage"]:
        add("coverage", 0.0, "Few phase markers",
            "Only %s of rank time is inside named phases; more markers would make the attribution more precise." % fmt(cov["value"], "%"),
            [cov["id"]], info=True)

    SYMPTOMS = {"mpi_wait"}
    is_symptom = lambda f: f["category"] in SYMPTOMS or f.get("symptom_of_imbalance")
    F.sort(key=lambda f: (f.get("info", False), bool(is_symptom(f)), -f["impact_s"]))
    for i, f in enumerate(F):
        f["id"] = "F%d" % (i + 1)
    causes = [f["id"] for f in F if f["category"] in ("load_imbalance", "slow_node", "gpu_imbalance", "shared_gpu")]
    for f in F:
        if is_symptom(f):
            f["symptom"] = True
            if causes:
                f["caused_by"] = causes
                f["text"] += " Likely a consequence of %s." % ", ".join(causes)
    for f in F:
        f["suggestion"] = suggestion(f)
    return F


def gpu_findings(A, M, add):
    G, wall, n = A["gpu"], A["wall"], A["n_ranks"]
    nodes, node_of = A["nodes"], A["node_of"]
    R = G["ranks"]
    mean = lambda xs: statistics.mean(xs) if xs else 0.0
    gi = G["ids"]
    specific = []   # (category, phase) of specific GPU findings, to avoid restating them as idle time

    # GPU imbalance: uneven kernel time per phase (by launching phase)
    for ph in A["phases"]:
        if "gpu_imbalance" not in ph["ids"]:
            continue
        imb, loss = M.get("phase[%s].gpu_imbalance" % ph["path"]), M.get("phase[%s].gpu_imbalance_loss" % ph["path"])
        if imb["value"] >= TH["gpu_imbalance_factor"] and loss["value"] >= TH["gpu_imbalance_loss"] * wall:
            sr = ph["gpu"]["slowest"]
            add("gpu_imbalance", loss["value"], "GPU work imbalance in phase '%s'" % ph["path"],
                "The kernels launched in '%s' run %s the mean GPU time on the slowest rank (%d, node %s), about %s more per rank "
                "than the mean; the other ranks' GPUs finish early and their ranks wait at the next synchronization or MPI call."
                % (ph["path"], fmt(imb["value"], "x"), sr, nodes[node_of[sr]], fmt(loss["value"], "s")),
                [imb["id"], loss["id"], ph["ids"]["gpu_kernel_max"], ph["ids"]["gpu_kernel_mean"]], phase=ph["path"], rank=sr)

    # ranks sharing a GPU while other GPUs on the node are unused
    groups = defaultdict(list)
    for i in range(n):
        for dev in G["dev_of"][i]:
            groups[dev[:2]].append(i)
    for (host, pci), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        on_host = {dev[1] for i in range(n) for dev in G["dev_of"][i] if dev[0] == host}
        visible = max(R[i]["visible"] for i in members)
        kb = mean([R[i]["kernel_busy"] for i in members])
        k = len(members)
        if visible > len(on_host):
            add("shared_gpu", kb * (k - 1) / k, "%d ranks share one GPU on %s while other GPUs there are unused" % (k, host),
                "Ranks %s on %s all run their kernels on GPU %s, although each sees %d GPUs and only %d of them are used. "
                "Their kernels compete for one device."
                % (", ".join(map(str, members)), host, pci, visible, len(on_host)), [gi["util"], gi["kernel"]], node=host, ranks=members)
        else:
            add("shared_gpu", 0.0, "%d ranks share GPU %s on %s" % (k, pci, host),
                "Ranks %s run their kernels on the same GPU. That can be intended (more ranks than GPUs, MPS), but the kernels "
                "compete for the device." % (", ".join(map(str, members)), pci), [gi["util"]], node=host, ranks=members, info=True)

    # MPI staged through host memory: host<->device copies around MPI in phases that move MPI data
    if M.get("gpu.mpi_device_frac") is None or M.get("gpu.mpi_device_frac")["value"] < 10:
        for ph in A["phases"]:
            path = ph["path"]
            cp = [sum(c["time"] for c in d["copies"] if c["phase"] == path and c["kind"] in ("DtoH", "HtoD") and not c.get("by_mpi")) for d in R]
            cb = sum(c["bytes"] for d in R for c in d["copies"] if c["phase"] == path and c["kind"] in ("DtoH", "HtoD") and not c.get("by_mpi"))
            mb = sum(a["bytes"][i] for a in A["sites"] if a["phase"] == path and a["kind"] in P2P_KINDS for i in range(n))
            if mb and cb and 0.3 <= cb / mb <= 3 and mean(cp) >= TH["gpu_staging"] * wall:
                specific.append(("copy", path))
                add("gpu_staging", mean(cp), "MPI data is staged through host memory in phase '%s'" % path,
                    "In '%s' the ranks copy about as many bytes between GPU and host as they send and receive with MPI, from host "
                    "buffers: the messages are staged through host memory. The copies take %s per rank on the GPU, plus the host "
                    "waiting for them." % (path, fmt(mean(cp), "s")), [gi["copy"]] + ([gi["mpi_dev"]] if "mpi_dev" in gi else []), phase=path)

    # pageable host memory in copies
    pg = M.get("gpu.pageable_copy_mean")
    if pg["value"] >= TH["gpu_pageable"] * wall:
        tops = sorted(G["copy_table"], key=lambda e: -e["time"])
        ev = [pg["id"]] + [e["ids"]["time"] for e in tops if e["host"] == "pageable"][:2]
        phs = sorted({c["phase"] for d in R for c in d["copies"] if "pageable" in (c["src"], c["dst"])})
        for p_ in phs:
            specific.append(("copy", p_))
        add("gpu_pageable", 0.6 * pg["value"], "Copies use pageable host memory",
            "Copies to or from pageable (not page-locked) host memory take %s per rank. Pageable copies are slower and block "
            "the host even when issued asynchronously (phases: %s)." % (fmt(pg["value"], "s"), ", ".join(phs)), ev, phases=phs)

    # CUDA allocations inside the run
    al = M.get("gpu.api_alloc_mean")
    if al["value"] >= TH["gpu_alloc"] * wall:
        per_ph = sorted(((mean(ph["gpu"]["api"]["alloc"]), ph["path"]) for ph in A["phases"] if "gpu" in ph), reverse=True)
        top = per_ph[0][1] if per_ph and per_ph[0][0] > 0 else "<main>"
        specific.append(("alloc", top))
        add("gpu_alloc", al["value"], "CUDA memory allocation costs %s per rank" % fmt(al["value"], "s"),
            "The ranks spend %s in cudaMalloc/cudaFree and similar calls (%s calls per rank), mostly in phase '%s'. cudaFree "
            "also synchronizes the device." % (fmt(al["value"], "s"), fmt(M.get("gpu.api_alloc_calls")["value"]), top),
            [al["id"], gi["api_n_alloc"]], phase=top)

    # launch-bound: many short kernels
    la = M.get("gpu.api_launch_mean")
    kdur = M.get("gpu.kernel_duration_mean")
    if la["value"] >= TH["gpu_launch"] * wall or (kdur and kdur["value"] < 20e-6 and M.get("gpu.kernels_mean")["value"] >= 1000
                                                 and M.get("gpu.util_mean")["value"] < TH["gpu_util_low"]):
        top = max((ph for ph in A["phases"] if "gpu" in ph), key=lambda ph: mean(ph["gpu"]["kcount"]), default=None)
        specific.append(("launch", top["path"] if top else ""))
        add("gpu_launch", la["value"], "Many short kernels: launch overhead matters",
            "The ranks launch %s kernels each, %s on average, and spend %s launching them%s. With kernels this short, launch "
            "latency leaves the GPU idle between them."
            % (fmt(M.get("gpu.kernels_mean")["value"]), fmt(kdur["value"], "s") if kdur else "n/a", fmt(la["value"], "s"),
               " (mostly in '%s')" % top["path"] if top else ""), [gi["kernels"], la["id"]] + ([kdur["id"]] if kdur else []),
            phase=top["path"] if top else None)

    # frequent synchronization: about one synchronizing call per kernel
    sync_n, launches = M.get("gpu.api_sync_calls")["value"], M.get("gpu.api_launch_calls")["value"]
    iters = max((p["instances"] for p in A["phases"]), default=0)
    if launches and sync_n >= TH["gpu_sync_per_kernel"] * launches and iters >= 8 and sync_n >= 8 * iters:
        sph = max((ph for ph in A["phases"] if "gpu" in ph), key=lambda ph: mean(ph["gpu"]["api_n"]["sync"]), default=None)
        idle = M.get("phase[%s].gpu_idle_mean" % sph["path"]) if sph else None
        impact = idle["value"] if idle else 0.0
        specific.append(("sync", sph["path"] if sph else ""))
        add("gpu_sync", impact, "The host synchronizes with the GPU after almost every kernel",
            "The ranks make %s synchronizing CUDA calls for %s kernel launches (%s per iteration), mostly in '%s'. Each one "
            "stops the host from queuing more work, so the GPU idles while the next launch is prepared."
            % (fmt(sync_n), fmt(launches), fmt(sync_n / iters), sph["path"] if sph else "?"),
            [gi["api_n_sync"], gi["api_n_launch"]] + ([idle["id"]] if idle else []), phase=sph["path"] if sph else None,
            info=impact < TH["gpu_idle"] * wall)

    # GPU idle while the host is busy elsewhere, per phase
    for ph in A["phases"]:
        if "gpu_idle_mean" not in ph["ids"]:
            continue
        idle = M.get("phase[%s].gpu_idle_mean" % ph["path"])
        if idle["value"] < TH["gpu_idle"] * wall:
            continue
        g = ph["gpu"]
        # what the host does while its GPU can be idle; time in explicit synchronization mostly overlaps GPU work,
        # so it explains idle time only when nothing else does
        # Blocking copies and synchronization mostly wait for running kernels: only the part not covered by GPU busy
        # time can explain idle time.
        blocking = mean(g["api"]["copy"]) + mean(g["api"]["sync"])
        residual = max(0.0, blocking - mean(g["busy"]))
        parts = {"cpu": mean(ph["comp"]), "mpi": mean(ph["mpi"]), "alloc": mean(g["api"]["alloc"]),
                 "copy": residual, "launch": mean(g["api"]["launch"])}
        dom = max(parts, key=parts.get)
        if parts[dom] <= 0.1 * idle["value"]:
            dom = "sync"
        parts["sync"] = mean(g["api"]["sync"])
        if (dom, ph["path"]) in specific or (dom in ("sync", "launch") and any(c in ("sync", "launch") for c, _ in specific)):
            continue
        wait_share = (statistics.mean(ph["mpi"]) - min(ph["mpi"])) / statistics.mean(ph["mpi"]) if statistics.mean(ph["mpi"]) else 0
        what = {"cpu": "computing on the CPU (%s per rank)" % fmt(parts["cpu"], "s"),
                "mpi": "inside MPI (%s per rank)" % fmt(parts["mpi"], "s"),
                "alloc": "allocating GPU memory (%s per rank)" % fmt(parts["alloc"], "s"),
                "copy": "in memory copies between host and GPU (%s per rank not overlapped by kernels)" % fmt(parts["copy"], "s"),
                "launch": "launching kernels (%s per rank)" % fmt(parts["launch"], "s"),
                "sync": "waiting in synchronizing CUDA calls (%s per rank)" % fmt(parts["sync"], "s")}[dom]
        add("gpu_idle", idle["value"], "GPU idle during phase '%s'" % ph["path"],
            "While the ranks are in '%s' their GPU is idle for %s per rank (%s of the phase). Meanwhile the host is mostly %s."
            % (ph["path"], fmt(idle["value"], "s"), fmt(M.get("phase[%s].gpu_idle_frac" % ph["path"])["value"], "%"), what),
            [idle["id"], ph["ids"]["gpu_idle_frac"], ph["ids"]["comp_mean"], ph["ids"]["mpi_mean"], ph["ids"]["cuda_mean"]],
            phase=ph["path"], dominant=dom, symptom_of_imbalance=(dom == "mpi" and wait_share >= 0.5))

    # summary notes
    u = M.get("gpu.util_mean")
    add("gpu_summary", 0.0, "GPUs are busy %s of the run" % fmt(u["value"], "%"),
        "Kernels keep the GPUs busy %s of the run on average (least busy rank %d: %s). Kernel time is %s per rank in %s launches; "
        "copies take %s." % (fmt(u["value"], "%"), M.get("gpu.util_min")["rank"], fmt(M.get("gpu.util_min")["value"], "%"),
                             fmt(M.get("gpu.kernel_time_mean")["value"], "s"), fmt(M.get("gpu.kernels_mean")["value"]),
                             fmt(M.get("gpu.copy_time_mean")["value"], "s")),
        [u["id"], gi["util_min"], gi["kernel"], gi["copy"]], info=True)
    md = M.get("gpu.mpi_device_frac")
    if md and md["value"] >= 10:
        add("gpu_aware_mpi", 0.0, "MPI uses GPU memory directly (GPU-aware MPI)",
            "%s of MPI payload bytes are passed from GPU memory, so messages are not staged through the host by the "
            "application." % fmt(md["value"], "%"), [md["id"]], info=True)


# ------------------------------------------------------------------------------------------------
def suggestion(f):
    """Concrete next step for a finding (rule-based; used without an LLM and given to the LLM as a hint)."""
    c = f["category"]
    ph, op, rank, node = f.get("phase"), f.get("op"), f.get("rank"), f.get("node")
    if c == "load_imbalance":
        return ("Rebalance the work in '%s': compare the work each rank gets there (cells, particles, rows) to confirm rank %s "
                "has more; use weighted partitioning (e.g. ParMETIS/Zoltan with work weights) or over-decompose and balance "
                "dynamically." % (ph, rank))
    if c == "slow_node":
        if f.get("confidence") == "low":
            return ("Rerun with the ranks mapped differently (e.g. srun --distribution=cyclic): if the slowness stays with node "
                    "%s it is the machine, if it moves with the ranks it is the decomposition." % node)
        return ("Check node %s (CPU frequency and temperature, other processes, memory/NUMA settings); rerun with "
                "sbatch --exclude=%s to confirm the run is faster without it." % (node, node))
    if c == "mpi_wait":
        return ("Fix the cause first (%s). To hide the remaining waits, overlap communication with computation: post "
                "receives and sends early and compute interior points while the messages are in flight."
                % ", ".join(f.get("caused_by", [])) if f.get("caused_by") else
                "Find which ranks arrive late at the MPI calls in '%s' (see the per-rank chart); overlap communication with "
                "computation to hide the waits." % ph)
    if c == "collective_cost":
        return ("Make fewer or smaller MPI_%s calls: combine several reductions into one call, use the non-blocking MPI_I%s "
                "and overlap it with independent work, or reduce how often it runs (e.g. check convergence every few "
                "iterations). Collective algorithm tuning in the MPI library may also help." % (op, op.lower()))
    if c == "small_messages":
        return ("Aggregate messages: pack the data for each neighbor into one buffer per exchange, or use persistent "
                "requests or neighborhood collectives.")
    if c == "synchronization":
        if f.get("info"):
            return "Remove the barriers unless they are needed for correctness; they turn any future imbalance into waiting."
        return "Remove barriers that are not needed for correctness, or replace them by the actual point-to-point dependencies."
    if c == "blocking_send":
        return ("Switch to MPI_Isend/MPI_Irecv with MPI_Waitall and post receives before sends; large messages above the "
                "eager limit block until the receiver arrives.")
    if c == "growth":
        return ("Look for growing work or data in '%s' (accumulating lists, memory growth, adaptive refinement); mark "
                "events with vibeprof_mark() to correlate the slowdown with them." % ph)
    if c == "communication_bound":
        return ("Communication dominates: try fewer ranks with more work each, MPI+threads, or overlapping communication "
                "with computation.")
    if c == "compute_bound":
        return ("Profile the computation itself with a node-level tool (perf, VTune, HPCToolkit): vectorization, memory "
                "bandwidth and cache use matter more than MPI here.")
    if c == "coverage":
        return "Add vibeprof_phase_begin/end markers around the main sections of the time loop."
    if c == "gpu_imbalance":
        return ("Balance the GPU work in '%s': give rank %s fewer cells/particles or weight the partitioning by measured "
                "kernel time; if the work is equal, check that GPU (clocks, throttling, PCIe/NVLink placement)." % (ph, rank))
    if c == "shared_gpu":
        if f.get("info"):
            return "If unintended, bind one rank per GPU (srun --gpus-per-task=1 --gpu-bind=closest, or cudaSetDevice(local_rank % ngpus))."
        return ("Bind one rank per GPU: srun --gpus-per-task=1 --gpu-bind=closest, or select the device in the code with "
                "cudaSetDevice(local_rank % num_gpus) using MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) for the local rank.")
    if c == "gpu_staging":
        return ("Pass device buffers to MPI directly (CUDA-aware MPI: Cray MPICH with MPICH_GPU_SUPPORT_ENABLED=1, Open MPI/UCX "
                "with CUDA, MVAPICH2-GDR), which removes these copies and lets the MPI library use GPUDirect RDMA.")
    if c == "gpu_pageable":
        return ("Allocate the host buffers used in copies with cudaMallocHost/cudaHostAlloc, or register existing ones with "
                "cudaHostRegister; then cudaMemcpyAsync overlaps with host work.")
    if c == "gpu_alloc":
        return ("Allocate GPU buffers once outside the time loop and reuse them, or use the stream-ordered allocator "
                "(cudaMallocAsync / a memory pool such as Kokkos' or RMM's).")
    if c == "gpu_launch":
        return ("Fuse small kernels, process more work per launch, or capture the launch sequence in a CUDA graph and replay it "
                "each iteration.")
    if c == "gpu_sync":
        return ("Remove synchronizations that are not needed between kernels on the same stream (work on one stream is already "
                "ordered); synchronize only before the host reads results or calls MPI, or use events.")
    if c == "gpu_idle":
        d = f.get("dominant")
        if d == "cpu":
            return ("Move this CPU work to the GPU, or overlap it: launch the next GPU kernels asynchronously before the CPU work "
                    "and synchronize after it.")
        if d == "mpi":
            return ("Overlap communication with GPU work: start the halo exchange, compute the interior on the GPU meanwhile "
                    "(separate stream), then the boundary; use GPU-aware MPI to avoid staging copies.")
        if d in ("sync", "launch"):
            return "Queue more work before synchronizing: fewer synchronizations, larger or fused kernels, CUDA graphs."
        if d == "alloc":
            return "Allocate once outside the loop or use cudaMallocAsync / memory pools."
        return "Use pinned host memory and asynchronous copies on a separate stream to overlap transfers with kernels."
    if c == "gpu_summary":
        return ("Compare GPU busy time with the GPU idle findings per phase to see where the GPU waits; Nsight Systems on one "
                "rank shows the timeline in detail.")
    if c == "gpu_aware_mpi":
        return "No action needed; make sure GPUDirect RDMA is enabled in the MPI library for inter-node messages."
    return ""


def facts_sheet(A, M, F, phase=None):
    """Compact text given to the LLM. Every number carries its metric id."""
    def line(key):
        m = M.get(key)
        return "[%s] %s: %s" % (m["id"], m["label"], fmt(m["value"], m["unit"])) if m else None

    out = []
    out.append("RUN")
    out.append("  program: %s" % (A["run"]["cmdline"] or A["run"]["exe"]))
    for k in ("run.ranks", "run.nodes", "run.wall", "run.comp_frac", "run.cuda_frac", "run.mpi_frac", "run.marker_coverage"):
        if line(k):
            out.append("  " + line(k))
    phs = [p for p in A["phases"] if phase is None or p["path"] == phase or p["path"].startswith(phase + "/")]
    out.append("PHASES (exclusive times unless noted; '/' = nesting)")
    for p in phs[:14]:
        out.append("  phase '%s':" % p["path"])
        for name in ("time_incl_mean", "share", "comp_mean", "comp_max", "imbalance", "imbalance_loss", "mpi_mean", "mpi_min",
                     "wait_mean", "instances", "growth", "cv", "cuda_mean", "cuda_sync_mean", "gpu_busy_mean", "gpu_idle_mean",
                     "gpu_idle_frac", "gpu_kernel_mean", "gpu_kernel_max", "gpu_imbalance", "gpu_imbalance_loss", "kernels"):
            l = line("phase[%s].%s" % (p["path"], name))
            if l:
                out.append("    " + l)
    if phase is None:
        out.append("MPI OPERATIONS (top by time)")
        for g in A["ops"][:8]:
            out.append("  MPI_%s (%s):" % (g["op"], g["kind"]))
            for name in ("share", "time_mean", "time_min", "time_max", "calls", "per_call", "bytes", "msg_mean", "small_frac"):
                l = line("op[%s].%s" % (g["op"], name))
                if l:
                    out.append("    " + l)
    if phase is None and A.get("gpu"):
        out.append("GPU (computation above is CPU work; CUDA calls are counted separately)")
        for m in M.items:
            if m["key"].startswith("gpu."):
                out.append("  " + line(m["key"]))
        out.append("  top kernels:")
        for a in A["gpu"]["kernels"][:8]:
            if "ids" in a:
                out.append("    kernel %s (phase '%s'): %s" % (a["display"], a["phase"], "; ".join(
                    line("kernel[%s].%s" % (a["display"], k)) for k in ("time_mean", "share", "count", "duration"))))
        out.append("  copies:")
        for e in A["gpu"]["copy_table"][:6]:
            out.append("    %s%s%s: %s; %s" % (e["kind"], " " + e["host"] + " host memory" if e["host"] else "", " (issued by MPI)" if e["by_mpi"] else "",
                                             line("copy[%s|%s|%s].time_mean" % (e["kind"], e["host"] or "-", "mpi" if e["by_mpi"] else "app")),
                                             line("copy[%s|%s|%s].bytes" % (e["kind"], e["host"] or "-", "mpi" if e["by_mpi"] else "app"))))
    out.append("CALL SITES (top by time)")
    for a in A["sites"][:12]:
        if "ids" not in a or (phase and not (a["phase"] == phase or a["phase"].startswith(phase + "/"))):
            continue
        key = "site[%s|%s|%s]" % (a["phase"], a["op"], a["site"])
        out.append("  MPI_%s at %s in phase '%s':" % (a["op"], a["site"], a["phase"]))
        for name in ("share", "time_mean", "wait", "calls", "bytes"):
            l = line(key + "." + name)
            if l:
                out.append("    " + l)
    if phase is None:
        node_lines = [line("node[%s].comp_rel" % nm) for nm in A["nodes"]] if len(A["nodes"]) > 1 else []
        node_lines = [l for l in node_lines if l]
        if node_lines:
            out.append("NODES")
            out += ["  " + l for l in node_lines]
        rl = [m for m in M.items if m["key"].startswith("rank[")]
        if rl:
            out.append("SLOWEST RANKS (computation)")
            out += ["  [%s] rank %d on node %s: %s" % (m["id"], m["rank"], m["node"], fmt(m["value"], "x") + " of the median rank") for m in rl]
        cl = [line(k) for k in ("comm.p2p_bytes", "comm.offnode_frac", "comm.partners_mean", "comm.partners_max")]
        cl = [l for l in cl if l]
        if cl:
            out.append("COMMUNICATION PATTERN")
            out += ["  " + l for l in cl]
    out.append("CANDIDATE FINDINGS (rule-based; impact = estimated seconds per rank)")
    for f in F:
        if phase and f.get("phase") not in (None, phase) and not str(f.get("phase", "")).startswith(phase + "/"):
            continue
        out.append("  [%s] %s%s. %s Evidence: %s. Suggested next step: %s" % (
            f["id"], f["title"], "" if f.get("info") else " (impact %s, %s of wall)" % (fmt(f["impact_s"], "s"), fmt(f["impact_pct"], "%")),
            f["text"], ", ".join(f["evidence"]), f.get("suggestion", "")))
    return "\n".join(out)


SYSTEM = ("You are an expert HPC performance engineer. You explain MPI profiling data to the scientists who wrote the "
          "application. You are precise, concise and never invent measurements.")

INSTRUCTIONS = """Write a performance report for this run, based only on the facts below.

Rules:
- Every number you mention must come from the facts and must be followed by its id in brackets, e.g. "41.2% [M7]".
- Refer to findings by id, e.g. [F2]. You may merge findings that describe the same cause, and say so.
- Do not invent numbers, ranks, nodes, phases or call sites. If the data cannot answer something, say so.
- Distinguish waiting (imbalance elsewhere) from the intrinsic cost of an MPI operation.

Format (Markdown):
## Summary
Two or three sentences: is the run limited by computation, communication or imbalance, and what matters most.
## Top issues
A numbered list, most important first. For each: what happens, evidence with ids, the likely cause, and one concrete next step (a code change or an experiment).
## What looks healthy
One or two bullets.
## Caveats
One or two bullets about the limits of this data.

FACTS
"""

PHASE_INSTRUCTIONS = """Summarize the performance of phase '{phase}' in at most 120 words, based only on the facts below.
Every number must be followed by its id in brackets, e.g. "1.4x [M12]". Do not invent anything.

FACTS
"""

BRIEF_INSTRUCTIONS = """Write a short performance brief (at most 200 words) for the scientist who ran this program, based only on the facts below.

Rules:
- Every number must come from the facts and be followed by its id in brackets, e.g. "41.2% [M7]". Refer to findings as [F2].
- Do not invent anything. Merge findings that share one cause.

Format (Markdown):
**Verdict:** one or two sentences on what limits this run.
**Top issues:** at most three numbered items, most important first; each is one sentence of evidence, then "Suggestion:" and one concrete action.

FACTS
"""

FINAL_INSTRUCTIONS = INSTRUCTIONS.replace("FACTS\n", "PHASE NOTES (written earlier from the same data; ids refer to the facts)\n{notes}\n\nFACTS\n")


def build_prompts(A, M, F, map_reduce=False, max_tokens=900, brief=False):
    sheet = facts_sheet(A, M, F)
    msgs = lambda u: {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]}
    if brief:
        r = msgs(BRIEF_INSTRUCTIONS + sheet)
        r["id"] = "brief"
        r["max_tokens"] = min(max_tokens, 400)
        return sheet, [r]
    if not map_reduce:
        r = msgs(INSTRUCTIONS + sheet)
        r["id"] = "report"
        r["max_tokens"] = max_tokens
        return sheet, [r]
    out = []
    for p in A["phases"]:
        if p["depth"] == 0:
            r = msgs(PHASE_INSTRUCTIONS.format(phase=p["path"]) + facts_sheet(A, M, F, phase=p["path"]))
            r["id"] = "phase:" + p["path"]
            r["max_tokens"] = 220
            out.append(r)
    return sheet, out


def brief_report(A, M, F, top=3):
    """Short interpretation + suggestions, no LLM."""
    g = lambda k: fmt(M.get(k)["value"], M.get(k)["unit"])
    causes = [f for f in F if not f.get("info") and not f.get("symptom")]
    symptoms = [f for f in F if f.get("symptom")]
    info = [f for f in F if f.get("info")]
    mf = M.get("run.mpi_frac")["value"]
    if causes:
        main = causes[0]
        limit = ("communication" if mf >= 100 * TH["mpi_bound"] else "computation") if not causes else (main["title"] if re.match(r"[A-Z]{2}|[A-Z]\w*_", main["title"]) else main["title"][0].lower() + main["title"][1:])
        verdict = ("%s ranks ran for %s, with %s of rank time inside MPI. The biggest opportunity: %s (about %s of the run)."
                   % (g("run.ranks"), g("run.wall"), g("run.mpi_frac"), limit, fmt(main["impact_pct"], "%")))
        if symptoms and any(s.get("caused_by") for s in symptoms):
            waited = sum(s["impact_s"] for s in symptoms if s.get("caused_by"))
            src = sorted({c for s_ in symptoms for c in s_.get("caused_by", [])}, key=lambda x: int(x[1:]))
            verdict += (" Most of the MPI time is ranks waiting for each other (≈%s per rank), a consequence of %s."
                        % (fmt(waited, "s"), "it" if src == [main["id"]] else ", ".join(src)))
    else:
        verdict = ("%s ranks ran for %s, with %s of rank time inside MPI; no significant imbalance or communication problem "
                   "was found." % (g("run.ranks"), g("run.wall"), g("run.mpi_frac")))
    out = ["# Performance brief", "", "`%s`" % (A["run"]["cmdline"] or A["run"]["exe"]), "", "**Verdict:** " + verdict, ""]
    items = causes[:top] or symptoms[:top]
    if items:
        out.append("**Top issues:**")
        out.append("")
        for i, f in enumerate(items, 1):
            out.append("%d. **%s** (%s, ≈%s of the run). %s" % (i, f["title"], f["id"], fmt(f["impact_pct"], "%"), f["text"]))
            out.append("   *Suggestion:* " + f["suggestion"])
        out.append("")
    if info:
        out.append("**Also noted:** " + " ".join("%s (%s)" % (f["title"] + ".", f["id"]) for f in info))
        out.append("")
    return "\n".join(out)


def deterministic_report(A, M, F):
    g = lambda k: fmt(M.get(k)["value"], M.get(k)["unit"])
    out = ["# vibeprof report (deterministic)", ""]
    out.append("**Program:** `%s`  " % (A["run"]["cmdline"] or A["run"]["exe"]))
    out.append("**Ranks:** %s on %s node(s), **wall time** %s, computation %s / MPI %s of rank time, %s of time in named phases."
               % (g("run.ranks"), g("run.nodes"), g("run.wall"), g("run.comp_frac"), g("run.mpi_frac"), g("run.marker_coverage")))
    out += ["", "## Findings", ""]
    for f in F:
        if f.get("info"):
            out.append("- **%s** (%s): %s" % (f["title"], f["id"], f["text"]))
        else:
            out.append("- **%s** (%s, impact ≈ %s per rank, %s of wall): %s" % (f["title"], f["id"], fmt(f["impact_s"], "s"),
                                                                              fmt(f["impact_pct"], "%"), f["text"]))
        if f.get("suggestion"):
            out.append("  *Suggestion:* " + f["suggestion"])
    out += ["", "## Phases", "", "| phase | share of wall | mean comp | max comp | imbalance | mean MPI | waiting | instances |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in A["phases"]:
        k = "phase[%s]." % p["path"]
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (p["path"], g(k + "share"), g(k + "comp_mean"), g(k + "comp_max"),
                                                                g(k + "imbalance"), g(k + "mpi_mean"), g(k + "wait_mean"), g(k + "instances")))
    out += ["", "## Top MPI call sites", "", "| MPI call | site | phase | share of wall | mean time | waiting | calls |", "|---|---|---|---:|---:|---:|---:|"]
    for a in A["sites"][:10]:
        if "ids" not in a:
            continue
        k = "site[%s|%s|%s]." % (a["phase"], a["op"], a["site"])
        out.append("| MPI_%s | `%s` | %s | %s | %s | %s | %s |" % (a["op"], a["site"], a["phase"], g(k + "share"), g(k + "time_mean"),
                                                                g(k + "wait"), g(k + "calls")))
    if A.get("gpu"):
        G = A["gpu"]
        out += ["", "## GPU", "", "GPU busy with kernels %s of the run (mean over ranks); %s of rank time inside CUDA calls." % (
            g("gpu.util_mean"), g("run.cuda_frac")), "",
            "| phase | GPU busy | GPU idle | kernel time launched | kernel imbalance | host in CUDA calls | kernels |", "|---|---:|---:|---:|---:|---:|---:|"]
        for p in A["phases"]:
            k = "phase[%s]." % p["path"]
            if M.get(k + "gpu_busy_mean"):
                gg = lambda x: g(k + x) if M.get(k + x) else ""
                out.append("| %s | %s | %s | %s | %s | %s | %s |" % (p["path"], gg("gpu_busy_mean"), gg("gpu_idle_mean"), gg("gpu_kernel_mean"),
                                                                    gg("gpu_imbalance"), gg("cuda_mean"), gg("kernels")))
        out += ["", "| kernel | phase | GPU time per rank | launches | mean duration |", "|---|---|---:|---:|---:|"]
        for a in G["kernels"][:10]:
            if "ids" in a:
                k = "kernel[%s]." % a["display"]
                out.append("| `%s` | %s | %s | %s | %s |" % (a["display"], a["phase"], g(k + "time_mean"), g(k + "count"), g(k + "duration")))
        out += ["", "| copy | host memory | by MPI | time per rank | bytes (all ranks) |", "|---|---|---|---:|---:|"]
        for e in G["copy_table"][:8]:
            out.append("| %s | %s | %s | %s | %s |" % (e["kind"], e["host"] or "-", "yes" if e["by_mpi"] else "", fmt(e["time"], "s"), fmt(e["bytes"], "B")))
    if A["warnings"]:
        out += ["", "## Collector warnings", ""] + ["- " + w for w in A["warnings"]]
    return "\n".join(out) + "\n"


def to_json(A, M, F):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    return clean({"analysis": A, "metrics": M.items, "findings": F})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--map-reduce", action="store_true", help="one prompt per top-level phase (pass 1 of a two-pass report)")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--brief", action="store_true", help="short report: verdict + top issues with suggestions (brief.md, brief prompt)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.profile_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    run, ranks = load(a.profile_dir)
    A, M = analyze(run, ranks)
    F = findings(A, M)
    sheet, prompts = build_prompts(A, M, F, a.map_reduce, a.max_tokens, a.brief)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(to_json(A, M, F), f, indent=1)
    with open(os.path.join(out, "findings.json"), "w") as f:
        json.dump(F, f, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(deterministic_report(A, M, F))
    with open(os.path.join(out, "brief.md"), "w") as f:
        f.write(brief_report(A, M, F))
    with open(os.path.join(out, "facts.txt"), "w") as f:
        f.write(sheet + "\n")
    with open(os.path.join(out, "prompts.jsonl"), "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print("vibeprof: %d ranks, %d phases, %d findings -> %s" % (A["n_ranks"], len(A["phases"]), len(F), out))
    for fd in F:
        print("  [%s] %s%s" % (fd["id"], fd["title"], "" if fd.get("info") else " (impact %s, %s)" % (fmt(fd["impact_s"], "s"), fmt(fd["impact_pct"], "%"))))


if __name__ == "__main__":
    main()
