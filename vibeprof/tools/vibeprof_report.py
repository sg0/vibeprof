#!/usr/bin/env python3
"""vibeprof_report.py - one command from a profile directory to a finished report.

Steps:
  1. analyze the profile (metrics with ids, rule-based findings)
  2. write summary.md (full, deterministic) and brief.md (verdict + top issues + suggestions)
  3. write the interactive report.html (and PNG figures with --png)
  4. optionally (--model) let an LLM interpret the facts, running the prompts through
     vibeprof_llm on the cluster itself (no server, no network)
  5. validate the LLM text: every [M#]/[F#] must exist and every number must appear in the data
  6. write report.md and report.html with the interpretation and the validation result

Examples:
  vibeprof_report.py $PROFILE                         # rule-based brief, plots
  vibeprof_report.py $PROFILE --brief --model m.gguf  # short LLM interpretation + suggestions
  vibeprof_report.py $PROFILE --model m.gguf --map-reduce --launcher "srun -n 4"
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vibeprof_analyze as LA  # noqa: E402
import vibeprof_plot as LP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LLM_BIN = os.path.normpath(os.path.join(HERE, "..", "..", "build", "vibeprof_llm"))


# ------------------------------------------------------------------------------------------------
# LLM execution through vibeprof_llm
def run_llm(prompts, a, work, tag):
    """prompts: list of {"id", "messages", "max_tokens"}; returns {id: text}."""
    pin = os.path.join(work, "llm-%s-in.jsonl" % tag)
    pout = os.path.join(work, "llm-%s-out.jsonl" % tag)
    with open(pin, "w") as f:
        for p in prompts:
            f.write(json.dumps({"messages": p["messages"], "max_tokens": p["max_tokens"]}) + "\n")
    cmd = shlex.split(a.launcher) + [a.llm_bin, "-m", a.model, "-p", pin, "-n", str(len(prompts)),
                                     "--chat", a.chat, "--temp", "0", "--osl-from-data", "--ignore-eos", "off",
                                     "-c", str(a.ctx), "--output", pout] + shlex.split(a.llm_args)
    print("vibeprof: running", " ".join(shlex.quote(c) for c in cmd), flush=True)
    log = os.path.join(work, "llm-%s.log" % tag)
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(pout):
        raise RuntimeError("LLM run failed (exit %d); see %s" % (rc, log))
    out = {}
    with open(pout) as f:
        for line in f:
            r = json.loads(line)
            out[prompts[r["record"]]["id"]] = r["text"].strip()
    missing = [p["id"] for p in prompts if p["id"] not in out]
    if missing:
        raise RuntimeError("LLM produced no text for %s; see %s" % (", ".join(missing), log))
    return out


# ------------------------------------------------------------------------------------------------
# validation
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def allowed_numbers(A, M, F, sheet):
    vals = set()
    for m in M.items:
        v = m["value"]
        if isinstance(v, (int, float)) and v == v:
            vals.update({float(v), float(v) * 1e3, float(v) * 1e6})
    for f in F:
        for k in ("impact_s", "impact_pct"):
            if isinstance(f.get(k), (int, float)):
                vals.update({float(f[k]), float(f[k]) * 1e3})
    for t in NUM.findall(sheet):
        try:
            vals.add(float(t.replace(",", "")))
        except ValueError:
            pass
    vals.update(float(i) for i in range(0, max(11, A["n_ranks"] + 1)))
    return sorted(vals)


def number_ok(tok, vals):
    s = tok.replace(",", "")
    try:
        x = float(s)
    except ValueError:
        return True
    # tolerance: half a unit of the last digit written, plus 0.6% for rounding of derived values
    dec = len(s.split(".")[1]) if "." in s and "e" not in s.lower() else 0
    tol = 0.5 * 10 ** (-dec) + 0.006 * abs(x)
    return any(abs(abs(x) - abs(v)) <= tol for v in vals)


def validate(text, A, M, F, sheet):
    known = {m["id"] for m in M.items} | {f["id"] for f in F}
    cites = re.findall(r"\[((?:[MF]\d+)(?:\s*,\s*[MF]\d+)*)\]", text)
    ids = [i.strip() for c in cites for i in c.split(",")]
    unknown = sorted({i for i in ids if i not in known}, key=lambda x: (x[0], int(x[1:])))
    # numbers outside citations, list markers, headings and code
    body = re.sub(r"`[^`]*`", " ", text)
    body = re.sub(r"\[[MF]\d+(?:\s*,\s*[MF]\d+)*\]", " ", body)
    body = re.sub(r"(?m)^\s*(?:\d+[.)]|#+)\s", " ", body)
    body = re.sub(r"\b[\w./-]+\.(?:c|cc|cpp|cxx|f|f90|F90|h|hpp):\d+", " ", body)  # source locations
    body = re.sub(r"\b(?:MPI|MPI_\w+|[A-Za-z]+\d+[A-Za-z\d]*)\b", " ", body)      # names like node12, MPI-3
    vals = allowed_numbers(A, M, F, sheet)
    unsupported = []
    for tok in NUM.findall(body):
        if not number_ok(tok, vals) and tok not in unsupported:
            unsupported.append(tok)
    return {"citations": len(ids), "unknown_ids": unknown, "unsupported": unsupported,
            "ok": bool(ids) and not unknown and not unsupported}


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_dir")
    ap.add_argument("-o", "--out", help="output directory (default: PROFILE/analysis)")
    ap.add_argument("--png", action="store_true", help="also write static PNG figures")
    ap.add_argument("--brief", action="store_true", help="short interpretation: verdict + top issues + suggestions")
    ap.add_argument("--model", help="GGUF model for the LLM interpretation (omit for the rule-based report only)")
    ap.add_argument("--map-reduce", action="store_true", help="LLM: one note per top-level phase, then a combined report")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--llm-bin", default=DEFAULT_LLM_BIN, help="path to vibeprof_llm (default: %(default)s)")
    ap.add_argument("--launcher", default=os.environ.get("VIBEPROF_LLM_LAUNCHER", "mpirun -np 2"),
                    help='MPI launcher for the LLM step, e.g. "srun -n 4 --overlap" (env VIBEPROF_LLM_LAUNCHER)')
    ap.add_argument("--ctx", type=int, default=8192, help="LLM context size")
    ap.add_argument("--chat", default="on", choices=["on", "off", "auto"], help="apply the model's chat template")
    ap.add_argument("--llm-args", default="", help="extra vibeprof_llm arguments, e.g. \"-ngl 99\"")
    a = ap.parse_args()

    out = a.out or os.path.join(a.profile_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    run, ranks = LA.load(a.profile_dir)
    A, M = LA.analyze(run, ranks)
    F = LA.findings(A, M)
    sheet = LA.facts_sheet(A, M, F)
    brief = LA.brief_report(A, M, F)
    w = lambda name, s: open(os.path.join(out, name), "w").write(s)
    w("metrics.json", json.dumps(LA.to_json(A, M, F), indent=1))
    w("findings.json", json.dumps(F, indent=1))
    w("summary.md", LA.deterministic_report(A, M, F))
    w("brief.md", brief)
    w("facts.txt", sheet + "\n")
    print("vibeprof: %d ranks, %d phases, %d findings -> %s" % (A["n_ranks"], len(A["phases"]), len(F), out))

    llm = None
    if a.model:
        try:
            if a.map_reduce and not a.brief:
                _, phase_prompts = LA.build_prompts(A, M, F, map_reduce=True, max_tokens=a.max_tokens)
                notes = run_llm(phase_prompts, a, out, "phases") if phase_prompts else {}
                note_txt = "\n".join("[%s]\n%s\n" % (k.split(":", 1)[1], v) for k, v in notes.items()) or "(none)"
                final = {"id": "report", "max_tokens": a.max_tokens,
                         "messages": [{"role": "system", "content": LA.SYSTEM},
                                      {"role": "user", "content": LA.FINAL_INSTRUCTIONS.replace("{notes}", note_txt) + sheet}]}
                text = run_llm([final], a, out, "final")["report"]
                w("phase_notes.md", note_txt)
            else:
                _, prompts = LA.build_prompts(A, M, F, max_tokens=a.max_tokens, brief=a.brief)
                text = run_llm(prompts, a, out, "brief" if a.brief else "report")[prompts[0]["id"]]
            check = validate(text, A, M, F, sheet)
            llm = {"text": text, "check": check, "model": os.path.basename(a.model)}
            print("vibeprof: LLM text: %d citations, %d unknown ids, %d unsupported numbers"
                  % (check["citations"], len(check["unknown_ids"]), len(check["unsupported"])))
        except RuntimeError as e:
            print("vibeprof: %s - falling back to the rule-based brief" % e, file=sys.stderr)

    md = []
    if llm:
        c = llm["check"]
        md += ["# Performance report", "", "*Interpretation written by %s from the measured facts.*" % llm["model"], "",
               llm["text"], "", "---", "",
               "**Validation:** %s%d citations; unknown ids: %s; numbers not found in the data: %s." % (
                   "" if c["citations"] else "NO CITATIONS - the text is not grounded in the data; ", c["citations"], ", ".join(c["unknown_ids"]) or "none", ", ".join(c["unsupported"]) or "none"),
               "Treat flagged statements with care; the rule-based findings below are computed directly from the profile.", ""]
    md.append(brief if not llm else "")
    md.append(LA.deterministic_report(A, M, F))
    w("report.md", "\n".join(md))
    LP.render_html(A, M, F, os.path.join(out, "report.html"), brief, llm)
    print("vibeprof: wrote", os.path.join(out, "report.md"))
    print("vibeprof: wrote", os.path.join(out, "report.html"))
    if a.png:
        for f in LP.render_png(A, M, F, out):
            print("vibeprof: wrote", f)


if __name__ == "__main__":
    main()
