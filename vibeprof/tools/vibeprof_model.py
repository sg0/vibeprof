#!/usr/bin/env python3
"""vibeprof_model.py - resolve a model name to a local GGUF file, downloading it once if needed.

A model is either a local path (returned unchanged) or a Hugging Face file:

    hf:<owner>/<repo>/<file.gguf>[@<revision>]
    hf:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf

Downloads go to $VIBEPROF_MODEL_DIR (default ~/.cache/vibeprof/models)/<owner>/<repo>/<file> and are reused
afterwards. Split models (...-00001-of-00003.gguf) fetch all parts; pass the first part. Downloads are atomic
(written to .part, then renamed) and resume an interrupted .part. HTTPS_PROXY / https_proxy are honored
(on ALCF Polaris compute nodes: export https_proxy=http://proxy.alcf.anl.gov:3128). HF_TOKEN is sent for gated
models.

Usage:
    vibeprof_model.py hf:owner/repo/file.gguf        # prints the local path (downloads if missing)
    vibeprof_llm -m "$(vibeprof_model.py hf:...)" ...
"""
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SPLIT = re.compile(r"^(?P<stem>.*)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")


def cache_dir():
    return os.environ.get("VIBEPROF_MODEL_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "vibeprof", "models")


def parse_hf(spec):
    """'hf:owner/repo/path/file.gguf@rev' -> (owner, repo, path, rev)"""
    body = spec[3:]
    rev = "main"
    if "@" in body:
        body, rev = body.rsplit("@", 1)
    parts = body.strip("/").split("/")
    if len(parts) < 3 or not parts[-1].endswith(".gguf"):
        raise ValueError("expected hf:<owner>/<repo>/<file.gguf>[@revision], got %r" % spec)
    return parts[0], parts[1], "/".join(parts[2:]), rev


def _download(url, dest, log):
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "vibeprof"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:            # .part already complete
            os.replace(tmp, dest)
            return
        hint = " (gated model? set HF_TOKEN)" if e.code in (401, 403) else ""
        raise RuntimeError("download failed: HTTP %d for %s%s" % (e.code, url, hint)) from None
    except urllib.error.URLError as e:
        raise RuntimeError("download failed: %s for %s (no internet on this node? set https_proxy, or "
                           "download on a login node)" % (e.reason, url)) from None
    if have and resp.status != 206:           # server ignored the range: start over
        have = 0
    total = resp.headers.get("Content-Length")
    total = int(total) + have if total else None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    done, t0, last = have, time.time(), 0.0
    with open(tmp, "ab" if have else "wb") as f:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last >= 5 or (total and done == total):
                last = now
                rate = (done - have) / max(now - t0, 1e-6) / 1e6
                pct = " %5.1f%%" % (100.0 * done / total) if total else ""
                log("vibeprof: %s %.2f GB%s  %.0f MB/s" % (os.path.basename(dest), done / 1e9, pct, rate))
    if total and done != total:
        raise RuntimeError("download incomplete: %d of %d bytes (run again to resume)" % (done, total))
    os.replace(tmp, dest)


def resolve(spec, log=lambda m: print(m, file=sys.stderr, flush=True)):
    """Local path for a model spec; downloads hf: models (all split parts) into the cache when missing."""
    if not spec.startswith("hf:"):
        return spec
    owner, repo, path, rev = parse_hf(spec)
    base = os.path.join(cache_dir(), owner, repo) if rev == "main" else os.path.join(cache_dir(), owner, repo + "@" + rev)
    m = SPLIT.match(path)
    files = [path] if not m else ["%s-%05d-of-%s.gguf" % (m["stem"], i, m["n"]) for i in range(1, int(m["n"]) + 1)]
    for fpath in files:
        dest = os.path.join(base, fpath)
        if os.path.exists(dest):
            continue
        url = "%s/%s/%s/resolve/%s/%s" % (HF_ENDPOINT, owner, repo, urllib.parse.quote(rev, safe=""), urllib.parse.quote(fpath))
        log("vibeprof: downloading %s -> %s" % (url, dest))
        _download(url, dest, log)
    return os.path.join(base, files[0])


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    try:
        print(resolve(sys.argv[1]))
    except (ValueError, RuntimeError) as e:
        print("vibeprof_model: %s" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
