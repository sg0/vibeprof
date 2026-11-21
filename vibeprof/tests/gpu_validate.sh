#!/usr/bin/env bash
# Validate vibeprof's GPU analysis: run gpu_toy with injected problems and check the findings.
#
#   tests/gpu_validate.sh sim  [OUTDIR]     simulated GPU (no GPU needed; uses build/gpu_toy_sim)
#   tests/gpu_validate.sh real [OUTDIR]     real NVIDIA GPUs (uses build/gpu_toy; run inside an allocation)
#
# Environment: LAUNCH (default "mpirun -np 2" for sim, "srun -n 2 --gpus-per-task=1" for real),
# GPU_AWARE=1 to include the GPU-aware MPI scenario on real systems (needs CUDA-aware MPI).
# Prints one line per scenario: expected finding category, PASS/FAIL, and the findings found.
# Send OUTDIR/summary.txt (and OUTDIR/*/analysis/findings.json) back for review.
set -uo pipefail
mode="${1:-sim}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="$(cd "$here/.." && pwd)"
B="${VIBEPROF_BUILD:-$L/build}"
out="${2:-gpu-validate-$mode}"
mkdir -p "$out"; out="$(cd "$out" && pwd)"
ITERS="${ITERS:-40}"

if [[ "$mode" == sim ]]; then
  toy="$B/gpu_toy_sim"
  LAUNCH="${LAUNCH:-mpirun -np 2}"
  export VIBEPROF_CUDA_LIB="$B/libfakecuda.so" VIBEPROF_CUPTI_LIB="$B/libfakecuda.so"
  export VIBEPROF_FAKE_PPN="${VIBEPROF_FAKE_PPN:-1}"   # 2 ranks on 2 pretend nodes
else
  toy="$B/gpu_toy"
  LAUNCH="${LAUNCH:-srun -n 2 --gpus-per-task=1}"
fi
[[ -x "$toy" ]] || { echo "missing $toy (build vibeprof; the real toy needs nvcc)" >&2; exit 2; }

summary="$out/summary.txt"
: > "$summary"
pass=0; fail=0

# run NAME EXPECT [ENV=VAL ...] -- toy args
#   EXPECT: comma-separated finding categories that must appear ("none" = no non-informational finding)
#   a category may be qualified with a phase: gpu_idle@step/host
run() {
  local name="$1" expect="$2"; shift 2
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  local d="$out/$name"
  rm -rf "$d"
  env "${envs[@]}" VIBEPROF_DIR="$d" LD_PRELOAD="$B/libvibeprof.so" $LAUNCH "$toy" --iters "$ITERS" "$@" > "$d.log" 2>&1
  python3 "$L/tools/vibeprof_analyze.py" "$d" > /dev/null 2>> "$d.log" || { echo "$name: analysis failed (see $d.log)" | tee -a "$summary"; fail=$((fail+1)); return; }
  local res
  res=$(python3 - "$d/analysis/findings.json" "$expect" <<'EOF'
import json, sys
F = json.load(open(sys.argv[1]))
exp = [e for e in sys.argv[2].split(",") if e]
real = [f for f in F if not f.get("info")]
got = lambda cat, ph=None: any(f["category"] == cat and (ph is None or f.get("phase") == ph) for f in F)
ok = True
for e in exp:
    if e == "none":
        ok &= not [f for f in real if f["category"].startswith("gpu") or f["category"] in ("shared_gpu", "load_imbalance")]
    else:
        cat, _, ph = e.partition("@")
        ok &= got(cat, ph or None)
desc = "; ".join("%s%s%s" % (f["category"], "@" + f["phase"] if f.get("phase") else "", "(info)" if f.get("info") else "") for f in F)
print(("PASS" if ok else "FAIL") + "  " + desc)
EOF
)
  printf '%-16s expect %-34s %s\n' "$name" "$expect" "$res" | tee -a "$summary"
  [[ "$res" == PASS* ]] && pass=$((pass+1)) || fail=$((fail+1))
}

W=20000
run baseline       none                               -- --work-us $W
run gpu_imbalance  gpu_imbalance@step/compute         -- --work-us $W --gpu-imbalance 0.5
run host_work      gpu_idle@step/host                 -- --work-us $W --host-us 3000
run staging        gpu_staging@step/halo              -- --work-us $W --halo-kb 32768
if [[ "$mode" == sim || "${GPU_AWARE:-0}" == 1 ]]; then
  run gpu_aware    gpu_aware_mpi                      -- --work-us $W --halo-kb 32768 --gpu-aware
fi
run pageable       gpu_pageable                       -- --work-us $W --halo-kb 8192 --pageable
run alloc          gpu_alloc                          -- --work-us $W --alloc-each
run tiny_sync      gpu_sync                           -- --work-us 2000 --tiny 40 --sync-each
if [[ "$mode" == sim ]]; then
  run shared_gpu   shared_gpu                         FAKECUDA_NDEV=2 VIBEPROF_FAKE_PPN=2 -- --work-us $W --bind zero
  run one_per_gpu  none                               FAKECUDA_NDEV=2 VIBEPROF_FAKE_PPN=2 -- --work-us $W --bind local
else
  # one node with >= 2 GPUs, both ranks on it, all GPUs visible to each rank
  saved="$LAUNCH"; LAUNCH="${SHARED_LAUNCH:-srun -N 1 -n 2 --gpus=2 --gpu-bind=none}"
  run shared_gpu   shared_gpu                         -- --work-us $W --bind zero
  LAUNCH="$saved"
fi
echo "passed $pass, failed $fail" | tee -a "$summary"
[[ $fail -eq 0 ]]
