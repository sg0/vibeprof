# vibeprof — MPI profiling with phase markers, findings and an optional LLM interpretation

vibeprof profiles unmodified MPI (SPMD) applications. It intercepts every MPI call through PMPI
with `LD_PRELOAD`, measures the computation between the calls, and attributes both to optional
application phases (`setup`, `step/halo`, …). The analysis step finds hotspots and their causes,
writes text reports and an interactive HTML report with charts, and can ask a local LLM, run
through `vibeprof_llm` on the same cluster, to interpret the results and suggest improvements.

```
 application ranks                       after the job step
 ┌──────────────────────┐                ┌──────────────────────────────────────────────┐
 │ app code             │                │ vibeprof_report.py                            │
 │  vibeprof_phase_*()   │  rank-*.json   │  analyze: metrics [M#] + findings [F#]       │
 │ libvibeprof.so (PMPI) ├───────────────►│  brief.md / summary.md / report.html / PNGs  │
 │ libmpi               │                │  optional: LLM text + validation (--model)   │
 └──────────────────────┘                └──────────────────────────────────────────────┘
```

## Build

```bash
cmake -S . -B build -DVIBEPROF_BUILD_LLM=OFF && cmake --build build -j    # from the repository root
# or on its own:   cd vibeprof && cmake -B build && cmake --build build -j
```

The repository's top-level build includes the profiler (in `build/vibeprof/`; `-DVIBEPROF_BUILD_LLM=OFF` skips
the llama.cpp engine). `vibeprof-srun`, `tests/gpu_validate.sh` and `apps/build.sh` find the libraries in either
place (or set `VIBEPROF_LIB` / `VIBEPROF_BUILD`). Below, `build/` means that directory.

Produces `libvibeprof.so` (collector, including the GPU backend), `libvibeprof_stub.so` (no-op
markers), `libvibeperturb.so` (fault injection for testing), `vibeprof_toy` (a test program with
known behavior), and for GPU testing `libfakecuda.so` + `gpu_toy_sim` (simulated GPU) and, when
CMake finds nvcc, `gpu_toy` (real GPU). Building needs no CUDA toolkit: CUPTI is loaded at run time.
The Python tools need only the standard library; `--png` needs matplotlib. Call sites are
symbolized with `addr2line`, so build the application with `-g` for file:line sites.

## Run

```bash
# Slurm
scripts/vibeprof-srun -- srun -N 4 -n 128 ./app args
# any launcher (MPICH/Hydra -genv or Open MPI -x is chosen automatically)
scripts/vibeprof-srun --png -- mpirun -np 8 ./app args
# short interpretation written by an LLM (runs vibeprof_llm as a separate job step)
scripts/vibeprof-srun --brief --model /models/qwen2.5-7b-instruct-q4_k_m.gguf -- srun -n 128 ./app
```

The wrapper sets `LD_PRELOAD` through the launcher, so only the ranks are profiled, writes the
profile to `vibeprof-<jobid>…/` (or `-o DIR`), then builds the report into `DIR/analysis/`.
Manual use: `LD_PRELOAD=build/libvibeprof.so VIBEPROF_DIR=prof srun ./app`, then
`tools/vibeprof_report.py prof`.

| Variable | Meaning |
|---|---|
| `VIBEPROF=0` | disable (wrappers forward directly) |
| `VIBEPROF_DIR` | profile directory |
| `VIBEPROF_MAX_INSTANCES` | phase instances kept per phase for trend charts (default 5000) |
| `VIBEPROF_FAKE_PPN=k` | testing only: pretend ranks run k per node |
| `VIBEPROF_LLM_LAUNCHER` | launcher for the LLM step (default `srun -n 2 --overlap` in an allocation, else `mpirun -np 2`) |
| `VIBEPROF_GPU=auto\|cuda\|none` | GPU measurement (default `auto`: on when an NVIDIA driver with GPUs is found) |
| `VIBEPROF_CUPTI_LIB`, `VIBEPROF_CUDA_LIB` | locations of `libcupti.so` / `libcuda.so.1` if the search below fails |
| `VIBEPROF_GPU_API=0` | do not trace CUDA API calls (lower overhead; no phase attribution of GPU work) |
| `VIBEPROF_KOKKOS_KERNELS=1` | also record each Kokkos `parallel_*` as a phase (host view) |

## Phase markers (optional)

```c
#include <vibeprof.h>
vibeprof_phase_begin("solve");        /* phases nest: "solve" > "halo" is reported as "solve/halo" */
  vibeprof_phase_begin("halo"); exchange(); vibeprof_phase_end("halo");
vibeprof_phase_end("solve");
vibeprof_mark("regrid");              /* one-off event */
```

C++: `VIBEPROF_SCOPE("halo");`. Fortran: `use vibeprof; call vibeprof_phase_begin("halo")`
(`include/vibeprof.f90`, untested: no Fortran compiler was available). Link `-lvibeprof_stub` to keep markers in
production builds at no cost; preloading `libvibeprof.so` turns them on. Markers do no communication.
Without markers the report still covers ranks, nodes, MPI calls, call sites and traffic.

## GPUs (NVIDIA, CUPTI)

GPU measurement turns on by itself when the node has an NVIDIA driver and GPUs. `libcupti.so` is
looked for in `$VIBEPROF_CUPTI_LIB`, `$CUDA_HOME`/`$CUDA_PATH`/`$NVHPC_CUDA_HOME` (`extras/CUPTI/lib64`),
next to the application's `libcudart`, the library path, and `/usr/local/cuda`. A `module load
cuda` usually suffices. CUDA 11, 12 and 13 are supported by the same binary.

Per rank it records:

- **Kernels:** count, GPU time, min/max, by kernel name and by the phase that *launched* them
  (through the CUPTI correlation id of the launching call).
- **Copies:** direction, host memory kind (pageable/pinned/managed), bytes, GPU time, and whether
  the MPI library issued them.
- **Host time in CUDA calls:** by phase and function. Calls are grouped as synchronize, copy
  (blocking copies wait for the GPU), launch, alloc and other.
- **GPU busy time:** the union of kernel intervals per device, a busy timeline, and GPU busy/idle
  time while the host is in each phase.
- **GPU buffers passed to MPI:** detects GPU-aware MPI.
- **Devices used:** name and PCI bus id, to detect ranks sharing a GPU.

For GPU runs "computation" means CPU work; host time in CUDA calls is counted separately, so CPU
imbalance is not confused with waiting for the GPU. GPU findings:

| Finding | Meaning | Typical fix |
|---|---|---|
| GPU idle during phase X | the GPU waits while the host computes, communicates, allocates or copies | offload or overlap; async kernels before CPU work |
| GPU work imbalance | kernel time per rank uneven in a phase | rebalance by measured kernel time; check the slow GPU |
| MPI staged through host memory | DtoH/HtoD copies matching the MPI volume | GPU-aware MPI with device buffers |
| Copies use pageable memory | slow, host-blocking copies | `cudaMallocHost` / `cudaHostRegister` |
| CUDA allocation costs | `cudaMalloc/Free` in the loop | allocate once, `cudaMallocAsync` pools |
| Many short kernels | launch-bound | fuse kernels, CUDA graphs |
| Synchronizes after every kernel | the host cannot queue work ahead | drop syncs between same-stream kernels |
| Ranks share one GPU | other GPUs on the node unused | `--gpus-per-task=1 --gpu-bind=closest` / `cudaSetDevice(local_rank)` |

The report adds charts for GPU busy/idle by phase, GPU busy per rank and over time, host time in
CUDA calls, top kernels, copies, and the GPUs each rank used. With `--png` it also writes
`gpu_phases.png`, `gpu_kernels.png` and `gpu_timeline.png`.

**Kokkos:** run with `scripts/vibeprof-srun --kokkos …` (sets `KOKKOS_TOOLS_LIBS`): Kokkos profile
regions become phases and kernels are reported under their Kokkos labels instead of the template
names. OpenMP offload and OpenACC kernels are seen through CUPTI with compiler-generated names.

**Validating on your GPUs:** inside an allocation with 2 GPUs on one node,

```bash
cmake -B build && cmake --build build -j          # builds build/gpu_toy with nvcc
tests/gpu_validate.sh real gpu-validate-real      # GPU_AWARE=1 adds the GPU-aware MPI case
```

The script runs `gpu_toy` with injected problems and prints PASS/FAIL per scenario.
Without a GPU, `tests/gpu_validate.sh sim` runs the same scenarios on the simulator.

## Outputs (`DIR/analysis/`)

| File | Content |
|---|---|
| `report.html` | self-contained interactive report: summary tiles, interpretation, findings, charts (phases, ranks, imbalance heatmap, traffic matrix, call sites, phase trends, message sizes, nodes), each with a table view; light/dark |
| `brief.md` | verdict, top 3 issues with suggestions (rule-based) |
| `summary.md` | all findings, phase table, top call sites |
| `report.md` | interpretation (LLM or rule-based) + summary |
| `metrics.json`, `findings.json` | every number with its id (`M#`), findings (`F#`) with evidence ids |
| `facts.txt`, `prompts.jsonl` | what the LLM sees |
| `*.png` | static figures (`--png`) |

### What the analysis separates

- **MPI cost vs waiting.** Per MPI operation the time of the rank that spends least is taken as
  the intrinsic cost; the excess on other ranks is waiting for late partners.
- **Causes before symptoms.** Waiting is reported as a consequence of the imbalance or slow node
  that causes it (`caused_by`).
- **Slow node vs uneven work.** A node whose ranks are slower in most phases with significant
  computation is a slow node (hardware, throttling, other jobs); slowness in one phase is
  imbalance of the decomposition. With a single significant phase the finding says it cannot tell.
- Other rules: expensive collectives, many small messages, costly or useless barriers,
  synchronous/blocking sends, phases that get slower over the run, communication- or
  compute-bound runs, marker coverage. Thresholds are in `TH` in `vibeprof_analyze.py`.

### LLM interpretation

`--model` writes the facts sheet into a chat prompt and runs it with `vibeprof_llm`
(`--chat on --temp 0`). `--brief` asks for a short verdict + suggestions; `--map-reduce` writes one
note per top-level phase first, then a combined report (for large profiles). The text is
validated: every `[M#]`/`[F#]` must exist and every number must occur in the data; the result is
shown under the text, and text without citations is flagged as ungrounded. Use an instruction-
tuned model of 7B parameters or more; small models do not follow the citation rules.

**Asking your own questions.** `--ask TEXT` (repeatable) puts free-form performance questions in front of the
model; it answers them first, in an `## Answers` section, then writes the usual report:

```bash
scripts/vibeprof-srun --brief --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --ask "Is this application communication or compute bound?" \
  --ask "Report the severity of load imbalance and possible mitigation strategies" \
  -- srun -n 128 ./app
```

`--ask-file FILE` reads questions from a file: plain text (one per line, `#` comments and blank lines
skipped), a JSON list of strings, or JSON lines with a `question`/`prompt`/`text` field or chat `messages`
(the last user message is used). The answers follow the same rules as the rest of the text (facts only,
numbers cited by id, checked by the validator); a question the measurements cannot answer gets a plain
"cannot tell" plus the measurement that would answer it. Each question adds room to the output budget (and
to the `--brief` word limit). The questions are listed above the text in `report.md`/`report.html`. With
`--map-reduce` they go to the final combined prompt only. On an existing profile:
`tools/vibeprof_report.py DIR --model ... --ask "..."`. (`vibeprof_llm -p` is a different thing: the
engine's prompts file.)

**Getting a model.** Download a single-file GGUF once, e.g. Qwen2.5-7B-Instruct Q4_K_M (4.7 GB). On clusters,
do this on a login node (compute nodes often have no internet) into project storage rather than `$HOME`:

```bash
mkdir -p /path/to/models && cd /path/to/models
curl -L -o Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
# or: hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir /path/to/models
```

and pass `--model /path/to/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`. Single-file repositories are simplest; some
official GGUF repositories (e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`) split larger quantizations into parts.

**Automatic download.** `--model` also takes a Hugging Face file, fetched on first use and cached:

```bash
--model hf:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf     # optional @revision suffix
```

Files go to `$VIBEPROF_MODEL_DIR` (default `~/.cache/vibeprof/models`); split models fetch every part (name
the first one); interrupted downloads resume; `HF_TOKEN` is sent for gated models and `https_proxy` is honored.
The report step runs where `vibeprof-srun` runs, usually a compute node, so either pre-fetch on a login node
with the same cache directory:

```bash
export VIBEPROF_MODEL_DIR=/path/to/models          # shared by login and compute nodes
tools/vibeprof_model.py hf:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf   # prints the path
```

or set the site proxy in the job (ALCF Polaris: `export https_proxy=http://proxy.alcf.anl.gov:3128`).
`tools/vibeprof_model.py` also works for the engine directly: `vibeprof_llm -m "$(tools/vibeprof_model.py hf:...)"`.

**Node-local copy.** `--model-stage DIR` (or `VIBEPROF_MODEL_STAGE`) copies the model to node-local storage once
per node before the LLM step loads it, e.g. `--model-stage /local/scratch` on Polaris. The model is memory-mapped
either way, so ranks on a node share one copy of the weights in the page cache; staging keeps the reads off the
shared file system and makes pages the kernel evicts under memory pressure cheap to read back.

## Validation

Ground truth comes from `libvibeperturb.so`, preloaded before the collector
(`--lib build/libvibeperturb.so:build/libvibeprof.so`): it slows chosen ranks or nodes (inside the
right phase), delays an MPI operation, inserts barriers, or makes sends synchronous.

Toy program, 2 ranks, 200 iterations, 3 repetitions each:

| Injected | Detected |
|---|---|
| none | nothing (3/3) |
| rank 1 computes 1.5× slower | slow node + imbalance (3/3) |
| application imbalance (`--imbalance 0.5`) | imbalance, never a slow node (3/3) |
| +4 ms per MPI_Allreduce | expensive Allreduce, ≈33–35% of run (3/3) |
| extra barrier per iteration | "barrier every iteration but cheap" note (3/3) |
| 256-byte halos | silent (cheap, correct) |

ECP proxy apps (`apps/build.sh` fetches, adds markers, builds), 2 ranks:

| App | Injected | Detected |
|---|---|---|
| miniFE 80³ | none | nothing / Allreduce cost on 1 of 2 runs (≈6%, real on this machine) |
| miniFE 80³ | rank 1 1.5× slower | slow node + imbalance in `cg/matvec` + waiting in `cg/dot` (2/2) |
| miniFE 80³ | +0.5 ms per Allreduce | expensive Allreduce (2/2) |
| miniFE 80³ | synchronous sends | silent (receives are pre-posted; correct) |
| Quicksilver CORAL2 P1, 50k particles | rank 1 1.5× slower | "node computes more (slow node or uneven work)" — one significant phase, so correctly undecided — + imbalance + waiting in `cycle/tracking` |
| AMG 100³ | none | nothing |
| AMG 100³ | rank 1 1.5× slower | slow node + imbalance in `pcg_solve`, `amg_setup` + waiting |
| AMG 100³ | +2 ms per Allreduce | silent: 130 calls ≈ 0.26 s ≈ 2% of run, below the 5% threshold (correct) |

These runs were on a 2-core VM with a blocking (ch3:sock) MPICH, so timing noise is large;
re-validate on real nodes.

GPU, on the simulated device (`tests/fakecuda`: models streams, kernel and copy durations, host
blocking; delivers CUPTI callbacks and activity records), `tests/gpu_validate.sh sim`, 2 ranks:

| Injected (`gpu_toy` option) | Detected |
|---|---|
| none | nothing (GPUs busy note only) |
| kernel time +50% on rank 1 (`--gpu-imbalance 0.5`) | GPU work imbalance in `step/compute`, MPI waiting as its consequence |
| CPU-only work per step (`--host-us 3000`) | GPU idle in `step/host`, host computing on the CPU |
| 32 MiB halos through host buffers | MPI staged through host memory in `step/halo` |
| the same with device buffers (`--gpu-aware`) | GPU-aware MPI note, no staging finding |
| pageable staging buffers (`--pageable`) | pageable copies |
| `cudaMalloc/Free` every step (`--alloc-each`) | CUDA allocation costs in `step/compute` |
| 40 tiny kernels with a sync after each | synchronizes after every kernel + many short kernels |
| 2 ranks on a 2-GPU node both on GPU 0 (`--bind zero`) | ranks share one GPU while another is unused |
| one rank per GPU (`--bind local`) | nothing |
| CUPTI callbacks unavailable (another tool attached) | warning; kernels counted but not attributed to phases |
| Kokkos regions and labels (`tests/kokkos_hooks_test.c`) | phases `solver`, `solver/assemble`; kernels named `Assemble::matrix`, `Dot` |

## Known limits

- **MPI wrapper layers hide call sites.** Libraries that wrap MPI (hypre's `hypre_MPI_*`,
  PETSc, Trilinos) show the wrapper as the call site (e.g. `mpistubs.c:1172`). Planned:
  optional stack capture (`VIBEPROF_STACK=depth`) and a skip list for wrapper frames.
- **Alternating lateness.** When ranks take turns arriving last (Monte Carlo tracking loops), the
  "fastest rank" estimate of intrinsic collective cost includes some waiting.
- **Not intercepted yet:** persistent requests (`MPI_Send_init`/`MPI_Start*`), MPI-IO, partitioned
  and MPI-4 large-count calls, communicator creation. OpenMP threads inside a rank are counted as
  that rank's computation.
- **MPI ABI:** built against MPICH-ABI headers (MPICH, Intel MPI, Cray MPICH); Open MPI needs its own build.
- **GPU backend not yet run on real GPUs.** CUPTI is loaded at run time and its record layouts
  are declared in `src/cupti_abi.h` from NVIDIA's documentation (checked with static asserts);
  the simulator exercises the logic, not the layouts. A run of `tests/gpu_validate.sh real`
  will confirm them.
- **GPU coverage:** peer-to-peer copies (`MEMCPY2` records), NVTX ranges, unified-memory page
  faults and hardware counters are not recorded. Several GPUs per rank are merged for the
  per-phase busy time. Only one CUPTI client can subscribe to API callbacks: with Nsight Systems
  or another tool attached, kernels are still counted but not attributed to phases (reported as
  a warning).
- **GPU overhead:** API tracing costs about a microsecond per CUDA call; set `VIBEPROF_GPU_API=0`
  for very launch-heavy codes (no phase attribution then). Not yet measured on hardware.
- **Other GPU vendors:** AMD (rocprofiler-sdk) and Intel (Level Zero) backends are not built;
  `src/gpu_backend.h` is the interface they would implement.
