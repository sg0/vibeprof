# vibeprof

vibeprof profiles MPI and MPI+GPU applications without changing them: a PMPI/CUPTI collector is
preloaded into the ranks, and an analysis step finds where the time goes and why (load imbalance,
slow nodes, waiting, expensive collectives, idle GPUs, staging copies, ...). It writes an
interactive HTML report with charts, a short brief with suggestions, and optionally an
interpretation written by an LLM that runs on the same cluster.

| Part | Where | What |
|---|---|---|
| profiler | [`vibeprof/`](vibeprof/README.md) | `libvibeprof.so` (MPI + NVIDIA GPU collector), phase markers, analysis, reports, `vibeprof-srun` |
| vibeprof-llm | `src/vibeprof_llm.cpp` (this directory's CMake project) | in-process llama.cpp inference over MPI with disaggregated prefill/decode; writes the optional LLM interpretation of profiles, and is a benchmark in its own right |

The rest of this file documents vibeprof-llm; the profiler is documented in
[vibeprof/README.md](vibeprof/README.md).

# vibeprof-llm

A C++/MPI wrapper around **llama.cpp (linked statically, run in-process — no server)** that runs continuously
batched, **disaggregated prefill/decode** inference across the CPUs and GPUs of a cluster, on synthetic or real
prompts:

* **Prefill ranks (normally one per GPU).** Each pulls work from a global task counter (**NXTVAL**, MPI-3 RMA
  `MPI_Fetch_and_op`) in an endless loop. It builds the prompt: a synthetic one, or a dataset record rendered
  with the model's chat template and tokenized. It then prefills with `llama_decode()`, samples the first token
  (TTFT), exports the KV cache with `llama_state_seq_get_data()` and ships it to a decode rank. With a draft
  model, the draft is prefilled on the same GPU and its KV cache is shipped too.
* **Decode ranks (CPU cores, and optionally GPUs).** Each keeps many sequences in one `llama_context` and
  advances them all together (**continuous batching**). Sequences join the moment their KV arrives and leave
  at EOS or their length limit.
* **Speculative decoding (optional).** Uses llama.cpp's `common_speculative` on the decode ranks: **n-gram
  self-speculation** (no extra model) and/or a **draft model**. Drafts for all active sequences are generated
  in one batched call and verified in the same target step.
* Any number of ranks ≥ 2, odd or even, and any number of GPUs per node (including zero).

```
                 rank 0 RMA window:  [ NXTVAL | seqs[0..P) | cells[0..P) ]
                         ▲ MPI_Fetch_and_op(+B)          ▲ Fetch_and_op(±1, ±cells), Get_accumulate(NO_OP)
                         │                               │
  ┌─ prefill rank (GPU k) ┴───────────────┐      ┌───────┴─ decode rank (CPU or GPU) ───────────────────┐
  │ loop:                                 │      │ loop:                                                 │
  │   t = NXTVAL(B); if t >= N: break     │ hdr  │   Improbe ctl msgs → post Irecv for payload           │
  │   prompt: synthetic | template+tokenize│────▶│   payload complete → set_data(target [, draft])       │
  │   llama_decode target [+ draft] (GPU) │      │   draft for all seqs   (common_speculative_draft)     │
  │   sample 1st token  → TTFT            │ toks │   batch = [last, d1..dk] × every active sequence      │
  │   state_seq_get_data target [+ draft] │ +KV  │   llama_process → verify/accept → roll back KV        │
  │   reserve slot+cells on decode rank   │─────▶│   EOS / length → record, free slot, release RMA       │
  └───────────────────────────────────────┘Isend └───────────────────────────────────────────────────────┘
```

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp      # any recent checkout
cmake -S . -B build -DLLAMA_CPP_DIR=../llama.cpp -DGGML_CUDA=ON      # or -DGGML_HIP=ON / SYCL / Vulkan; omit for CPU-only
cmake --build build -j
```

llama.cpp is added with `add_subdirectory` and built as static libraries: `llama`, `ggml`, and `llama-common`
for speculative decoding, sampling and Jinja chat templates. GGML options are passed straight through. Requires
an MPI-3 library (MPICH, Open MPI, Cray MPICH, Intel MPI, MVAPICH…) and a C++17 compiler. Needs llama.cpp from
2026-09-24 or later (extended batch API: `common_batch`, `llama_process`); builds without warnings up to `5917b39`
(2026-11-25).

**Optional: GPU-direct KV transfer.** To keep KV data in GPU memory end to end (`--kv-transport device`, see
*KV transfer* below), apply the small llama.cpp patch before configuring:

```bash
patches/apply.sh ../llama.cpp      # path to your llama.cpp checkout
```

The script is safe to run twice (an already patched tree is detected and left alone), cleans up `.rej` files
from earlier attempts, and falls back to a 3-way merge when llama.cpp has moved on. The patch applies cleanly
to llama.cpp from mid-June 2026 through `5917b39` (2026-11-25), and needs the 3-way merge for older checkouts.
If you patch by hand, note that `git -C ../llama.cpp apply PATCH` resolves `PATCH` relative to `../llama.cpp`,
so use an absolute path. CMake reports whether the split-state API was found. Without the patch everything else
works, and KV transfer uses host memory.

### ALCF Polaris

Polaris defaults to `PrgEnv-nvhpc`, whose `cc`/`CC` wrappers call `nvc`/`nvc++`; llama.cpp needs GCC or Clang as
the host compiler (`nvc++` rejects its warning flags, e.g. `Unknown switch: -Wmissing-declarations`). Build with
the GNU environment for host code and `nvcc` for the CUDA kernels, keeping the Cray wrappers so Cray MPICH and
its GPU support (GTL) are linked:

```bash
module load PrgEnv-gnu                     # Cray wrappers cc/CC now call gcc/g++ instead of nvc/nvc++
module load cuda/13.0
module load craype-x86-milan craype-accel-nvidia80     # accel module: links Cray MPICH's GPU (GTL) support
module use /soft/modulefiles

cd ~/projects/vibeprof && rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_HOST_COMPILER=$(which g++) \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DLLAMA_CPP_DIR=$HOME/sources/llama.cpp
cmake --build build -j 16
```

At run time, `export MPICH_GPU_SUPPORT_ENABLED=1` enables GPU-aware MPI (needed for `--kv-transport device`).

**CMake.** The site-built CMake 4.5.1 on Polaris fails to configure llama.cpp with
`target_compile_features specified unknown feature ""` on every llama.cpp target, although the arguments are
correct. The official Kitware binaries (3.18 through 4.5.1) configure cleanly; use one (unpack and run
`bin/cmake`, no install needed) in place of `cmake` above:

```bash
cd ~ && curl -LO https://github.com/Kitware/CMake/releases/download/v4.2.0/cmake-4.2.0-linux-x86_64.tar.gz
tar xzf cmake-4.2.0-linux-x86_64.tar.gz
export PATH=$HOME/cmake-4.2.0-linux-x86_64/bin:$PATH
```

## Run

```bash
# synthetic: 1 node, 4 GPUs, 12 ranks -> ranks 0-3 prefill on GPUs 0-3, ranks 4-11 decode on CPU cores
mpirun -np 12 ./build/vibeprof_llm -m model.gguf -n 512 --isl 1024 --isl-std 256 --osl 256 --decode-batch 16

# real prompts (ShareGPT), chat template applied, output lengths from the reference answers, stop at EOS
mpirun -np 12 ./build/vibeprof_llm -m model.gguf -p ShareGPT_V3_unfiltered_cleaned_split.json \
       --osl-from-data --shuffle -n 2000 --temp 0.7 --output answers.jsonl --csv requests.csv

# n-gram self-speculation (no draft model) - most useful when outputs repeat prompt content (code, RAG, edits)
mpirun -np 12 ./build/vibeprof_llm -m model.gguf -p prompts.jsonl --spec-type ngram-mod

# draft model; with temp > 0 use rejection sampling (see "Speculative decoding" below)
mpirun -np 12 ./build/vibeprof_llm -m llama-3.1-8b.gguf -md llama-3.2-1b.gguf --spec-draft-max 8 \
       -p prompts.jsonl --temp 0.7 --spec-rejection

# GPU decode with large batches: 4 GPUs -> 2 prefill GPUs + 2 decode GPUs (128 seqs each) + 8 CPU decode ranks
mpirun -np 12 ./build/vibeprof_llm -m model.gguf -p prompts.jsonl --decode-gpus 2 --decode-batch-gpu 128

# Slurm, 2 nodes x 8 ranks, decode ranks bound to disjoint cores
srun -N2 --ntasks-per-node=8 --cpu-bind=cores ./build/vibeprof_llm -m model.gguf -n 2000 --isl 512 --osl 128
```

### Options

| option | meaning |
|---|---|
| `-m` | target GGUF model |
| `-n`, `--duration` | total requests (default 64, or the number of records with `-p`), or wall time to keep claiming (`-n 0`) |
| `--isl`, `--isl-std` | synthetic input length mean / stddev (normal, clipped to `[1, ctx-osl]`) |
| `--osl`, `--osl-std` | output length = max new tokens |
| `--prompt-mode tokens/words` | synthetic prompts: random vocab ids (exact length), or random English words |
| `--ignore-eos auto/on/off` | `auto` (default): EOS is ignored for synthetic prompts and stops generation with `-p` |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `-s` | sampling (temp ≤ 0 → greedy) |
| **real prompts** | |
| `-p, --prompts FILE` | `.jsonl`, `.json` (array) or `.txt` (one prompt per line), see *Datasets* |
| `--chat auto/on/off` | apply the model's chat template: `auto` = for message records, `on` = also wrap plain prompts as a user turn |
| `--chat-template-file F` | Jinja template overriding the model's |
| `--system-prompt STR` | system message added to chat prompts that have none |
| `--conv-turns first/all` | conversations: first user turn (default, like vLLM's ShareGPT benchmark) or the whole history up to the last assistant turn |
| `--osl-from-data` | output length from the record: `max_tokens`-style field, or the token length of the reference answer |
| `--shuffle`, `--max-prompts N` | seeded record order; load at most N records. Requests cycle through records if `-n` > records |
| `--output FILE` | generated text per request (JSONL: task, record, token counts, stop reason, text) |
| **batching** | |
| `-c` | context window per sequence. Longer prompts keep BOS + the most recent tokens (counted as *truncated*) |
| `--prefill-batch B` | max prompts prefilled together |
| `--prefill-tokens T` | token budget per prefill batch: tasks are claimed one at a time until `T` prompt tokens or `B` prompts |
| `--claim fixed/guided`, `--claim-factor K` | guided (default): cap each prefill batch at `ceil(remaining / (K × prefill processes))` so batches shrink as the task pool drains (K default 2). Applies with `-n`; `--duration` runs use fixed claims |
| `--decode-batch SPEC` | slots per **CPU** decode rank: `N`, `N1,N2,...` (per CPU decode rank, in rank order) or `per-thread:K` |
| `--decode-kv-cells SPEC` | KV cell (token) budget per CPU decode rank (default `ctx × slots`, each ≥ `-c`) |
| `--decode-batch-gpu SPEC`, `--decode-kv-cells-gpu SPEC` | same for **GPU** decode ranks (default 128 slots; max 256 per rank, a llama.cpp limit) |
| `--decode-ubatch N` | physical batch on decode ranks (default 512) |
| `-b`, `-ub` | logical / physical batch for prefill |
| **speculative decoding** | |
| `--spec-type LIST` | `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`, `draft-simple` (comma list; first that drafts wins) |
| `-md, --draft-model F` | draft GGUF with a compatible vocabulary (implies `draft-simple`) |
| `--spec-draft-max N` | max drafted tokens per sequence and step, all methods (default 8) |
| `--spec-draft-min N`, `--spec-draft-p-min X` | draft model: discard short drafts; stop drafting below this probability |
| `--spec-ngram-n/-m/-min-hits` | n-gram key length, draft length, min hits (`ngram-simple/map-k/map-k4v`) |
| `--spec-rejection` | draft model + temp > 0: sample drafts and verify by rejection sampling |
| `--spec-synth-len L` | **benchmarking only**: accept drafts synthetically with mean accepted length L (output text is not valid) |
| `--draft-prefill prefill/decode` | where the draft model processes the prompt (default: on the prefill GPU, shipped as KV) |
| **topology** | |
| `--decode-gpus N` | GPU decode ranks per node (default 0) |
| `--prefill-gpus N` | GPUs per node used for prefill (default: all GPUs not given to decode; all GPUs if decode takes all, i.e. shared) |
| `--prefill-per-node N`, `--gpus-per-node N` | override the defaults / GPU auto-detection |
| `-ngl` | layers offloaded on GPU ranks (default all) |
| `--threads-prefill`, `--threads-decode` | llama.cpp threads (auto: GPU ranks 2, CPU ranks share the remaining cores) |
| `-fa on/off`, `--kv-type f16/q8_0/f32/bf16` | must match on both sides, so they are global options |
| `--progress none/root/all` | helper thread that drives MPI progress while the main thread is inside llama.cpp |
| **KV transfer** | |
| `--kv-transport auto/host/device` | `host`: one serialized blob per sequence in host memory. `device`: metadata in host memory plus KV tensor data in GPU memory, sent with GPU-aware MPI (needs the patch). `auto` (default): `device` when the patch is built and some GPU rank has GPU-aware MPI, else `host` |
| `--kv-staging pinned/pageable` | host staging buffers on GPU ranks (default pinned) |
| `--gpu-aware-mpi auto/yes/no` | override the runtime check (Open MPI CUDA/ROCm query, MPICH `MPIX_GPU_query_support`, Cray `MPICH_GPU_SUPPORT_ENABLED=1`) |
| **output** | |
| `--csv FILE` | per-request timestamps, lengths, stop reason, drafted/accepted tokens |
| `--verify` | with `--temp 0`: the prefill rank also decodes each request locally **without speculation**; the decode rank checks the output is token-identical |

### Datasets

Records are detected per line (`.jsonl`) or per array element (`.json`). Records without a usable prompt are
skipped and counted.

| record shape | becomes |
|---|---|
| `{"messages": [{"role": "...", "content": "..."}]}` (OpenAI; content may be a list of text parts) | chat prompt |
| `{"conversations": [{"from": "human"/"gpt"/"system", "value": "..."}]}` (ShareGPT) | chat prompt; the assistant reply is the reference answer |
| `{"prompt"/"text"/"input"/"question"/"instruction": "..."}` or a line of `.txt` | plain prompt (chat-wrapped with `--chat on`) |
| optional `max_tokens` / `max_new_tokens` / `output_len` / `output_tokens` / `osl` | output length (with `--osl-from-data`) |
| optional `output` / `response` / `completion` / `answer` / `reference` | reference answer → output length (with `--osl-from-data`) |

The first prefill rank parses the file once and broadcasts compact records (messages + output-length hint) to
the prefill ranks. Rendering with the chat template (llama.cpp's Jinja engine) and tokenization happen when a
request is claimed, so they count toward TTFT, as in a serving engine. Decode ranks never see the dataset; they
receive the prompt tokens with the KV payload, which n-gram speculation needs. `examples/` has a small sample of
each format.

## How it works

**Roles.** `MPI_Comm_split_type(SHARED)` gives each node's ranks. The node leader enumerates GGML GPU devices
(CUDA/HIP/SYCL/Vulkan/Metal) and broadcasts the count. Per node, local ranks are laid out as:

```
[0, Pp)            prefill   - GPU i for i < prefill-gpus, CPU beyond
[Pp, Pp + Dg)      decode on GPU (prefill-gpus + j) mod #GPUs        (Dg = --decode-gpus)
[Pp + Dg, ranks)   decode on CPU
```

By default `Pp` = number of prefill GPUs, and at least one rank stays a decode rank. With no GPUs, half the
ranks (rounded down) prefill. Degenerate layouts (e.g. 1 rank per node) fall back to even=prefill, odd=decode.
GPU ranks load the model with `devices = {that GPU}`, `split_mode = NONE`. CPU decode ranks use
`devices = {NULL}`, no KQV/op offload, and `CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES=-1` set before backend
init. When `prefill-gpus + decode-gpus` exceeds the GPU count, prefill and decode ranks share GPUs (a note is
printed; CUDA MPS helps).

**NXTVAL.** A window allocated on rank 0 (`MPI_Win_allocate`) holds the task counter. Every rank opens one
passive-target epoch (`MPI_Win_lock_all`) for the entire run. A claim is `MPI_Fetch_and_op(+B, MPI_SUM)` +
`MPI_Win_flush`. With `--prefill-tokens`, prefill ranks claim one task at a time and stop at the token budget.
A task that would overflow the budget is carried into the next batch, never dropped.

**Guided claiming (tail-effect mitigation).** With fixed claims, a prefill process takes a whole
`--prefill-batch` per claim. Near the end of a run, one process can hold the last large chunk while the others
sit idle. With `--claim guided`, each prefill round first reads the counter atomically (`MPI_Fetch_and_op` with
`MPI_NO_OP`) and caps its batch at `ceil(remaining / (K × prefill processes))`. Batches stay at
`--prefill-batch` while the pool is large and shrink toward one task as it drains (guided self-scheduling).
The read is only a hint: another process may claim in between, so a batch can be slightly larger than ideal,
but tasks are still handed out exactly once by the `+n` fetch-and-add. The report shows tasks per claim, each
process's finish time, and the *finish spread* (last minus first process to finish), the direct measure of
the tail. Task ids map to synthetic
prompts or dataset records deterministically from `(seed, task_id)`, so only ids cross the network.

**Per-rank KV capacity.** Each decode rank has its own slot count and KV cell budget: `--decode-batch*` for CPU
ranks, `--decode-batch-gpu*` for GPU ranks, possibly scaled by thread count. Ranks share them with
`MPI_Allgather`. A rank's context is created with `n_seq_max = slots` and `n_ctx = cells`. The unified KV
buffer makes cells one shared pool, so slots and memory are independent: e.g. `--decode-batch 64
--decode-kv-cells 16384` holds 64 short requests in the memory that `ctx × slots` would size for 4 at
`-c 4096`.

**Routing & back-pressure.** The rank-0 window holds, per decode rank, resident sequences and reserved KV cells
(`isl + max_out` each). A prefill rank snapshots both arrays atomically (`MPI_Get_accumulate` + `MPI_NO_OP`). It
considers ranks with a free slot *and* enough free cells, and picks the lowest relative load
`max(seqs/slots, cells/kv_cells)`. Idle ranks fill first, and under pressure work is shared in proportion to
capacity, so a 128-slot GPU decode rank takes far more than a 16-slot CPU rank. Reservation uses two
`Fetch_and_op`s, undone if a concurrent reservation won, and is released when the sequence finishes.

**Handoff.** A header (task id, lengths, first token, timestamps) goes on a control communicator. The payload
`[prompt tokens | target KV | draft KV]` follows on a separate communicator as non-blocking ≤1 GiB chunks
(`MPI_Isend`). Decode ranks poll with `MPI_Improbe`/`MPI_Mrecv` between steps and admit a sequence once
`MPI_Testall` completes, so transfers overlap with decoding. Requests that end at the first token (EOS, or
`max_out = 1`) send no payload. All contexts use a **unified KV buffer** (`kv_unified = true`). The state
format records the number of KV streams, so prefill (B sequences) and decode (M sequences) contexts must both
have exactly one, and llama.cpp restores cells into any free, not necessarily contiguous, slots.

**KV transfer.** There are two transports:

* **Host (default without the patch).** `llama_state_seq_get_data()` serializes each sequence's state, KV
  data included, into host memory. On GPU ranks the staging buffers are **pinned** (allocated through ggml's
  host buffer type for the rank's GPU, e.g. `cudaMallocHost`), so the GPU-to-host copy runs at full DMA speed
  and the MPI library can send the buffer without an extra bounce copy. Staging buffers are pooled and reused,
  because pinned allocation is slow.
* **Device (`patches/llama.cpp-state-seq-split.patch`).** llama.cpp's public API can only serialize KV data
  into host memory. Its `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` flag only keeps a local checkpoint inside the same
  context, and the exported bytes hold no KV data, so it can't move a sequence between processes. The patch
  adds `llama_state_seq_{get_size,get_data,set_data}_split()`: the state's small metadata goes to host memory,
  and the KV tensor bytes are copied device-to-device into any ggml backend buffer the caller provides. Each
  rank that has a GPU and GPU-aware MPI puts that buffer in GPU memory. The prefill rank `MPI_Isend`s the
  device pointer, and the decode rank `MPI_Irecv`s into its own device buffer and imports device-to-device.
  KV data never touches host memory when both ends are such ranks; the MPI library uses GPUDirect RDMA or
  CUDA IPC.

  Ranks without a GPU or without GPU-aware MPI keep their half in pinned or plain host memory, and the format
  is the same either way. A GPU prefill rank can therefore send straight from GPU memory to a CPU decode rank's
  host memory, which removes the sender-side copy.

  The patch uses two-sided `MPI_Isend`/`MPI_Irecv`, the most widely supported GPU-aware MPI path, rather than
  RMA on device-memory windows, which fewer MPI implementations support.
**Speculative decoding.** Each decode rank owns one `common_speculative` instance for all M sequences (seq id =
slot). It is fed with `common_speculative_begin(prompt tokens)` at admission. Every step does the following:

1. Every active sequence gets draft parameters with `n_max = min(--spec-draft-max, remaining − 1)`, so a
   sequence never needs more KV cells than it reserved.
2. `common_speculative_draft()` drafts for all of them in one call. A draft model runs these drafts batched
   across sequences.
3. The draft KV is rolled back to the target's position.
4. One `llama_process()` evaluates `[last, d1..dk]` for every sequence.
5. `common_speculative_process()` updates the draft model's KV with the same batch.
6. The drafts are verified, the accepted tokens committed and the rest of the KV cache removed
   (`llama_memory_seq_rm`) on both target and draft.

Verification is `common_sampler_sample_and_accept_n` (exact for any temperature: a draft token is kept only if
the target samples it). `--spec-rejection` uses `..._rejection` with the draft's distribution instead, which
keeps far more drafts at temp > 0.

**Clocks.** All ranks reset their clock right after a common `MPI_Barrier`, so cross-rank intervals are
consistent to within barrier skew (microseconds).

## Metrics

Per request: **TTFT** (claim → first token, includes templating, tokenization and prefill), **TPOT**
((done − first token) / (generated − 1), includes KV transfer and batching), E2E, prefill compute, decode-slot
wait, KV transfer, KV size, input/output lengths, decode rate, and with speculation the **draft acceptance
rate** and **tokens per decode step**.

Per process: requests, prefill tok/s, decode tok/s, utilization, batch occupancy, slot and KV-cell utilization
and peak, requests per slot, step time, and with speculation acceptance, tokens per sequence-step and the share
of decode time spent drafting.

Each is reported as **min, max, avg, stddev, p50, p90, p99**, plus aggregate throughput and EOS / length /
truncation counts.

## Speculative decoding: what to expect

Measured on CPU with the tiny random test model (3 ranks, 48 requests × 128 tokens, greedy):

| method | accepted | tokens/step | output tok/s |
|---|---|---|---|
| none | – | 1.00 | 817 |
| `ngram-cache` | 66% | 4.18 | 887 |
| `ngram-mod` | 90% | 1.41 | 942 |
| unrelated random draft model | 0.1% | 1.01 | 99 |

These numbers say little about real models; they illustrate the mechanics:

* **Decode becomes compute-bound.** Verifying k drafted tokens for each of M sequences makes every step
  `M·(k+1)` tokens. This pays off where decode is memory-bandwidth-bound, i.e. GPUs or few sequences. On CPU
  with large batches, 4.2 tokens/step bought only 9%. Lower `--spec-draft-max` or the decode batch to tune.
* **A poor draft is expensive.** It drafts 8 tokens per step that are almost all rejected. Use a draft from
  the same family, ideally ≥5–10× smaller.
* **N-gram methods need repetition** between prompt and output (code edits, RAG, summarization, multi-turn).
  Synthetic random prompts give them nothing to match. Use real prompts, or `--spec-synth-len` to benchmark
  the system at an assumed acceptance rate.
* **Temperature.** At temp > 0, plain verification keeps a draft only if the target samples the same token.
  With the target as its own draft and top-k 10, that kept 2% of drafts, while `--spec-rejection` kept 99%.

## Tested

Built against llama.cpp `338fce6` (2026-10-27) with MPICH 4.2.3 (`MPI_THREAD_MULTIPLE`) on CPU with 2–7 ranks
(odd and even), using tiny random-weight Llama models from `tools/make_tiny_gguf.py` (one with a ChatML
template; a second "draft" with the same vocabulary and different weights).

* **Exactness (`--verify`, greedy, `-fa off`):** token-identical to plain local decoding in every
  configuration tried:
  - no speculation;
  - draft model = target (100% acceptance; this also confirms the shipped draft KV is correct);
  - an unrelated draft model;
  - every n-gram type, and n-gram + draft chained;
  - draft prefill on the decode rank;
  - batched prefill with a draft;
  - f16 and f32 KV;
  - truncated prompts;
  - token-budget prefill;
  - ShareGPT single- and multi-turn.
* **With flash attention on**, 0–4 of 24 sequences can differ from the local reference, with or without
  speculation. CPU flash attention's rounding depends on the KV cache size and the number of query tokens per
  sequence, and both differ between the decode context and the reference. Matching the KV size
  (`--decode-kv-cells` = reference size) made the non-speculative run identical; multi-token verification
  steps then still differ. Which sequences share a batch depends on timing, so the count varies between runs.
  This is the usual batch-invariance issue in inference engines, magnified by the random model's near-flat
  logits. `--verify` warns about it; use `-fa off` for a strict check.
* **Guided claiming** was measured on 6 processes (3 prefill + 3 decode) on a 2-core machine, three runs each:
  - Small prefill-heavy job (40 prompts × 400 tokens, `--prefill-batch 16`): fixed claims split the tasks
    16/16/8, with makespan 2.7–3.2 s and prefill finish spread 1.6–2.1 s. Guided: 12–14 per process,
    makespan 1.72–1.80 s, spread 0.04–0.11 s.
  - Large job (400 prompts, `--prefill-batch 8`): guided matched fixed throughput (33.7 vs 32.4 req/s), and
    the prefill spread fell from 0.74 s to 0.01 s.
  - `--verify` stayed exact with guided claiming, also combined with `--prefill-tokens` and datasets.
* **Sampling:** `--spec-synth-len 3` produced 2.89 tokens/step. Rejection sampling with matched samplers
  accepted 99.4%.
* **Real prompts:** all three formats, chat on/off (the template adds ~48 tokens per prompt), system prompt,
  `max_tokens`, reference-answer lengths, shuffle with cycling, EOS stops (30 of 60 at temp 1), `--output` and
  CSV.
* **KV transfer:**
  - The patch applies cleanly to unmodified llama.cpp from mid-June 2026 through `5917b39` (2026-11-25), and the
    project builds with and without it against `5917b39`; `--verify` is exact there with the patch built.
  - The split format was tested with its data half in CPU buffers, since there is no GPU here. That covers the
    patch's export/import code and the two-part messages. `--verify` was exact with flash attention off
    (f32 and f16 KV) and on (f16 and q8_0 KV, with matched KV size); with a draft model (100% acceptance, two
    states sharing one data buffer); with draft prefill on the decode rank; with 5 processes and batched
    prefill; with real prompts; and with the emulated GPU-decode layout.
  - On CPU, split export/import cost was within noise of the host blob, with an identical 1.0 MiB payload per
    512-token request.
  - **Not run:** real GPU buffers, pinned staging, and GPU-aware MPI (device pointers in `MPI_Isend`/`MPI_Irecv`).
* **GPU decode** was tested for layout, per-class capacities, routing, sharing and validation errors using
  `VIBEPROF_LLM_EMULATE_GPUS=1 --gpus-per-node N`. This is a test hook that keeps the GPU role assignment but runs
  those ranks on the CPU. The build machine has no GPU, so **actual GPU execution (prefill or decode) has not
  been run**.

```bash
PYTHONPATH=../llama.cpp/gguf-py python3 tools/make_tiny_gguf.py tiny.gguf --chat-template
PYTHONPATH=../llama.cpp/gguf-py python3 tools/make_tiny_gguf.py draft.gguf --layers 2 --seed 7
mpirun -np 4 ./build/vibeprof_llm -m tiny.gguf -md tiny.gguf -n 24 --isl 80 --osl 40 -c 256 --temp 0 --verify
mpirun -np 3 ./build/vibeprof_llm -m tiny.gguf -p examples/prompts_chat.jsonl --osl-from-data -c 512 --output out.jsonl
VIBEPROF_LLM_EMULATE_GPUS=1 mpirun -np 6 ./build/vibeprof_llm -m tiny.gguf --gpus-per-node 2 --decode-gpus 1 -n 60 -c 256
```

## Notes and limits

* Every rank loads its model(s). CPU ranks on a node share the mmapped weights via the page cache. With
  `--draft-prefill prefill` both prefill and decode ranks load the draft model.
* Speculative types `draft-eagle3`, `draft-mtp`, `draft-dflash` and `draft-dspark` are rejected for now. They
  carry extra per-sequence state (target hidden states, `common_speculative_get_state`) that the handoff does
  not ship yet.
* Speculation requires partial KV removal, so recurrent/hybrid models are refused when it is enabled. Without
  it, such models go through the same `llama_state_seq_*` API but were not tested.
* Large GPU decode batches with speculation allocate logits for `M·(k+1)` outputs per step, e.g. 128 × 9 ×
  128k vocabulary ≈ 590 MB in f32. Size `--decode-batch-gpu` and `--spec-draft-max` accordingly. More than 256
  sequences per GPU needs several decode ranks on that GPU.
* Chat-templated prompts are tokenized as one string with special-token parsing on. Fine for benchmarking, but
  a serving system should not parse special tokens inside user content.
* GPU-direct KV transfer needs the bundled llama.cpp patch (tested up to llama.cpp `5917b39`, 2026-11-25).
  Future llama.cpp changes to the state-saving code may need small adjustments; `patches/apply.sh` tries a
  3-way merge first. Without flash attention, llama.cpp stores V transposed and exports it row by
  row; on a GPU that means many small synchronized copies per request in both transports. Keep flash
  attention on (the default) for large transfers.
* In-flight device staging buffers use GPU memory on the sender until the send completes, roughly one KV state
  per request being handed off.
* Passive-target RMA needs the target (rank 0) to make progress. Keep `--progress all` (the default) unless
  your MPI has hardware RMA atomics or asynchronous progress (e.g. `MPICH_ASYNC_PROGRESS=1`).
* Not yet implemented: prefix caching across requests sharing a system prompt, open-loop (Poisson) arrival
  times, and chunked prefill mixed into decode batches.
