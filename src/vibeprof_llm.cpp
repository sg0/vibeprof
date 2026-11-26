// vibeprof_llm.cpp
//
// MPI wrapper around llama.cpp (linked statically, used in-process; no server) that runs
// continuously batched, disaggregated prefill/decode inference on synthetic or real prompts:
//
//   * PREFILL ranks (normally one per GPU) pull work from a global task counter (NXTVAL,
//     MPI-3 RMA MPI_Fetch_and_op on rank 0), build the prompt (synthetic generator, or a
//     dataset record rendered with the model's chat template and tokenized), prefill one or
//     more prompts per llama_process(), sample the first token (TTFT), export the KV cache of
//     each sequence with llama_state_seq_get_data() and ship it to a decode rank. With a
//     draft model, the draft is prefilled on the same device and its KV cache is shipped too.
//   * DECODE ranks run continuous batching: up to N sequences live in one llama_context,
//     new sequences join as soon as their KV arrives, finished ones leave immediately, and
//     every step is one llama_process() over all active sequences. Decode ranks run on CPU
//     cores by default and optionally on GPUs (--decode-gpus) with large batches.
//   * Optional speculative decoding on the decode ranks via llama.cpp's common_speculative:
//     n-gram self-speculation (no extra model) and/or a draft model. Drafts for all active
//     sequences are generated in one batched call and verified in the same llama_process().
//   * Decode capacity (sequence slots + KV cells) is reserved through a second RMA array,
//     giving capacity-proportional routing and back-pressure.
//
// Works with any (odd or even) number of ranks >= 2 and any number of GPUs per node.
// Build: see CMakeLists.txt / README.md.

#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>          // MPIX_Query_cuda_support / MPIX_Query_rocm_support
#endif

#include "ggml-backend.h"
#include "llama.h"

#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;

// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------

static int    g_rank  = 0;
static int    g_world = 1;
static double g_t0    = 0.0;     // common epoch (set right after a barrier)
static bool   g_verbose = false;

static double now_s() { return MPI_Wtime() - g_t0; }

[[noreturn]] static void fatal(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[rank %d] FATAL: ", g_rank);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void logr(const char * fmt, ...) {
    if (!g_verbose) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[rank %d %9.4f] ", g_rank, now_s());
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static void llama_log_cb(ggml_log_level level, const char * text, void *) {
    static ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (g_verbose || last == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s%s", level == GGML_LOG_LEVEL_CONT ? "" : ("[rank " + std::to_string(g_rank) + " llama] ").c_str(), text);
    }
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, ',')) if (!cur.empty()) out.push_back(cur);
    return out;
}

// ----------------------------------------------------------------------------
// Arguments
// ----------------------------------------------------------------------------

struct Args {
    std::string model;
    std::string model_stage;   // node-local directory to copy the model(s) into before loading ("" = load in place)
    // workload
    int64_t n_requests   = 64;     // total requests (0 => run until --duration)
    bool    n_set        = false;  // -n given explicitly
    double  duration     = 0.0;    // seconds; 0 => unlimited
    int     isl          = 128;    // synthetic: mean input length (tokens)
    double  isl_std      = 0.0;
    int     osl          = 64;     // output length (max new tokens)
    double  osl_std      = 0.0;
    std::string prompt_mode = "tokens"; // synthetic generator: tokens | words
    uint32_t seed        = 1234;
    // real prompts
    std::string prompts;                 // dataset file (enables real-prompt mode)
    std::string prompt_format = "auto";  // auto | jsonl | json | txt
    std::string chat          = "auto";  // auto | on | off
    std::string chat_template_file;      // jinja template override
    std::string system_prompt;
    std::string conv_turns    = "first"; // first | all
    bool    osl_from_data = false;
    bool    shuffle       = false;
    int64_t max_prompts   = 0;
    std::string ignore_eos_s = "auto";   // auto | on | off
    bool    ignore_eos    = true;        // resolved
    std::string output;                  // generated text (JSONL)
    // sampling
    float temp   = 0.8f;
    int   top_k  = 40;
    float top_p  = 0.95f;
    float min_p  = 0.0f;
    // llama.cpp context
    int ctx            = 4096;  // per-sequence context window
    int n_batch        = 2048;  // logical batch for prefill
    int n_ubatch       = 512;   // physical batch for prefill
    int prefill_batch  = 1;     // max prompts prefilled together
    int prefill_tokens = 0;     // token budget per prefill batch (0 = off)
    std::string claim  = "guided"; // fixed | guided (shrink claims as the task pool drains)
    double claim_factor = 2.0;  // guided: batch <= ceil(remaining / (factor * #prefill ranks))
    std::string decode_batch        = "16";   // CPU decode ranks: N | N1,N2,... | per-thread:K
    std::string decode_kv_cells;              // CPU decode ranks: C | C1,C2,...  (default ctx*slots)
    std::string decode_batch_gpu    = "128";  // GPU decode ranks: N | N1,N2,...
    std::string decode_kv_cells_gpu;          // GPU decode ranks: C | C1,C2,...
    int decode_ubatch  = 512;   // physical batch on decode ranks
    int threads_prefill = 0;    // 0 => auto
    int threads_decode  = 0;    // 0 => auto
    int  n_gpu_layers   = -1;   // GPU ranks: layers on GPU (-1 => all)
    bool flash_attn     = true;
    std::string kv_type = "f16";
    // speculative decoding
    std::string spec_type = "none";      // comma list of llama.cpp speculative types
    std::string draft_model;
    int   spec_n_max  = 8;
    int   spec_n_min  = 0;
    float spec_p_min  = 0.0f;
    int   spec_ngram_n = -1, spec_ngram_m = -1, spec_ngram_hits = -1;
    double spec_synth_len = -1.0;
    bool  spec_rejection = false;
    std::string draft_prefill = "prefill"; // prefill | decode
    // topology
    int gpus_per_node    = -1;  // -1 => auto-detect
    int prefill_per_node = -1;  // -1 => #prefill GPUs (or half the ranks if no GPU)
    int prefill_gpus     = -1;  // GPUs per node used by prefill ranks (-1 => auto)
    int decode_gpus      = 0;   // GPU decode ranks per node
    bool hide_gpus_on_decode = true;
    std::string progress = "all";
    // KV transfer
    std::string kv_transport = "auto";   // auto | host | device
    std::string kv_staging   = "pinned"; // pinned | pageable
    std::string gpu_aware    = "auto";   // auto | yes | no
    // output
    std::string csv;
    int warmup = 1;
    bool verbose = false;
    bool verify  = false;
    // derived
    std::vector<common_speculative_type> spec_types;
    bool use_spec  = false;
    bool use_draft = false;
    std::string kv_desc;                 // resolved KV transport, for the report
};

static void usage(const char * prog) {
    if (g_rank != 0) return;
    printf(
        "usage: mpirun -np N %s -m model.gguf [options]\n\n"
        "workload:\n"
        "  -m,  --model PATH          GGUF model (required)\n"
        "       --model-stage DIR     copy the model (and draft model) to node-local DIR first, once per node, and\n"
        "                             load from there (e.g. /local/scratch); default $VIBEPROF_MODEL_STAGE\n"
        "  -n,  --num-requests N      total requests (default 64, or #records with --prompts; 0 = until --duration)\n"
        "       --duration SEC        stop claiming new requests after SEC seconds\n"
        "       --isl N / --isl-std X synthetic input length mean / stddev (default 128 / 0)\n"
        "       --osl N / --osl-std X output length (max new tokens) mean / stddev (default 64 / 0)\n"
        "       --prompt-mode M       synthetic prompts: tokens | words (default tokens)\n"
        "  -s,  --seed N              RNG seed (default 1234)\n"
        "real prompts:\n"
        "  -p,  --prompts FILE        dataset: .jsonl / .json (array) / .txt (one prompt per line). Records may be\n"
        "                             {\"messages\":[{role,content}]}, ShareGPT {\"conversations\":[{from,value}]},\n"
        "                             or {\"prompt\"|\"text\"|\"input\"|\"question\": str}; optional max_tokens /\n"
        "                             output_len, or a reference answer (output/response/completion/answer)\n"
        "       --prompt-format F     auto | jsonl | json | txt (default auto)\n"
        "       --chat auto|on|off    apply the model's chat template: auto = for message records only (default)\n"
        "       --chat-template-file F  Jinja chat template overriding the model's\n"
        "       --system-prompt STR   system message added to chat prompts that have none\n"
        "       --conv-turns first|all  conversations: first user turn (default) or full history\n"
        "       --osl-from-data       output length from the record (max_tokens or reference answer length)\n"
        "       --shuffle             shuffle record order (seeded); requests cycle through records\n"
        "       --max-prompts N       load at most N records\n"
        "       --ignore-eos auto|on|off  auto = on for synthetic, off for --prompts (default auto)\n"
        "       --output FILE         write generated text per request as JSONL\n"
        "sampling:\n"
        "       --temp X              temperature (default 0.8; <=0 => greedy)\n"
        "       --top-k N / --top-p X / --min-p X  (default 40 / 0.95 / 0)\n"
        "llama.cpp:\n"
        "  -c,  --ctx N               context window per sequence (default 4096)\n"
        "  -b,  --n-batch N           logical batch for prefill (default 2048)\n"
        "  -ub, --n-ubatch N          physical batch for prefill (default 512)\n"
        "       --prefill-batch N     max prompts prefilled together (default 1)\n"
        "       --prefill-tokens N    token budget per prefill batch; claims one task at a time (default 0 = off)\n"
        "       --claim fixed|guided  guided (default): near the end of a -n run, cap each prefill batch at\n"
        "                             ceil(remaining / (K * #prefill ranks)) so no rank holds a large last chunk\n"
        "       --claim-factor K      guided claiming factor (default 2; larger = smaller late batches)\n"
        "       --decode-batch SPEC   slots per CPU decode rank: N | N1,N2,... | per-thread:K (default 16)\n"
        "       --decode-kv-cells SPEC  KV cells per CPU decode rank: C | C1,C2,... (default ctx*slots)\n"
        "       --decode-batch-gpu SPEC slots per GPU decode rank: N | N1,N2,... (default 128, max 256)\n"
        "       --decode-kv-cells-gpu SPEC KV cells per GPU decode rank (default ctx*slots)\n"
        "       --decode-ubatch N     physical batch on decode ranks (default 512)\n"
        "       --threads-prefill N / --threads-decode N  (default auto)\n"
        "  -ngl,--n-gpu-layers N      layers offloaded on GPU ranks (default -1 = all)\n"
        "  -fa, --flash-attn on|off   must match on both sides (default on)\n"
        "       --kv-type f16|q8_0|f32|bf16  KV cache type (default f16)\n"
        "speculative decoding (decode ranks):\n"
        "       --spec-type LIST      comma list: none, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod,\n"
        "                             ngram-cache, draft-simple (default none; -md implies draft-simple)\n"
        "  -md, --draft-model PATH    draft model GGUF (same vocabulary as the target)\n"
        "       --spec-draft-max N    max drafted tokens per sequence and step (default 8)\n"
        "       --spec-draft-min N    discard draft-model drafts shorter than N (default 0)\n"
        "       --spec-draft-p-min X  stop drafting below this draft probability (default 0)\n"
        "       --spec-ngram-n N / --spec-ngram-m M / --spec-ngram-min-hits K  n-gram lookup/draft sizes\n"
        "       --spec-rejection      sample drafts and verify by rejection sampling (draft model, temp > 0)\n"
        "       --spec-synth-len L    benchmarking only: synthetic acceptance with mean accepted length L\n"
        "       --draft-prefill prefill|decode  where the draft model prefills the prompt (default prefill)\n"
        "topology:\n"
        "       --gpus-per-node N     override GPU auto-detection\n"
        "       --prefill-gpus N      GPUs per node used for prefill (default all minus --decode-gpus)\n"
        "       --decode-gpus N       GPU decode ranks per node (default 0); GPUs are shared with prefill\n"
        "                             ranks if prefill-gpus + decode-gpus > GPUs\n"
        "       --prefill-per-node N  prefill ranks per node (default = prefill GPUs, or half if none)\n"
        "       --no-hide-gpus        do not hide GPUs from CPU decode ranks\n"
        "       --progress none|root|all  MPI async-progress helper thread (default all)\n"
        "KV transfer:\n"
        "       --kv-transport auto|host|device  host = one serialized blob in host memory; device = metadata on\n"
        "                             the host + KV tensor data in GPU memory, sent with GPU-aware MPI (needs the\n"
        "                             patched llama.cpp, see patches/). auto = device if available (default)\n"
        "       --kv-staging pinned|pageable  host staging buffers on GPU ranks (default pinned)\n"
        "       --gpu-aware-mpi auto|yes|no  override GPU-aware MPI detection (default auto)\n"
        "output:\n"
        "       --csv PATH            per-request records\n"
        "       --warmup N            warmup iterations per rank (default 1)\n"
        "       --verify              (needs --temp 0) prefill rank also decodes each request locally, without\n"
        "                             speculation; decode output must be token-identical (slow; for testing)\n"
        "  -v,  --verbose\n",
        prog);
}

static Args parse_args(int argc, char ** argv) {
    Args a;
    auto need = [&](int & i) -> const char * {
        if (i + 1 >= argc) { usage(argv[0]); fatal("missing value for %s", argv[i]); }
        return argv[++i];
    };
    auto onoff = [&](const char * v) { std::string s = v; return s == "on" || s == "1" || s == "true"; };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "-m"  || k == "--model")          a.model = need(i);
        else if (k == "--model-stage")                  a.model_stage = need(i);
        else if (k == "-n"  || k == "--num-requests") { a.n_requests = atoll(need(i)); a.n_set = true; }
        else if (k == "--duration")                     a.duration = atof(need(i));
        else if (k == "--isl")                          a.isl = atoi(need(i));
        else if (k == "--isl-std")                      a.isl_std = atof(need(i));
        else if (k == "--osl")                          a.osl = atoi(need(i));
        else if (k == "--osl-std")                      a.osl_std = atof(need(i));
        else if (k == "--prompt-mode")                  a.prompt_mode = need(i);
        else if (k == "-s"  || k == "--seed")           a.seed = (uint32_t) atoll(need(i));
        else if (k == "-p"  || k == "--prompts")        a.prompts = need(i);
        else if (k == "--prompt-format")                a.prompt_format = need(i);
        else if (k == "--chat")                         a.chat = need(i);
        else if (k == "--chat-template-file")           a.chat_template_file = need(i);
        else if (k == "--system-prompt")                a.system_prompt = need(i);
        else if (k == "--conv-turns")                   a.conv_turns = need(i);
        else if (k == "--osl-from-data")                a.osl_from_data = true;
        else if (k == "--shuffle")                      a.shuffle = true;
        else if (k == "--max-prompts")                  a.max_prompts = atoll(need(i));
        else if (k == "--ignore-eos")                   a.ignore_eos_s = need(i);
        else if (k == "--output")                       a.output = need(i);
        else if (k == "--temp")                         a.temp = (float) atof(need(i));
        else if (k == "--top-k")                        a.top_k = atoi(need(i));
        else if (k == "--top-p")                        a.top_p = (float) atof(need(i));
        else if (k == "--min-p")                        a.min_p = (float) atof(need(i));
        else if (k == "-c"  || k == "--ctx")            a.ctx = atoi(need(i));
        else if (k == "-b"  || k == "--n-batch")        a.n_batch = atoi(need(i));
        else if (k == "-ub" || k == "--n-ubatch")       a.n_ubatch = atoi(need(i));
        else if (k == "--prefill-batch")                a.prefill_batch = atoi(need(i));
        else if (k == "--prefill-tokens")               a.prefill_tokens = atoi(need(i));
        else if (k == "--claim")                        a.claim = need(i);
        else if (k == "--claim-factor")                 a.claim_factor = atof(need(i));
        else if (k == "--decode-batch")                 a.decode_batch = need(i);
        else if (k == "--decode-kv-cells")              a.decode_kv_cells = need(i);
        else if (k == "--decode-batch-gpu")             a.decode_batch_gpu = need(i);
        else if (k == "--decode-kv-cells-gpu")          a.decode_kv_cells_gpu = need(i);
        else if (k == "--decode-ubatch")                a.decode_ubatch = atoi(need(i));
        else if (k == "--threads-prefill")              a.threads_prefill = atoi(need(i));
        else if (k == "--threads-decode")               a.threads_decode = atoi(need(i));
        else if (k == "-ngl" || k == "--n-gpu-layers")  a.n_gpu_layers = atoi(need(i));
        else if (k == "-fa" || k == "--flash-attn")     a.flash_attn = onoff(need(i));
        else if (k == "--kv-type")                      a.kv_type = need(i);
        else if (k == "--spec-type")                    a.spec_type = need(i);
        else if (k == "-md" || k == "--draft-model")    a.draft_model = need(i);
        else if (k == "--spec-draft-max")               a.spec_n_max = atoi(need(i));
        else if (k == "--spec-draft-min")               a.spec_n_min = atoi(need(i));
        else if (k == "--spec-draft-p-min")             a.spec_p_min = (float) atof(need(i));
        else if (k == "--spec-ngram-n")                 a.spec_ngram_n = atoi(need(i));
        else if (k == "--spec-ngram-m")                 a.spec_ngram_m = atoi(need(i));
        else if (k == "--spec-ngram-min-hits")          a.spec_ngram_hits = atoi(need(i));
        else if (k == "--spec-rejection")               a.spec_rejection = true;
        else if (k == "--spec-synth-len")               a.spec_synth_len = atof(need(i));
        else if (k == "--draft-prefill")                a.draft_prefill = need(i);
        else if (k == "--gpus-per-node")                a.gpus_per_node = atoi(need(i));
        else if (k == "--prefill-per-node")             a.prefill_per_node = atoi(need(i));
        else if (k == "--prefill-gpus")                 a.prefill_gpus = atoi(need(i));
        else if (k == "--decode-gpus")                  a.decode_gpus = atoi(need(i));
        else if (k == "--no-hide-gpus")                 a.hide_gpus_on_decode = false;
        else if (k == "--progress")                     a.progress = need(i);
        else if (k == "--kv-transport")                 a.kv_transport = need(i);
        else if (k == "--kv-staging")                   a.kv_staging = need(i);
        else if (k == "--gpu-aware-mpi")                a.gpu_aware = need(i);
        else if (k == "--csv")                          a.csv = need(i);
        else if (k == "--warmup")                       a.warmup = atoi(need(i));
        else if (k == "-v"  || k == "--verbose")        a.verbose = true;
        else if (k == "--verify")                       a.verify = true;
        else if (k == "-h"  || k == "--help")         { usage(argv[0]); MPI_Finalize(); exit(0); }
        else { usage(argv[0]); fatal("unknown argument: %s", k.c_str()); }
    }
    if (a.model.empty()) { usage(argv[0]); fatal("--model is required"); }
    if (a.model_stage.empty()) { const char * e = getenv("VIBEPROF_MODEL_STAGE"); if (e) a.model_stage = e; }
    const bool real = !a.prompts.empty();
    if (!real && a.n_requests <= 0 && a.duration <= 0) fatal("need --num-requests > 0 or --duration > 0");
    if (a.osl < 1)   fatal("--osl must be >= 1");
    if (a.osl >= a.ctx) fatal("--osl (%d) must be below --ctx (%d)", a.osl, a.ctx);
    if (!real) {
        if (a.isl < 1) fatal("--isl must be >= 1");
        if (a.isl + a.osl > a.ctx) fatal("isl + osl (%d) exceeds --ctx (%d)", a.isl + a.osl, a.ctx);
    }
    if (a.prefill_batch < 1)  fatal("--prefill-batch must be >= 1");
    if (a.prefill_tokens < 0) fatal("--prefill-tokens must be >= 0");
    if (a.claim != "fixed" && a.claim != "guided") fatal("--claim must be fixed|guided");
    if (a.kv_transport != "auto" && a.kv_transport != "host" && a.kv_transport != "device") fatal("--kv-transport must be auto|host|device");
    if (a.kv_staging != "pinned" && a.kv_staging != "pageable") fatal("--kv-staging must be pinned|pageable");
    if (a.gpu_aware != "auto" && a.gpu_aware != "yes" && a.gpu_aware != "no") fatal("--gpu-aware-mpi must be auto|yes|no");
    if (!(a.claim_factor > 0)) fatal("--claim-factor must be > 0");
    if (a.decode_ubatch < 1)  fatal("--decode-ubatch must be >= 1");
    if (a.prompt_mode != "tokens" && a.prompt_mode != "words") fatal("--prompt-mode must be tokens|words");
    if (a.chat != "auto" && a.chat != "on" && a.chat != "off") fatal("--chat must be auto|on|off");
    if (a.conv_turns != "first" && a.conv_turns != "all")       fatal("--conv-turns must be first|all");
    if (a.prompt_format != "auto" && a.prompt_format != "jsonl" && a.prompt_format != "json" && a.prompt_format != "txt")
        fatal("--prompt-format must be auto|jsonl|json|txt");
    if (a.ignore_eos_s == "auto") a.ignore_eos = !real;
    else if (a.ignore_eos_s == "on" || a.ignore_eos_s == "off") a.ignore_eos = a.ignore_eos_s == "on";
    else fatal("--ignore-eos must be auto|on|off");
    if (a.draft_prefill != "prefill" && a.draft_prefill != "decode") fatal("--draft-prefill must be prefill|decode");

    // ---- speculative types ----
    for (const auto & name : split_csv(a.spec_type)) {
        common_speculative_type t;
        try { t = common_speculative_type_from_name(name); }
        catch (const std::exception &) { fatal("unknown --spec-type '%s' (known: %s)", name.c_str(), common_speculative_all_types_str()); }
        if (t == COMMON_SPECULATIVE_TYPE_NONE) continue;
        switch (t) {
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
                break;
            default:
                fatal("--spec-type %s is not supported by the disaggregated handoff yet (supported: ngram-*, draft-simple)", name.c_str());
        }
        if (std::find(a.spec_types.begin(), a.spec_types.end(), t) == a.spec_types.end()) a.spec_types.push_back(t);
    }
    const bool has_draft_type = std::find(a.spec_types.begin(), a.spec_types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) != a.spec_types.end();
    if (!a.draft_model.empty() && !has_draft_type) a.spec_types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
    a.use_spec  = !a.spec_types.empty();
    a.use_draft = std::find(a.spec_types.begin(), a.spec_types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) != a.spec_types.end();
    if (a.use_draft && a.draft_model.empty()) fatal("--spec-type draft-simple needs --draft-model");
    if (a.use_spec && a.spec_n_max < 1)       fatal("--spec-draft-max must be >= 1");
    if (a.spec_rejection && (!a.use_draft || a.temp <= 0)) fatal("--spec-rejection needs a draft model and --temp > 0");
    if (a.spec_synth_len != -1.0 && !a.use_spec) fatal("--spec-synth-len needs speculative decoding enabled");
    if (a.spec_synth_len != -1.0 && a.spec_synth_len < 1.0) fatal("--spec-synth-len must be >= 1 (it includes the target token)");
    if (a.verify && a.temp > 0) fatal("--verify requires --temp 0 (greedy)");
    if (a.verify && a.spec_synth_len != -1.0) fatal("--verify cannot be combined with --spec-synth-len (synthetic output is not valid)");
    if (a.verify && a.flash_attn && g_rank == 0)
        fprintf(stderr, "warning: --verify with flash attention: FA rounding depends on the batch shape and KV size, which differ\n"
                        "         between the decode context and the local reference, so rare argmax near-ties can flip.\n"
                        "         Use -fa off for a strict bit-exactness check.\n");
    return a;
}

static ggml_type kv_ggml_type(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "bf16") return GGML_TYPE_BF16;
    fatal("unsupported --kv-type %s", s.c_str());
}

// Parse "N", "N1,N2,..." or (if allow_per_thread) "per-thread:K" into the value for rank #idx of n.
static int64_t pick_capacity(const std::string & spec, const char * opt, int idx, int n, int threads, bool allow_per_thread) {
    if (allow_per_thread && spec.rfind("per-thread:", 0) == 0) {
        const int64_t k = atoll(spec.c_str() + 11);
        if (k < 1) fatal("%s: per-thread:K needs K >= 1", opt);
        return k * std::max(1, threads);
    }
    std::vector<int64_t> v;
    for (const auto & tok : split_csv(spec)) {
        char * end = nullptr;
        const long long x = strtoll(tok.c_str(), &end, 10);
        if (*end != '\0' || x < 1) fatal("%s: invalid value '%s' in '%s'", opt, tok.c_str(), spec.c_str());
        v.push_back(x);
    }
    if (v.empty()) fatal("%s: empty value", opt);
    if (v.size() == 1) return v[0];
    if ((int) v.size() != n) fatal("%s lists %zu values but there are %d such decode ranks", opt, v.size(), n);
    return v[idx];
}

// ----------------------------------------------------------------------------
// Global RMA counters: NXTVAL + per-decode-rank load (capacity reservation)
//
// Window lives on rank 0:  [0] = next task id,  [1 + r] = in-flight seqs on rank r,
//                          [1 + P + r] = KV cells reserved on rank r   (P = #ranks)
// A passive-target epoch (MPI_Win_lock_all) is held for the whole run; each
// operation is an atomic MPI_Fetch_and_op / MPI_Get_accumulate followed by a flush.
// ----------------------------------------------------------------------------

class GlobalCounters {
public:
    void init(MPI_Comm comm) {
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);
        const int n = 1 + 2 * size_;
        MPI_Aint bytes = (rank_ == 0) ? (MPI_Aint) n * sizeof(int64_t) : 0;
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ops", "same_op_no_op");
        MPI_Win_allocate(bytes, sizeof(int64_t), info, comm, &base_, &win_);
        MPI_Info_free(&info);
        if (rank_ == 0) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_);
            for (int i = 0; i < n; ++i) base_[i] = 0;
            MPI_Win_unlock(0, win_);
        }
        MPI_Barrier(comm);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
    }

    // NXTVAL: atomically fetch the counter and add `inc`. Returns the old value.
    int64_t nxtval(int64_t inc = 1) { return fetch_add(0, inc); }
    // Atomic read of the task counter (MPI_Fetch_and_op with MPI_NO_OP).
    int64_t peek() {
        int64_t dummy = 0, v = -1;
        MPI_Fetch_and_op(&dummy, &v, MPI_INT64_T, 0, 0, MPI_NO_OP, win_);
        MPI_Win_flush(0, win_);
        return v;
    }
    int64_t load_add(int r, int64_t delta)  { return fetch_add(1 + r, delta); }
    int64_t cells_add(int r, int64_t delta) { return fetch_add(1 + size_ + r, delta); }

    // Atomic (per element) snapshot: out[0..P) = seq loads, out[P..2P) = reserved cells.
    void load_snapshot(std::vector<int64_t> & out) {
        out.resize(2 * size_);
        MPI_Get_accumulate(nullptr, 0, MPI_INT64_T, out.data(), 2 * size_, MPI_INT64_T,
                           0, 1, 2 * size_, MPI_INT64_T, MPI_NO_OP, win_);
        MPI_Win_flush(0, win_);
    }

    void finalize() {
        MPI_Win_unlock_all(win_);
        MPI_Win_free(&win_);
    }

private:
    int64_t fetch_add(MPI_Aint disp, int64_t inc) {
        int64_t old = -1;
        MPI_Fetch_and_op(&inc, &old, MPI_INT64_T, 0, disp, MPI_SUM, win_);
        MPI_Win_flush(0, win_);
        return old;
    }
    MPI_Win   win_  = MPI_WIN_NULL;
    int64_t * base_ = nullptr;
    int rank_ = 0, size_ = 1;
};

// ----------------------------------------------------------------------------
// Optional MPI progress helper thread (makes passive-target RMA on rank 0 and
// rendezvous transfers progress while the main thread is inside llama.cpp)
// ----------------------------------------------------------------------------

class ProgressThread {
public:
    void start(MPI_Comm comm) {
        MPI_Comm_dup(comm, &comm_);
        MPI_Comm_rank(comm_, &me_);
        MPI_Irecv(&dummy_, 1, MPI_INT, me_, 7, comm_, &req_);
        th_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed)) {
                int flag = 0;
                MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
                if (flag) break;
                usleep(200);
            }
        });
        running_ = true;
    }
    void stop() {
        if (!running_) return;
        stop_ = true;
        th_.join();
        int one = 1;
        MPI_Send(&one, 1, MPI_INT, me_, 7, comm_);   // complete the dummy receive
        MPI_Wait(&req_, MPI_STATUS_IGNORE);
        MPI_Comm_free(&comm_);
        running_ = false;
    }
private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    MPI_Request req_ = MPI_REQUEST_NULL;
    int dummy_ = 0, me_ = 0;
    std::thread th_;
    std::atomic<bool> stop_{false};
    bool running_ = false;
};

// ----------------------------------------------------------------------------
// Synthetic prompt generator (deterministic per task id)
// ----------------------------------------------------------------------------

static const char * k_words[] = {
    "the","of","and","to","in","is","was","that","for","on","with","as","by","at","from","his","her",
    "they","we","you","this","which","or","an","be","are","have","had","not","but","what","all","were",
    "when","there","can","more","if","no","out","so","said","time","about","many","then","them","these",
    "would","like","into","system","model","data","network","energy","river","mountain","history",
    "science","music","market","computer","language","city","garden","engine","signal","memory",
    "parallel","process","matrix","vector","kernel","cluster","node","message","window","atomic",
    "counter","latency","throughput","token","sequence","layer","attention","cache","prefill","decode",
    "ocean","forest","planet","silver","quiet","rapid","ancient","modern","bright","heavy","simple",
    "complex","global","local","random","stable","winter","summer","morning","evening","story","answer",
    "question","reason","result","method","theory","practice","research","design","pattern","structure",
    "function","value","number","table","chair","window","bridge","road","train","station","library",
    "student","teacher","doctor","artist","writer","builder","farmer","pilot","sailor","scientist",
    "explain","describe","compare","analyze","summarize","translate","compute","measure","observe",
};
static constexpr int k_n_words = sizeof(k_words) / sizeof(k_words[0]);

struct LenSpec { int isl; int osl; };

static LenSpec sample_lengths(const Args & a, int64_t id) {
    std::mt19937_64 rng(((uint64_t) a.seed << 32) ^ (uint64_t) (id * 0x9E3779B97F4A7C15ULL));
    LenSpec t{a.isl, a.osl};
    if (a.isl_std > 0) { std::normal_distribution<double> nd(a.isl, a.isl_std); t.isl = (int) std::lround(nd(rng)); }
    if (a.osl_std > 0) { std::normal_distribution<double> nd(a.osl, a.osl_std); t.osl = (int) std::lround(nd(rng)); }
    t.osl = std::clamp(t.osl, 1, a.ctx - 1);
    t.isl = std::clamp(t.isl, 1, a.ctx - t.osl);
    return t;
}

class PromptGen {
public:
    void init(const llama_vocab * vocab, const Args & a) {
        vocab_ = vocab;
        args_ = &a;
        add_bos_ = llama_vocab_get_add_bos(vocab);
        bos_ = llama_vocab_bos(vocab);
        const int n = llama_vocab_n_tokens(vocab);
        for (llama_token t = 0; t < n; ++t) {
            if (llama_vocab_is_control(vocab, t) || llama_vocab_is_eog(vocab, t)) continue;
            ordinary_.push_back(t);
        }
        if (ordinary_.empty()) fatal("vocabulary has no ordinary tokens");
    }

    std::vector<llama_token> make(int64_t id, int isl) const {
        std::mt19937_64 rng(((uint64_t) args_->seed * 0x2545F4914F6CDD1DULL) ^ (uint64_t) (id + 1));
        std::vector<llama_token> out;
        out.reserve(isl);
        if (args_->prompt_mode == "words") {
            std::uniform_int_distribution<int> wd(0, k_n_words - 1);
            std::string text;
            for (;;) {
                for (int i = 0; i < 64; ++i) { text += k_words[wd(rng)]; text += (i % 13 == 12) ? ". " : " "; }
                auto toks = common_tokenize(vocab_, text, true, false);
                if ((int) toks.size() >= isl) { out.assign(toks.begin(), toks.begin() + isl); break; }
            }
        } else {
            std::uniform_int_distribution<size_t> td(0, ordinary_.size() - 1);
            if (add_bos_ && bos_ != LLAMA_TOKEN_NULL) out.push_back(bos_);
            while ((int) out.size() < isl) out.push_back(ordinary_[td(rng)]);
        }
        return out;
    }

private:
    const llama_vocab * vocab_ = nullptr;
    const Args * args_ = nullptr;
    bool add_bos_ = false;
    llama_token bos_ = LLAMA_TOKEN_NULL;
    std::vector<llama_token> ordinary_;
};

// ----------------------------------------------------------------------------
// Real-prompt datasets
//
// The first prefill rank parses the file into compact records (messages + an output
// length hint), which are broadcast to all prefill ranks. Rendering with the chat
// template and tokenization happen when a request is claimed (counted in TTFT, as in
// a serving engine).
// ----------------------------------------------------------------------------

struct Msg    { std::string role, content; };
struct Record { std::vector<Msg> msgs; bool chat = false; int32_t osl_hint = 0; };

static std::string json_text(const json & v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_array()) {                       // OpenAI content parts
        std::string s;
        for (const auto & p : v) {
            if (p.is_string()) s += p.get<std::string>();
            else if (p.is_object() && p.contains("text") && p["text"].is_string()) s += p["text"].get<std::string>();
        }
        return s;
    }
    return "";
}

static std::string norm_role(std::string r) {
    std::transform(r.begin(), r.end(), r.begin(), ::tolower);
    if (r == "human" || r == "user")                                   return "user";
    if (r == "gpt" || r == "assistant" || r == "chatgpt" || r == "bard" || r == "bing" || r == "model") return "assistant";
    if (r == "system")                                                 return "system";
    return r;
}

static bool parse_record(const json & o, const Args & a, const llama_vocab * vocab, Record & rec) {
    auto ntok = [&](const std::string & s) { return (int32_t) common_tokenize(vocab, s, false, false).size(); };
    if (!o.is_object()) {
        if (o.is_string()) { rec.msgs = {{"", o.get<std::string>()}}; return !rec.msgs[0].content.empty(); }
        return false;
    }
    std::vector<Msg> conv;
    if (o.contains("messages") && o["messages"].is_array()) {
        for (const auto & m : o["messages"]) if (m.is_object()) conv.push_back({norm_role(m.value("role", "user")), json_text(m.value("content", json()))});
    } else if (o.contains("conversations") && o["conversations"].is_array()) {
        for (const auto & m : o["conversations"]) if (m.is_object()) conv.push_back({norm_role(m.value("from", "human")), json_text(m.value("value", json()))});
    }
    std::string reference;
    if (!conv.empty()) {
        rec.chat = true;
        if (a.conv_turns == "first") {
            size_t u = 0;
            while (u < conv.size() && conv[u].role != "user") ++u;
            if (u == conv.size()) return false;
            for (size_t i = 0; i < u; ++i) if (conv[i].role == "system") rec.msgs.push_back(conv[i]);
            rec.msgs.push_back(conv[u]);
            if (u + 1 < conv.size() && conv[u + 1].role == "assistant") reference = conv[u + 1].content;
        } else {
            if (conv.back().role == "assistant") { reference = conv.back().content; conv.pop_back(); }
            rec.msgs = conv;
            if (rec.msgs.empty()) return false;
        }
    } else {
        for (const char * key : {"prompt", "text", "input", "question", "instruction"}) {
            if (o.contains(key) && o[key].is_string()) { rec.msgs = {{"", o[key].get<std::string>()}}; break; }
        }
        if (rec.msgs.empty() || rec.msgs[0].content.empty()) return false;
    }
    for (const char * key : {"max_tokens", "max_new_tokens", "output_len", "output_tokens", "osl"}) {
        if (o.contains(key) && o[key].is_number_integer()) { rec.osl_hint = o[key].get<int32_t>(); break; }
    }
    if (rec.osl_hint <= 0 && a.osl_from_data) {
        if (reference.empty()) {
            for (const char * key : {"output", "response", "completion", "answer", "reference"}) {
                if (o.contains(key) && o[key].is_string()) { reference = o[key].get<std::string>(); break; }
            }
        }
        if (!reference.empty()) rec.osl_hint = ntok(reference);
    }
    return true;
}

static std::vector<Record> load_dataset(const Args & a, const llama_vocab * vocab) {
    std::ifstream f(a.prompts, std::ios::binary);
    if (!f) fatal("cannot open --prompts %s", a.prompts.c_str());
    std::string fmt = a.prompt_format;
    if (fmt == "auto") {
        const auto dot = a.prompts.find_last_of('.');
        const std::string ext = dot == std::string::npos ? "" : a.prompts.substr(dot + 1);
        if (ext == "jsonl" || ext == "ndjson") fmt = "jsonl";
        else if (ext == "json") {
            int c;
            while ((c = f.peek()) != EOF && isspace(c)) f.get();
            fmt = (c == '[') ? "json" : "jsonl";
        } else fmt = "txt";
    }
    std::vector<Record> out;
    int64_t n_bad = 0;
    auto take = [&](const json & o) {
        Record r;
        if (parse_record(o, a, vocab, r)) out.push_back(std::move(r)); else n_bad++;
        return a.max_prompts <= 0 || (int64_t) out.size() < a.max_prompts;
    };
    if (fmt == "json") {
        json all;
        try { all = json::parse(f); } catch (const std::exception & e) { fatal("failed to parse %s: %s", a.prompts.c_str(), e.what()); }
        if (!all.is_array()) fatal("%s: top-level JSON value must be an array", a.prompts.c_str());
        for (const auto & o : all) if (!take(o)) break;
    } else {
        std::string line;
        int64_t ln = 0;
        while (std::getline(f, line)) {
            ++ln;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.find_first_not_of(" \t") == std::string::npos) continue;
            if (fmt == "txt") { if (!take(json(line))) break; continue; }
            json o;
            try { o = json::parse(line); } catch (const std::exception & e) { fatal("%s:%" PRId64 ": invalid JSON: %s", a.prompts.c_str(), ln, e.what()); }
            if (!take(o)) break;
        }
    }
    if (out.empty()) fatal("no usable prompts in %s", a.prompts.c_str());
    fprintf(stderr, "[rank %d] loaded %zu prompts from %s (format %s, %" PRId64 " skipped)\n",
            g_rank, out.size(), a.prompts.c_str(), fmt.c_str(), n_bad);
    return out;
}

// compact serialization for MPI_Bcast
static void put_i32(std::vector<char> & b, int32_t v) { b.insert(b.end(), (char *) &v, (char *) &v + 4); }
static void put_str(std::vector<char> & b, const std::string & s) { put_i32(b, (int32_t) s.size()); b.insert(b.end(), s.begin(), s.end()); }

static std::vector<char> serialize(const std::vector<Record> & R) {
    std::vector<char> b;
    put_i32(b, (int32_t) R.size());
    for (const auto & r : R) {
        put_i32(b, r.osl_hint); put_i32(b, r.chat ? 1 : 0); put_i32(b, (int32_t) r.msgs.size());
        for (const auto & m : r.msgs) { put_str(b, m.role); put_str(b, m.content); }
    }
    return b;
}

static std::vector<Record> deserialize(const std::vector<char> & b) {
    size_t p = 0;
    auto i32 = [&]() { int32_t v; memcpy(&v, b.data() + p, 4); p += 4; return v; };
    auto str = [&]() { const int32_t n = i32(); std::string s(b.data() + p, n); p += n; return s; };
    std::vector<Record> R(i32());
    for (auto & r : R) {
        r.osl_hint = i32(); r.chat = i32() != 0;
        r.msgs.resize(i32());
        for (auto & m : r.msgs) { m.role = str(); m.content = str(); }
    }
    return R;
}

static void bcast_bytes(std::vector<char> & buf, int root, MPI_Comm comm) {
    int64_t n = (int64_t) buf.size();
    MPI_Bcast(&n, 1, MPI_INT64_T, root, comm);
    buf.resize(n);
    const int64_t CH = 1LL << 30;
    for (int64_t off = 0; off < n; off += CH) MPI_Bcast(buf.data() + off, (int) std::min(CH, n - off), MPI_CHAR, root, comm);
}

// Renders a record with the chat template (when applicable) and tokenizes it.
class PromptBuilder {
public:
    void init(const llama_model * model, const Args & a) {
        args_ = &a;
        vocab_ = llama_model_get_vocab(model);
        bos_ = llama_vocab_bos(vocab_);
        std::string override_tmpl;
        if (!a.chat_template_file.empty()) {
            std::ifstream t(a.chat_template_file);
            if (!t) fatal("cannot read --chat-template-file %s", a.chat_template_file.c_str());
            override_tmpl.assign(std::istreambuf_iterator<char>(t), std::istreambuf_iterator<char>());
        }
        if (a.chat != "off") {
            try { tmpls_ = common_chat_templates_init(model, override_tmpl); }
            catch (const std::exception & e) { fatal("chat template init failed: %s", e.what()); }
        }
    }

    std::vector<llama_token> build(const Record & r) const {
        const bool use_chat = tmpls_ && (r.chat || args_->chat == "on");
        std::vector<llama_token> toks;
        if (use_chat) {
            common_chat_templates_inputs in;
            in.add_generation_prompt = true;
            in.use_jinja = true;
            bool has_sys = false;
            for (const auto & m : r.msgs) has_sys |= m.role == "system";
            if (!args_->system_prompt.empty() && !has_sys) { common_chat_msg s; s.role = "system"; s.content = args_->system_prompt; in.messages.push_back(s); }
            for (const auto & m : r.msgs) { common_chat_msg cm; cm.role = m.role.empty() ? "user" : m.role; cm.content = m.content; in.messages.push_back(cm); }
            std::string text;
            try { text = common_chat_templates_apply(tmpls_.get(), in).prompt.str(); }
            catch (const std::exception & e) { fatal("chat template failed: %s", e.what()); }
            toks = common_tokenize(vocab_, text, true, true);
            if (toks.size() >= 2 && toks[0] == bos_ && toks[1] == bos_) toks.erase(toks.begin());   // template + tokenizer BOS
        } else {
            std::string text;
            for (size_t i = 0; i < r.msgs.size(); ++i) { if (i) text += "\n"; text += r.msgs[i].content; }
            toks = common_tokenize(vocab_, text, true, false);
        }
        if (toks.empty()) toks.push_back(bos_ != LLAMA_TOKEN_NULL ? bos_ : 0);
        return toks;
    }

private:
    const Args * args_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    llama_token bos_ = LLAMA_TOKEN_NULL;
    common_chat_templates_ptr tmpls_;
};

// ----------------------------------------------------------------------------
// Sampling (llama.cpp common_sampler, shared by prefill and decode)
// ----------------------------------------------------------------------------

static common_sampler * make_sampler(const llama_model * model, const Args & a, uint32_t seed) {
    common_params_sampling sp;
    sp.seed    = seed;
    sp.temp    = a.temp;
    sp.top_k   = a.top_k;
    sp.top_p   = a.top_p;
    sp.min_p   = a.min_p;
    sp.no_perf = true;
    sp.ignore_eos = a.ignore_eos;
    if (a.ignore_eos) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        for (llama_token t = 0; t < llama_vocab_n_tokens(vocab); ++t)
            if (llama_vocab_is_eog(vocab, t)) sp.logit_bias.push_back({t, -INFINITY});
    }
    common_sampler * s = common_sampler_init(model, sp);
    if (!s) fatal("common_sampler_init failed");
    return s;
}

// Synthetic acceptance (benchmarking): accept draft token i with probability probs[i].
static std::vector<llama_token> accept_synthetic(common_sampler * smpl, llama_context * ctx, const std::vector<int> & idxs,
                                                 const llama_tokens & draft, const std::vector<double> & probs, std::mt19937 & rng) {
    std::vector<llama_token> out;
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        if (i < probs.size() && dist(rng) < probs[i] && !llama_vocab_is_eog(vocab, draft[i])) {
            common_sampler_accept(smpl, draft[i], false);
            out.push_back(draft[i]);
            continue;
        }
        common_sampler_accept(smpl, id, true);
        out.push_back(id);
        return out;
    }
    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    out.push_back(id);
    return out;
}

// ----------------------------------------------------------------------------
// Messages and records
// ----------------------------------------------------------------------------

enum : int { TAG_HDR = 1, TAG_DONE = 2 };
static constexpr int64_t KV_CHUNK = 1LL << 30;   // payload chunk (bytes) per MPI message
enum : int32_t { STOP_LENGTH = 0, STOP_EOS = 1 };

// prefill -> decode, on comm_ctl. Payload on comm_kv:
//   tag 0 (host memory):  [prompt tokens | target state | draft state]   (state = full blob, or metadata if split)
//   tag 1 (split only):   [target tensor data | draft tensor data]       (GPU memory with GPU-aware MPI)
struct KvHeader {
    int64_t  task_id;
    int64_t  rec_idx;         // dataset record (-1 = synthetic)
    int64_t  tok_bytes, kv_tgt_bytes, kv_dft_bytes;
    int64_t  dat_tgt_bytes, dat_dft_bytes;   // split: tensor data sizes (0 otherwise)
    int32_t  isl, max_out, first_token, prefill_rank;
    int32_t  done;            // 1 = finished at prefill (first token was EOG, or max_out == 1): no payload
    int32_t  truncated;       // prompt was truncated to fit the context
    int32_t  split;           // 1 = split format
    int32_t  pad_;
    double   t_claim, t_prefill_start, t_first, t_send;
    double   t_pack;          // seconds spent exporting the state on the prefill rank
    uint64_t ref_hash;        // --verify: hash of the locally decoded reference (0 = none)
};

struct ReqRecord {
    int64_t task_id, rec_idx, kv_bytes;
    int32_t isl, max_out, n_gen, prefill_rank, decode_rank, stop, verify, truncated;
    int32_t n_steps, n_drafted, n_accepted, pad_;
    double  t_claim, t_prefill_start, t_first, t_send, t_kv_ready, t_done;
    double  t_pack, t_unpack;   // state export (prefill) / import (decode) time
};

enum Role : int32_t { ROLE_PREFILL = 0, ROLE_DECODE = 1 };

struct RankStat {
    int32_t rank, role, node, local_rank, gpu, threads;
    int64_t requests;       // prefilled (prefill rank) or decoded (decode rank)
    int64_t tokens_in;      // prompt tokens processed
    int64_t tokens_out;     // tokens generated (first token counted on prefill side)
    int64_t steps;          // llama_process calls in the timed region
    double  busy;           // seconds inside llama.cpp compute + sampling
    double  wall;           // seconds from epoch to rank finish
    double  sum_batch;      // decode: sum of active sequences over steps
    double  slot_wait;      // prefill: time waiting for decode capacity
    int64_t slots, kv_cells, peak_seqs, peak_cells;
    double  sum_cells;
    int64_t truncated;      // prefill: prompts truncated to fit
    int64_t eos_stops;      // decode: requests that ended on EOG
    int64_t drafted, accepted, seq_steps, verify_tokens;
    double  t_draft;        // decode: drafting + draft-model updates
    double  t_draft_prefill;// draft-model prompt processing
    int64_t claims;         // prefill: NXTVAL claim operations
    int64_t claimed;        // prefill: tasks obtained by those claims
    int32_t kv_mode;        // payload buffers: 0 host, 1 pinned host, 2 GPU memory
    int32_t pad2_;
    int64_t stage_bytes;    // staging buffer memory allocated (pools)
};

static uint64_t fnv1a(uint64_t h, llama_token t) {
    for (int i = 0; i < 4; ++i) { h ^= (uint64_t) ((t >> (8 * i)) & 0xff); h *= 0x100000001b3ULL; }
    return h;
}
static constexpr uint64_t FNV0 = 0xcbf29ce484222325ULL;

// ----------------------------------------------------------------------------
// llama.cpp helpers
// ----------------------------------------------------------------------------

// Batches use llama.cpp's extended batch API (common_batch + llama_process). A batch belongs to the context it
// was created for (it keeps that context's memory and vocabulary), so the draft model gets its own batch.
static bool decode_batch(llama_context * ctx, common_batch & b) {
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0;
}

static llama_context * make_context(llama_model * model, const Args & a, int n_seq, int64_t n_cells, int n_batch,
                                    int n_ubatch, int threads, bool cpu_only) {
    auto cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_cells;
    cp.n_seq_max       = (uint32_t) n_seq;
    // Unified KV buffer everywhere: the per-sequence state blob records the number of KV streams,
    // so prefill and decode contexts (different n_seq) must agree. With n_stream == 1,
    // llama_state_seq_set_data() places imported cells in any free cells.
    cp.kv_unified      = true;
    cp.n_batch         = (uint32_t) n_batch;
    cp.n_ubatch        = (uint32_t) n_ubatch;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = a.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k          = kv_ggml_type(a.kv_type);
    cp.type_v          = kv_ggml_type(a.kv_type);
    cp.no_perf         = true;
    if (cpu_only) {
        cp.offload_kqv = false;
        cp.op_offload  = false;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) fatal("llama_init_from_model failed (n_ctx=%u, n_seq=%d, n_batch=%d)", cp.n_ctx, n_seq, n_batch);
    return ctx;
}

static std::vector<ggml_backend_dev_t> gpu_devices() {
    std::vector<ggml_backend_dev_t> v;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) v.push_back(d);
    }
    return v;
}

// Testing hook: VIBEPROF_LLM_EMULATE_GPUS=1 keeps GPU role/device assignment (layout, capacities,
// routing, report) but runs those ranks on the CPU, so GPU topologies can be exercised anywhere.
static bool emulate_gpus() { const char * e = getenv("VIBEPROF_LLM_EMULATE_GPUS"); return e && atoi(e) != 0; }

// ----------------------------------------------------------------------------
// Model staging: copy GGUF files to node-local storage (e.g. NVMe) once per node, then load from there.
// The model is memory-mapped either way (one shared copy in the page cache per node, not one per rank);
// staging moves the reads off the shared file system and makes evicted pages cheap to read back.
// ----------------------------------------------------------------------------

// all files of a model: the file itself, or every part of a split GGUF (...-00001-of-00003.gguf)
static std::vector<std::string> model_files(const std::string & path) {
    static const std::regex split("^(.*)-([0-9]{5})-of-([0-9]{5})\\.gguf$");
    std::smatch m;
    if (!std::regex_match(path, m, split)) return {path};
    std::vector<std::string> out;
    const int n = std::stoi(m[3].str());
    for (int i = 1; i <= n; ++i) {
        char part[32];
        snprintf(part, sizeof(part), "-%05d-of-%s.gguf", i, m[3].str().c_str());
        out.push_back(m[1].str() + part);
    }
    return out;
}

// copy src to dst unless an identical copy (same size and mtime) is already there; returns "" or an error
static std::string stage_file(const std::string & src, const std::string & dst, double * bytes_copied) {
    struct stat ss, ds;
    if (stat(src.c_str(), &ss) != 0) return "cannot stat " + src + ": " + strerror(errno);
    if (stat(dst.c_str(), &ds) == 0 && ds.st_size == ss.st_size && ds.st_mtime == ss.st_mtime) return "";
    std::error_code ec;
    const auto sp = std::filesystem::space(std::filesystem::path(dst).parent_path(), ec);
    if (!ec && sp.available < (uintmax_t) ss.st_size)
        return "not enough space in " + std::filesystem::path(dst).parent_path().string() + " for " + src;
    const std::string tmp = dst + ".tmp." + std::to_string(getpid());
    const int in = open(src.c_str(), O_RDONLY);
    if (in < 0) return "cannot open " + src + ": " + strerror(errno);
    const int out = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) { close(in); return "cannot create " + tmp + ": " + strerror(errno); }
    std::vector<char> buf(64u << 20);
    std::string err;
    for (;;) {
        const ssize_t r = read(in, buf.data(), buf.size());
        if (r < 0) { err = "read error on " + src + ": " + strerror(errno); break; }
        if (r == 0) break;
        for (ssize_t off = 0; off < r;) {
            const ssize_t w = write(out, buf.data() + off, (size_t) (r - off));
            if (w < 0) { err = "write error on " + tmp + ": " + strerror(errno); break; }
            off += w;
        }
        if (!err.empty()) break;
        *bytes_copied += (double) r;
    }
    close(in);
    if (close(out) != 0 && err.empty()) err = "write error on " + tmp + ": " + strerror(errno);
    if (err.empty()) {
        struct timespec times[2] = {ss.st_atim, ss.st_mtim};   // keep the source mtime: marks the copy as current
        utimensat(AT_FDCWD, tmp.c_str(), times, 0);
        if (rename(tmp.c_str(), dst.c_str()) != 0) err = "cannot rename " + tmp + ": " + strerror(errno);
    }
    if (!err.empty()) unlink(tmp.c_str());
    return err;
}

// Node leaders copy the model's files into dir; returns the staged path of the (first) model file on every rank.
static std::string stage_model(const std::string & path, const std::string & dir, MPI_Comm node, int lrank) {
    const auto files = model_files(path);
    const std::string base = (std::filesystem::path(dir) / std::filesystem::path(path).filename()).string();
    if (std::filesystem::path(path).parent_path() == std::filesystem::path(dir)) return path;   // already local
    std::string err;
    double bytes = 0, t0 = now_s();
    if (lrank == 0) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) err = "cannot create " + dir + ": " + ec.message();
        for (size_t i = 0; err.empty() && i < files.size(); ++i)
            err = stage_file(files[i], (std::filesystem::path(dir) / std::filesystem::path(files[i]).filename()).string(), &bytes);
    }
    int ok = err.empty() ? 1 : 0;
    MPI_Bcast(&ok, 1, MPI_INT, 0, node);
    if (!ok) {
        if (lrank == 0) fatal("--model-stage: %s", err.c_str());
        MPI_Barrier(node);   // the leader aborts the job
        fatal("--model-stage failed on this node's leader");
    }
    if (g_rank == 0) {
        char what[64];
        if (bytes > 0) snprintf(what, sizeof(what), "%.1f MB copied", bytes / 1e6);
        else snprintf(what, sizeof(what), "already present");
        printf("model staged: %s -> %s (%s, %.1f s on node 0)\n", path.c_str(), dir.c_str(), what, now_s() - t0);
    }
    return base;
}

// Loads a model on one GPU (dev >= 0) or CPU only (dev < 0).
static llama_model * load_model(const std::string & path, int dev, int ngl) {
    auto mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    if (dev >= 0 && !emulate_gpus()) {
        auto g = gpu_devices();
        if (g.empty()) fatal("rank assigned GPU %d but no GPU backend device is visible", dev);
        devs = {g[dev % (int) g.size()], nullptr};
        mp.n_gpu_layers = ngl;
        mp.split_mode   = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu     = 0;
    } else {
        devs = {nullptr};
        mp.n_gpu_layers = 0;
    }
    mp.devices = devs.data();
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    if (!m) fatal("failed to load model %s", path.c_str());
    return m;
}

// ----------------------------------------------------------------------------
// KV transfer: staging buffer pools, GPU-aware MPI detection, split-state wrappers
// ----------------------------------------------------------------------------

// Pool of ggml backend buffers of one type (GPU memory, pinned host memory or plain CPU memory). Pinned
// and device allocations are expensive, so buffers are reused across requests; a new buffer is only
// allocated when no free one is large enough (sizes are rounded up by at most ~12.5% to help reuse).
class BufPool {
public:
    void init(ggml_backend_buffer_type_t buft) { buft_ = buft; }
    ggml_backend_buffer_t acquire(size_t size) {
        size = std::max<size_t>(size, 1);
        auto best = free_.end();
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            const size_t sz = ggml_backend_buffer_get_size(*it);
            if (sz >= size && (best == free_.end() || sz < ggml_backend_buffer_get_size(*best))) best = it;
        }
        if (best != free_.end()) { ggml_backend_buffer_t b = *best; free_.erase(best); return b; }
        size_t q = 64 * 1024;
        while (q * 8 < size) q *= 2;
        const size_t alloc = (size + q - 1) / q * q;
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft_, alloc);
        if (!b) fatal("failed to allocate a %.1f MiB %s staging buffer", alloc / 1048576.0, ggml_backend_buft_name(buft_));
        bytes_ += ggml_backend_buffer_get_size(b);
        return b;
    }
    void release(ggml_backend_buffer_t b) { if (b) free_.push_back(b); }
    void free_all() { for (auto * b : free_) ggml_backend_buffer_free(b); free_.clear(); }
    size_t bytes() const { return bytes_; }
private:
    ggml_backend_buffer_type_t buft_ = nullptr;
    std::vector<ggml_backend_buffer_t> free_;
    size_t bytes_ = 0;
};

static uint8_t * buf_ptr(ggml_backend_buffer_t b) { return (uint8_t *) ggml_backend_buffer_get_base(b); }

struct KvXfer {
    bool split = false;                            // split format: metadata (host) + tensor data (data_buft)
    bool pinned = false;                           // host staging uses pinned memory
    bool data_on_device = false;                   // split tensor data lives in GPU memory (GPU-aware MPI)
    ggml_backend_buffer_type_t host_buft = nullptr;
    ggml_backend_buffer_type_t data_buft = nullptr;
};

// Best-effort runtime check for GPU-aware MPI (Open MPI CUDA/ROCm, MPICH >= 4 incl. derivatives, Cray MPICH).
static bool detect_gpu_aware_mpi(std::string & how) {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    if (MPIX_Query_cuda_support()) { how = "Open MPI (CUDA)"; return true; }
#endif
#if defined(MPIX_ROCM_AWARE_SUPPORT) && MPIX_ROCM_AWARE_SUPPORT
    if (MPIX_Query_rocm_support()) { how = "Open MPI (ROCm)"; return true; }
#endif
#if defined(MPIX_GPU_SUPPORT_CUDA)
    {
        const std::pair<int, const char *> kinds[] = {{MPIX_GPU_SUPPORT_CUDA, "CUDA"}, {MPIX_GPU_SUPPORT_HIP, "HIP"}, {MPIX_GPU_SUPPORT_ZE, "Level Zero"}};
        for (const auto & [kind, name] : kinds) {
            int ok = 0;
            if (MPIX_GPU_query_support(kind, &ok) == MPI_SUCCESS && ok) { how = std::string("MPICH (") + name + ")"; return true; }
        }
    }
#endif
    const char * cray = getenv("MPICH_GPU_SUPPORT_ENABLED");
    if (cray && atoi(cray) == 1) { how = "MPICH_GPU_SUPPORT_ENABLED=1"; return true; }
    how = "not detected";
    return false;
}

#ifdef LLAMA_STATE_SEQ_SPLIT_API
static constexpr bool k_split_api = true;
static size_t kv_size_split(llama_context * c, llama_seq_id s, size_t * data) {
    return llama_state_seq_get_size_split(c, s, data, LLAMA_STATE_SEQ_FLAGS_NONE);
}
static size_t kv_get_split(llama_context * c, uint8_t * meta, size_t n, ggml_backend_buffer_t b, size_t off, size_t * data, llama_seq_id s) {
    return llama_state_seq_get_data_split(c, meta, n, b, off, data, s, LLAMA_STATE_SEQ_FLAGS_NONE);
}
static size_t kv_set_split(llama_context * c, const uint8_t * meta, size_t n, ggml_backend_buffer_t b, size_t off, llama_seq_id s) {
    return llama_state_seq_set_data_split(c, meta, n, b, off, s, LLAMA_STATE_SEQ_FLAGS_NONE);
}
#else
static constexpr bool k_split_api = false;
static size_t kv_size_split(llama_context *, llama_seq_id, size_t *) { fatal("built without the llama.cpp split-state API"); }
static size_t kv_get_split(llama_context *, uint8_t *, size_t, ggml_backend_buffer_t, size_t, size_t *, llama_seq_id) { fatal("built without the llama.cpp split-state API"); }
static size_t kv_set_split(llama_context *, const uint8_t *, size_t, ggml_backend_buffer_t, size_t, llama_seq_id) { fatal("built without the llama.cpp split-state API"); }
#endif

// ----------------------------------------------------------------------------
// PREFILL rank
// ----------------------------------------------------------------------------

struct Outgoing {
    ggml_backend_buffer_t    hbuf = nullptr, dbuf = nullptr;   // host part / split tensor data
    size_t                   hbytes = 0, dbytes = 0;
    KvHeader                 hdr{};
    std::vector<MPI_Request> reqs;
};

struct PrefillCtx {
    const Args * a;
    GlobalCounters * gc;
    MPI_Comm comm_ctl, comm_kv;
    std::vector<int> decode_ranks;
    std::deque<Outgoing> outq;
    std::vector<int64_t> loads;
    std::vector<int64_t> cap_seq, cap_cells;   // per global rank (0 for prefill ranks)
    int rr = 0;
    RankStat * st;
    BufPool hpool, dpool;
};

static void reap_sends(PrefillCtx & P, bool wait_all) {
    for (auto it = P.outq.begin(); it != P.outq.end();) {
        int done = 0;
        if (wait_all) { MPI_Waitall((int) it->reqs.size(), it->reqs.data(), MPI_STATUSES_IGNORE); done = 1; }
        else          { MPI_Testall((int) it->reqs.size(), it->reqs.data(), &done, MPI_STATUSES_IGNORE); }
        if (done) { P.hpool.release(it->hbuf); P.dpool.release(it->dbuf); it = P.outq.erase(it); } else ++it;
    }
}

// Reserve one sequence slot plus `need` KV cells on the decode rank with the lowest relative
// load, min over d of max(seqs/slots, cells/kv_cells): idle ranks fill first, and under
// pressure ranks with more slots or a larger cell budget (e.g. GPU decode ranks) receive
// proportionally more work. Both counters are updated atomically via RMA; if a concurrent
// reservation wins the race, ours is undone and we retry.
static int reserve_decode_slot(PrefillCtx & P, int64_t need) {
    const int nd = (int) P.decode_ranks.size();
    const int W  = (int) P.cap_seq.size();
    const double t0 = now_s();
    for (int spins = 0;; ++spins) {
        P.gc->load_snapshot(P.loads);
        int best = -1;
        double best_score = 1e300;
        for (int k = 0; k < nd; ++k) {
            const int d = P.decode_ranks[(P.rr + k) % nd];
            const int64_t seqs = P.loads[d], cells = P.loads[W + d];
            if (seqs >= P.cap_seq[d] || cells + need > P.cap_cells[d]) continue;
            const double score = std::max((double) seqs / P.cap_seq[d], (double) cells / P.cap_cells[d]);
            if (score < best_score) { best = d; best_score = score; }
        }
        if (best >= 0) {
            const int64_t old_s = P.gc->load_add(best, +1);
            const int64_t old_c = P.gc->cells_add(best, need);
            if (old_s < P.cap_seq[best] && old_c + need <= P.cap_cells[best]) {
                P.rr = (P.rr + 1) % nd;
                P.st->slot_wait += now_s() - t0;
                return best;
            }
            P.gc->load_add(best, -1);      // lost the race, undo both
            P.gc->cells_add(best, -need);
        }
        reap_sends(P, false);
        usleep(spins < 100 ? 20 : 200);
    }
}

struct Task {
    int64_t id, rec;
    int isl, max_out;
    bool truncated;
    double t_claim;
    std::vector<llama_token> toks;
};

static void run_prefill(const Args & a, llama_model * model, llama_model * model_dft, GlobalCounters & gc,
                        MPI_Comm comm_ctl, MPI_Comm comm_kv, const std::vector<int> & decode_ranks,
                        const std::vector<int64_t> & cap_seq, const std::vector<int64_t> & cap_cells,
                        const std::vector<Record> & dataset, int threads, bool on_gpu, const KvXfer & X, RankStat & st) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int B = a.prefill_batch;
    const int n_batch = std::max(a.n_batch, 32);
    llama_context * ctx = make_context(model, a, B, (int64_t) a.ctx * B, n_batch, std::min(a.n_ubatch, n_batch), threads, !on_gpu);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_context * ctx_dft = model_dft ? make_context(model_dft, a, B, (int64_t) a.ctx * B, n_batch, std::min(a.n_ubatch, n_batch), threads, !on_gpu) : nullptr;
    llama_memory_t mem_dft = ctx_dft ? llama_get_memory(ctx_dft) : nullptr;
    common_batch batch(ctx);
    common_batch batch_dft;
    if (ctx_dft) batch_dft = common_batch(ctx_dft);
    // the same tokens go to the target and (if any) the draft batch
    auto add_both = [&](llama_token tok, llama_pos pos, llama_seq_id seq, bool out) {
        const int32_t idx = batch.add(tok, pos, seq, out);
        if (ctx_dft) batch_dft.add(tok, pos, seq, out);
        return idx;
    };
    auto clear_both = [&]() { batch.clear(); if (ctx_dft) batch_dft.clear(); };

    PromptGen pg;
    pg.init(vocab, a);
    PromptBuilder pb;
    std::vector<int64_t> perm;
    if (!dataset.empty()) {
        pb.init(model, a);
        perm.resize(dataset.size());
        std::iota(perm.begin(), perm.end(), 0);
        if (a.shuffle) { std::mt19937_64 rng(a.seed); std::shuffle(perm.begin(), perm.end(), rng); }
    }

    // warmup (not timed)
    for (int w = 0; w < a.warmup; ++w) {
        auto toks = pg.make(-1 - w, std::min(64, a.ctx - 1));
        clear_both();
        for (int i = 0; i < (int) toks.size(); ++i) add_both(toks[i], i, 0, i + 1 == (int) toks.size());
        if (!decode_batch(ctx, batch)) fatal("warmup prefill failed");
        llama_memory_seq_rm(mem, 0, -1, -1);
        if (ctx_dft) { if (!decode_batch(ctx_dft, batch_dft)) fatal("warmup draft prefill failed"); llama_memory_seq_rm(mem_dft, 0, -1, -1); }
    }

    PrefillCtx P{&a, &gc, comm_ctl, comm_kv, decode_ranks, {}, {}, cap_seq, cap_cells, g_rank % (int) decode_ranks.size(), &st, {}, {}};
    P.hpool.init(X.host_buft);
    P.dpool.init(X.data_buft);

    auto make_task = [&](int64_t id, double t_claim) {
        Task t{id, -1, 0, 0, false, t_claim, {}};
        const LenSpec ls = sample_lengths(a, id);
        if (dataset.empty()) {
            t.isl = ls.isl; t.max_out = ls.osl;
            t.toks = pg.make(id, t.isl);
        } else {
            t.rec = perm[id % (int64_t) perm.size()];
            const Record & r = dataset[t.rec];
            t.toks = pb.build(r);
            t.max_out = (a.osl_from_data && r.osl_hint > 0) ? r.osl_hint : ls.osl;
            t.max_out = std::clamp(t.max_out, 1, a.ctx - 1);
            const int keep = a.ctx - t.max_out;
            if ((int) t.toks.size() > keep) {   // keep BOS (if any) + the most recent tokens
                const bool bos = llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL && t.toks[0] == llama_vocab_bos(vocab);
                std::vector<llama_token> tr;
                if (bos) tr.push_back(t.toks[0]);
                tr.insert(tr.end(), t.toks.end() - (keep - (int) tr.size()), t.toks.end());
                t.toks.swap(tr);
                t.truncated = true;
                st.truncated++;
            }
            t.isl = (int) t.toks.size();
        }
        return t;
    };

    MPI_Barrier(MPI_COMM_WORLD);          // epoch barrier (matches decode ranks)
    g_t0 = MPI_Wtime();

    std::deque<Task> carry;               // claimed but did not fit the token budget
    bool exhausted = false;
    const int n_prefill = std::max(1, g_world - (int) decode_ranks.size());
    const bool guided = a.claim == "guided" && a.n_requests > 0;

    // ---- the work loop: claim -> prefill -> hand off -> repeat ----
    for (;;) {
        std::vector<Task> tasks;
        int64_t ntok = 0;
        if (!carry.empty()) { tasks.push_back(std::move(carry.front())); carry.pop_front(); ntok = tasks.back().isl; }
        // Guided self-scheduling: cap this round's batch at ceil(remaining / (K * #prefill ranks)), so
        // batches stay at --prefill-batch while the pool is large and shrink toward 1 as it drains.
        // The read is a hint (other ranks may claim concurrently); correctness only relies on NXTVAL.
        int64_t cap = B;
        if (guided && B > 1 && !exhausted) {
            const int64_t remaining = a.n_requests - gc.peek();
            if (remaining > 0) cap = std::clamp<int64_t>((int64_t) std::ceil(remaining / (a.claim_factor * n_prefill)), 1, B);
        }
        while (!exhausted && (int64_t) tasks.size() < cap && carry.empty()) {
            if (a.prefill_tokens > 0 && ntok >= a.prefill_tokens) break;
            if (a.duration > 0 && now_s() >= a.duration) { exhausted = true; break; }
            const int64_t want = a.prefill_tokens > 0 ? 1 : cap - (int64_t) tasks.size();
            const int64_t first = gc.nxtval(want);                                 // NXTVAL
            if (a.n_requests > 0 && first >= a.n_requests) { exhausted = true; break; }
            st.claims++;
            const int64_t got = a.n_requests > 0 ? std::min<int64_t>(want, a.n_requests - first) : want;
            st.claimed += got;
            const double tc = now_s();
            for (int64_t k = 0; k < got; ++k) {
                Task t = make_task(first + k, tc);
                if (a.prefill_tokens > 0 && !tasks.empty() && ntok + t.isl > a.prefill_tokens) { carry.push_back(std::move(t)); break; }
                ntok += t.isl;
                tasks.push_back(std::move(t));
            }
        }
        if (tasks.empty()) break;
        const int nb = (int) tasks.size();
        logr("prefill batch: %d prompts, %" PRId64 " tokens (first task %" PRId64 ")", nb, ntok, tasks[0].id);

        std::vector<common_sampler *> smpl(nb);
        for (int s = 0; s < nb; ++s) smpl[s] = make_sampler(model, a, a.seed + (uint32_t) (tasks[s].id * 7919));

        // pack all prompts into as few llama_process calls as possible (target, then draft)
        std::vector<int32_t> first_tok(nb, -1);
        std::vector<double>  t_first(nb, 0.0);
        const double t_pf = now_s();
        clear_both();
        std::vector<std::pair<int, int>> pending_logits;  // (seq, batch index)
        auto flush = [&]() {
            if (batch.size() == 0) return;
            const double tb = now_s();
            if (!decode_batch(ctx, batch)) fatal("prefill llama_process failed");
            for (auto [s, idx] : pending_logits) {
                first_tok[s] = common_sampler_sample(smpl[s], ctx, idx);
                common_sampler_accept(smpl[s], first_tok[s], true);
                t_first[s] = now_s();
            }
            st.busy += now_s() - tb;
            if (ctx_dft) {
                const double td = now_s();
                if (!decode_batch(ctx_dft, batch_dft)) fatal("draft prefill llama_process failed");
                st.t_draft_prefill += now_s() - td;
                st.busy += now_s() - td;
            }
            st.steps++;
            pending_logits.clear();
            clear_both();
        };
        for (int s = 0; s < nb; ++s) {
            const auto & p = tasks[s].toks;
            for (int i = 0; i < (int) p.size(); ++i) {
                const bool last = (i + 1 == (int) p.size());
                const int32_t idx = add_both(p[i], i, s, last);
                if (last) pending_logits.push_back({s, idx});
                if (batch.size() == n_batch) flush();
            }
            st.tokens_in += (int64_t) p.size();
        }
        flush();

        // hand off each sequence (prompt tokens + KV caches) to a decode rank
        for (int s = 0; s < nb; ++s) {
            const Task & t = tasks[s];
            const bool done = t.max_out <= 1 || (!a.ignore_eos && llama_vocab_is_eog(vocab, first_tok[s]));

            uint64_t ref = 0;
            if (a.verify) {                  // colocated greedy reference without speculation
                ref = fnv1a(FNV0, first_tok[s]);
                llama_token last = first_tok[s];
                for (int k = 1; k < t.max_out && !(!a.ignore_eos && llama_vocab_is_eog(vocab, last)); ++k) {
                    batch.clear();
                    batch.add(last, t.isl + k - 1, s, true);
                    if (!decode_batch(ctx, batch)) fatal("verify decode failed");
                    last = common_sampler_sample(smpl[s], ctx, 0);
                    common_sampler_accept(smpl[s], last, true);
                    ref = fnv1a(ref, last);
                }
                batch.clear();
                // reference decoding appended to the KV cache; restore the prompt-only state
                llama_memory_seq_rm(mem, s, t.isl, -1);
            }

            // export: prompt tokens + target [+ draft] state into staging buffers
            Outgoing o;
            size_t got_t = 0, got_d = 0, dat_t = 0, dat_d = 0;
            const size_t ntb = done ? 0 : t.toks.size() * sizeof(llama_token);
            const double t_pack0 = now_s();
            if (!done) {
                if (!X.split) {
                    const size_t sz_t = llama_state_seq_get_size(ctx, s);
                    const size_t sz_d = ctx_dft ? llama_state_seq_get_size(ctx_dft, s) : 0;
                    o.hbuf = P.hpool.acquire(ntb + sz_t + sz_d);
                    uint8_t * h = buf_ptr(o.hbuf);
                    memcpy(h, t.toks.data(), ntb);
                    got_t = llama_state_seq_get_data(ctx, h + ntb, sz_t, s);
                    if (got_t == 0) fatal("llama_state_seq_get_data failed for task %" PRId64, t.id);
                    if (ctx_dft) {
                        got_d = llama_state_seq_get_data(ctx_dft, h + ntb + got_t, sz_d, s);
                        if (got_d == 0) fatal("draft llama_state_seq_get_data failed for task %" PRId64, t.id);
                    }
                } else {
                    size_t need_t = 0, need_d = 0;
                    const size_t mt = kv_size_split(ctx, s, &need_t);
                    const size_t md = ctx_dft ? kv_size_split(ctx_dft, s, &need_d) : 0;
                    o.hbuf = P.hpool.acquire(ntb + mt + md);
                    o.dbuf = P.dpool.acquire(need_t + need_d);
                    uint8_t * h = buf_ptr(o.hbuf);
                    memcpy(h, t.toks.data(), ntb);
                    got_t = kv_get_split(ctx, h + ntb, mt, o.dbuf, 0, &dat_t, s);
                    if (got_t == 0) fatal("split state export failed for task %" PRId64, t.id);
                    if (ctx_dft) {
                        got_d = kv_get_split(ctx_dft, h + ntb + got_t, md, o.dbuf, dat_t, &dat_d, s);
                        if (got_d == 0) fatal("draft split state export failed for task %" PRId64, t.id);
                    }
                }
                o.hbytes = ntb + got_t + got_d;
                o.dbytes = dat_t + dat_d;
            }
            const double t_pack = now_s() - t_pack0;
            llama_memory_seq_rm(mem, s, -1, -1);
            if (ctx_dft) llama_memory_seq_rm(mem_dft, s, -1, -1);

            const int dst = reserve_decode_slot(P, (int64_t) t.isl + t.max_out);
            KvHeader & H = o.hdr;
            H.task_id = t.id;           H.rec_idx = t.rec;
            H.tok_bytes = (int64_t) ntb; H.kv_tgt_bytes = (int64_t) got_t; H.kv_dft_bytes = (int64_t) got_d;
            H.dat_tgt_bytes = (int64_t) dat_t; H.dat_dft_bytes = (int64_t) dat_d;
            H.isl = t.isl;              H.max_out = t.max_out;   H.first_token = first_tok[s]; H.prefill_rank = g_rank;
            H.done = done ? 1 : 0;      H.truncated = t.truncated ? 1 : 0; H.split = (X.split && !done) ? 1 : 0;
            H.t_claim = t.t_claim;      H.t_prefill_start = t_pf; H.t_first = t_first[s]; H.t_send = now_s();
            H.t_pack = t_pack;          H.ref_hash = ref;
            P.outq.push_back(std::move(o));
            Outgoing & q = P.outq.back();
            MPI_Request r;
            MPI_Isend(&q.hdr, (int) sizeof(KvHeader), MPI_BYTE, dst, TAG_HDR, comm_ctl, &r);
            q.reqs.push_back(r);
            auto send_part = [&](ggml_backend_buffer_t b, size_t bytes, int tag) {
                for (int64_t off = 0; off < (int64_t) bytes; off += KV_CHUNK) {
                    const int n = (int) std::min<int64_t>(KV_CHUNK, (int64_t) bytes - off);
                    MPI_Isend(buf_ptr(b) + off, n, MPI_BYTE, dst, tag, comm_kv, &r);   // device pointer if data_on_device
                    q.reqs.push_back(r);
                }
            };
            if (q.hbytes) send_part(q.hbuf, q.hbytes, 0);
            if (q.dbytes) send_part(q.dbuf, q.dbytes, 1);
            st.requests++;
            st.tokens_out++;   // first token
            logr("task %" PRId64 " isl=%d max_out=%d -> decode rank %d (%.1f KiB host + %.1f KiB data)", t.id, t.isl, t.max_out, dst,
                 q.hbytes / 1024.0, q.dbytes / 1024.0);
        }
        for (auto * s : smpl) common_sampler_free(s);
        reap_sends(P, false);
    }

    reap_sends(P, true);
    for (int d : decode_ranks) {
        int z = 0;
        MPI_Send(&z, 1, MPI_INT, d, TAG_DONE, comm_ctl);
    }
    st.wall = now_s();
    st.stage_bytes = (int64_t) (P.hpool.bytes() + (X.split ? P.dpool.bytes() : 0));
    P.hpool.free_all();
    P.dpool.free_all();

    batch = common_batch();       // release the batches before their contexts
    batch_dft = common_batch();
    if (ctx_dft) llama_free(ctx_dft);
    llama_free(ctx);
}

// ----------------------------------------------------------------------------
// DECODE rank (continuous batching + optional speculative decoding)
// ----------------------------------------------------------------------------

struct Incoming {
    KvHeader hdr;
    int src;
    ggml_backend_buffer_t hbuf = nullptr, dbuf = nullptr;
    std::vector<MPI_Request> reqs;
};

struct Seq {
    KvHeader hdr;
    int slot = -1;
    int n_past = 0;                 // tokens in the KV cache (== hist.size())
    int n_gen = 0;                  // tokens generated (incl. the prefill-side first token)
    llama_token last = 0;           // newest token, not yet in the KV cache
    common_sampler * smpl = nullptr;
    double t_kv_ready = 0;
    double t_unpack = 0;
    uint64_t hash = FNV0;
    int stop = STOP_LENGTH;
    std::vector<llama_token> hist;  // tokens in the KV cache: prompt + generated except `last`
    llama_tokens draft;
    std::vector<std::vector<llama_token_data>> draft_q;
    std::vector<int> idxs;
    std::vector<llama_token> out;   // generated tokens (for --output)
    int n_steps = 0, n_drafted = 0, n_accepted = 0;
    std::mt19937 rng;
};

struct OutText { int64_t task_id, rec_idx; int32_t isl, n_gen, stop; std::string text; };

static void run_decode(const Args & a, llama_model * model, llama_model * model_dft, GlobalCounters & gc,
                       MPI_Comm comm_ctl, MPI_Comm comm_kv, int n_prefill_ranks, int threads, bool on_gpu, const KvXfer & X,
                       int M, int64_t n_cells, RankStat & st, std::vector<ReqRecord> & recs, std::vector<OutText> & outs) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_draft_max = a.use_spec ? a.spec_n_max : 0;
    const int n_batch  = std::max(32, M * (1 + n_draft_max));
    const int n_ubatch = std::min(n_batch, std::max(a.decode_ubatch, M));
    llama_context * ctx = make_context(model, a, M, n_cells, n_batch, n_ubatch, threads, !on_gpu);
    if ((int64_t) llama_n_ctx(ctx) < n_cells) fatal("decode context has %u cells, budget is %" PRId64, llama_n_ctx(ctx), n_cells);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_context * ctx_dft = model_dft ? make_context(model_dft, a, M, n_cells, n_batch, n_ubatch, threads, !on_gpu) : nullptr;
    llama_memory_t mem_dft = ctx_dft ? llama_get_memory(ctx_dft) : nullptr;

    // ---- speculative decoding (llama.cpp common_speculative; one instance, M sequences) ----
    common_speculative * spec = nullptr;
    std::vector<double> synth_probs;
    if (a.use_spec) {
        if (common_context_can_seq_rm(ctx) != COMMON_CONTEXT_SEQ_RM_TYPE_PART)
            fatal("speculative decoding needs partial KV removal, which this model's memory does not support");
        common_params_speculative sp;
        sp.types = a.spec_types;
        sp.draft.n_max = a.spec_n_max;
        sp.draft.n_min = a.spec_n_min;
        sp.draft.p_min = a.spec_p_min;
        sp.draft.probabilistic = a.spec_rejection;
        sp.draft.ctx_tgt = ctx;
        sp.draft.ctx_dft = ctx_dft;
        sp.draft.cache_type_k = sp.draft.cache_type_v = kv_ggml_type(a.kv_type);
        for (auto * m : {&sp.ngram_simple, &sp.ngram_map_k, &sp.ngram_map_k4v}) {
            if (a.spec_ngram_n > 0)    m->size_n = (uint16_t) a.spec_ngram_n;
            if (a.spec_ngram_m > 0)    m->size_m = (uint16_t) a.spec_ngram_m;
            if (a.spec_ngram_hits > 0) m->min_hits = (uint16_t) a.spec_ngram_hits;
        }
        sp.synth_len = a.spec_synth_len;
        try { spec = common_speculative_init(sp, (uint32_t) M); }
        catch (const std::exception & e) { fatal("speculative decoding init failed: %s", e.what()); }
        if (!spec) fatal("speculative decoding init returned no implementation for --spec-type %s", a.spec_type.c_str());
        synth_probs = common_speculative_get_synth_probs(spec);
    }

    // warmup (not timed)
    {
        common_batch wb(ctx);
        common_batch wb_dft;
        if (ctx_dft) wb_dft = common_batch(ctx_dft);
        const int nv = llama_vocab_n_tokens(vocab);
        for (int w = 0; w < a.warmup; ++w) {
            wb.clear();
            if (ctx_dft) wb_dft.clear();
            for (int s = 0; s < M; ++s) {
                wb.add((1 + s) % nv, 0, s, true);
                if (ctx_dft) wb_dft.add((1 + s) % nv, 0, s, true);
            }
            if (!decode_batch(ctx, wb)) fatal("warmup decode failed");
            if (ctx_dft && !decode_batch(ctx_dft, wb_dft)) fatal("warmup draft decode failed");
            for (int s = 0; s < M; ++s) {
                llama_memory_seq_rm(mem, s, -1, -1);
                if (mem_dft) llama_memory_seq_rm(mem_dft, s, -1, -1);
            }
        }
    }
    common_batch bt(ctx);
    common_batch dpb;                                     // draft prefill on the decode rank
    if (ctx_dft) dpb = common_batch(ctx_dft);

    MPI_Barrier(MPI_COMM_WORLD);          // epoch barrier
    g_t0 = MPI_Wtime();

    std::vector<int> free_slots;
    for (int s = M - 1; s >= 0; --s) free_slots.push_back(s);
    BufPool hpool, dpool;
    hpool.init(X.host_buft);
    dpool.init(X.data_buft);
    std::deque<Incoming> incoming;
    std::vector<Seq> active;
    int done_from = 0;
    int64_t cur_cells = 0;

    auto finish = [&](Seq & q, double t_kv_ready) {
        const double t = now_s();
        const int verify = q.hdr.ref_hash ? (q.hash == q.hdr.ref_hash ? 1 : 0) : -1;
        recs.push_back(ReqRecord{q.hdr.task_id, q.hdr.rec_idx, q.hdr.kv_tgt_bytes + q.hdr.kv_dft_bytes + q.hdr.dat_tgt_bytes + q.hdr.dat_dft_bytes,
                                 q.hdr.isl, q.hdr.max_out, q.n_gen, q.hdr.prefill_rank, g_rank, q.stop, verify, q.hdr.truncated,
                                 q.n_steps, q.n_drafted, q.n_accepted, 0,
                                 q.hdr.t_claim, q.hdr.t_prefill_start, q.hdr.t_first, q.hdr.t_send, t_kv_ready, t,
                                 q.hdr.t_pack, q.t_unpack});
        if (!a.output.empty()) {
            std::string text;
            for (llama_token tk : q.out) if (!llama_vocab_is_eog(vocab, tk)) text += common_token_to_piece(vocab, tk, false);
            outs.push_back(OutText{q.hdr.task_id, q.hdr.rec_idx, q.hdr.isl, q.n_gen, q.stop, std::move(text)});
        }
        if (q.slot >= 0) {
            llama_memory_seq_rm(mem, q.slot, -1, -1);
            if (mem_dft) llama_memory_seq_rm(mem_dft, q.slot, -1, -1);
            free_slots.push_back(q.slot);
        }
        if (q.smpl) common_sampler_free(q.smpl);
        q.smpl = nullptr;
        const int64_t need = (int64_t) q.hdr.isl + q.hdr.max_out;
        gc.load_add(g_rank, -1);          // release slot + cells (RMA)
        gc.cells_add(g_rank, -need);
        cur_cells -= need;
        st.requests++;
        if (q.stop == STOP_EOS) st.eos_stops++;
        st.drafted += q.n_drafted; st.accepted += q.n_accepted; st.seq_steps += q.n_steps;
        logr("task %" PRId64 " done (%d tokens, %s)", q.hdr.task_id, q.n_gen, q.stop == STOP_EOS ? "eos" : "length");
    };

    for (;;) {
        // 1) control messages (headers / done), in per-source order
        for (;;) {
            int flag = 0;
            MPI_Message msg;
            MPI_Status stt;
            MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_ctl, &flag, &msg, &stt);
            if (!flag) break;
            if (stt.MPI_TAG == TAG_HDR) {
                incoming.emplace_back();
                Incoming & in = incoming.back();
                in.src = stt.MPI_SOURCE;
                MPI_Mrecv(&in.hdr, (int) sizeof(KvHeader), MPI_BYTE, &msg, MPI_STATUS_IGNORE);
                const int64_t htotal = in.hdr.tok_bytes + in.hdr.kv_tgt_bytes + in.hdr.kv_dft_bytes;
                const int64_t dtotal = in.hdr.dat_tgt_bytes + in.hdr.dat_dft_bytes;
                if (in.hdr.split && !X.split) fatal("received a split-format payload but this rank uses the host format");
                auto recv_part = [&](ggml_backend_buffer_t b, int64_t bytes, int tag) {
                    for (int64_t off = 0; off < bytes; off += KV_CHUNK) {
                        const int n = (int) std::min<int64_t>(KV_CHUNK, bytes - off);
                        MPI_Request r;
                        MPI_Irecv(buf_ptr(b) + off, n, MPI_BYTE, in.src, tag, comm_kv, &r);   // device pointer if data_on_device
                        in.reqs.push_back(r);
                    }
                };
                if (htotal > 0) { in.hbuf = hpool.acquire(htotal); recv_part(in.hbuf, htotal, 0); }
                if (dtotal > 0) { in.dbuf = dpool.acquire(dtotal); recv_part(in.dbuf, dtotal, 1); }
            } else {
                int z;
                MPI_Mrecv(&z, 1, MPI_INT, &msg, MPI_STATUS_IGNORE);
                done_from++;
            }
        }

        // 2) admit sequences whose payload has fully arrived
        for (auto it = incoming.begin(); it != incoming.end();) {
            int ok = 0;
            MPI_Testall((int) it->reqs.size(), it->reqs.data(), &ok, MPI_STATUSES_IGNORE);
            if (!ok) { ++it; continue; }
            Seq q;
            q.hdr = it->hdr;
            q.n_gen = 1;
            q.last = q.hdr.first_token;
            q.hash = fnv1a(FNV0, q.hdr.first_token);
            if (!a.output.empty()) q.out.push_back(q.hdr.first_token);
            cur_cells += (int64_t) q.hdr.isl + q.hdr.max_out;
            st.peak_cells = std::max(st.peak_cells, cur_cells);
            const double t_ready = now_s();
            if (q.hdr.done) {                                // finished on the prefill rank
                q.stop = (q.hdr.max_out > 1) ? STOP_EOS : STOP_LENGTH;
                finish(q, t_ready);
                it = incoming.erase(it);
                continue;
            }
            if (free_slots.empty()) fatal("decode capacity exceeded (reservation protocol violated)");
            q.slot = free_slots.back();
            free_slots.pop_back();
            const uint8_t * p = buf_ptr(it->hbuf);
            const llama_token * tp = (const llama_token *) p;
            q.hist.assign(tp, tp + q.hdr.isl);
            q.hist.reserve(q.hdr.isl + q.hdr.max_out + 1);
            const double tu0 = now_s();
            const bool ok_t = q.hdr.split
                ? kv_set_split(ctx, p + q.hdr.tok_bytes, q.hdr.kv_tgt_bytes, it->dbuf, 0, q.slot) != 0
                : llama_state_seq_set_data(ctx, p + q.hdr.tok_bytes, q.hdr.kv_tgt_bytes, q.slot) != 0;
            if (!ok_t) fatal("state import failed for task %" PRId64 " (KV layout mismatch? check -fa/--kv-type)", q.hdr.task_id);
            if (ctx_dft && q.hdr.kv_dft_bytes > 0) {
                const uint8_t * pd = p + q.hdr.tok_bytes + q.hdr.kv_tgt_bytes;
                const bool ok_d = q.hdr.split
                    ? kv_set_split(ctx_dft, pd, q.hdr.kv_dft_bytes, it->dbuf, q.hdr.dat_tgt_bytes, q.slot) != 0
                    : llama_state_seq_set_data(ctx_dft, pd, q.hdr.kv_dft_bytes, q.slot) != 0;
                if (!ok_d) fatal("draft state import failed for task %" PRId64, q.hdr.task_id);
            }
            q.t_unpack = now_s() - tu0;
            hpool.release(it->hbuf); it->hbuf = nullptr;
            dpool.release(it->dbuf); it->dbuf = nullptr;
            if (ctx_dft) {
                if (q.hdr.kv_dft_bytes == 0) {                // --draft-prefill decode
                    const double td = now_s();
                    for (int i0 = 0; i0 < q.hdr.isl; i0 += n_batch) {
                        dpb.clear();
                        const int i1 = std::min(q.hdr.isl, i0 + n_batch);
                        for (int i = i0; i < i1; ++i) dpb.add(q.hist[i], i, q.slot, i + 1 == q.hdr.isl);
                        if (!decode_batch(ctx_dft, dpb)) fatal("draft prefill on decode rank failed");
                    }
                    st.t_draft_prefill += now_s() - td;
                    st.busy += now_s() - td;
                }
            }
            q.n_past = q.hdr.isl;
            q.smpl = make_sampler(model, a, a.seed + (uint32_t) (q.hdr.task_id * 7919) + 1);
            q.rng.seed((uint32_t) (a.seed ^ (uint32_t) (q.hdr.task_id * 2654435761u)));
            q.t_kv_ready = t_ready;
            if (spec) common_speculative_begin(spec, q.slot, q.hist);
            st.tokens_in += q.hdr.isl;
            st.peak_seqs = std::max<int64_t>(st.peak_seqs, (int64_t) (M - free_slots.size()));
            it = incoming.erase(it);
            active.push_back(std::move(q));
        }

        if (active.empty()) {
            if (done_from == n_prefill_ranks && incoming.empty()) break;
            usleep(50);
            continue;
        }

        const double tb = now_s();
        // 3a) drafting for all active sequences in one batched call
        if (spec) {
            for (auto & q : active) {
                q.draft.clear();
                q.draft_q.clear();
                const int nd = std::min(a.spec_n_max, q.hdr.max_out - q.n_gen - 1);
                if (nd <= 0) continue;
                auto & dp = common_speculative_get_draft_params(spec, q.slot);
                dp.drafting = true;
                dp.n_max    = nd;
                dp.pos0     = q.n_past;
                dp.id_last  = q.last;
                dp.prompt   = &q.hist;
                dp.result   = &q.draft;
                dp.result_q = a.spec_rejection ? &q.draft_q : nullptr;
                dp.temp     = a.temp;
                dp.seed     = a.seed + (uint32_t) q.hdr.task_id;
            }
            common_speculative_draft(spec);
            if (mem_dft) for (auto & q : active) llama_memory_seq_rm(mem_dft, q.slot, q.n_past, -1);
        }
        const double t_drafted = now_s();

        // 3b) one target step: [last, draft...] for every active sequence
        bt.clear();
        for (auto & q : active) {
            q.idxs.clear();
            q.idxs.push_back(bt.add(q.last, q.n_past, q.slot, true));
            for (size_t i = 0; i < q.draft.size(); ++i) q.idxs.push_back(bt.add(q.draft[i], q.n_past + 1 + (llama_pos) i, q.slot, true));
        }
        const int n_bt = bt.size();
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, bt.get()) != 0) fatal("decode llama_process failed (batch=%d)", n_bt);
        const double t_target = now_s();
        if (spec && !common_speculative_process(spec, bt)) fatal("speculative process failed");
        const double t_spec_proc = now_s();

        // 3c) verify drafts, commit accepted tokens, roll back the rest
        int64_t produced = 0;
        for (auto & q : active) {
            std::vector<llama_token> ids;
            if (!synth_probs.empty())                         ids = accept_synthetic(q.smpl, ctx, q.idxs, q.draft, synth_probs, q.rng);
            else if (a.spec_rejection && !q.draft_q.empty())  ids = common_sampler_sample_and_accept_n_rejection(q.smpl, ctx, q.idxs, q.draft, q.draft_q);
            else                                              ids = common_sampler_sample_and_accept_n(q.smpl, ctx, q.idxs, q.draft);
            const int n_acc = (int) ids.size() - 1;
            if (spec) common_speculative_accept(spec, q.slot, (uint16_t) n_acc);
            q.n_steps++;
            q.n_drafted  += (int) q.draft.size();
            q.n_accepted += n_acc;
            int keep = 0;
            for (llama_token tk : ids) {
                if (q.n_gen >= q.hdr.max_out) break;
                q.hist.push_back(q.last);
                q.last = tk;
                q.n_gen++;
                q.hash = fnv1a(q.hash, tk);
                if (!a.output.empty()) q.out.push_back(tk);
                keep++;
                if (!a.ignore_eos && llama_vocab_is_eog(vocab, tk)) { q.stop = STOP_EOS; break; }
            }
            q.n_past += keep;
            llama_memory_seq_rm(mem, q.slot, q.n_past, -1);
            if (mem_dft) llama_memory_seq_rm(mem_dft, q.slot, q.n_past, -1);
            produced += keep;
        }
        const double te = now_s();
        st.busy += te - tb;
        st.t_draft += (t_drafted - tb) + (t_spec_proc - t_target);
        st.steps++;
        st.sum_batch += (double) active.size();
        st.sum_cells += (double) cur_cells;
        st.tokens_out += produced;
        st.verify_tokens += n_bt;
        for (auto it = active.begin(); it != active.end();) {
            if (it->n_gen >= it->hdr.max_out || it->stop == STOP_EOS) { finish(*it, it->t_kv_ready); it = active.erase(it); } else ++it;
        }
    }
    st.wall = now_s();
    st.stage_bytes = (int64_t) (hpool.bytes() + (X.split ? dpool.bytes() : 0));
    hpool.free_all();
    dpool.free_all();
    if (spec && g_verbose) common_speculative_print_stats(spec);
    common_speculative_free(spec);
    bt = common_batch();          // release the batches before their contexts
    dpb = common_batch();
    if (ctx_dft) llama_free(ctx_dft);
    llama_free(ctx);
}

// ----------------------------------------------------------------------------
// Statistics / report
// ----------------------------------------------------------------------------

struct Summary { double min, max, avg, std, p50, p90, p99; size_t n; };

static Summary summarize(std::vector<double> v) {
    Summary s{0, 0, 0, 0, 0, 0, 0, v.size()};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    s.avg = sum / v.size();
    double var = 0; for (double x : v) var += (x - s.avg) * (x - s.avg);
    s.std = v.size() > 1 ? std::sqrt(var / (v.size() - 1)) : 0.0;
    auto pct = [&](double p) {
        const double idx = p / 100.0 * (v.size() - 1);
        const size_t lo = (size_t) std::floor(idx), hi = (size_t) std::ceil(idx);
        return v[lo] + (v[hi] - v[lo]) * (idx - lo);
    };
    s.min = v.front(); s.max = v.back();
    s.p50 = pct(50); s.p90 = pct(90); s.p99 = pct(99);
    return s;
}

static void print_row(const char * name, const char * unit, const std::vector<double> & v) {
    Summary s = summarize(v);
    printf("  %-30s %-7s %6zu %11.3f %11.3f %11.3f %11.3f %11.3f %11.3f %11.3f\n",
           name, unit, s.n, s.min, s.max, s.avg, s.std, s.p50, s.p90, s.p99);
}

static void print_header(const char * title) {
    printf("\n%s\n", title);
    printf("  %-30s %-7s %6s %11s %11s %11s %11s %11s %11s %11s\n",
           "metric", "unit", "n", "min", "max", "avg", "stddev", "p50", "p90", "p99");
    printf("  %s\n", std::string(132, '-').c_str());
}

static void report(const Args & a, const std::vector<ReqRecord> & R, const std::vector<RankStat> & S, int64_t n_records) {
    std::vector<double> ttft, tpot, e2e, pf, wait, xfer, kvmb, isl, osl, dec_tps, acc_rate, tok_step, pack, unpack;
    double makespan = 0;
    int64_t sum_in = 0, sum_out = 0, n_eos = 0, n_trunc = 0;
    for (const auto & r : R) {
        ttft.push_back((r.t_first - r.t_claim) * 1e3);
        if (r.n_gen > 1) tpot.push_back((r.t_done - r.t_first) / (r.n_gen - 1) * 1e3);
        e2e.push_back((r.t_done - r.t_claim) * 1e3);
        pf.push_back((r.t_first - r.t_prefill_start) * 1e3);
        wait.push_back((r.t_send - r.t_first) * 1e3);
        xfer.push_back((r.t_kv_ready - r.t_send) * 1e3);
        if (r.kv_bytes > 0) { pack.push_back(r.t_pack * 1e3); unpack.push_back(r.t_unpack * 1e3); }
        kvmb.push_back(r.kv_bytes / (1024.0 * 1024.0));
        isl.push_back(r.isl);
        osl.push_back(r.n_gen);
        if (r.n_gen > 1 && r.t_done > r.t_kv_ready) dec_tps.push_back((r.n_gen - 1) / (r.t_done - r.t_kv_ready));
        if (r.n_drafted > 0) acc_rate.push_back(100.0 * r.n_accepted / r.n_drafted);
        if (r.n_steps > 0) tok_step.push_back((double) (r.n_gen - 1) / r.n_steps);
        makespan = std::max(makespan, r.t_done);
        sum_in += r.isl; sum_out += r.n_gen;
        n_eos += r.stop == STOP_EOS; n_trunc += r.truncated;
    }

    int npf = 0, ndc = 0, ndg = 0;
    for (const auto & s : S) { if (s.role == ROLE_PREFILL) npf++; else { ndc++; ndg += s.gpu >= 0; } }

    printf("\n==================================================================================================================\n");
    printf(" llama.cpp MPI disaggregated prefill/decode benchmark\n");
    printf("==================================================================================================================\n");
    printf(" model            : %s\n", a.model.c_str());
    printf(" ranks            : %d total = %d prefill + %d decode (%d on GPU, %d on CPU)\n", (int) S.size(), npf, ndc, ndg, ndc - ndg);
    if (a.prompts.empty())
        printf(" workload         : synthetic, isl %d (std %.1f), osl %d (std %.1f), prompt-mode %s\n",
               a.isl, a.isl_std, a.osl, a.osl_std, a.prompt_mode.c_str());
    else
        printf(" workload         : %s (%" PRId64 " records%s), chat %s, osl %s, conv-turns %s\n", a.prompts.c_str(), n_records,
               a.shuffle ? ", shuffled" : "", a.chat.c_str(), a.osl_from_data ? "from data" : std::to_string(a.osl).c_str(), a.conv_turns.c_str());
    printf(" generation       : EOS %s, temp %.2f, top-k %d, top-p %.2f\n", a.ignore_eos ? "ignored" : "stops", a.temp, a.top_k, a.top_p);
    printf(" llama.cpp        : ctx/seq %d, prefill-batch %d (%s claims), prefill-tokens %d, decode-batch cpu %s / gpu %s, fa %s, kv %s\n",
           a.ctx, a.prefill_batch, a.claim == "guided" ? ("guided, K=" + std::to_string(a.claim_factor).substr(0, 4)).c_str() : "fixed", a.prefill_tokens, a.decode_batch.c_str(), a.decode_batch_gpu.c_str(), a.flash_attn ? "on" : "off", a.kv_type.c_str());
    printf(" KV transport     : %s\n", a.kv_desc.c_str());
    if (a.use_spec)
        printf(" speculative      : %s, draft-max %d%s%s%s\n", common_speculative_type_name_str(a.spec_types).c_str(), a.spec_n_max,
               a.use_draft ? (", draft model " + a.draft_model + ", draft prefill on " + a.draft_prefill).c_str() : "",
               a.spec_rejection ? ", rejection sampling" : "",
               a.spec_synth_len != -1.0 ? (", SYNTHETIC acceptance len " + std::to_string(a.spec_synth_len)).c_str() : "");
    printf("\n Rank layout:\n");
    printf("  %5s %-8s %5s %6s %5s %6s %8s %6s %9s %9s %9s %10s %10s %10s %9s %9s %7s %8s\n",
           "rank", "role", "node", "local", "gpu", "kvbuf", "threads", "slots", "kv_cells", "peak_seq", "requests",
           "tok_in", "tok_out", "steps", "busy[s]", "util%", "acc%", "tok/step");
    for (const auto & s : S) {
        const bool dc = s.role == ROLE_DECODE;
        char acc[16] = "-", tps[16] = "-";
        if (dc && s.drafted > 0)   snprintf(acc, sizeof(acc), "%.1f", 100.0 * s.accepted / s.drafted);
        if (dc && s.seq_steps > 0) snprintf(tps, sizeof(tps), "%.2f", (double) (s.accepted + s.seq_steps) / s.seq_steps);
        printf("  %5d %-8s %5d %6d %5s %6s %8d %6s %9s %9s %9" PRId64 " %10" PRId64 " %10" PRId64 " %10" PRId64 " %9.3f %8.1f%% %7s %8s\n",
               s.rank, dc ? "decode" : "prefill", s.node, s.local_rank,
               s.gpu >= 0 ? std::to_string(s.gpu).c_str() : "cpu", s.kv_mode == 2 ? "gpu" : (s.kv_mode == 1 ? "pinned" : "host"), s.threads,
               dc ? std::to_string(s.slots).c_str() : "-", dc ? std::to_string(s.kv_cells).c_str() : "-",
               dc ? std::to_string(s.peak_seqs).c_str() : "-", s.requests, s.tokens_in, s.tokens_out,
               s.steps, s.busy, s.wall > 0 ? 100.0 * s.busy / s.wall : 0.0, acc, tps);
    }

    print_header(" Per-request latency (across all requests):");
    print_row("TTFT (claim->1st token)", "ms", ttft);
    print_row("TPOT (1st->last token)", "ms", tpot);
    print_row("E2E latency", "ms", e2e);
    print_row("prefill compute (batch)", "ms", pf);
    print_row("decode-slot wait + KV pack", "ms", wait);
    print_row("KV export (pack, prefill)", "ms", pack);
    print_row("KV transfer (send->ready)", "ms", xfer);
    print_row("KV import (unpack, decode)", "ms", unpack);
    print_row("KV state size", "MiB", kvmb);
    print_row("input length", "tok", isl);
    print_row("output length", "tok", osl);
    print_row("per-request decode rate", "tok/s", dec_tps);
    if (a.use_spec) {
        print_row("draft acceptance rate", "%", acc_rate);
        print_row("tokens per decode step", "tok", tok_step);
    }

    std::vector<double> pf_req, pf_tps, pf_util, pf_wait, dc_req, dc_tps, dc_util, dc_occ, dc_step, all_req,
                        dc_slot_util, dc_cell_util, dc_cell_peak, dc_req_per_slot, dc_draft, dc_acc, dc_tpstep,
                        pf_claim, pf_end, dc_end;
    for (const auto & s : S) {
        all_req.push_back((double) s.requests);
        if (s.role == ROLE_PREFILL) {
            pf_req.push_back((double) s.requests);
            pf_tps.push_back(s.wall > 0 ? s.tokens_in / s.wall : 0);
            pf_util.push_back(s.wall > 0 ? 100.0 * s.busy / s.wall : 0);
            pf_wait.push_back(s.slot_wait * 1e3);
            if (s.claims > 0) pf_claim.push_back((double) s.claimed / s.claims);
            pf_end.push_back(s.wall);
        } else {
            dc_req.push_back((double) s.requests);
            dc_tps.push_back(s.wall > 0 ? s.tokens_out / s.wall : 0);
            dc_util.push_back(s.wall > 0 ? 100.0 * s.busy / s.wall : 0);
            dc_occ.push_back(s.steps > 0 ? s.sum_batch / s.steps : 0);
            dc_slot_util.push_back(s.steps > 0 && s.slots > 0 ? 100.0 * s.sum_batch / s.steps / s.slots : 0);
            dc_cell_util.push_back(s.steps > 0 && s.kv_cells > 0 ? 100.0 * s.sum_cells / s.steps / s.kv_cells : 0);
            dc_cell_peak.push_back(s.kv_cells > 0 ? 100.0 * s.peak_cells / s.kv_cells : 0);
            dc_req_per_slot.push_back(s.slots > 0 ? (double) s.requests / s.slots : 0);
            dc_step.push_back(s.steps > 0 ? s.busy / s.steps * 1e3 : 0);
            dc_end.push_back(s.wall);
            dc_draft.push_back(s.busy > 0 ? 100.0 * s.t_draft / s.busy : 0);
            if (s.drafted > 0) dc_acc.push_back(100.0 * s.accepted / s.drafted);
            if (s.seq_steps > 0) dc_tpstep.push_back((double) (s.accepted + s.seq_steps) / s.seq_steps);
        }
    }
    print_header(" Per-process statistics (across ranks):");
    print_row("requests handled / rank", "req", all_req);
    print_row("requests / prefill rank", "req", pf_req);
    print_row("requests / decode rank", "req", dc_req);
    print_row("prefill throughput / rank", "tok/s", pf_tps);
    print_row("prefill utilization", "%", pf_util);
    print_row("prefill decode-slot wait", "ms", pf_wait);
    print_row("tasks per NXTVAL claim", "task", pf_claim);
    print_row("prefill rank finish time", "s", pf_end);
    print_row("decode throughput / rank", "tok/s", dc_tps);
    print_row("decode utilization", "%", dc_util);
    print_row("decode batch occupancy", "seq", dc_occ);
    print_row("decode slot utilization", "%", dc_slot_util);
    print_row("decode KV cell utilization", "%", dc_cell_util);
    print_row("decode KV cell peak", "%", dc_cell_peak);
    print_row("requests / decode slot", "req", dc_req_per_slot);
    print_row("decode step time", "ms", dc_step);
    print_row("decode rank finish time", "s", dc_end);
    if (a.use_spec) {
        print_row("draft acceptance / rank", "%", dc_acc);
        print_row("tokens / seq-step / rank", "tok", dc_tpstep);
        print_row("drafting share of decode", "%", dc_draft);
    }

    printf("\n Aggregate:\n");
    printf("  requests completed      : %zu (%" PRId64 " stopped at EOS, %" PRId64 " at length limit, %" PRId64 " prompts truncated)\n",
           R.size(), n_eos, (int64_t) R.size() - n_eos, n_trunc);
    printf("  makespan                : %.3f s\n", makespan);
    {
        const Summary pe = summarize(pf_end), de = summarize(dc_end);
        printf("  finish spread (tail)    : prefill ranks %.3f s, decode ranks %.3f s (last - first rank to finish)\n",
               pe.max - pe.min, de.max - de.min);
    }
    printf("  request throughput      : %.3f req/s\n", makespan > 0 ? R.size() / makespan : 0.0);
    printf("  input  token throughput : %.1f tok/s\n", makespan > 0 ? sum_in / makespan : 0.0);
    printf("  output token throughput : %.1f tok/s\n", makespan > 0 ? sum_out / makespan : 0.0);
    printf("  total  token throughput : %.1f tok/s\n", makespan > 0 ? (sum_in + sum_out) / makespan : 0.0);
    if (a.use_spec) {
        int64_t dr = 0, ac = 0, ss = 0;
        for (const auto & s : S) { dr += s.drafted; ac += s.accepted; ss += s.seq_steps; }
        printf("  speculative             : %" PRId64 " drafted, %" PRId64 " accepted (%.1f%%), %.2f tokens per sequence-step\n",
               dr, ac, dr > 0 ? 100.0 * ac / dr : 0.0, ss > 0 ? (double) (ac + ss) / ss : 0.0);
    }
    int n_chk = 0, n_ok = 0;
    for (const auto & r : R) if (r.verify >= 0) { n_chk++; n_ok += r.verify; }
    if (n_chk > 0) printf("  verify (disagg == local) : %d / %d sequences token-identical%s\n", n_ok, n_chk, n_ok == n_chk ? "  [PASS]" : "  [MISMATCH]");
    printf("==================================================================================================================\n");
    fflush(stdout);

    if (!a.csv.empty()) {
        FILE * f = fopen(a.csv.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", a.csv.c_str()); return; }
        fprintf(f, "task_id,record,prefill_rank,decode_rank,isl,max_out,n_gen,stop,truncated,kv_bytes,steps,drafted,accepted,"
                   "t_claim,t_prefill_start,t_first,t_send,t_kv_ready,t_done,ttft_ms,tpot_ms,e2e_ms\n");
        for (const auto & r : R) {
            fprintf(f, "%" PRId64 ",%" PRId64 ",%d,%d,%d,%d,%d,%s,%d,%" PRId64 ",%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                    r.task_id, r.rec_idx, r.prefill_rank, r.decode_rank, r.isl, r.max_out, r.n_gen, r.stop == STOP_EOS ? "eos" : "length",
                    r.truncated, r.kv_bytes, r.n_steps, r.n_drafted, r.n_accepted,
                    r.t_claim, r.t_prefill_start, r.t_first, r.t_send, r.t_kv_ready, r.t_done,
                    (r.t_first - r.t_claim) * 1e3, r.n_gen > 1 ? (r.t_done - r.t_first) / (r.n_gen - 1) * 1e3 : 0.0,
                    (r.t_done - r.t_claim) * 1e3);
        }
        fclose(f);
        printf(" per-request records written to %s\n", a.csv.c_str());
    }
}

// gather generated texts to rank 0 and write them as JSONL, ordered by task id
static void write_outputs(const Args & a, const std::vector<OutText> & outs) {
    std::vector<char> b;
    for (const auto & o : outs) {
        b.insert(b.end(), (const char *) &o.task_id, (const char *) &o.task_id + 8);
        b.insert(b.end(), (const char *) &o.rec_idx, (const char *) &o.rec_idx + 8);
        put_i32(b, o.isl); put_i32(b, o.n_gen); put_i32(b, o.stop); put_str(b, o.text);
    }
    int n = (int) b.size();
    std::vector<int> counts(g_world), displs(g_world);
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> all;
    if (g_rank == 0) {
        int tot = 0;
        for (int r = 0; r < g_world; ++r) { displs[r] = tot; tot += counts[r]; }
        all.resize(tot);
    }
    MPI_Gatherv(b.data(), n, MPI_CHAR, all.data(), counts.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
    if (g_rank != 0) return;
    std::vector<OutText> R;
    for (size_t p = 0; p < all.size();) {
        OutText o;
        memcpy(&o.task_id, all.data() + p, 8); p += 8;
        memcpy(&o.rec_idx, all.data() + p, 8); p += 8;
        auto i32 = [&]() { int32_t v; memcpy(&v, all.data() + p, 4); p += 4; return v; };
        o.isl = i32(); o.n_gen = i32(); o.stop = i32();
        const int32_t len = i32();
        o.text.assign(all.data() + p, len); p += len;
        R.push_back(std::move(o));
    }
    std::sort(R.begin(), R.end(), [](const OutText & x, const OutText & y) { return x.task_id < y.task_id; });
    FILE * f = fopen(a.output.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", a.output.c_str()); return; }
    for (const auto & o : R) {
        json j = {{"task_id", o.task_id}, {"record", o.rec_idx}, {"prompt_tokens", o.isl}, {"output_tokens", o.n_gen},
                  {"stop", o.stop == STOP_EOS ? "eos" : "length"}, {"text", o.text}};
        fprintf(f, "%s\n", j.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }
    fclose(f);
    printf(" generated text for %zu requests written to %s\n", R.size(), a.output.c_str());
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

int main(int argc, char ** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world);
    g_t0 = MPI_Wtime();

    Args a = parse_args(argc, argv);
    g_verbose = a.verbose;
    common_log_set_verbosity_thold(a.verbose ? LOG_LEVEL_DEBUG : LOG_LEVEL_WARN);
    if (g_world < 2) fatal("need at least 2 MPI processes (>=1 prefill + >=1 decode)");

    // ---- node-local topology ----
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
    int lrank, lsize;
    MPI_Comm_rank(node, &lrank);
    MPI_Comm_size(node, &lsize);
    int node_id = g_rank;               // global rank of the node leader identifies the node
    MPI_Bcast(&node_id, 1, MPI_INT, 0, node);

    // ---- optional: copy the model(s) to node-local storage once per node, then load from there ----
    if (!a.model_stage.empty()) {
        a.model = stage_model(a.model, a.model_stage, node, lrank);
        if (!a.draft_model.empty()) a.draft_model = stage_model(a.draft_model, a.model_stage, node, lrank);
    }

    // ---- GPU detection on the node leader only (CPU decode ranks may hide GPUs before backend init) ----
    bool backend_up = false;
    int ngpu = a.gpus_per_node;
    if (ngpu < 0) {
        if (lrank == 0) {
            llama_log_set(llama_log_cb, nullptr);
            llama_backend_init();
            ggml_backend_load_all();
            backend_up = true;
            ngpu = (int) gpu_devices().size();
        }
        MPI_Bcast(&ngpu, 1, MPI_INT, 0, node);
    }

    // ---- role + device assignment (per node) ----
    //   local ranks [0, Pp)        prefill   (GPU i for i < Pg, CPU beyond)
    //   local ranks [Pp, Pp + Dg)  decode on GPU (Pg + j) % ngpu
    //   remaining local ranks      decode on CPU
    if (emulate_gpus() && a.gpus_per_node < 0) fatal("VIBEPROF_LLM_EMULATE_GPUS needs --gpus-per-node");
    const int Dg = std::max(0, a.decode_gpus);
    if (Dg > 0 && ngpu == 0) fatal("--decode-gpus %d requested but no GPUs were detected (see --gpus-per-node)", Dg);
    const int Pg = a.prefill_gpus >= 0 ? std::min(a.prefill_gpus, ngpu) : (ngpu == 0 ? 0 : (Dg < ngpu ? ngpu - Dg : ngpu));
    int Pp = a.prefill_per_node >= 0 ? a.prefill_per_node : (Pg > 0 ? Pg : lsize / 2);
    if (a.prefill_per_node < 0 && lsize > 1) Pp = std::max(0, std::min(Pp, lsize - std::max(Dg, 1)));
    if (Pp + Dg > lsize) fatal("node has %d ranks but needs %d prefill + %d GPU-decode ranks", lsize, Pp, Dg);
    int role = lrank < Pp ? ROLE_PREFILL : ROLE_DECODE;
    int dev  = -1;
    if (lrank < Pp)            dev = lrank < Pg ? lrank : -1;
    else if (lrank < Pp + Dg)  dev = (Pg + (lrank - Pp)) % ngpu;

    std::vector<int> rd(2 * g_world), mine = {role, dev};
    MPI_Allgather(mine.data(), 2, MPI_INT, rd.data(), 2, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> roles(g_world), devs(g_world);
    for (int r = 0; r < g_world; ++r) { roles[r] = rd[2 * r]; devs[r] = rd[2 * r + 1]; }
    int n_pf = (int) std::count(roles.begin(), roles.end(), (int) ROLE_PREFILL);
    if (n_pf == 0 || n_pf == g_world) {  // degenerate layouts (e.g. 1 rank per node): alternate globally
        if (Dg > 0) fatal("cannot place prefill and GPU-decode ranks with %d ranks per node", lsize);
        for (int r = 0; r < g_world; ++r) roles[r] = (r % 2 == 0) ? ROLE_PREFILL : ROLE_DECODE;
        role = roles[g_rank];
        dev = (role == ROLE_PREFILL && ngpu > 0 && lrank < ngpu) ? lrank : -1;
        MPI_Allgather(&dev, 1, MPI_INT, devs.data(), 1, MPI_INT, MPI_COMM_WORLD);
    }
    std::vector<int> decode_ranks, cpu_dec, gpu_dec;
    for (int r = 0; r < g_world; ++r) if (roles[r] == ROLE_DECODE) {
        decode_ranks.push_back(r);
        (devs[r] >= 0 ? gpu_dec : cpu_dec).push_back(r);
    }
    n_pf = g_world - (int) decode_ranks.size();

    if (role == ROLE_DECODE && dev < 0 && a.hide_gpus_on_decode && !backend_up) {
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
        setenv("HIP_VISIBLE_DEVICES", "-1", 1);
    }
    if (!backend_up) {
        llama_log_set(llama_log_cb, nullptr);
        llama_backend_init();
        ggml_backend_load_all();
    }

    // ---- threads: GPU-driving ranks get 2, CPU decode ranks share the remaining cores ----
    const int hw = (int) std::max(1u, std::thread::hardware_concurrency());
    int n_gpu_ranks_node = 0, n_cpu_dec_node = 0, n_cpu_pf_node = 0;
    {
        int my[3] = {dev >= 0 ? 1 : 0, (role == ROLE_DECODE && dev < 0) ? 1 : 0, (role == ROLE_PREFILL && dev < 0) ? 1 : 0}, sum[3];
        MPI_Allreduce(my, sum, 3, MPI_INT, MPI_SUM, node);
        n_gpu_ranks_node = sum[0]; n_cpu_dec_node = sum[1]; n_cpu_pf_node = sum[2];
    }
    const int cpu_left = std::max(1, hw - 2 * n_gpu_ranks_node);
    const int per_cpu_rank = std::max(1, cpu_left / std::max(1, n_cpu_dec_node + n_cpu_pf_node));
    int threads;
    if (role == ROLE_PREFILL) threads = a.threads_prefill > 0 ? a.threads_prefill : (dev >= 0 ? 2 : per_cpu_rank);
    else                      threads = a.threads_decode  > 0 ? a.threads_decode  : (dev >= 0 ? 2 : per_cpu_rank);

    // ---- load models (each rank, in-process; mmap shares pages between CPU ranks on a node) ----
    llama_model * model = load_model(a.model, dev, a.n_gpu_layers);
    const bool need_dft = a.use_draft && (role == ROLE_DECODE || a.draft_prefill == "prefill");
    llama_model * model_dft = need_dft ? load_model(a.draft_model, dev, a.n_gpu_layers) : nullptr;
    if (model_dft && !common_speculative_are_compatible(model, model_dft)) fatal("draft model vocabulary is not compatible with the target");

    if (g_rank == 0) {
        printf("MPI: %d ranks, thread level %s; GPUs/node: %d; prefill ranks: %d, decode ranks: %d (%zu GPU, %zu CPU)\n", g_world,
               provided == MPI_THREAD_MULTIPLE ? "MULTIPLE" : "below MULTIPLE", ngpu, n_pf, (int) decode_ranks.size(), gpu_dec.size(), cpu_dec.size());
        if (Dg > 0 && Pg + Dg > ngpu) printf("note: %d prefill + %d decode GPU ranks share %d GPUs per node (consider CUDA MPS)\n", Pg, Dg, ngpu);
        fflush(stdout);
    }

    // ---- real prompts: parsed once on the first prefill rank, broadcast to prefill ranks ----
    MPI_Comm comm_pf;
    MPI_Comm_split(MPI_COMM_WORLD, role == ROLE_PREFILL ? 0 : 1, g_rank, &comm_pf);
    std::vector<Record> dataset;
    int64_t n_records = 0;
    if (!a.prompts.empty()) {
        if (role == ROLE_PREFILL) {
            int pr;
            MPI_Comm_rank(comm_pf, &pr);
            std::vector<char> buf;
            if (pr == 0) buf = serialize(load_dataset(a, llama_model_get_vocab(model)));
            bcast_bytes(buf, 0, comm_pf);
            dataset = deserialize(buf);
            n_records = (int64_t) dataset.size();
        }
        MPI_Allreduce(MPI_IN_PLACE, &n_records, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
        if (!a.n_set) a.n_requests = n_records;
    }

    // ---- KV transport: host blob, or split (metadata on host + tensor data in GPU memory via GPU-aware MPI) ----
    KvXfer X;
    {
        std::string how;
        const bool detected  = detect_gpu_aware_mpi(how);
        const bool real_gpu  = dev >= 0 && !emulate_gpus();
        const bool aware     = a.gpu_aware == "yes" || (a.gpu_aware == "auto" && detected);
        const int  capable   = (real_gpu && aware) ? 1 : 0;
        int any_capable = 0, n_capable = 0;
        MPI_Allreduce(&capable, &any_capable, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&capable, &n_capable,   1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (a.kv_transport == "device" && !k_split_api)
            fatal("--kv-transport device needs llama.cpp with the split-state API: apply patches/llama.cpp-state-seq-split.patch and rebuild");
        X.split = a.kv_transport == "device" || (a.kv_transport == "auto" && k_split_api && any_capable);
        ggml_backend_dev_t gdev = nullptr;
        if (real_gpu) { auto g = gpu_devices(); if (!g.empty()) gdev = g[dev % (int) g.size()]; }
        ggml_backend_buffer_type_t pinned = (gdev && a.kv_staging == "pinned") ? ggml_backend_dev_host_buffer_type(gdev) : nullptr;
        X.pinned         = pinned != nullptr;
        X.host_buft      = pinned ? pinned : ggml_backend_cpu_buffer_type();
        X.data_on_device = X.split && capable && gdev;
        X.data_buft      = X.data_on_device ? ggml_backend_dev_buffer_type(gdev) : X.host_buft;
        int n_pinned = X.pinned ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &n_pinned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        char d[512];
        if (X.split)
            snprintf(d, sizeof(d), "split (metadata in host memory, KV data %s; GPU-aware MPI on rank 0: %s%s)",
                     n_capable > 0 ? (std::to_string(n_capable) + " rank(s) keep KV data in GPU memory").c_str() : "in host memory on all ranks",
                     a.gpu_aware == "auto" ? how.c_str() : ("forced " + a.gpu_aware).c_str(), k_split_api ? "" : "");
        else
            snprintf(d, sizeof(d), "host (one serialized blob per sequence; %s%s)",
                     n_pinned > 0 ? (std::to_string(n_pinned) + " GPU rank(s) stage in pinned memory").c_str() : "pageable staging",
                     k_split_api ? "" : "; split-state API not built, --kv-transport device unavailable");
        a.kv_desc = d;
        if (g_rank == 0 && a.kv_transport == "device" && !any_capable)
            fprintf(stderr, "warning: --kv-transport device, but no rank can keep KV data in GPU memory (no GPU, or GPU-aware MPI\n"
                            "         not detected; see --gpu-aware-mpi). Using the split format with host buffers.\n");
    }

    // ---- RMA counters + communicators ----
    GlobalCounters gc;
    gc.init(MPI_COMM_WORLD);
    MPI_Comm comm_ctl, comm_kv;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_ctl);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_kv);

    ProgressThread prog;
    const bool want_prog = (a.progress == "all") || (a.progress == "root" && g_rank == 0);
    if (want_prog) {
        if (provided == MPI_THREAD_MULTIPLE) prog.start(MPI_COMM_WORLD);
        else if (g_rank == 0) fprintf(stderr, "warning: MPI_THREAD_MULTIPLE unavailable, progress thread disabled\n");
    }

    // ---- per-decode-rank KV capacities (slots + cell budget), shared with every rank ----
    int64_t my_cap[2] = {0, 0};
    if (role == ROLE_DECODE) {
        const bool g = dev >= 0;
        const auto & cls = g ? gpu_dec : cpu_dec;
        const int di = (int) (std::find(cls.begin(), cls.end(), g_rank) - cls.begin());
        const std::string & bspec = g ? a.decode_batch_gpu : a.decode_batch;
        const std::string & cspec = g ? a.decode_kv_cells_gpu : a.decode_kv_cells;
        my_cap[0] = pick_capacity(bspec, g ? "--decode-batch-gpu" : "--decode-batch", di, (int) cls.size(), threads, !g);
        my_cap[1] = cspec.empty() ? (int64_t) a.ctx * my_cap[0]
                  : pick_capacity(cspec, g ? "--decode-kv-cells-gpu" : "--decode-kv-cells", di, (int) cls.size(), threads, false);
        if (my_cap[0] > 256) fatal("rank %d: %" PRId64 " decode slots exceeds llama.cpp's limit of 256 sequences per context (run more decode ranks)", g_rank, my_cap[0]);
        if (my_cap[1] < a.ctx)
            fatal("rank %d has %" PRId64 " KV cells, below -c %d (a max-length request could never be admitted); lower -c or raise the budget",
                g_rank, my_cap[1], a.ctx);
    }
    std::vector<int64_t> caps(2 * g_world);
    MPI_Allgather(my_cap, 2, MPI_INT64_T, caps.data(), 2, MPI_INT64_T, MPI_COMM_WORLD);
    std::vector<int64_t> cap_seq(g_world), cap_cells(g_world);
    for (int r = 0; r < g_world; ++r) { cap_seq[r] = caps[2 * r]; cap_cells[r] = caps[2 * r + 1]; }

    RankStat st{};
    st.rank = g_rank; st.role = role; st.node = node_id; st.local_rank = lrank; st.gpu = dev; st.threads = threads;
    st.slots = my_cap[0]; st.kv_cells = my_cap[1];
    st.kv_mode = X.data_on_device ? 2 : (X.pinned ? 1 : 0);
    std::vector<ReqRecord> recs;
    std::vector<OutText> outs;

    if (role == ROLE_PREFILL) run_prefill(a, model, model_dft, gc, comm_ctl, comm_kv, decode_ranks, cap_seq, cap_cells, dataset, threads, dev >= 0, X, st);
    else                      run_decode(a, model, model_dft, gc, comm_ctl, comm_kv, n_pf, threads, dev >= 0, X, (int) my_cap[0], my_cap[1], st, recs, outs);

    MPI_Barrier(MPI_COMM_WORLD);
    prog.stop();

    // ---- gather stats ----
    std::vector<RankStat> all_st(g_rank == 0 ? g_world : 0);
    MPI_Gather(&st, (int) sizeof(RankStat), MPI_BYTE, all_st.data(), (int) sizeof(RankStat), MPI_BYTE, 0, MPI_COMM_WORLD);

    int my_bytes = (int) (recs.size() * sizeof(ReqRecord));
    std::vector<int> counts(g_world), displs(g_world);
    MPI_Gather(&my_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<ReqRecord> all_recs;
    if (g_rank == 0) {
        int tot = 0;
        for (int r = 0; r < g_world; ++r) { displs[r] = tot; tot += counts[r]; }
        all_recs.resize(tot / sizeof(ReqRecord));
    }
    MPI_Gatherv(recs.data(), my_bytes, MPI_BYTE, all_recs.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::sort(all_recs.begin(), all_recs.end(), [](const ReqRecord & x, const ReqRecord & y) { return x.task_id < y.task_id; });
        report(a, all_recs, all_st, n_records);
    }
    if (!a.output.empty()) write_outputs(a, outs);

    gc.finalize();
    MPI_Comm_free(&comm_ctl);
    MPI_Comm_free(&comm_kv);
    MPI_Comm_free(&comm_pf);
    MPI_Comm_free(&node);
    if (model_dft) llama_model_free(model_dft);
    llama_model_free(model);
    llama_backend_free();
    MPI_Finalize();
    return 0;
}
