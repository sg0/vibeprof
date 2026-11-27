// vibeprof collector: PMPI wrappers + phase markers + per-rank profiles.
//
// Loaded with LD_PRELOAD=libvibeprof.so, or linked with -lvibeprof. For every wrapped MPI call it
// records, per (phase, MPI operation, call site): call count, time inside MPI, max time, payload bytes
// and a log2 message-size histogram. Time between MPI calls is computation, attributed to the
// innermost open phase. Each phase instance (begin..end) is recorded with its computation and MPI
// time, which gives per-iteration behavior. Sends also record bytes per destination world rank.
//
// At MPI_Init rank 0 writes <dir>/started; at MPI_Finalize every rank writes <dir>/rank-NNNNNN.json
// and rank 0 also writes <dir>/run.json.
//
// Environment:
//   VIBEPROF=0               disable (wrappers forward directly)
//   VIBEPROF_DIR=path        output directory (default vibeprof-<SLURM_JOB_ID[.STEP]|PBS job> or vibeprof-<time>-<pid>)
//   VIBEPROF_MAX_INSTANCES   phase instances kept per phase (default 5000)
//   VIBEPROF_FAKE_PPN=k      testing: report ranks as k-per-node on fake hosts (rank/k)
//   VIBEPROF_GPU=auto|cuda|none   GPU backend (see gpu_cupti.cpp)
//   VIBEPROF_KOKKOS_KERNELS=1     also record Kokkos parallel_for/reduce/scan as phases (host view)
//
// Kokkos: with KOKKOS_TOOLS_LIBS=libvibeprof.so, Kokkos profile regions become phases and Kokkos
// kernel labels name the GPU kernels they launch.

#include "vibeprof.h"
#include "gpu_backend.h"
#include "ops.h"
#include "mpi_bufs.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

double now_mono() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + ts.tv_nsec * 1e-9;
}

double now_epoch() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double) ts.tv_sec + ts.tv_nsec * 1e-9;
}

constexpr int k_hist = 40;   // bucket b: bytes in [2^(b-1), 2^b), bucket 0 = 0 bytes

struct OpStat {
    uint64_t count = 0;
    double   time = 0, tmax = 0;
    uint64_t bytes = 0;
    uint64_t dev_bytes = 0, dev_count = 0;   // buffers in GPU memory (GPU-aware MPI)
    uint32_t hist[k_hist] = {0};
};

struct Key {
    uint32_t  phase;
    uint32_t  op;
    uintptr_t site;
    bool operator==(const Key & o) const { return phase == o.phase && op == o.op && site == o.site; }
};
struct KeyHash {
    size_t operator()(const Key & k) const {
        return std::hash<uint64_t>()(((uint64_t) k.phase << 40) ^ ((uint64_t) k.op << 32) ^ (uint64_t) k.site);
    }
};

struct Inst { double t0, t1, comp, mpi; };

struct Phase {
    std::string path;
    double comp = 0, mpi = 0;        // exclusive: time while this phase was the innermost one
    uint64_t n = 0;
    std::vector<Inst> inst;          // inclusive per-instance times
    uint64_t dropped = 0;
};

struct Frame { uint32_t id; double t0, comp0, mpi0; };

struct Mark { uint64_t n = 0; std::vector<double> t; };

struct State {
    bool enabled = true;             // from VIBEPROF
    bool on = false;                 // MPI initialized and enabled
    bool use_mu = false;             // MPI_THREAD_MULTIPLE
    std::mutex mu;
    int rank = -1, size = 0;
    double t0 = 0, t0_epoch = 0, t_last = 0;
    double comp_total = 0, mpi_total = 0;
    uint64_t calls_total = 0;
    std::vector<Phase> phases;
    std::unordered_map<std::string, uint32_t> phase_ids;
    std::vector<Frame> stack;
    std::unordered_map<Key, OpStat, KeyHash> ops;
    std::unordered_map<int, std::pair<uint64_t, uint64_t>> peers;   // world dest -> (bytes, messages)
    std::map<MPI_Comm, std::vector<int>> comm_map;                   // comm rank -> world rank
    std::map<std::string, Mark> marks;
    std::vector<std::string> warnings;
    uint64_t markers_outside_mpi = 0, marker_mismatches = 0;
    std::string dir;
    size_t max_inst = 5000;
    int fake_ppn = 0;
    int thread_level = MPI_THREAD_SINGLE;
    GpuBackend * gpu = nullptr;
    bool kokkos_kernels = false;
};

State & S() {
    static State * s = new State();   // intentionally leaked: usable during static destruction
    return *s;
}

// Call-site hint from an interposing layer between the application and the collector (e.g. the
// perturbation library): the original caller's return address, used instead of our own.
thread_local uintptr_t t_site_hint = 0;

struct Lock {
    State & s;
    explicit Lock(State & st) : s(st) { if (s.use_mu) s.mu.lock(); }
    ~Lock() { if (s.use_mu) s.mu.unlock(); }
};

uint32_t current_phase(const State & s) { return s.stack.empty() ? 0 : s.stack.back().id; }

uint32_t intern_phase(State & s, const std::string & path) {
    auto it = s.phase_ids.find(path);
    if (it != s.phase_ids.end()) return it->second;
    const uint32_t id = (uint32_t) s.phases.size();
    s.phases.emplace_back();
    s.phases.back().path = path;
    s.phase_ids.emplace(path, id);
    return id;
}

// account the computation interval [t_last, t) to the innermost phase
void account_comp(State & s, double t) {
    const double dt = t - s.t_last;
    if (dt > 0) {
        s.comp_total += dt;
        s.phases[current_phase(s)].comp += dt;
    }
    s.t_last = t;
}

int hist_bucket(uint64_t bytes) {
    if (bytes == 0) return 0;
    int b = 64 - __builtin_clzll(bytes);   // floor(log2) + 1
    return std::min(b, k_hist - 1);
}

int world_peer(State & s, MPI_Comm comm, int r) {
    if (r < 0 || comm == MPI_COMM_NULL) return -1;
    if (comm == MPI_COMM_WORLD) return r;
    auto it = s.comm_map.find(comm);
    if (it == s.comm_map.end()) {
        std::vector<int> v;
        int inter = 0;
        PMPI_Comm_test_inter(comm, &inter);
        if (!inter) {
            int n = 0;
            PMPI_Comm_size(comm, &n);
            MPI_Group g, wg;
            PMPI_Comm_group(comm, &g);
            PMPI_Comm_group(MPI_COMM_WORLD, &wg);
            std::vector<int> a(n);
            std::iota(a.begin(), a.end(), 0);
            v.resize(n);
            PMPI_Group_translate_ranks(g, n, a.data(), wg, v.data());
            PMPI_Group_free(&g);
            PMPI_Group_free(&wg);
        }
        it = s.comm_map.emplace(comm, std::move(v)).first;
    }
    return r < (int) it->second.size() ? it->second[r] : -1;
}

}  // namespace

// ---- byte-count helpers used by the call table ----
static inline int64_t tsz(int count, MPI_Datatype dt) {
    if (count <= 0 || dt == MPI_DATATYPE_NULL) return 0;
    int sz = 0;
    PMPI_Type_size(dt, &sz);
    return (int64_t) count * sz;
}
static inline int64_t csz(MPI_Comm comm) {
    int n = 0, inter = 0;
    PMPI_Comm_test_inter(comm, &inter);
    if (inter) PMPI_Comm_remote_size(comm, &n); else PMPI_Comm_size(comm, &n);
    return n;
}
static inline int64_t vsz(const int * counts, MPI_Datatype dt, MPI_Comm comm) {
    if (!counts) return 0;
    const int64_t n = csz(comm);
    int64_t c = 0;
    for (int64_t i = 0; i < n; ++i) c += counts[i];
    return c > INT_MAX ? tsz(1, dt) * c : tsz((int) c, dt);
}

// ---- wrapper core ----
static inline double llmp_enter(State & s) {
    const double t = now_mono();
    account_comp(s, t);
    return t;
}

static inline void llmp_leave(State & s, int op, uintptr_t site, double t_enter, MPI_Comm comm, int64_t bytes, int peer, const void * buf) {
    const double t = now_mono();
    const double dt = t - t_enter;
    const uint32_t ph = current_phase(s);
    s.mpi_total += dt;
    s.calls_total++;
    s.phases[ph].mpi += dt;
    OpStat & o = s.ops[Key{ph, (uint32_t) op, site}];
    o.count++;
    o.time += dt;
    o.tmax = std::max(o.tmax, dt);
    if (bytes > 0) o.bytes += (uint64_t) bytes;
    if (k_op_kinds[op] != K_WAIT && k_op_kinds[op] != K_PROBE && k_op_kinds[op] != K_RMASYNC && op != OP_Barrier && op != OP_Ibarrier)
        o.hist[hist_bucket((uint64_t) std::max<int64_t>(bytes, 0))]++;
    if (peer >= 0 && (k_op_kinds[op] == K_SEND || k_op_kinds[op] == K_SENDRECV)) {
        const int w = world_peer(s, comm, peer);
        if (w >= 0) { auto & p = s.peers[w]; p.first += (uint64_t) std::max<int64_t>(bytes, 0); p.second++; }
    }
    s.t_last = t;
    if (s.gpu && buf && bytes > 0 && buf != MPI_BOTTOM && s.gpu->is_device_pointer(buf)) {
        o.dev_bytes += (uint64_t) bytes;
        o.dev_count++;
        s.t_last = now_mono();   // the pointer query is tool overhead, not computation
    }
}

#define LLMP_RA() ((uintptr_t) __builtin_return_address(0))

#define X(NAME, PARAMS, ARGS, COMM, BYTES, PEER, KIND)                                   \
    extern "C" int MPI_##NAME PARAMS {                                                   \
        const uintptr_t site = t_site_hint ? t_site_hint : LLMP_RA();                   \
        t_site_hint = 0;                                                                 \
        State & s = S();                                                                 \
        if (!s.on) return PMPI_##NAME ARGS;                                              \
        Lock lk(s);                                                                      \
        const double t0 = llmp_enter(s);                                                 \
        vibeprof_t_in_mpi = true;                                                         \
        const int rc = PMPI_##NAME ARGS;                                                 \
        vibeprof_t_in_mpi = false;                                                        \
        llmp_leave(s, OP_##NAME, site, t0, COMM, (int64_t) (BYTES), (PEER), LLMP_BUF_##NAME); \
        return rc;                                                                       \
    }
#include "mpi_calls_send.def"
#include "mpi_calls.def"
#undef X

// ---- init / finalize ----
namespace {

std::string default_dir() {
    const char * job = getenv("SLURM_JOB_ID");
    const char * step = getenv("SLURM_STEP_ID");
    std::string pbs;
    if (!job && (job = getenv("PBS_JOBID"))) {   // PBS: "1234567.polaris-pbs-01..." -> "1234567"
        pbs = job; pbs = pbs.substr(0, pbs.find('.')); job = pbs.c_str(); step = nullptr;
    }
    char buf[256];
    if (job) {
        snprintf(buf, sizeof(buf), "vibeprof-%s%s%s", job, step ? "." : "", step ? step : "");
    } else {
        time_t t = time(nullptr);
        struct tm tmv;
        localtime_r(&t, &tmv);
        char ts[64];
        strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", &tmv);
        snprintf(buf, sizeof(buf), "vibeprof-%s-%d", ts, (int) getpid());
    }
    return buf;
}

void mkdir_p(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) mkdir(cur.c_str(), 0755);
    }
}

void tool_init(int provided) {
    State & s = S();
    const char * e = getenv("VIBEPROF");
    s.enabled = !(e && (strcmp(e, "0") == 0 || strcmp(e, "off") == 0));
    if (!s.enabled) return;
    PMPI_Comm_rank(MPI_COMM_WORLD, &s.rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &s.size);
    s.thread_level = provided;
    s.use_mu = provided == MPI_THREAD_MULTIPLE;
    if (const char * m = getenv("VIBEPROF_MAX_INSTANCES")) s.max_inst = (size_t) std::max(0L, atol(m));
    if (const char * f = getenv("VIBEPROF_FAKE_PPN")) s.fake_ppn = std::max(0, atoi(f));

    // every rank must agree on the output directory: rank 0 decides and broadcasts
    char dir[4096] = {0};
    if (s.rank == 0) {
        const char * d = getenv("VIBEPROF_DIR");
        snprintf(dir, sizeof(dir), "%s", d && *d ? d : default_dir().c_str());
        mkdir_p(dir);
        // marker for vibeprof-srun: the collector loaded and MPI_Init ran (profiles come at MPI_Finalize)
        std::string mk = std::string(dir) + "/started";
        if (FILE * f = fopen(mk.c_str(), "w")) {
            fprintf(f, "ranks %d\npid %d\nthread_level %d\n", s.size, (int) getpid(), provided);
            fclose(f);
        }
    }
    PMPI_Bcast(dir, (int) sizeof(dir), MPI_CHAR, 0, MPI_COMM_WORLD);
    PMPI_Barrier(MPI_COMM_WORLD);   // directory exists; also aligns the ranks' time origins
    s.dir = dir;

    s.t0 = now_mono();
    s.t0_epoch = now_epoch();
    s.t_last = s.t0;
    intern_phase(s, "<main>");
    std::string why;
    s.gpu = vibeprof_make_gpu_backend(&why);
    if (!why.empty()) s.warnings.push_back(why);
    if (s.gpu) s.gpu->on_init(s.rank, s.t0);
    if (const char * k = getenv("VIBEPROF_KOKKOS_KERNELS")) s.kokkos_kernels = atoi(k) != 0;
    if (s.use_mu) s.warnings.push_back("MPI_THREAD_MULTIPLE: computation intervals are approximate when several threads call MPI");
    s.on = true;
}

// ---------- JSON output ----------
void jstr(FILE * f, const std::string & v) {
    fputc('"', f);
    for (unsigned char c : v) {
        switch (c) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f); break;
            case '\t': fputs("\\t", f); break;
            default:
                if (c < 0x20) fprintf(f, "\\u%04x", c); else fputc(c, f);
        }
    }
    fputc('"', f);
}

std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return "";
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

std::string cmdline() {
    std::string c = read_file("/proc/self/cmdline");
    for (auto & ch : c) if (ch == '\0') ch = ' ';
    while (!c.empty() && c.back() == ' ') c.pop_back();
    return c;
}

std::string exe_path() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = 0;
    return buf;
}

void write_profile() {
    State & s = S();
    char path[4600];
    snprintf(path, sizeof(path), "%s/rank-%06d.json", s.dir.c_str(), s.rank);
    FILE * f = fopen(path, "w");
    if (!f) { fprintf(stderr, "vibeprof: cannot write %s\n", path); return; }

    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    std::string hostname = host;
    if (s.fake_ppn > 0) hostname = "fakenode" + std::to_string(s.rank / s.fake_ppn);
    char ver[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
    int vlen = 0;
    PMPI_Get_library_version(ver, &vlen);
    std::string mpiver(ver, vlen);
    if (auto nl = mpiver.find('\n'); nl != std::string::npos) mpiver = mpiver.substr(0, nl);

    const double wall = now_mono() - s.t0;
    fprintf(f, "{\n\"format\": \"vibeprof-rank\", \"version\": 1,\n");
    fprintf(f, "\"rank\": %d, \"size\": %d, \"host\": ", s.rank, s.size); jstr(f, hostname);
    fprintf(f, ", \"pid\": %d, \"thread_level\": %d,\n\"mpi_library\": ", (int) getpid(), s.thread_level); jstr(f, mpiver);
    fprintf(f, ",\n\"exe\": "); jstr(f, exe_path());
    fprintf(f, ", \"cmdline\": "); jstr(f, cmdline());
    fprintf(f, ",\n\"t0_epoch\": %.6f, \"wall\": %.9f, \"comp_total\": %.9f, \"mpi_total\": %.9f, \"calls_total\": %llu,\n",
            s.t0_epoch, wall, s.comp_total, s.mpi_total, (unsigned long long) s.calls_total);

    fprintf(f, "\"phases\": [\n");
    for (size_t i = 0; i < s.phases.size(); ++i) {
        const Phase & p = s.phases[i];
        fprintf(f, "  {\"id\": %zu, \"path\": ", i); jstr(f, p.path);
        fprintf(f, ", \"comp\": %.9f, \"mpi\": %.9f, \"n\": %llu, \"dropped\": %llu, \"inst\": [",
                p.comp, p.mpi, (unsigned long long) p.n, (unsigned long long) p.dropped);
        for (size_t k = 0; k < p.inst.size(); ++k)
            fprintf(f, "%s[%.6f,%.6f,%.6f,%.6f]", k ? "," : "", p.inst[k].t0, p.inst[k].t1, p.inst[k].comp, p.inst[k].mpi);
        fprintf(f, "]}%s\n", i + 1 < s.phases.size() ? "," : "");
    }
    fprintf(f, "],\n\"ops\": [\n");
    size_t k = 0;
    std::map<uintptr_t, int> sites;
    for (const auto & [key, o] : s.ops) {
        sites[key.site] = 0;
        fprintf(f, "  {\"phase\": %u, \"op\": \"%s\", \"kind\": \"%s\", \"site\": \"0x%llx\", \"count\": %llu, \"time\": %.9f, "
                   "\"tmax\": %.9f, \"bytes\": %llu, ",
                key.phase, k_op_names[key.op], k_kind_names[k_op_kinds[key.op]], (unsigned long long) key.site,
                (unsigned long long) o.count, o.time, o.tmax, (unsigned long long) o.bytes);
        if (o.dev_count) fprintf(f, "\"dev_bytes\": %llu, \"dev_count\": %llu, ", (unsigned long long) o.dev_bytes, (unsigned long long) o.dev_count);
        fprintf(f, "\"hist\": {");
        bool first = true;
        for (int b = 0; b < k_hist; ++b) if (o.hist[b]) { fprintf(f, "%s\"%d\": %u", first ? "" : ", ", b, o.hist[b]); first = false; }
        fprintf(f, "}}%s\n", ++k < s.ops.size() ? "," : "");
    }
    // call sites: object file + offset for addr2line, plus nearest dynamic symbol
    fprintf(f, "],\n\"sites\": {\n");
    const std::string exe = exe_path();
    k = 0;
    for (const auto & [addr, unused] : sites) {
        (void) unused;
        Dl_info di;
        std::string obj, sym;
        uintptr_t off = addr;
        if (dladdr((void *) addr, &di) && di.dli_fname) {
            obj = di.dli_fname;
            if (obj.empty() || access(obj.c_str(), R_OK) != 0) obj = exe;
            off = addr - (uintptr_t) di.dli_fbase - 1;   // return address -> call instruction
            if (di.dli_sname) sym = di.dli_sname;
        }
        fprintf(f, "  \"0x%llx\": {\"obj\": ", (unsigned long long) addr); jstr(f, obj);
        fprintf(f, ", \"offset\": \"0x%llx\", \"sym\": ", (unsigned long long) off); jstr(f, sym);
        fprintf(f, "}%s\n", ++k < sites.size() ? "," : "");
    }
    fprintf(f, "},\n\"peers\": [");
    k = 0;
    for (const auto & [dst, p] : s.peers)
        fprintf(f, "%s[%d,%llu,%llu]", k++ ? "," : "", dst, (unsigned long long) p.first, (unsigned long long) p.second);
    fprintf(f, "],\n\"marks\": [");
    k = 0;
    for (const auto & [name, m] : s.marks) {
        fprintf(f, "%s\n  {\"name\": ", k++ ? "," : ""); jstr(f, name);
        fprintf(f, ", \"n\": %llu, \"t\": [", (unsigned long long) m.n);
        for (size_t i = 0; i < m.t.size(); ++i) fprintf(f, "%s%.6f", i ? "," : "", m.t[i]);
        fprintf(f, "]}");
    }
    fprintf(f, "],\n\"gpu\": {");
    if (s.gpu) { fprintf(f, "\"backend\": "); jstr(f, s.gpu->name()); s.gpu->write_json(f); }
    fprintf(f, "},\n\"warnings\": [");
    for (size_t i = 0; i < s.warnings.size(); ++i) { if (i) fputc(',', f); jstr(f, s.warnings[i]); }
    fprintf(f, "]\n}\n");
    fclose(f);

    if (s.rank == 0) {
        snprintf(path, sizeof(path), "%s/run.json", s.dir.c_str());
        if (FILE * r = fopen(path, "w")) {
            fprintf(r, "{\"format\": \"vibeprof-run\", \"version\": 1, \"size\": %d, \"t0_epoch\": %.6f, \"exe\": ", s.size, s.t0_epoch);
            jstr(r, exe_path());
            fprintf(r, ", \"cmdline\": "); jstr(r, cmdline());
            fprintf(r, ", \"mpi_library\": "); jstr(r, mpiver);
            fprintf(r, ", \"slurm_job_id\": "); jstr(r, getenv("SLURM_JOB_ID") ? getenv("SLURM_JOB_ID") : "");
            fprintf(r, "}\n");
            fclose(r);
        }
    }
}

void tool_finalize() {
    State & s = S();
    if (!s.on) return;
    {
        Lock lk(s);
        const double t = now_mono();
        account_comp(s, t);
        if (!s.stack.empty()) {
            s.warnings.push_back(std::to_string(s.stack.size()) + " phase(s) still open at MPI_Finalize; closed automatically");
            while (!s.stack.empty()) vibeprof_phase_end(nullptr);
        }
        if (s.marker_mismatches)
            s.warnings.push_back(std::to_string(s.marker_mismatches) + " vibeprof_phase_end() name(s) did not match the innermost open phase");
        if (s.markers_outside_mpi)
            s.warnings.push_back(std::to_string(s.markers_outside_mpi) + " marker call(s) before MPI_Init or after MPI_Finalize were ignored");
        if (s.gpu) s.gpu->on_finalize();
        s.on = false;
    }
    write_profile();
}

}  // namespace

extern "C" int MPI_Init(int * argc, char *** argv) {
    const int rc = PMPI_Init(argc, argv);
    if (rc == MPI_SUCCESS) tool_init(MPI_THREAD_SINGLE);
    return rc;
}

extern "C" int MPI_Init_thread(int * argc, char *** argv, int required, int * provided) {
    const int rc = PMPI_Init_thread(argc, argv, required, provided);
    if (rc == MPI_SUCCESS) tool_init(*provided);
    return rc;
}

extern "C" int MPI_Finalize(void) {
    tool_finalize();
    return PMPI_Finalize();
}

extern "C" int MPI_Comm_free(MPI_Comm * comm) {
    State & s = S();
    if (s.on && comm) { Lock lk(s); s.comm_map.erase(*comm); }
    return PMPI_Comm_free(comm);
}

// ---- markers ----
extern "C" void vibeprof_phase_begin(const char * name) {
    State & s = S();
    if (!s.on) { if (s.enabled) s.markers_outside_mpi++; return; }
    Lock lk(s);
    const double t = now_mono();
    account_comp(s, t);
    const std::string nm = name && *name ? name : "(unnamed)";
    const std::string path = s.stack.empty() ? nm : s.phases[s.stack.back().id].path + "/" + nm;
    const uint32_t id = intern_phase(s, path);
    s.stack.push_back(Frame{id, t, s.comp_total, s.mpi_total});
    s.phases[id].n++;
    if (s.gpu) s.gpu->on_phase_change(id);
}

extern "C" void vibeprof_phase_end(const char * name) {
    State & s = S();
    // called with name == nullptr only from tool_finalize (lock held, s.on still true)
    if (name && !s.on) { if (s.enabled) s.markers_outside_mpi++; return; }
    std::unique_lock<std::mutex> lk(s.mu, std::defer_lock);
    if (name && s.use_mu) lk.lock();
    const double t = now_mono();
    account_comp(s, t);
    if (s.stack.empty()) { s.marker_mismatches++; return; }
    const Frame fr = s.stack.back();
    Phase & p = s.phases[fr.id];
    if (name) {
        const std::string & path = p.path;
        const size_t slash = path.rfind('/');
        const std::string leaf = slash == std::string::npos ? path : path.substr(slash + 1);
        if (leaf != name) s.marker_mismatches++;
    }
    s.stack.pop_back();
    if (p.inst.size() < s.max_inst) p.inst.push_back(Inst{fr.t0 - s.t0, t - s.t0, s.comp_total - fr.comp0, s.mpi_total - fr.mpi0});
    else p.dropped++;
    if (s.gpu) s.gpu->on_phase_change(current_phase(s));
}

extern "C" void vibeprof_mark(const char * name) {
    State & s = S();
    if (!s.on) { if (s.enabled) s.markers_outside_mpi++; return; }
    Lock lk(s);
    Mark & m = s.marks[name && *name ? name : "(unnamed)"];
    m.n++;
    if (m.t.size() < 1000) m.t.push_back(now_mono() - s.t0);
}

extern "C" int vibeprof_enabled(void) { return S().on ? 1 : 0; }

// For interposing tools: report `ra` as the call site of the next MPI call made by this thread.
extern "C" void vibeprof_set_caller_site(void * ra) { t_site_hint = (uintptr_t) ra; }

// ---- Kokkos Tools interface (KOKKOS_TOOLS_LIBS=libvibeprof.so) ----
namespace {
// '/' separates nested phases, so it cannot appear inside a phase name
std::string kk_clean(const char * name) {
    std::string n = name && *name ? name : "(unnamed)";
    std::replace(n.begin(), n.end(), '/', '|');
    return n;
}
struct KkNames {
    std::mutex mu;
    std::vector<std::string> names;                 // kID -> "kokkos:<label>"
    std::unordered_map<std::string, uint64_t> ids;
};
KkNames & kk() { static KkNames * k = new KkNames(); return *k; }
}

extern "C" void kokkosp_init_library(const int, const uint64_t, const uint32_t, void *) {}
extern "C" void kokkosp_finalize_library() {}

extern "C" void kokkosp_push_profile_region(const char * name) { vibeprof_phase_begin(kk_clean(name).c_str()); }

extern "C" void kokkosp_pop_profile_region() {
    State & s = S();
    if (!s.on) { if (s.enabled) s.markers_outside_mpi++; return; }
    std::string leaf;
    {
        Lock lk(s);
        if (s.stack.empty()) { s.marker_mismatches++; return; }
        const std::string & path = s.phases[s.stack.back().id].path;
        const size_t slash = path.rfind('/');
        leaf = slash == std::string::npos ? path : path.substr(slash + 1);
    }
    vibeprof_phase_end(leaf.c_str());
}

static void kk_begin(const char * name, uint64_t * kid) {
    State & s = S();
    if (kid) *kid = UINT64_MAX;
    if (s.gpu) s.gpu->on_kernel_label(name);
    if (s.kokkos_kernels && s.on && kid) {
        KkNames & k = kk();
        std::string full = "kokkos:" + kk_clean(name);
        {
            std::lock_guard<std::mutex> lk(k.mu);
            auto it = k.ids.find(full);
            if (it == k.ids.end()) { it = k.ids.emplace(full, k.names.size()).first; k.names.push_back(full); }
            *kid = it->second;
        }
        vibeprof_phase_begin(full.c_str());
    }
}
static void kk_end(uint64_t kid) {
    State & s = S();
    if (s.gpu) s.gpu->on_kernel_label(nullptr);
    if (s.kokkos_kernels && s.on && kid != UINT64_MAX) {
        std::string n;
        {
            KkNames & k = kk();
            std::lock_guard<std::mutex> lk(k.mu);
            if (kid >= k.names.size()) return;
            n = k.names[kid];
        }
        vibeprof_phase_end(n.c_str());
    }
}
extern "C" void kokkosp_begin_parallel_for(const char * name, const uint32_t, uint64_t * kid) { kk_begin(name, kid); }
extern "C" void kokkosp_end_parallel_for(const uint64_t kid) { kk_end(kid); }
extern "C" void kokkosp_begin_parallel_reduce(const char * name, const uint32_t, uint64_t * kid) { kk_begin(name, kid); }
extern "C" void kokkosp_end_parallel_reduce(const uint64_t kid) { kk_end(kid); }
extern "C" void kokkosp_begin_parallel_scan(const char * name, const uint32_t, uint64_t * kid) { kk_begin(name, kid); }
extern "C" void kokkosp_end_parallel_scan(const uint64_t kid) { kk_end(kid); }
