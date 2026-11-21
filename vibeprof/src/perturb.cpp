// vibeperturb: inject known performance problems into unmodified MPI programs (ground truth for
// evaluating vibeprof reports).
//
// Load it in front of the collector:  LD_PRELOAD=libvibeperturb.so:libvibeprof.so
// It interposes two layers, so the profiler observes the perturbation like a real problem:
//   MPI_*  (outer, before the profiler): slows computation, inserts extra barriers.
//   PMPI_* (inner, inside the profiler's timing): delays chosen MPI operations, makes sends synchronous.
//
// Environment (all optional):
//   VIBEPERTURB_SLOW_RANKS=1,5        ranks whose computation runs VIBEPERTURB_SLOW_FACTOR times longer
//   VIBEPERTURB_SLOW_FACTOR=1.5
//   VIBEPERTURB_SLOW_NODE=1           node index (0-based, hostnames sorted; or rank/VIBEPROF_FAKE_PPN)
//   VIBEPERTURB_NODE_FACTOR=1.5       slowdown of computation on that node
//   VIBEPERTURB_EXTRA_BARRIER=Allreduce   insert MPI_Barrier after every Nth call of this operation
//   VIBEPERTURB_EXTRA_BARRIER_EVERY=1
//   VIBEPERTURB_COMM_DELAY=Allreduce:500  add 500 us inside every call of this operation (all ranks)
//   VIBEPERTURB_SYNC_SENDS=1          MPI_Send/MPI_Isend behave as MPI_Ssend/MPI_Issend
//   VIBEPERTURB_TRUTH=path.json       rank 0 writes the active perturbations here (for scoring)

#include "ops.h"

#include <dlfcn.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + ts.tv_nsec * 1e-9;
}

void spin(double seconds) {
    if (seconds <= 0) return;
    const double end = now() + seconds;
    volatile double x = 1.0;
    while (now() < end) x = x * 1.0000001 + 1e-9;   // burn CPU like real computation
}

void * next_sym(const char * name) {
    void * p = dlsym(RTLD_NEXT, name);
    if (!p) { fprintf(stderr, "vibeperturb: cannot resolve %s\n", name); abort(); }
    return p;
}

struct Perturb {
    bool ready = false;
    int rank = -1, size = 0, node = -1, n_nodes = 0;
    std::vector<int> slow_ranks;
    double slow_factor = 1.0;
    int slow_node = -1;
    double node_factor = 1.0;
    double factor = 1.0;              // effective computation slowdown on this rank
    int barrier_op = -1, barrier_every = 1;
    long barrier_count = 0;
    int delay_op = -1;
    double delay_s = 0;
    bool sync_sends = false;
    double last_exit = 0;
};

Perturb & P() { static Perturb * p = new Perturb(); return *p; }

std::vector<int> parse_ints(const char * s) {
    std::vector<int> v;
    if (!s) return v;
    std::string cur;
    for (const char * c = s;; ++c) {
        if (*c == ',' || *c == 0) { if (!cur.empty()) v.push_back(atoi(cur.c_str())); cur.clear(); if (!*c) break; }
        else cur += *c;
    }
    return v;
}

void setup() {
    Perturb & p = P();
    PMPI_Comm_rank(MPI_COMM_WORLD, &p.rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &p.size);

    p.slow_ranks = parse_ints(getenv("VIBEPERTURB_SLOW_RANKS"));
    if (const char * f = getenv("VIBEPERTURB_SLOW_FACTOR")) p.slow_factor = atof(f);
    if (const char * n = getenv("VIBEPERTURB_SLOW_NODE")) p.slow_node = atoi(n);
    if (const char * f = getenv("VIBEPERTURB_NODE_FACTOR")) p.node_factor = atof(f);
    if (const char * b = getenv("VIBEPERTURB_EXTRA_BARRIER")) p.barrier_op = op_by_name(b);
    if (const char * e = getenv("VIBEPERTURB_EXTRA_BARRIER_EVERY")) p.barrier_every = std::max(1, atoi(e));
    if (const char * d = getenv("VIBEPERTURB_COMM_DELAY")) {
        std::string s = d;
        const size_t c = s.find(':');
        p.delay_op = op_by_name(s.substr(0, c).c_str());
        p.delay_s = c == std::string::npos ? 1e-4 : atof(s.substr(c + 1).c_str()) * 1e-6;
    }
    if (const char * y = getenv("VIBEPERTURB_SYNC_SENDS")) p.sync_sends = atoi(y) != 0;

    // node index: fake grouping (same as the collector) or sorted unique hostnames
    if (const char * ppn = getenv("VIBEPROF_FAKE_PPN"); ppn && atoi(ppn) > 0) {
        p.node = p.rank / atoi(ppn);
        p.n_nodes = (p.size + atoi(ppn) - 1) / atoi(ppn);
    } else {
        char host[64] = {0};
        gethostname(host, sizeof(host) - 1);
        std::vector<char> all((size_t) p.size * 64);
        PMPI_Allgather(host, 64, MPI_CHAR, all.data(), 64, MPI_CHAR, MPI_COMM_WORLD);
        std::vector<std::string> names;
        for (int r = 0; r < p.size; ++r) names.emplace_back(&all[(size_t) r * 64]);
        std::vector<std::string> uniq = names;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        p.node = (int) (std::lower_bound(uniq.begin(), uniq.end(), names[p.rank]) - uniq.begin());
        p.n_nodes = (int) uniq.size();
    }

    p.factor = 1.0;
    if (std::find(p.slow_ranks.begin(), p.slow_ranks.end(), p.rank) != p.slow_ranks.end()) p.factor *= p.slow_factor;
    if (p.slow_node >= 0 && p.node == p.slow_node) p.factor *= p.node_factor;

    if (p.rank == 0) {
        std::string truth = "{";
        auto add = [&](const std::string & kv) { truth += (truth.size() > 1 ? ", " : "") + kv; };
        if (!p.slow_ranks.empty() && p.slow_factor != 1.0) {
            std::string r;
            for (size_t i = 0; i < p.slow_ranks.size(); ++i) r += (i ? "," : "") + std::to_string(p.slow_ranks[i]);
            add("\"slow_ranks\": {\"ranks\": [" + r + "], \"factor\": " + std::to_string(p.slow_factor) + "}");
        }
        if (p.slow_node >= 0 && p.node_factor != 1.0)
            add("\"slow_node\": {\"node\": " + std::to_string(p.slow_node) + ", \"factor\": " + std::to_string(p.node_factor) + "}");
        if (p.barrier_op >= 0)
            add("\"extra_barrier\": {\"after\": \"" + std::string(k_op_names[p.barrier_op]) + "\", \"every\": " + std::to_string(p.barrier_every) + "}");
        if (p.delay_op >= 0)
            add("\"comm_delay\": {\"op\": \"" + std::string(k_op_names[p.delay_op]) + "\", \"us\": " + std::to_string(p.delay_s * 1e6) + "}");
        if (p.sync_sends) add("\"sync_sends\": true");
        truth += "}";
        fprintf(stderr, "vibeperturb: active perturbations %s (%d nodes)\n", truth.c_str(), p.n_nodes);
        if (const char * path = getenv("VIBEPERTURB_TRUTH")) {
            if (FILE * f = fopen(path, "w")) { fprintf(f, "%s\n", truth.c_str()); fclose(f); }
        }
    }
    p.last_exit = now();
    p.ready = true;
}

// pass the application's call site on to the collector (if loaded), so sites are not attributed to us
inline void hint_site(void * ra) {
    static auto set = (void (*)(void *)) dlsym(RTLD_DEFAULT, "vibeprof_set_caller_site");
    if (set) set(ra);
}

inline void pre_outer() {
    Perturb & p = P();
    if (p.ready && p.factor > 1.0) spin((p.factor - 1.0) * (now() - p.last_exit));
}

inline void post_outer(int op, MPI_Comm comm, void * ra) {
    Perturb & p = P();
    if (p.ready && op == p.barrier_op && (++p.barrier_count % p.barrier_every) == 0) {
        static auto barrier = (decltype(&PMPI_Barrier)) next_sym("MPI_Barrier");
        hint_site(ra);   // the inserted barrier appears at the application's call site
        barrier(comm != MPI_COMM_NULL ? comm : MPI_COMM_WORLD);   // completion calls (Wait*) carry no communicator
    }
    p.last_exit = now();
}

inline void inner_delay(int op) {
    Perturb & p = P();
    if (p.ready && op == p.delay_op) spin(p.delay_s);
}

}  // namespace

// ---- outer layer: MPI_* (calls the next MPI_* in the chain, i.e. the profiler) ----
#define X(NAME, PARAMS, ARGS, COMM, BYTES, PEER, KIND)                                  \
    extern "C" int MPI_##NAME PARAMS {                                                  \
        static auto next = (decltype(&PMPI_##NAME)) next_sym("MPI_" #NAME);             \
        void * ra = __builtin_return_address(0);                                        \
        pre_outer();                                                                    \
        hint_site(ra);                                                                  \
        const int rc = next ARGS;                                                       \
        post_outer(OP_##NAME, COMM, ra);                                                \
        return rc;                                                                      \
    }
#include "mpi_calls_send.def"
#include "mpi_calls.def"
#undef X

// ---- inner layer: PMPI_* (calls the MPI library) ----
#define X(NAME, PARAMS, ARGS, COMM, BYTES, PEER, KIND)                                  \
    extern "C" int PMPI_##NAME PARAMS {                                                 \
        static auto next = (decltype(&PMPI_##NAME)) next_sym("PMPI_" #NAME);            \
        inner_delay(OP_##NAME);                                                         \
        return next ARGS;                                                               \
    }
#include "mpi_calls.def"
#undef X

extern "C" int PMPI_Send(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    static auto send  = (decltype(&PMPI_Send)) next_sym("PMPI_Send");
    static auto ssend = (decltype(&PMPI_Ssend)) next_sym("PMPI_Ssend");
    inner_delay(OP_Send);
    return P().sync_sends ? ssend(buf, count, datatype, dest, tag, comm) : send(buf, count, datatype, dest, tag, comm);
}

extern "C" int PMPI_Isend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * req) {
    static auto isend  = (decltype(&PMPI_Isend)) next_sym("PMPI_Isend");
    static auto issend = (decltype(&PMPI_Issend)) next_sym("PMPI_Issend");
    inner_delay(OP_Isend);
    return P().sync_sends ? issend(buf, count, datatype, dest, tag, comm, req) : isend(buf, count, datatype, dest, tag, comm, req);
}

// ---- init / finalize ----
extern "C" int MPI_Init(int * argc, char *** argv) {
    static auto next = (decltype(&PMPI_Init)) next_sym("MPI_Init");
    const int rc = next(argc, argv);
    if (rc == MPI_SUCCESS) setup();
    return rc;
}

extern "C" int MPI_Init_thread(int * argc, char *** argv, int required, int * provided) {
    static auto next = (decltype(&PMPI_Init_thread)) next_sym("MPI_Init_thread");
    const int rc = next(argc, argv, required, provided);
    if (rc == MPI_SUCCESS) setup();
    return rc;
}

extern "C" int MPI_Finalize(void) {
    static auto next = (decltype(&PMPI_Finalize)) next_sym("MPI_Finalize");
    pre_outer();   // the last computation segment is slowed too
    P().ready = false;
    return next();
}

// ---- phase markers: apply the computation slowdown at phase boundaries too, so the injected time
// lands in the phase whose computation it extends (not in the next phase's first MPI call) ----
extern "C" void vibeprof_phase_begin(const char * name) {
    static auto next = (void (*)(const char *)) dlsym(RTLD_NEXT, "vibeprof_phase_begin");
    pre_outer();
    if (next) next(name);
    P().last_exit = now();
}

extern "C" void vibeprof_phase_end(const char * name) {
    static auto next = (void (*)(const char *)) dlsym(RTLD_NEXT, "vibeprof_phase_end");
    pre_outer();
    if (next) next(name);
    P().last_exit = now();
}
