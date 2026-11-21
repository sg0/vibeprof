// NVIDIA GPU backend: CUPTI activity + callback API, loaded at run time.
//
// What is measured, per rank:
//   * kernels: count, GPU time, min/max per (launching phase, kernel name, Kokkos label)
//   * memory copies: count, bytes, time per (phase, direction, source/destination memory kind),
//     flagged when the MPI library issued them (staging inside MPI calls)
//   * memsets
//   * host time inside CUDA API calls per (phase, function), classified as sync (the host waits for
//     the GPU), alloc, launch, copy or other; calls made inside MPI are excluded
//   * GPU busy time (union of kernel intervals per device) and a busy timeline
//   * devices used (name, PCI bus id) and visible device count
//   * MPI buffers in device memory (GPU-aware MPI) via cuPointerGetAttribute
//
// Device work is attributed to the phase that was open when the host launched it, through the
// CUPTI correlation id of the launching API call.
//
// Environment:
//   VIBEPROF_GPU=auto|cuda|none     (default auto: enable when a CUDA driver with devices is present)
//   VIBEPROF_CUPTI_LIB=path         libcupti to use (default: search CUDA_HOME, the loaded libcudart, ...)
//   VIBEPROF_CUDA_LIB=path          libcuda to use (default libcuda.so.1)
//   VIBEPROF_GPU_API=0              do not trace CUDA API calls (no phase attribution, no sync time)

#include "cupti_abi.h"
#include "gpu_backend.h"

#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace cupti;

namespace {

double mono() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + ts.tv_nsec * 1e-9;
}

void jstr(FILE * f, const std::string & v) {
    fputc('"', f);
    for (unsigned char c : v) {
        if (c == '"') fputs("\\\"", f);
        else if (c == '\\') fputs("\\\\", f);
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

const char * copy_kind_name(uint8_t k) {
    static const char * n[] = {"unknown", "HtoD", "DtoH", "HtoA", "AtoH", "AtoA", "AtoD", "DtoA", "DtoD", "HtoH", "PtoP"};
    return k <= MEMCPY_PTOP ? n[k] : "unknown";
}
const char * mem_kind_name(uint8_t k) {
    static const char * n[] = {"unknown", "pageable", "pinned", "device", "array", "managed", "device", "managed"};
    return k <= MEM_MANAGED_STATIC ? n[k] : "unknown";
}

// "cudaMemcpyAsync_v3020" -> "cudaMemcpyAsync"; "cuLaunchKernel_ptsz" -> "cuLaunchKernel"
std::string api_base(const char * fn) {
    std::string s = fn ? fn : "?";
    for (;;) {
        const size_t u = s.rfind('_');
        if (u == std::string::npos || u == 0) break;
        const std::string tail = s.substr(u + 1);
        const bool ver = tail.size() > 1 && tail[0] == 'v' && std::all_of(tail.begin() + 1, tail.end(), ::isdigit);
        if (ver || tail == "ptsz" || tail == "ptds") s.resize(u); else break;
    }
    return s;
}

bool starts(const std::string & s, const char * p) { return s.compare(0, strlen(p), p) == 0; }
bool has(const std::string & s, const char * p) { return s.find(p) != std::string::npos; }

const char * api_category(const std::string & n) {
    if (has(n, "Synchronize")) return "sync";
    if (starts(n, "cudaMalloc") || starts(n, "cudaFree") || starts(n, "cudaHostAlloc") || starts(n, "cudaHostRegister") ||
        starts(n, "cudaHostUnregister") || starts(n, "cuMemAlloc") || starts(n, "cuMemFree") || starts(n, "cuMemHostAlloc") ||
        starts(n, "cuMemHostRegister") || starts(n, "cuMemCreate") || starts(n, "cuMemRelease") || starts(n, "cuMemHostUnregister"))
        return "alloc";
    if (starts(n, "cudaLaunch") || starts(n, "cuLaunch") || starts(n, "cudaGraphLaunch") || starts(n, "cuGraphLaunch")) return "launch";
    if (starts(n, "cudaMemcpy") || starts(n, "cuMemcpy") || starts(n, "cudaMemset") || starts(n, "cuMemset")) {
        // non-async copies block the host until the copy (and prior work in the stream) is done
        if (has(n, "Memset")) return "copy";
        return has(n, "Async") ? "copy" : "sync";
    }
    return "other";
}

bool is_work_api(const char * fn) {
    if (!fn) return false;
    if (fn[0] != 'c' || fn[1] != 'u') return false;
    const char * p = fn + 2;
    if (*p == 'd' && p[1] == 'a') p += 2;   // cuda...
    return !strncmp(p, "Launch", 6) || !strncmp(p, "Memcpy", 6) || !strncmp(p, "Memset", 6) || !strncmp(p, "GraphLaunch", 11) ||
           !strncmp(p, "MemPrefetch", 11);
}

// union of busy intervals, with a coarse timeline
struct Busy {
    std::vector<std::pair<double, double>> pend;
    std::vector<std::pair<double, double>> merged;   // sorted, disjoint (kept for the per-phase sweep)
    double gap_eps = 0;                              // gaps shorter than this were closed to bound memory
    double total = 0;
    std::vector<float> bins;
    double w = 0.01;
    static constexpr size_t k_cap = 100000;
    void add(double s, double e) {
        if (e > s) pend.emplace_back(s, e);
        if (pend.size() >= (1u << 20)) flush();
    }
    void bin(double s, double e) {
        if (e <= 0) return;
        s = std::max(s, 0.0);
        while ((size_t) (e / w) + 1 > k_cap) {   // coarsen: merge pairs
            std::vector<float> nb((bins.size() + 1) / 2, 0.f);
            for (size_t i = 0; i < bins.size(); ++i) nb[i / 2] += bins[i];
            bins.swap(nb);
            w *= 2;
        }
        const size_t b1 = (size_t) (e / w);
        if (bins.size() <= b1) bins.resize(b1 + 1, 0.f);
        for (size_t b = (size_t) (s / w); b <= b1; ++b) {
            const double lo = std::max(s, b * w), hi = std::min(e, (b + 1) * w);
            if (hi > lo) bins[b] += (float) (hi - lo);
        }
    }
    void flush() {
        if (pend.empty()) return;
        std::sort(pend.begin(), pend.end());
        double cs = pend[0].first, ce = pend[0].second;
        for (size_t i = 1; i <= pend.size(); ++i) {
            if (i < pend.size() && pend[i].first <= ce) { ce = std::max(ce, pend[i].second); continue; }
            // clip against intervals already counted (late records overlapping an earlier flush)
            if (!merged.empty() && cs < merged.back().second) cs = std::min(ce, merged.back().second);
            if (ce > cs) {
                total += ce - cs;
                bin(cs, ce);
                if (!merged.empty() && cs - merged.back().second <= gap_eps) merged.back().second = ce;
                else merged.emplace_back(cs, ce);
            }
            if (i < pend.size()) { cs = pend[i].first; ce = pend[i].second; }
        }
        pend.clear();
        while (merged.size() > (4u << 20)) {   // bound memory: close gaps below a growing threshold
            gap_eps = gap_eps > 0 ? gap_eps * 2 : 1e-6;
            std::vector<std::pair<double, double>> m;
            for (auto & iv : merged)
                if (!m.empty() && iv.first - m.back().second <= gap_eps) m.back().second = iv.second; else m.push_back(iv);
            merged.swap(m);
        }
    }
};

struct KStat { uint64_t n = 0; double t = 0, tmax = 0, tmin = 1e30; };
struct CStat { uint64_t n = 0, bytes = 0; double t = 0; };
struct AStat { uint64_t n = 0; double t = 0; };

class CuptiBackend;
CuptiBackend * g_self = nullptr;
thread_local int t_depth = 0;
thread_local double t_api_enter = 0;
thread_local const char * t_api_fn = nullptr;
thread_local bool t_tool_call = false;   // our own driver calls (pointer queries)

class CuptiBackend final : public GpuBackend {
  public:
    Api a{};
    Driver d{};
    void * hcupti = nullptr;
    void * hcuda = nullptr;
    std::string cupti_path, status = "ok";
    uint32_t cupti_version = 0;
    int driver_version = 0, visible = 0;
    double t0 = 0;           // host time origin (collector's MPI_Init time)
    int64_t ts_offset = 0;   // cupti_ns - mono_ns
    bool api_trace = true;
    SubscriberHandle sub = nullptr;

    std::atomic<uint32_t> cur_phase{0};
    std::vector<std::pair<double, uint32_t>> phase_log;   // (host time rel. t0, innermost phase from then on)
    bool phase_log_full = false;
    std::map<uint32_t, double> phase_busy;                 // phase -> GPU busy time while the host was in it
    std::atomic<uint32_t> cur_label{0};
    static constexpr size_t k_ring = 1u << 20;
    std::vector<std::atomic<uint64_t>> ring;   // corr(32) | phase(20) | label(11) | in_mpi(1)

    std::mutex mu;
    std::vector<std::string> labels{""};        // Kokkos kernel labels, id 0 = none
    std::unordered_map<std::string, uint32_t> label_ids;
    std::vector<std::string> knames;
    std::unordered_map<std::string, uint32_t> kname_ids;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, KStat> kernels;          // phase, label, name
    std::map<std::tuple<uint32_t, uint8_t, uint8_t, uint8_t, bool>, CStat> copies;   // phase, kind, src, dst, by_mpi
    std::map<uint32_t, CStat> memsets;
    std::map<std::pair<uint32_t, std::string>, AStat> apis;
    std::map<uint32_t, Busy> kbusy, cbusy;     // per device: kernels, copies
    std::map<uint32_t, std::pair<double, double>> dev_time;   // device -> kernel time, copy time
    uint64_t unattributed = 0, records = 0, incomplete = 0;
    size_t dropped = 0;
    double api_time_total = 0;

    CuptiBackend() : ring(k_ring) {}

    const char * name() const override { return "cupti"; }

    uint32_t phase_of(uint32_t corr, uint32_t * label, bool * in_mpi) {
        const uint64_t v = ring[corr & (k_ring - 1)].load(std::memory_order_relaxed);
        if (corr == 0 || (uint32_t) (v >> 32) != corr) { unattributed++; *label = 0; *in_mpi = false; return 0; }
        *label = (uint32_t) ((v >> 1) & 0x7ff);
        *in_mpi = v & 1;
        return (uint32_t) ((v >> 12) & 0xfffff);
    }

    double host_time(uint64_t ts) const { return ((double) ((int64_t) ts - ts_offset)) * 1e-9 - t0; }

    uint32_t kname_id(const char * n) {
        std::string s = n ? n : "?";
        auto it = kname_ids.find(s);
        if (it != kname_ids.end()) return it->second;
        const uint32_t id = (uint32_t) knames.size();
        knames.push_back(s);
        kname_ids.emplace(s, id);
        return id;
    }

    void process(uint8_t * buf, size_t valid) {
        std::lock_guard<std::mutex> lk(mu);
        Activity * r = nullptr;
        while (a.ActivityGetNextRecord(buf, valid, &r) == 0) {
            records++;
            uint32_t label = 0;
            bool in_mpi = false;
            if (r->kind == KIND_CONCURRENT_KERNEL || r->kind == KIND_KERNEL) {
                auto * k = (ActivityKernel *) r;
                if (k->end <= k->start) { incomplete++; continue; }   // forced flush: kernel still running
                const uint32_t ph = phase_of(k->correlationId, &label, &in_mpi);
                const double s = host_time(k->start), e = host_time(k->end), dt = (double) (k->end - k->start) * 1e-9;
                KStat & st = kernels[{ph, label, kname_id(k->name)}];
                st.n++; st.t += dt; st.tmax = std::max(st.tmax, dt); st.tmin = std::min(st.tmin, dt);
                kbusy[k->deviceId].add(s, e);
                dev_time[k->deviceId].first += dt;
            } else if (r->kind == KIND_MEMCPY) {
                auto * c = (ActivityMemcpy *) r;
                if (c->end < c->start) { incomplete++; continue; }
                const uint32_t ph = phase_of(c->correlationId, &label, &in_mpi);
                const double dt = (double) (c->end - c->start) * 1e-9;
                CStat & st = copies[{ph, c->copyKind, c->srcKind, c->dstKind, in_mpi}];
                st.n++; st.bytes += c->bytes; st.t += dt;
                cbusy[c->deviceId].add(host_time(c->start), host_time(c->end));
                dev_time[c->deviceId].second += dt;
            } else if (r->kind == KIND_MEMSET) {
                auto * m = (ActivityMemset *) r;
                if (m->end < m->start) { incomplete++; continue; }
                const uint32_t ph = phase_of(m->correlationId, &label, &in_mpi);
                CStat & st = memsets[ph];
                st.n++; st.bytes += m->bytes; st.t += (double) (m->end - m->start) * 1e-9;
            }
        }
    }

    static void buf_request(uint8_t ** buffer, size_t * size, size_t * max_records) {
        constexpr size_t sz = 8u << 20;
        *buffer = (uint8_t *) aligned_alloc(8, sz);
        *size = *buffer ? sz : 0;
        *max_records = 0;
    }
    static void buf_complete(CUcontext ctx, uint32_t stream, uint8_t * buffer, size_t, size_t valid) {
        if (g_self && buffer && valid) {
            g_self->process(buffer, valid);
            size_t dr = 0;
            if (g_self->a.ActivityGetNumDroppedRecords && g_self->a.ActivityGetNumDroppedRecords(ctx, stream, &dr) == 0 && dr) {
                std::lock_guard<std::mutex> lk(g_self->mu);
                g_self->dropped += dr;
            }
        }
        free(buffer);
    }

    static void api_callback(void *, uint32_t domain, uint32_t, const void * cbdata) {
        CuptiBackend * b = g_self;
        if (!b || (domain != DOMAIN_RUNTIME_API && domain != DOMAIN_DRIVER_API) || t_tool_call) return;
        auto * cd = (const CallbackData *) cbdata;
        if (cd->callbackSite == API_ENTER) {
            if (t_depth++ == 0) { t_api_enter = mono(); t_api_fn = cd->functionName; }
            if (is_work_api(cd->functionName)) {
                const uint64_t v = ((uint64_t) cd->correlationId << 32) | ((uint64_t) (b->cur_phase.load(std::memory_order_relaxed) & 0xfffff) << 12) |
                                   ((uint64_t) (b->cur_label.load(std::memory_order_relaxed) & 0x7ff) << 1) | (vibeprof_t_in_mpi ? 1u : 0u);
                b->ring[cd->correlationId & (k_ring - 1)].store(v, std::memory_order_relaxed);
            }
        } else if (t_depth > 0 && --t_depth == 0) {
            if (vibeprof_t_in_mpi) return;   // CUDA calls made by the MPI library count as MPI time
            const double dt = mono() - t_api_enter;
            const std::string n = api_base(t_api_fn);
            std::lock_guard<std::mutex> lk(b->mu);
            AStat & st = b->apis[{b->cur_phase.load(std::memory_order_relaxed), n}];
            st.n++;
            st.t += dt;
            b->api_time_total += dt;
        }
    }

    bool load(std::string * why) {
        const char * cl = getenv("VIBEPROF_CUDA_LIB");
        hcuda = dlopen(cl && *cl ? cl : "libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!hcuda) { *why = "no CUDA driver (libcuda.so.1)"; return false; }
#define LD(field, sym) d.field = (decltype(d.field)) dlsym(hcuda, sym)
        LD(Init, "cuInit"); LD(DriverGetVersion, "cuDriverGetVersion"); LD(DeviceGetCount, "cuDeviceGetCount");
        LD(DeviceGet, "cuDeviceGet"); LD(DeviceGetName, "cuDeviceGetName"); LD(DeviceGetPCIBusId, "cuDeviceGetPCIBusId");
        LD(PointerGetAttribute, "cuPointerGetAttribute");
#undef LD
        if (!d.Init || !d.DeviceGetCount || d.Init(0) != 0) { *why = "cuInit failed"; return false; }
        if (d.DeviceGetCount(&visible) != 0 || visible <= 0) { *why = "no CUDA devices visible"; return false; }
        if (d.DriverGetVersion) d.DriverGetVersion(&driver_version);

        std::vector<std::string> cand;
        if (const char * p = getenv("VIBEPROF_CUPTI_LIB")) cand.push_back(p);
        for (const char * env : {"CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDATOOLKIT_HOME", "NVHPC_CUDA_HOME"})
            if (const char * h = getenv(env)) {
                cand.push_back(std::string(h) + "/extras/CUPTI/lib64/libcupti.so");
                cand.push_back(std::string(h) + "/lib64/libcupti.so");
            }
        // next to the application's libcudart, if dynamically linked
        for (const char * rt : {"libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so"}) {
            if (void * h = dlopen(rt, RTLD_NOW | RTLD_NOLOAD)) {
                Dl_info di;
                void * sym = dlsym(h, "cudaRuntimeGetVersion");
                if (sym && dladdr(sym, &di) && di.dli_fname) {
                    std::string dir = di.dli_fname;
                    dir = dir.substr(0, dir.rfind('/'));
                    cand.push_back(dir + "/libcupti.so");
                    cand.push_back(dir + "/../extras/CUPTI/lib64/libcupti.so");
                }
                dlclose(h);
            }
        }
        for (const char * n : {"libcupti.so", "libcupti.so.13", "libcupti.so.12", "libcupti.so.11.8", "libcupti.so.11.7",
                               "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so", "/usr/local/cuda/lib64/libcupti.so"})
            cand.push_back(n);
        for (const auto & c : cand)
            if ((hcupti = dlopen(c.c_str(), RTLD_NOW | RTLD_GLOBAL))) { cupti_path = c; break; }
        if (!hcupti) { *why = "libcupti not found (set VIBEPROF_CUPTI_LIB or CUDA_HOME)"; return false; }
#define LS(field, sym) a.field = (decltype(a.field)) dlsym(hcupti, sym)
        LS(GetVersion, "cuptiGetVersion"); LS(ActivityEnable, "cuptiActivityEnable"); LS(ActivityDisable, "cuptiActivityDisable");
        LS(ActivityRegisterCallbacks, "cuptiActivityRegisterCallbacks"); LS(ActivityGetNextRecord, "cuptiActivityGetNextRecord");
        LS(ActivityFlushAll, "cuptiActivityFlushAll"); LS(ActivityGetNumDroppedRecords, "cuptiActivityGetNumDroppedRecords");
        LS(GetTimestamp, "cuptiGetTimestamp"); LS(Subscribe, "cuptiSubscribe"); LS(Unsubscribe, "cuptiUnsubscribe");
        LS(EnableDomain, "cuptiEnableDomain"); LS(GetResultString, "cuptiGetResultString");
#undef LS
        if (!a.ActivityEnable || !a.ActivityRegisterCallbacks || !a.ActivityGetNextRecord || !a.ActivityFlushAll || !a.GetTimestamp) {
            *why = "libcupti lacks the activity API: " + cupti_path;
            return false;
        }
        if (a.GetVersion) a.GetVersion(&cupti_version);
        return true;
    }

    std::string err(CUptiResult r) {
        const char * s = nullptr;
        if (a.GetResultString) a.GetResultString(r, &s);
        return s ? s : ("CUPTI error " + std::to_string(r));
    }

    void on_init(int, double t0_host) override {
        t0 = t0_host;
        phase_log.emplace_back(0.0, 0);   // the host starts in "<main>"
        uint64_t ts = 0;
        const double m0 = mono();
        a.GetTimestamp(&ts);
        const double m1 = mono();
        ts_offset = (int64_t) ts - (int64_t) llround((m0 + m1) * 0.5e9);
        g_self = this;
        CUptiResult r = a.ActivityRegisterCallbacks(buf_request, buf_complete);
        if (r != 0) { status = "activity callbacks: " + err(r); return; }
        r = a.ActivityEnable(KIND_CONCURRENT_KERNEL);
        if (r != 0) { status = "kernel activity: " + err(r); return; }
        a.ActivityEnable(KIND_MEMCPY);
        a.ActivityEnable(KIND_MEMSET);
        const char * e = getenv("VIBEPROF_GPU_API");
        api_trace = !(e && !strcmp(e, "0"));
        if (api_trace && a.Subscribe && a.EnableDomain) {
            r = a.Subscribe(&sub, api_callback, this);
            if (r != 0) {
                status = "API tracing unavailable (" + err(r) + "; another CUPTI tool attached?): kernels not attributed to phases";
                api_trace = false;
                sub = nullptr;
            } else {
                a.EnableDomain(1, sub, DOMAIN_RUNTIME_API);
                a.EnableDomain(1, sub, DOMAIN_DRIVER_API);
            }
        }
    }

    void on_phase_change(uint32_t phase) override {
        cur_phase.store(phase, std::memory_order_relaxed);
        if (phase_log.size() < (2u << 20)) phase_log.emplace_back(mono() - t0, phase);
        else phase_log_full = true;
    }

    // GPU busy time (union over this rank's devices) while the host was in each phase
    void sweep_phases() {
        std::vector<std::pair<double, double>> iv;
        for (auto & [dev, b] : kbusy) iv.insert(iv.end(), b.merged.begin(), b.merged.end());
        std::sort(iv.begin(), iv.end());
        std::vector<std::pair<double, double>> u;
        for (auto & x : iv) if (!u.empty() && x.first <= u.back().second) u.back().second = std::max(u.back().second, x.second); else u.push_back(x);
        if (phase_log.empty()) phase_log.emplace_back(0.0, 0);
        size_t j = 0;
        for (size_t k = 0; k < phase_log.size(); ++k) {
            const double a = phase_log[k].first;
            const double b = k + 1 < phase_log.size() ? phase_log[k + 1].first : 1e300;
            while (j < u.size() && u[j].second <= a) ++j;
            double busy = 0;
            for (size_t m = j; m < u.size() && u[m].first < b; ++m) busy += std::min(b, u[m].second) - std::max(a, u[m].first);
            if (busy > 0) phase_busy[phase_log[k].second] += busy;
        }
    }

    void on_kernel_label(const char * label) override {
        if (!label) { cur_label.store(0, std::memory_order_relaxed); return; }
        std::lock_guard<std::mutex> lk(mu);
        auto it = label_ids.find(label);
        uint32_t id;
        if (it != label_ids.end()) id = it->second;
        else if (labels.size() < 2047) { id = (uint32_t) labels.size(); labels.push_back(label); label_ids.emplace(label, id); }
        else id = 0;
        cur_label.store(id, std::memory_order_relaxed);
    }

    bool is_device_pointer(const void * p) override {
        if (!p || !d.PointerGetAttribute) return false;
        unsigned mt = 0;
        t_tool_call = true;
        const CUresult r = d.PointerGetAttribute(&mt, POINTER_ATTRIBUTE_MEMORY_TYPE, (unsigned long long) (uintptr_t) p);
        t_tool_call = false;
        return r == 0 && (mt == MEMORYTYPE_DEVICE || mt == MEMORYTYPE_UNIFIED);
    }

    void on_finalize() override {
        if (a.ActivityFlushAll) a.ActivityFlushAll(FLUSH_FORCED);
        if (sub && a.Unsubscribe) { a.Unsubscribe(sub); sub = nullptr; }
        std::lock_guard<std::mutex> lk(mu);
        for (auto & [dev, b] : kbusy) b.flush();
        for (auto & [dev, b] : cbusy) b.flush();
        sweep_phases();
    }

    void write_json(FILE * f) override {
        std::lock_guard<std::mutex> lk(mu);
        fprintf(f, ", \"status\": "); jstr(f, status);
        fprintf(f, ", \"cupti_version\": %u, \"driver_version\": %d, \"cupti_lib\": ", cupti_version, driver_version); jstr(f, cupti_path);
        fprintf(f, ", \"visible_devices\": %d, \"api_trace\": %s, \"records\": %llu, \"dropped\": %zu, \"unattributed\": %llu, \"incomplete\": %llu, \"api_time\": %.9f",
                visible, api_trace ? "true" : "false", (unsigned long long) records, dropped, (unsigned long long) unattributed,
                (unsigned long long) incomplete, api_time_total);
        // devices
        std::set<uint32_t> devs;
        for (auto & [k, v] : dev_time) devs.insert(k);
        fprintf(f, ",\n  \"devices\": [");
        size_t i = 0;
        for (uint32_t id : devs) {
            char nm[256] = "", pci[64] = "";
            CUdevice dv = 0;
            t_tool_call = true;
            if (d.DeviceGet && d.DeviceGet(&dv, (int) id) == 0) {
                if (d.DeviceGetName) d.DeviceGetName(nm, sizeof(nm), dv);
                if (d.DeviceGetPCIBusId) d.DeviceGetPCIBusId(pci, sizeof(pci), dv);
            }
            t_tool_call = false;
            const Busy & kb = kbusy[id];
            const Busy & cb = cbusy[id];
            fprintf(f, "%s{\"id\": %u, \"name\": ", i++ ? ", " : "", id); jstr(f, nm);
            fprintf(f, ", \"pci\": "); jstr(f, pci);
            fprintf(f, ", \"kernel_time\": %.9f, \"copy_time\": %.9f, \"kernel_busy\": %.9f, \"copy_busy\": %.9f, \"timeline_bin\": %.6f, \"timeline\": [",
                    dev_time[id].first, dev_time[id].second, kb.total, cb.total, kb.w);
            // at most ~400 points: busy fraction of the GPU per bin
            const size_t n = kb.bins.size(), step = std::max<size_t>(1, (n + 399) / 400);
            for (size_t b = 0; b < n; b += step) {
                double s = 0;
                for (size_t j = b; j < std::min(n, b + step); ++j) s += kb.bins[j];
                fprintf(f, "%s%.3f", b ? "," : "", std::min(1.0, s / (kb.w * step)));
            }
            fprintf(f, "], \"timeline_step\": %.6f}", kb.w * step);
        }
        fprintf(f, "],\n  \"phase_busy\": [");
        i = 0;
        for (auto & [ph, t] : phase_busy) fprintf(f, "%s[%u, %.9f]", i++ ? ", " : "", ph, t);
        double eps = 0;
        for (auto & [dev, b] : kbusy) eps = std::max(eps, b.gap_eps);
        fprintf(f, "], \"phase_busy_partial\": %s, \"busy_gap_eps\": %.9f", phase_log_full ? "true" : "false", eps);
        fprintf(f, ",\n  \"kernels\": [");
        i = 0;
        for (auto & [k, s] : kernels) {
            fprintf(f, "%s\n    {\"phase\": %u, \"name\": ", i++ ? "," : "", std::get<0>(k)); jstr(f, knames[std::get<2>(k)]);
            fprintf(f, ", \"label\": "); jstr(f, labels[std::get<1>(k)]);
            fprintf(f, ", \"count\": %llu, \"time\": %.9f, \"tmax\": %.9f, \"tmin\": %.9f}", (unsigned long long) s.n, s.t, s.tmax, s.tmin);
        }
        fprintf(f, "],\n  \"copies\": [");
        i = 0;
        for (auto & [k, s] : copies) {
            fprintf(f, "%s\n    {\"phase\": %u, \"kind\": \"%s\", \"src\": \"%s\", \"dst\": \"%s\", \"by_mpi\": %s, \"count\": %llu, \"bytes\": %llu, \"time\": %.9f}",
                    i++ ? "," : "", std::get<0>(k), copy_kind_name(std::get<1>(k)), mem_kind_name(std::get<2>(k)), mem_kind_name(std::get<3>(k)),
                    std::get<4>(k) ? "true" : "false", (unsigned long long) s.n, (unsigned long long) s.bytes, s.t);
        }
        fprintf(f, "],\n  \"memsets\": [");
        i = 0;
        for (auto & [ph, s] : memsets)
            fprintf(f, "%s{\"phase\": %u, \"count\": %llu, \"bytes\": %llu, \"time\": %.9f}", i++ ? ", " : "", ph,
                    (unsigned long long) s.n, (unsigned long long) s.bytes, s.t);
        fprintf(f, "],\n  \"api\": [");
        i = 0;
        for (auto & [k, s] : apis) {
            fprintf(f, "%s\n    {\"phase\": %u, \"name\": ", i++ ? "," : "", k.first); jstr(f, k.second);
            fprintf(f, ", \"cat\": \"%s\", \"count\": %llu, \"time\": %.9f}", api_category(k.second), (unsigned long long) s.n, s.t);
        }
        fprintf(f, "]");
    }
};

}  // namespace

thread_local bool vibeprof_t_in_mpi = false;

GpuBackend * vibeprof_make_gpu_backend(std::string * why) {
    const char * m = getenv("VIBEPROF_GPU");
    const std::string mode = m && *m ? m : "auto";
    if (mode == "none" || mode == "0" || mode == "off") { *why = ""; return nullptr; }
    auto * b = new CuptiBackend();
    std::string reason;
    if (!b->load(&reason)) {
        const bool gpus = b->visible > 0;
        delete b;
        // auto mode on a CPU-only node is the normal case: stay quiet; warn when GPUs are present
        *why = (mode == "auto" && !gpus) ? "" : "GPU backend disabled: " + reason;
        return nullptr;
    }
    return b;
}
