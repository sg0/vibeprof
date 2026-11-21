// fakecuda: simulated CUDA runtime/driver + CUPTI subset (see fakecuda.h).
#include "fakecuda.h"
#include "../../src/cupti_abi.h"

#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using namespace cupti;

namespace {

constexpr size_t k_slot = 256;   // every record occupies one fixed slot in a buffer

int64_t now_ns() {   // CUPTI timestamps: deliberately CLOCK_REALTIME (the backend calibrates)
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void wait_until(int64_t t) {   // sleep, then spin the last 300 us (sleep overshoot would distort timings)
    for (;;) {
        const int64_t d = t - now_ns();
        if (d <= 0) return;
        if (d > 300000) {
            const int64_t sl = d - 300000;
            timespec ts{(time_t) (sl / 1000000000LL), (long) (sl % 1000000000LL)};
            nanosleep(&ts, nullptr);
        }
    }
}

void spin_ns(int64_t d) { const int64_t e = now_ns() + d; while (now_ns() < e) {} }

struct Fake {
    std::recursive_mutex mu;
    int ndev = 1, dev = 0;
    std::map<std::pair<int, int>, int64_t> free_at;       // (device, stream) -> time
    std::map<uintptr_t, std::pair<size_t, int>> dmem;     // device allocations: base -> (size, device)
    std::map<uintptr_t, size_t> pinned;
    std::set<std::string> names;
    uint32_t corr = 0;
    // CUPTI state
    BufferRequestFn req = nullptr;
    BufferCompleteFn done = nullptr;
    CallbackFn cb = nullptr;
    void * cb_user = nullptr;
    bool dom_rt = false, dom_drv = false;
    bool en_kernel = false, en_memcpy = false, en_memset = false;
    uint8_t * buf = nullptr;
    size_t buf_size = 0, buf_used = 0;
    int calls_in_api = 0;
};
Fake & F() { static Fake * f = new Fake(); return *f; }

int env_int(const char * n, int d) { const char * v = getenv(n); return v ? atoi(v) : d; }

void ensure_init() {
    static bool once = false;
    if (once) return;
    once = true;
    F().ndev = std::max(1, env_int("FAKECUDA_NDEV", 1));
}

// --- CUPTI delivery ---
void flush_buffer() {
    Fake & f = F();
    if (f.buf && f.buf_used && f.done) {
        uint8_t * b = f.buf;
        const size_t used = f.buf_used, size = f.buf_size;
        f.buf = nullptr;
        f.buf_used = 0;
        f.done(nullptr, 0, b, size, used);
    }
}

void * slot() {
    Fake & f = F();
    if (!f.req) return nullptr;
    if (f.buf && f.buf_used + k_slot > f.buf_size) flush_buffer();
    if (!f.buf) {
        size_t sz = 0, maxr = 0;
        f.req(&f.buf, &sz, &maxr);
        f.buf_size = std::min<size_t>(sz, 64 * k_slot);   // small buffers: exercise many deliveries
        f.buf_used = 0;
        if (!f.buf) return nullptr;
    }
    void * p = f.buf + f.buf_used;
    memset(p, 0, k_slot);
    f.buf_used += k_slot;
    return p;
}

// one API call: runtime ENTER (+ nested driver ENTER for launches/copies), body, EXITs
struct ApiScope {
    CallbackData rt{}, drv{};
    bool nested;
    uint32_t drv_corr = 0;
    ApiScope(const char * rt_name, const char * drv_name) : nested(drv_name != nullptr) {
        Fake & f = F();
        rt.functionName = rt_name;
        rt.correlationId = ++f.corr;
        rt.callbackSite = API_ENTER;
        if (f.cb && f.dom_rt) f.cb(f.cb_user, DOMAIN_RUNTIME_API, 1, &rt);
        if (nested) {
            drv.functionName = drv_name;
            drv.correlationId = drv_corr = ++f.corr;
            drv.callbackSite = API_ENTER;
            if (f.cb && f.dom_drv) f.cb(f.cb_user, DOMAIN_DRIVER_API, 1, &drv);
        }
    }
    ~ApiScope() {
        Fake & f = F();
        if (nested) { drv.callbackSite = API_EXIT; if (f.cb && f.dom_drv) f.cb(f.cb_user, DOMAIN_DRIVER_API, 1, &drv); }
        rt.callbackSite = API_EXIT;
        if (f.cb && f.dom_rt) f.cb(f.cb_user, DOMAIN_RUNTIME_API, 1, &rt);
    }
    // device work is correlated with the innermost (driver) call
    uint32_t work_corr() const { return nested ? drv_corr : rt.correlationId; }
};

int64_t enqueue(int stream, int64_t dur_ns) {   // returns start; stream busy until start+dur
    Fake & f = F();
    int64_t & fa = f.free_at[{f.dev, stream}];
    const int64_t s = std::max(now_ns(), fa);
    fa = s + dur_ns;
    return s;
}

int64_t device_free() {
    Fake & f = F();
    int64_t t = 0;
    for (auto & [k, v] : f.free_at) if (k.first == f.dev) t = std::max(t, v);
    return t;
}

uint8_t mem_kind(const void * p) {
    Fake & f = F();
    const uintptr_t a = (uintptr_t) p;
    auto it = f.dmem.upper_bound(a);
    if (it != f.dmem.begin()) { --it; if (a < it->first + it->second.first) return MEM_DEVICE; }
    auto jt = f.pinned.upper_bound(a);
    if (jt != f.pinned.begin()) { --jt; if (a < jt->first + jt->second) return MEM_PINNED; }
    return MEM_PAGEABLE;
}

double bandwidth(uint8_t src, uint8_t dst) {   // bytes/s
    if (src == MEM_DEVICE && dst == MEM_DEVICE) return 800e9;
    if (src == MEM_PAGEABLE || dst == MEM_PAGEABLE) return 6e9;
    return 22e9;
}

uint8_t copy_kind(uint8_t src, uint8_t dst) {
    const bool sd = src == MEM_DEVICE, dd = dst == MEM_DEVICE;
    return sd && dd ? MEMCPY_DTOD : sd ? MEMCPY_DTOH : dd ? MEMCPY_HTOD : MEMCPY_HTOH;
}

int do_copy(void * dst, const void * src, size_t bytes, int stream, bool async, const ApiScope & api) {
    Fake & f = F();
    memmove(dst, src, bytes);
    const uint8_t sk = mem_kind(src), dk = mem_kind(dst);
    const int64_t dur = 2000 + (int64_t) (bytes / bandwidth(sk, dk) * 1e9);   // 2 us setup
    const int64_t s = enqueue(stream, dur);
    if (f.en_memcpy) {
        if (auto * r = (ActivityMemcpy *) slot()) {
            r->kind = KIND_MEMCPY; r->copyKind = copy_kind(sk, dk); r->srcKind = sk; r->dstKind = dk;
            r->bytes = bytes; r->start = (uint64_t) s; r->end = (uint64_t) (s + dur);
            r->deviceId = (uint32_t) f.dev; r->streamId = (uint32_t) stream; r->correlationId = api.work_corr();
        }
    }
    // synchronous copies, and any copy touching pageable host memory, block the host
    if (!async || sk == MEM_PAGEABLE || dk == MEM_PAGEABLE) wait_until(s + dur);
    return 0;
}

}  // namespace

extern "C" {

// ---------------- simulated runtime ----------------
int fcudaGetDeviceCount(int * n) { ensure_init(); *n = F().ndev; return 0; }

int fcudaSetDevice(int dev) {
    ensure_init();
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaSetDevice_v3020", nullptr);
    if (dev < 0 || dev >= F().ndev) return 101;   // cudaErrorInvalidDevice
    F().dev = dev;
    return 0;
}

int fcudaMalloc(void ** p, size_t bytes) {
    ensure_init();
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaMalloc_v3020", "cuMemAlloc_v2");
    spin_ns(40000);   // allocation cost
    *p = aligned_alloc(256, (bytes + 255) / 256 * 256);
    F().dmem[(uintptr_t) *p] = {bytes, F().dev};
    return *p ? 0 : 2;
}

int fcudaFree(void * p) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaFree_v3020", "cuMemFree_v2");
    wait_until(device_free());   // cudaFree synchronizes the device
    spin_ns(20000);
    F().dmem.erase((uintptr_t) p);
    free(p);
    return 0;
}

int fcudaMallocHost(void ** p, size_t bytes) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaMallocHost_v3020", "cuMemAllocHost_v2");
    spin_ns(100000);
    *p = aligned_alloc(4096, (bytes + 4095) / 4096 * 4096);
    F().pinned[(uintptr_t) *p] = bytes;
    return *p ? 0 : 2;
}

int fcudaFreeHost(void * p) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaFreeHost_v3020", "cuMemFreeHost");
    F().pinned.erase((uintptr_t) p);
    free(p);
    return 0;
}

int fcudaLaunch(const char * name, double seconds, int stream) {
    Fake & f = F();
    std::lock_guard<std::recursive_mutex> lk(f.mu);
    ApiScope api("cudaLaunchKernel_v7000", "cuLaunchKernel");
    spin_ns(4000);   // launch overhead on the host
    const int64_t dur = std::max<int64_t>(1500, (int64_t) (seconds * 1e9));
    const int64_t s = enqueue(stream, dur);
    if (f.en_kernel) {
        if (auto * r = (ActivityKernel *) slot()) {
            r->kind = KIND_CONCURRENT_KERNEL;
            r->start = (uint64_t) s; r->end = (uint64_t) (s + dur); r->completed = r->end;
            r->deviceId = (uint32_t) f.dev; r->streamId = (uint32_t) stream;
            r->gridX = 1024; r->gridY = r->gridZ = 1; r->blockX = 256; r->blockY = r->blockZ = 1;
            r->correlationId = api.work_corr();
            r->name = f.names.insert(name ? name : "kernel").first->c_str();
        }
    }
    return 0;
}

int fcudaMemcpy(void * dst, const void * src, size_t bytes, int) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaMemcpy_v3020", "cuMemcpy");
    wait_until(device_free());   // the legacy default stream waits for all prior work
    return do_copy(dst, src, bytes, 0, false, api);
}

int fcudaMemcpyAsync(void * dst, const void * src, size_t bytes, int, int stream) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaMemcpyAsync_v3020", "cuMemcpyAsync");
    return do_copy(dst, src, bytes, stream, true, api);
}

int fcudaMemset(void * p, int v, size_t bytes) {
    (void) p;
    Fake & f = F();
    std::lock_guard<std::recursive_mutex> lk(f.mu);
    ApiScope api("cudaMemset_v3020", "cuMemsetD8_v2");
    (void) v;   // contents of simulated device memory are irrelevant; a real memset runs on the GPU, not the host
    const int64_t dur = 1500 + (int64_t) (bytes / 1000e9 * 1e9);
    const int64_t s = enqueue(0, dur);
    if (f.en_memset)
        if (auto * r = (ActivityMemset *) slot()) {
            r->kind = KIND_MEMSET; r->value = (uint32_t) v; r->bytes = bytes; r->start = (uint64_t) s; r->end = (uint64_t) (s + dur);
            r->deviceId = (uint32_t) f.dev; r->correlationId = api.work_corr(); r->memoryKind = MEM_DEVICE;
        }
    return 0;
}

int fcudaDeviceSynchronize(void) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaDeviceSynchronize_v3020", "cuCtxSynchronize");
    wait_until(device_free());
    return 0;
}

int fcudaStreamSynchronize(int stream) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    ApiScope api("cudaStreamSynchronize_v3020", "cuStreamSynchronize");
    wait_until(F().free_at[{F().dev, stream}]);
    return 0;
}

// ---------------- driver API subset ----------------
int cuInit(unsigned) { ensure_init(); return 0; }
int cuDriverGetVersion(int * v) { *v = 12040; return 0; }
int cuDeviceGetCount(int * n) { ensure_init(); *n = F().ndev; return 0; }
int cuDeviceGet(int * d, int ordinal) { if (ordinal < 0 || ordinal >= F().ndev) return 101; *d = ordinal; return 0; }
int cuDeviceGetName(char * name, int len, int dev) { snprintf(name, (size_t) len, "Fake GPU %d (simulated)", dev); return 0; }
int cuDeviceGetPCIBusId(char * bus, int len, int dev) { snprintf(bus, (size_t) len, "0000:%02X:00.0", 0x41 + dev); return 0; }
int cuPointerGetAttribute(void * data, int attr, unsigned long long ptr) {
    if (attr != POINTER_ATTRIBUTE_MEMORY_TYPE) return 1;
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    const uint8_t k = mem_kind((const void *) (uintptr_t) ptr);
    if (k == MEM_DEVICE) { *(unsigned *) data = MEMORYTYPE_DEVICE; return 0; }
    if (k == MEM_PINNED) { *(unsigned *) data = MEMORYTYPE_HOST; return 0; }
    return 1;   // CUDA_ERROR_INVALID_VALUE: not a CUDA pointer
}

// ---------------- CUPTI subset ----------------
int cuptiGetVersion(uint32_t * v) { *v = 24; return 0; }
int cuptiGetTimestamp(uint64_t * t) { *t = (uint64_t) now_ns(); return 0; }
int cuptiGetResultString(int r, const char ** s) { *s = r ? "CUPTI_ERROR (fake)" : "CUPTI_SUCCESS"; return 0; }
int cuptiActivityRegisterCallbacks(BufferRequestFn rq, BufferCompleteFn dn) { F().req = rq; F().done = dn; return 0; }
int cuptiActivityEnable(uint32_t k) {
    Fake & f = F();
    if (k == KIND_CONCURRENT_KERNEL || k == KIND_KERNEL) f.en_kernel = true;
    else if (k == KIND_MEMCPY) f.en_memcpy = true;
    else if (k == KIND_MEMSET) f.en_memset = true;
    return 0;
}
int cuptiActivityDisable(uint32_t) { return 0; }
int cuptiActivityGetNextRecord(uint8_t * buffer, size_t valid, Activity ** rec) {
    uint8_t * next = *rec ? (uint8_t *) *rec + k_slot : buffer;
    if (next + k_slot > buffer + valid) return 3;   // CUPTI_ERROR_MAX_LIMIT_REACHED
    *rec = (Activity *) next;
    return 0;
}
int cuptiActivityFlushAll(uint32_t) {
    std::lock_guard<std::recursive_mutex> lk(F().mu);
    flush_buffer();
    return 0;
}
int cuptiActivityGetNumDroppedRecords(void *, uint32_t, size_t * d) { *d = 0; return 0; }
int cuptiSubscribe(void ** sub, CallbackFn fn, void * user) {
    if (getenv("FAKECUDA_SUBSCRIBE_FAIL")) return 39;   // CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
    F().cb = fn; F().cb_user = user; *sub = (void *) 1; return 0;
}
int cuptiUnsubscribe(void *) { F().cb = nullptr; return 0; }
int cuptiEnableDomain(uint32_t en, void *, uint32_t dom) {
    if (dom == DOMAIN_RUNTIME_API) F().dom_rt = en;
    if (dom == DOMAIN_DRIVER_API) F().dom_drv = en;
    return 0;
}

}  // extern "C"
