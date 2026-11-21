// gpu_toy: an MPI + GPU program with known behavior, for validating vibeprof's GPU analysis.
//
// Both are built by CMake: build/gpu_toy (real GPU, when nvcc is found) and build/gpu_toy_sim
// (simulated device, tests/fakecuda). By hand:
//   nvcc -O2 -ccbin mpicxx tests/gpu_toy.cu -Iinclude -Lbuild -lvibeprof_stub -o gpu_toy
//   mpicxx -x c++ -O2 -DVIBEPROF_FAKECUDA tests/gpu_toy.cu -Iinclude -Itests/fakecuda -Lbuild -lfakecuda -lvibeprof_stub
//
// Each step: phase "compute" (stencil kernel), "halo" (exchange with ring neighbors, staged through
// host memory or GPU-aware), "reduce" (dot kernel + MPI_Allreduce). Switches inject problems:
//   --work-us U         kernel time per step (default 2000)
//   --gpu-imbalance X   kernel time scales by 1 + X*rank/(size-1)
//   --tiny K            split the stencil into K tiny kernels (launch-bound)
//   --sync-each         cudaDeviceSynchronize after every kernel
//   --host-us U         host-only work per step while the GPU idles (phase "host")
//   --gpu-aware         pass device buffers to MPI (needs CUDA-aware MPI on a real system)
//   --pageable          host staging buffers from malloc instead of cudaMallocHost
//   --alloc-each        cudaMalloc/cudaFree a scratch buffer every step
//   --bind zero|local   all ranks use device 0, or device local_rank % ndev (default local)
//   --halo-kb K         halo size (default 512 KiB)
//   --iters N           steps (default 100)
#include <mpi.h>
#include <vibeprof.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef VIBEPROF_FAKECUDA
#include "fakecuda.h"
static int dev_count() { int n = 0; fcudaGetDeviceCount(&n); return n; }
static void dev_set(int d) { fcudaSetDevice(d); }
static void * dev_alloc(size_t b) { void * p = nullptr; fcudaMalloc(&p, b); return p; }
static void dev_free(void * p) { fcudaFree(p); }
static void * host_pinned(size_t b) { void * p = nullptr; fcudaMallocHost(&p, b); return p; }
static void host_pinned_free(void * p) { fcudaFreeHost(p); }
static void kernel(const char * name, double sec, double *, size_t) { fcudaLaunch(name, sec, 0); }
static void d2h(void * dst, const void * src, size_t b) { fcudaMemcpy(dst, src, b, FCUDA_D2H); }
static void h2d(void * dst, const void * src, size_t b) { fcudaMemcpy(dst, src, b, FCUDA_H2D); }
static void dev_sync() { fcudaDeviceSynchronize(); }
static void dev_memset(void * p, size_t b) { fcudaMemset(p, 0, b); }
static void calibrate() {}
#else
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
static double g_cycles_per_sec = 1.4e9;
// distinct kernels so profiles show them by name; each spins for a given number of GPU cycles
#define SPIN_KERNEL(NAME)                                                        \
    __global__ void NAME(double * u, size_t n, long long cycles) {             \
        const long long t0 = clock64();                                          \
        const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;          \
        double acc = 0;                                                          \
        while (clock64() - t0 < cycles) acc += 1e-9;                             \
        if (i < n) u[i] += acc * 1e-30;                                          \
    }
SPIN_KERNEL(stencil)
SPIN_KERNEL(pack)
SPIN_KERNEL(unpack)
SPIN_KERNEL(dot)
static int dev_count() { int n = 0; CK(cudaGetDeviceCount(&n)); return n; }
static void dev_set(int d) { CK(cudaSetDevice(d)); }
static void * dev_alloc(size_t b) { void * p = nullptr; CK(cudaMalloc(&p, b)); return p; }
static void dev_free(void * p) { CK(cudaFree(p)); }
static void * host_pinned(size_t b) { void * p = nullptr; CK(cudaMallocHost(&p, b)); return p; }
static void host_pinned_free(void * p) { CK(cudaFreeHost(p)); }
static void kernel(const char * name, double sec, double * u, size_t n) {
    const long long c = (long long) (sec * g_cycles_per_sec);
    // 108 blocks x 256 threads: occupies the GPU without depending on its size
    if (!strcmp(name, "pack")) pack<<<108, 256>>>(u, n, c);
    else if (!strcmp(name, "unpack")) unpack<<<108, 256>>>(u, n, c);
    else if (!strcmp(name, "dot")) dot<<<108, 256>>>(u, n, c);
    else stencil<<<108, 256>>>(u, n, c);
    CK(cudaGetLastError());
}
static void d2h(void * dst, const void * src, size_t b) { CK(cudaMemcpy(dst, src, b, cudaMemcpyDeviceToHost)); }
static void h2d(void * dst, const void * src, size_t b) { CK(cudaMemcpy(dst, src, b, cudaMemcpyHostToDevice)); }
static void dev_sync() { CK(cudaDeviceSynchronize()); }
static void dev_memset(void * p, size_t b) { CK(cudaMemset(p, 0, b)); }
static void calibrate() {   // GPU clock cycles per second, measured with events
    double * u = (double *) dev_alloc(1024 * sizeof(double));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    stencil<<<1, 32>>>(u, 0, 1000000);   // warm-up
    CK(cudaEventRecord(a));
    stencil<<<1, 32>>>(u, 0, 50000000);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    if (ms > 0) g_cycles_per_sec = 50000000.0 / (ms * 1e-3);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    dev_free(u);
}
#endif

static void host_work(double us) {   // busy host computation
    const double t_end = MPI_Wtime() + us * 1e-6;
    volatile double x = 0;
    while (MPI_Wtime() < t_end) for (int i = 0; i < 1000; ++i) x += i * 1e-9;
}

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double work_us = 2000, imb = 0, host_us = 0;
    int tiny = 0, iters = 100, halo_kb = 512;
    bool sync_each = false, gpu_aware = false, pageable = false, alloc_each = false;
    std::string bind = "local";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return i + 1 < argc ? argv[++i] : (char *) "0"; };
        if (a == "--work-us") work_us = atof(nxt());
        else if (a == "--gpu-imbalance") imb = atof(nxt());
        else if (a == "--tiny") tiny = atoi(nxt());
        else if (a == "--sync-each") sync_each = true;
        else if (a == "--host-us") host_us = atof(nxt());
        else if (a == "--gpu-aware") gpu_aware = true;
        else if (a == "--pageable") pageable = true;
        else if (a == "--alloc-each") alloc_each = true;
        else if (a == "--bind") bind = nxt();
        else if (a == "--halo-kb") halo_kb = atoi(nxt());
        else if (a == "--iters") iters = atoi(nxt());
        else { if (!rank) fprintf(stderr, "unknown option %s\n", a.c_str()); MPI_Abort(MPI_COMM_WORLD, 2); }
    }

    // node-local rank from the host name (portable; some MPI builds report each process as its own
    // shared-memory domain with MPI_COMM_TYPE_SHARED)
    char host[MPI_MAX_PROCESSOR_NAME] = {0};
    int hl = 0;
    MPI_Get_processor_name(host, &hl);
    unsigned color = 2166136261u;
    for (int i = 0; i < hl; ++i) color = (color ^ (unsigned char) host[i]) * 16777619u;
    MPI_Comm node;
    MPI_Comm_split(MPI_COMM_WORLD, (int) (color & 0x7fffffff), rank, &node);
    int local = 0;
    MPI_Comm_rank(node, &local);
    const int ndev = dev_count();
    dev_set(bind == "zero" ? 0 : local % ndev);

    vibeprof_phase_begin("setup");
    calibrate();
    const size_t n = 1 << 22;                       // 32 MiB field
    const size_t halo = (size_t) halo_kb * 1024 / sizeof(double);
    double * u = (double *) dev_alloc(n * sizeof(double));
    double * dsend = (double *) dev_alloc(halo * sizeof(double));
    double * drecv = (double *) dev_alloc(halo * sizeof(double));
    double * dred = (double *) dev_alloc(sizeof(double));
    dev_memset(u, n * sizeof(double));
    double * hsend = pageable ? (double *) malloc(halo * sizeof(double)) : (double *) host_pinned(halo * sizeof(double));
    double * hrecv = pageable ? (double *) malloc(halo * sizeof(double)) : (double *) host_pinned(halo * sizeof(double));
    dev_sync();
    vibeprof_phase_end("setup");

    const double my_work = work_us * 1e-6 * (1.0 + (size > 1 ? imb * rank / (size - 1) : 0.0));
    const int right = (rank + 1) % size, left = (rank - 1 + size) % size;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int it = 0; it < iters; ++it) {
        vibeprof_phase_begin("step");

        vibeprof_phase_begin("compute");
        void * scratch = alloc_each ? dev_alloc(8 << 20) : nullptr;
        const int k = tiny > 0 ? tiny : 1;
        for (int j = 0; j < k; ++j) {
            kernel("stencil", my_work / k, u, n);
            if (sync_each) dev_sync();
        }
        if (scratch) dev_free(scratch);
        vibeprof_phase_end("compute");

        if (host_us > 0) {
            vibeprof_phase_begin("host");
            dev_sync();
            host_work(host_us);
            vibeprof_phase_end("host");
        }

        vibeprof_phase_begin("halo");
        kernel("pack", 20e-6, dsend, halo);
        MPI_Request rq[2];
        if (gpu_aware) {
            dev_sync();
            MPI_Irecv(drecv, (int) halo, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &rq[0]);
            MPI_Isend(dsend, (int) halo, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &rq[1]);
            MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
        } else {
            d2h(hsend, dsend, halo * sizeof(double));
            MPI_Irecv(hrecv, (int) halo, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &rq[0]);
            MPI_Isend(hsend, (int) halo, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &rq[1]);
            MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
            h2d(drecv, hrecv, halo * sizeof(double));
        }
        kernel("unpack", 20e-6, drecv, halo);
        vibeprof_phase_end("halo");

        vibeprof_phase_begin("reduce");
        kernel("dot", 50e-6, u, n);
        double local_dot = 0, global_dot = 0;
        d2h(&local_dot, dred, sizeof(double));
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vibeprof_phase_end("reduce");

        vibeprof_phase_end("step");
    }
    dev_sync();
    MPI_Barrier(MPI_COMM_WORLD);
    const double el = MPI_Wtime() - t0;
    if (!rank) printf("gpu_toy: %d ranks, %d devices/node, %d steps, %.4f s\n", size, ndev, iters, el);

    if (pageable) { free(hsend); free(hrecv); } else { host_pinned_free(hsend); host_pinned_free(hrecv); }
    dev_free(u); dev_free(dsend); dev_free(drecv); dev_free(dred);
    MPI_Comm_free(&node);
    MPI_Finalize();
    return 0;
}
