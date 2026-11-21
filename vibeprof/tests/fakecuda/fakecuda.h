/* fakecuda: a simulated CUDA device + CUPTI for testing vibeprof's GPU backend without a GPU.
 *
 * Each process gets FAKECUDA_NDEV (default 1) devices. Work is modelled in time only: a kernel or
 * copy occupies its stream from max(now, stream free) for its duration; synchronizing calls block
 * the host until the stream is free; copies with pageable memory block the host as in CUDA.
 * Memory is ordinary host memory (so MPI can send "device" buffers), but cuPointerGetAttribute
 * reports fcudaMalloc memory as device memory. Every call goes through the CUPTI callback API with
 * real CUDA function names, and activity records are delivered through the CUPTI buffer API.
 *
 * Use: VIBEPROF_CUDA_LIB=VIBEPROF_CUPTI_LIB=libfakecuda.so; the application links libfakecuda.so.
 */
#ifndef FAKECUDA_H
#define FAKECUDA_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { FCUDA_H2D = 1, FCUDA_D2H = 2, FCUDA_D2D = 3, FCUDA_DEFAULT = 4 };

int fcudaGetDeviceCount(int * n);
int fcudaSetDevice(int dev);
int fcudaMalloc(void ** p, size_t bytes);
int fcudaFree(void * p);
int fcudaMallocHost(void ** p, size_t bytes);
int fcudaFreeHost(void * p);
/* a kernel that runs `seconds` on the device */
int fcudaLaunch(const char * name, double seconds, int stream);
int fcudaMemcpy(void * dst, const void * src, size_t bytes, int kind);
int fcudaMemcpyAsync(void * dst, const void * src, size_t bytes, int kind, int stream);
int fcudaMemset(void * p, int v, size_t bytes);
int fcudaDeviceSynchronize(void);
int fcudaStreamSynchronize(int stream);

#ifdef __cplusplus
}
#endif
#endif
