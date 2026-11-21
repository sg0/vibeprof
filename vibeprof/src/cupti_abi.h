// Minimal CUPTI / CUDA driver ABI used by the vibeprof GPU backend.
//
// vibeprof does not compile against the CUDA toolkit: libcupti and libcuda are opened at run time,
// so one libvibeprof.so works on CPU-only systems and with CUDA 11, 12 and 13. This header
// declares only the stable C ABI we use: function signatures, enum values and the leading fields
// of the activity records. CUPTI keeps old record versions and only appends fields to new ones;
// the layouts below follow the CUPTI documentation (CUpti_ActivityKernel4 prefix of all later
// kernel records, CUpti_ActivityMemcpy5/6, CUpti_ActivityMemset4/5, CUpti_CallbackData).
#pragma once
#include <cstddef>
#include <cstdint>

namespace cupti {

using CUptiResult = int;          // CUPTI_SUCCESS = 0
using CUresult = int;             // CUDA_SUCCESS = 0
using CUcontext = void *;
using CUdevice = int;
using SubscriberHandle = void *;

enum : uint32_t {
    KIND_MEMCPY = 1,
    KIND_MEMSET = 2,
    KIND_KERNEL = 3,
    KIND_DRIVER = 4,
    KIND_RUNTIME = 5,
    KIND_CONCURRENT_KERNEL = 10,
    KIND_OVERHEAD = 17,
};

enum : uint32_t { FLUSH_FORCED = 1 };

enum : uint32_t {
    DOMAIN_DRIVER_API = 1,
    DOMAIN_RUNTIME_API = 2,
    DOMAIN_RESOURCE = 3,
};

enum : uint32_t { API_ENTER = 0, API_EXIT = 1 };

// CUpti_ActivityMemcpyKind
enum : uint8_t {
    MEMCPY_UNKNOWN = 0, MEMCPY_HTOD = 1, MEMCPY_DTOH = 2, MEMCPY_HTOA = 3, MEMCPY_ATOH = 4, MEMCPY_ATOA = 5,
    MEMCPY_ATOD = 6, MEMCPY_DTOA = 7, MEMCPY_DTOD = 8, MEMCPY_HTOH = 9, MEMCPY_PTOP = 10,
};

// CUpti_ActivityMemoryKind
enum : uint8_t {
    MEM_UNKNOWN = 0, MEM_PAGEABLE = 1, MEM_PINNED = 2, MEM_DEVICE = 3, MEM_ARRAY = 4, MEM_MANAGED = 5,
    MEM_DEVICE_STATIC = 6, MEM_MANAGED_STATIC = 7,
};

#define LLMP_PACKED __attribute__((packed, aligned(8)))

struct LLMP_PACKED Activity {     // CUpti_Activity
    uint32_t kind;
};

struct LLMP_PACKED ActivityKernel {   // leading fields of CUpti_ActivityKernel4 .. Kernel13
    uint32_t kind;
    uint8_t  cacheConfig;
    uint8_t  sharedMemoryConfig;
    uint16_t registersPerThread;
    uint32_t partitionedGlobalCacheRequested;
    uint32_t partitionedGlobalCacheExecuted;
    uint64_t start;
    uint64_t end;
    uint64_t completed;
    uint32_t deviceId;
    uint32_t contextId;
    uint32_t streamId;
    int32_t  gridX, gridY, gridZ;
    int32_t  blockX, blockY, blockZ;
    int32_t  staticSharedMemory;
    int32_t  dynamicSharedMemory;
    uint32_t localMemoryPerThread;
    uint32_t localMemoryTotal;
    uint32_t correlationId;
    int64_t  gridId;
    const char * name;
};
static_assert(offsetof(ActivityKernel, start) == 16, "kernel.start");
static_assert(offsetof(ActivityKernel, deviceId) == 40, "kernel.deviceId");
static_assert(offsetof(ActivityKernel, gridX) == 52, "kernel.gridX");
static_assert(offsetof(ActivityKernel, correlationId) == 92, "kernel.correlationId");
static_assert(offsetof(ActivityKernel, name) == 104, "kernel.name");

struct LLMP_PACKED ActivityMemcpy {   // leading fields of CUpti_ActivityMemcpy .. Memcpy6
    uint32_t kind;
    uint8_t  copyKind;
    uint8_t  srcKind;
    uint8_t  dstKind;
    uint8_t  flags;
    uint64_t bytes;
    uint64_t start;
    uint64_t end;
    uint32_t deviceId;
    uint32_t contextId;
    uint32_t streamId;
    uint32_t correlationId;
    uint32_t runtimeCorrelationId;
};
static_assert(offsetof(ActivityMemcpy, bytes) == 8, "memcpy.bytes");
static_assert(offsetof(ActivityMemcpy, deviceId) == 32, "memcpy.deviceId");
static_assert(offsetof(ActivityMemcpy, correlationId) == 44, "memcpy.correlationId");

struct LLMP_PACKED ActivityMemset {   // leading fields of CUpti_ActivityMemset .. Memset5
    uint32_t kind;
    uint32_t value;
    uint64_t bytes;
    uint64_t start;
    uint64_t end;
    uint32_t deviceId;
    uint32_t contextId;
    uint32_t streamId;
    uint32_t correlationId;
    uint16_t flags;
    uint16_t memoryKind;
};
static_assert(offsetof(ActivityMemset, correlationId) == 44, "memset.correlationId");

struct CallbackData {             // CUpti_CallbackData (natural alignment)
    uint32_t     callbackSite;
    const char * functionName;
    const void * functionParams;
    void *       functionReturnValue;
    const char * symbolName;
    CUcontext    context;
    uint32_t     contextUid;
    uint64_t *   correlationData;
    uint32_t     correlationId;
};

using BufferRequestFn = void (*)(uint8_t ** buffer, size_t * size, size_t * maxNumRecords);
using BufferCompleteFn = void (*)(CUcontext ctx, uint32_t streamId, uint8_t * buffer, size_t size, size_t validSize);
using CallbackFn = void (*)(void * userdata, uint32_t domain, uint32_t cbid, const void * cbdata);

struct Api {   // filled with dlsym
    CUptiResult (*GetVersion)(uint32_t *);
    CUptiResult (*ActivityEnable)(uint32_t);
    CUptiResult (*ActivityDisable)(uint32_t);
    CUptiResult (*ActivityRegisterCallbacks)(BufferRequestFn, BufferCompleteFn);
    CUptiResult (*ActivityGetNextRecord)(uint8_t *, size_t, Activity **);
    CUptiResult (*ActivityFlushAll)(uint32_t);
    CUptiResult (*ActivityGetNumDroppedRecords)(CUcontext, uint32_t, size_t *);
    CUptiResult (*GetTimestamp)(uint64_t *);
    CUptiResult (*Subscribe)(SubscriberHandle *, CallbackFn, void *);
    CUptiResult (*Unsubscribe)(SubscriberHandle);
    CUptiResult (*EnableDomain)(uint32_t, SubscriberHandle, uint32_t);
    CUptiResult (*GetResultString)(CUptiResult, const char **);
};

// CUDA driver API (pointer and device queries)
enum : int { POINTER_ATTRIBUTE_MEMORY_TYPE = 2 };
enum : unsigned { MEMORYTYPE_HOST = 1, MEMORYTYPE_DEVICE = 2, MEMORYTYPE_ARRAY = 3, MEMORYTYPE_UNIFIED = 4 };

struct Driver {
    CUresult (*Init)(unsigned);
    CUresult (*DriverGetVersion)(int *);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetPCIBusId)(char *, int, CUdevice);
    CUresult (*PointerGetAttribute)(void *, int, unsigned long long);
};

}  // namespace cupti
