// GPU measurement backend interface.
//
// A backend is created once per rank at MPI_Init (vibeprof_make_gpu_backend). The collector tells it
// when the active phase changes, so device work can be attributed to the phase that launched it,
// and when a Kokkos kernel label is active. MPI wrappers ask whether a buffer is device memory
// (GPU-aware MPI). At MPI_Finalize the backend flushes and appends members to the rank's "gpu" object.
//
// Backends: gpu_cupti.cpp (NVIDIA, CUPTI loaded at run time). Planned: rocprofiler-sdk (AMD),
// Level Zero (Intel).
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

struct GpuBackend {
    virtual ~GpuBackend() = default;
    virtual const char * name() const = 0;
    virtual void on_init(int /*world_rank*/, double /*t0_host*/) {}
    virtual void on_phase_change(uint32_t /*phase_id*/) {}
    // Kokkos (or other model) kernel label for the device work launched next; nullptr clears it.
    virtual void on_kernel_label(const char * /*label*/) {}
    virtual bool is_device_pointer(const void * /*p*/) { return false; }
    // Write members of the "gpu" JSON object, each starting with ", " (the collector writes "backend").
    virtual void write_json(FILE * /*f*/) {}
    virtual void on_finalize() {}
};

// Returns a backend for this process (VIBEPROF_GPU=auto|cuda|none), or nullptr. *why receives a
// message for the profile's warnings when a requested or expected backend could not start.
GpuBackend * vibeprof_make_gpu_backend(std::string * why);

// True while the calling thread is inside a wrapped MPI call (device work issued by the MPI
// library itself, e.g. staging copies, is flagged and its API time is left to MPI).
extern thread_local bool vibeprof_t_in_mpi;
