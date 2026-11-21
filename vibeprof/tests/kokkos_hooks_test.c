/* Calls the Kokkos Tools hooks the way Kokkos does (resolved at run time, as Kokkos does with
 * KOKKOS_TOOLS_LIBS), launching simulated kernels inside a labeled parallel_for and a profile region.
 * Expect: phase "solver" and "solver/assemble", kernels labeled "Assemble::matrix" / "Dot". */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include "fakecuda.h"

typedef void (*push_t)(const char *);
typedef void (*pop_t)(void);
typedef void (*begin_t)(const char *, uint32_t, uint64_t *);
typedef void (*end_t)(uint64_t);

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);
    push_t push = (push_t) dlsym(RTLD_DEFAULT, "kokkosp_push_profile_region");
    pop_t pop = (pop_t) dlsym(RTLD_DEFAULT, "kokkosp_pop_profile_region");
    begin_t bfor = (begin_t) dlsym(RTLD_DEFAULT, "kokkosp_begin_parallel_for");
    end_t efor = (end_t) dlsym(RTLD_DEFAULT, "kokkosp_end_parallel_for");
    begin_t bred = (begin_t) dlsym(RTLD_DEFAULT, "kokkosp_begin_parallel_reduce");
    end_t ered = (end_t) dlsym(RTLD_DEFAULT, "kokkosp_end_parallel_reduce");
    if (!push || !pop || !bfor || !efor || !bred || !ered) { fprintf(stderr, "kokkos hooks not found (preload libvibeprof.so)\n"); return 1; }
    for (int it = 0; it < 20; ++it) {
        push("solver");
        push("assemble");
        uint64_t k;
        bfor("Assemble::matrix", 0, &k);
        fcudaLaunch("_ZN6Kokkos4Impl33cuda_parallel_launch_local_memoryINS0_11ParallelForI7FunctorEEEEvT_", 2e-3, 0);
        efor(k);
        pop();
        bred("Dot", 0, &k);
        fcudaLaunch("_ZN6Kokkos4Impl33cuda_parallel_launch_local_memoryINS0_14ParallelReduceI3DotEEEEvT_", 5e-4, 0);
        ered(k);
        fcudaDeviceSynchronize();
        double x = 1, y;
        MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        pop();
    }
    MPI_Finalize();
    return 0;
}
