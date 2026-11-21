/* Toy SPMD program with known behavior, for validating vibeprof.
 *
 *   setup:  computation + MPI_Bcast
 *   loop "step" (N iterations):
 *     "compute": fixed amount of arithmetic (optionally rank-dependent: --imbalance X adds X*rank/size)
 *     "halo":    ring exchange with both neighbors (Isend/Irecv + Waitall), --halo-bytes each way
 *     "reduce":  MPI_Allreduce of one double
 *   final: MPI_Gather of a result
 *
 * usage: toy [--iters N] [--work W] [--setup-work W0] [--halo-bytes B] [--imbalance X] [--regrid-every K]
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vibeprof.h"

static double work(long units) {
    double s = 0.0;
    for (long i = 0; i < units * 1000; ++i) s += sqrt((double) (i % 1000) + 1.0);
    return s;
}

int main(int argc, char ** argv) {
    int iters = 50, halo = 64 * 1024, regrid = 0;
    long w = 2000, setup_w = -1;
    double imb = 0.0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--work") && i + 1 < argc) w = atol(argv[++i]);
        else if (!strcmp(argv[i], "--halo-bytes") && i + 1 < argc) halo = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--imbalance") && i + 1 < argc) imb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--regrid-every") && i + 1 < argc) regrid = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--setup-work") && i + 1 < argc) setup_w = atol(argv[++i]);
    }

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char * sbuf = malloc((size_t) halo * 2), * rbuf = malloc((size_t) halo * 2);
    memset(sbuf, rank, (size_t) halo * 2);
    double acc = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();
    vibeprof_phase_begin("setup");
    acc += work(setup_w >= 0 ? setup_w : w / 2);   /* identical on all ranks */
    double param = rank == 0 ? 3.14 : 0.0;
    MPI_Bcast(&param, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    vibeprof_phase_end("setup");

    const int left = (rank - 1 + size) % size, right = (rank + 1) % size;
    const long my_work = (long) (w * (1.0 + imb * rank / (size > 1 ? size - 1 : 1)));
    for (int it = 0; it < iters; ++it) {
        vibeprof_phase_begin("step");

        vibeprof_phase_begin("compute");
        acc += work(my_work);
        vibeprof_phase_end("compute");

        vibeprof_phase_begin("halo");
        MPI_Request req[4];
        MPI_Irecv(rbuf, halo, MPI_CHAR, left, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(rbuf + halo, halo, MPI_CHAR, right, 2, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(sbuf, halo, MPI_CHAR, right, 1, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(sbuf + halo, halo, MPI_CHAR, left, 2, MPI_COMM_WORLD, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        vibeprof_phase_end("halo");

        vibeprof_phase_begin("reduce");
        double local = acc * 1e-9, global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vibeprof_phase_end("reduce");

        if (regrid > 0 && (it + 1) % regrid == 0) vibeprof_mark("regrid");
        vibeprof_phase_end("step");
    }

    double res = acc, * all = rank == 0 ? malloc(sizeof(double) * size) : NULL;
    MPI_Gather(&res, 1, MPI_DOUBLE, all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { printf("toy: %d ranks, %d iterations, %.4f s\n", size, iters, MPI_Wtime() - t_start); free(all); }
    free(sbuf);
    free(rbuf);
    MPI_Finalize();
    return 0;
}
