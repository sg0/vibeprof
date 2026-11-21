/* Generated from mpi_calls*.def: the user buffer of each wrapped MPI call (for GPU-aware MPI
 * detection: is the buffer device memory?). nullptr when the call has no data buffer. */
#pragma once
#define LLMP_BUF_Send                   (const void *) buf
#define LLMP_BUF_Isend                  (const void *) buf
#define LLMP_BUF_Ssend                  (const void *) buf
#define LLMP_BUF_Rsend                  (const void *) buf
#define LLMP_BUF_Bsend                  (const void *) buf
#define LLMP_BUF_Issend                 (const void *) buf
#define LLMP_BUF_Irsend                 (const void *) buf
#define LLMP_BUF_Ibsend                 (const void *) buf
#define LLMP_BUF_Recv                   (const void *) buf
#define LLMP_BUF_Irecv                  (const void *) buf
#define LLMP_BUF_Sendrecv               (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Sendrecv_replace       (const void *) buf
#define LLMP_BUF_Probe                  nullptr
#define LLMP_BUF_Iprobe                 nullptr
#define LLMP_BUF_Wait                   nullptr
#define LLMP_BUF_Waitall                nullptr
#define LLMP_BUF_Waitany                nullptr
#define LLMP_BUF_Waitsome               nullptr
#define LLMP_BUF_Test                   nullptr
#define LLMP_BUF_Testall                nullptr
#define LLMP_BUF_Testany                nullptr
#define LLMP_BUF_Testsome               nullptr
#define LLMP_BUF_Barrier                nullptr
#define LLMP_BUF_Bcast                  (const void *) buffer
#define LLMP_BUF_Reduce                 (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Allreduce              (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Gather                 (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Gatherv                (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Scatter                (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Scatterv               (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Allgather              (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Allgatherv             (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Alltoall               (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Alltoallv              (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Reduce_scatter         (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Reduce_scatter_block   (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Scan                   (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Exscan                 (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Neighbor_alltoall      (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Ibarrier               nullptr
#define LLMP_BUF_Ibcast                 (const void *) buffer
#define LLMP_BUF_Ireduce                (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Iallreduce             (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Iallgather             (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Ialltoall              (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Ialltoallv             (sendbuf == MPI_IN_PLACE ? (const void *) recvbuf : (const void *) sendbuf)
#define LLMP_BUF_Put                    (const void *) origin_addr
#define LLMP_BUF_Get                    (const void *) origin_addr
#define LLMP_BUF_Accumulate             (const void *) origin_addr
#define LLMP_BUF_Fetch_and_op           (const void *) origin_addr
#define LLMP_BUF_Win_fence              nullptr
#define LLMP_BUF_Win_flush              nullptr
#define LLMP_BUF_Win_lock               nullptr
#define LLMP_BUF_Win_unlock             nullptr
