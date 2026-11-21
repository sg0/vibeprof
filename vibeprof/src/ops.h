// Operation ids and kinds generated from the MPI call tables.
#pragma once
#include <mpi.h>
#include <cstdint>

enum OpKind : int { K_SEND, K_RECV, K_SENDRECV, K_WAIT, K_COLL, K_ICOLL, K_PROBE, K_RMA, K_RMASYNC, K_MARKER_ };

enum OpId : int {
#define X(NAME, PARAMS, ARGS, COMM, BYTES, PEER, KIND) OP_##NAME,
#include "mpi_calls_send.def"
#include "mpi_calls.def"
#undef X
    OP_COUNT_
};

static const char * const k_op_names[] = {
#define X(NAME, PARAMS, ARGS, COMM, BYTES, PEER, KIND) #NAME,
#include "mpi_calls_send.def"
#include "mpi_calls.def"
#undef X
};

static const int k_op_kinds[] = {
#define X(NAME, PARAMS, ARGS, COMM, BYTES, PEER, KIND) KIND,
#include "mpi_calls_send.def"
#include "mpi_calls.def"
#undef X
};

static const char * const k_kind_names[] = {"send", "recv", "sendrecv", "wait", "collective", "nb-collective", "probe", "rma", "rma-sync"};

static inline int op_by_name(const char * name) {
    for (int i = 0; i < OP_COUNT_; ++i) {
        const char * a = k_op_names[i];
        const char * b = name;
        if (b[0] == 'M' && b[1] == 'P' && b[2] == 'I' && b[3] == '_') b += 4;   // accept "MPI_Allreduce" or "Allreduce"
        int j = 0;
        while (a[j] && b[j] && (a[j] | 32) == (b[j] | 32)) ++j;
        if (!a[j] && !b[j]) return i;
    }
    return -1;
}
