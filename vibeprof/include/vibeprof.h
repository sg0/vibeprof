/*
 * vibeprof: MPI profiling with phase markers (C API).
 *
 * Link with -lvibeprof (markers + MPI wrappers), or link -lvibeprof_stub (no-op markers) and
 * enable profiling at run time with LD_PRELOAD=libvibeprof.so. Profiling is on whenever
 * libvibeprof is loaded, unless VIBEPROF=0.
 *
 * Markers are per rank, nestable, and do no communication. Calls made before MPI_Init or
 * after MPI_Finalize are ignored.
 */
#ifndef VIBEPROF_H
#define VIBEPROF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Begin / end a named phase. Phases nest: "solver" containing "halo" is reported as "solver/halo".
 * vibeprof_phase_end() should name the innermost open phase; a mismatch is recorded as a warning. */
void vibeprof_phase_begin(const char * name);
void vibeprof_phase_end(const char * name);

/* Record a one-off event (e.g. "regrid", "checkpoint"): counted and time-stamped. */
void vibeprof_mark(const char * name);

/* 1 if profiling is active in this process (library loaded, MPI initialized, VIBEPROF != 0). */
int vibeprof_enabled(void);

#ifdef __cplusplus
}

/* C++ convenience: RAII phase scope. */
struct vibeprof_scope {
    const char * name_;
    explicit vibeprof_scope(const char * name) : name_(name) { vibeprof_phase_begin(name_); }
    ~vibeprof_scope() { vibeprof_phase_end(name_); }
    vibeprof_scope(const vibeprof_scope &) = delete;
    vibeprof_scope & operator=(const vibeprof_scope &) = delete;
};
#define VIBEPROF_CAT2_(a, b) a##b
#define VIBEPROF_CAT_(a, b) VIBEPROF_CAT2_(a, b)
#define VIBEPROF_SCOPE(name) vibeprof_scope VIBEPROF_CAT_(vibeprof_scope_, __LINE__)(name)
#endif

#endif /* VIBEPROF_H */
