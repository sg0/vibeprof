/* No-op phase markers. Link applications with -lvibeprof_stub so the markers can stay in the code
 * permanently; run with LD_PRELOAD=libvibeprof.so to profile (the preloaded library overrides these). */
#include "vibeprof.h"

void vibeprof_phase_begin(const char * name) { (void) name; }
void vibeprof_phase_end(const char * name) { (void) name; }
void vibeprof_mark(const char * name) { (void) name; }
int vibeprof_enabled(void) { return 0; }
