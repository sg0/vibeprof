#!/usr/bin/env bash
# Fetch, mark up and build the ECP proxy apps used to validate vibeprof.
#
#   apps/build.sh [WORKDIR] [minife] [quicksilver] [amg]     (default WORKDIR=./proxy, all apps)
#
# Each app is pinned to the commit the patch was made against, gets phase markers from
# apps/<app>/*.patch, and links libvibeprof_stub (no-op markers), so the binaries run normally
# and are profiled by preloading libvibeprof.so (scripts/vibeprof-srun). Needs mpicc/mpicxx in
# PATH and vibeprof built (top-level build, standalone vibeprof/build, or VIBEPROF_BUILD=...).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="$(cd "$here/.." && pwd)"
# profiler build: the top-level build (<repo>/build/vibeprof) or a standalone one (vibeprof/build)
vibeprof_build() {
  local d
  for d in "$1/../build/vibeprof" "$1/build"; do
    [[ -f "$d/libvibeprof.so" ]] && { (cd "$d" && pwd); return; }
  done
  echo "$(cd "$1/.." && pwd)/build/vibeprof"
}
B="${VIBEPROF_BUILD:-$(vibeprof_build "$L")}"
work="${1:-proxy}"; shift || true
apps=("$@"); [[ ${#apps[@]} -gt 0 ]] || apps=(minife quicksilver amg)
mkdir -p "$work"; work="$(cd "$work" && pwd)"
[[ -f "$B/libvibeprof_stub.so" ]] || { echo "build vibeprof first (missing $B/libvibeprof_stub.so)" >&2; exit 2; }
STUB="-L$B -lvibeprof_stub -Wl,-rpath,$B"
JOBS="${JOBS:-$(nproc)}"

fetch() { # url dir commit patch
  if [[ ! -d "$work/$2" ]]; then git clone -q "$1" "$work/$2"; fi
  git -C "$work/$2" checkout -q "$3"
  git -C "$work/$2" checkout -q -- .
  git -C "$work/$2" apply "$4"
}

for app in "${apps[@]}"; do
  case "$app" in
    minife)
      fetch https://github.com/Mantevo/miniFE.git miniFE c79e0f7 "$here/minife/minife-vibeprof.patch"
      make -C "$work/miniFE/ref/src" -j"$JOBS" CXX=mpicxx CC=mpicc \
        CPPFLAGS="-I. -I../utils -I../fem -I$L/include -DMINIFE_SCALAR=double -DMINIFE_LOCAL_ORDINAL=int -DMINIFE_GLOBAL_ORDINAL=int -DMINIFE_CSR_MATRIX -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK" \
        LIBS="$STUB"
      echo "built $work/miniFE/ref/src/miniFE.x" ;;
    quicksilver)
      fetch https://github.com/LLNL/Quicksilver.git Quicksilver eb68bb8 "$here/quicksilver/quicksilver-vibeprof.patch"
      d="$work/Quicksilver/src"
      # the upstream link line puts LDFLAGS before the objects; --no-as-needed keeps the stub library
      make -C "$d" -j"$JOBS" CXX=mpicxx CXXFLAGS="-g -O2 -include cstdint" CPPFLAGS="-DHAVE_MPI -I$L/include" \
        LDFLAGS="-Wl,--no-as-needed $STUB" qs >/dev/null
      echo "built $d/qs" ;;
    amg)
      fetch https://github.com/LLNL/AMG.git AMG 3ada8a1 "$here/amg/amg-vibeprof.patch"
      # MPI only (no OpenMP, no persistent requests: vibeprof does not intercept MPI_Start* yet)
      sed -i "s|^INCLUDE_CFLAGS = .*|INCLUDE_CFLAGS = -O2 -g -DTIMER_USE_MPI -DHYPRE_HOPSCOTCH -DHYPRE_BIGINT -I$L/include|; s|^INCLUDE_LFLAGS = .*|INCLUDE_LFLAGS = -lm $STUB|" \
        "$work/AMG/Makefile.include"
      make -C "$work/AMG" -j"$JOBS" >/dev/null
      echo "built $work/AMG/test/amg" ;;
    *) echo "unknown app $app" >&2; exit 2 ;;
  esac
done
