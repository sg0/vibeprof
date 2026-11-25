#!/usr/bin/env bash
# Apply the split-state patch to a llama.cpp checkout (optional: enables --kv-transport device).
#
#   patches/apply.sh [LLAMA_CPP_DIR]        (default: ../llama.cpp)
#
# Safe to run more than once: an already patched tree is detected and left alone. If the patch
# does not apply cleanly (llama.cpp moved on), a 3-way merge is tried. vibeprof-llm also builds
# against an unpatched llama.cpp; only --kv-transport device is then unavailable.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch="$here/llama.cpp-state-seq-split.patch"
dir="${1:-$here/../../llama.cpp}"
[[ -f "$dir/include/llama.h" ]] || { echo "not a llama.cpp checkout: $dir" >&2; exit 2; }
cd "$dir"
rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if grep -q "LLAMA_STATE_SEQ_SPLIT_API" include/llama.h; then
  echo "already applied in $dir ($rev); nothing to do"
  exit 0
fi
rm -f include/llama.h.rej src/llama-context.cpp.rej src/llama-context.h.rej   # leftovers of a failed attempt
if git apply --check "$patch" 2>/dev/null; then
  git apply "$patch"
  echo "applied to $dir ($rev)"
elif git apply --3way "$patch"; then
  echo "applied to $dir ($rev) with a 3-way merge; check 'git diff' and rebuild"
else
  echo "the patch does not apply to llama.cpp $rev." >&2
  echo "Undo any partial changes (git -C $dir checkout -- include src), then either build without it" >&2
  echo "(vibeprof-llm works; only --kv-transport device is unavailable) or check out a tested" >&2
  echo "llama.cpp revision listed in README.md." >&2
  exit 1
fi
