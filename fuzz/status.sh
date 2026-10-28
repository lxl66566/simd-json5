#!/usr/bin/env bash
# Print per-target campaign status for run-fuzz.sh runs.
#
# Usage: status.sh [TARGET ...]     (defaults to all targets)
#       FUZZ_BUDGET=<seconds> status.sh   (budget for remaining-time math)
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
budget=${FUZZ_BUDGET:-18000}
targets=${*:-"parse de_dom_parity typed_shapes"}

total_execs=0
for t in $targets; do
  acc=$(cat "$root/fuzz/.runs/$t/accumulated" 2>/dev/null || echo 0)
  remaining=$((budget > acc ? budget - acc : 0))
  workers=$(pgrep -cx "$t" 2>/dev/null)
  workers=${workers:-0}

  execs=0
  latest=""
  for log in "$root"/fuzz/logs/"$t"-g*-w*.log; do
    [ -e "$log" ] || continue
    n=$(grep -aoE '^#[0-9]+' "$log" | tail -1 | tr -d '#')
    [ -n "${n:-}" ] && execs=$((execs + n))
    latest=$log
  done
  cov=$(grep -aoE 'cov: [0-9]+ ft: [0-9]+' "$latest" 2>/dev/null | tail -1)
  corpus_n=$(ls "$root/fuzz/corpus/$t" 2>/dev/null | wc -l)
  corpus_sz=$(du -sh "$root/fuzz/corpus/$t" 2>/dev/null | cut -f1)
  arts=$(find "$root/fuzz/artifacts/$t" -type f 2>/dev/null | wc -l)

  state=stopped
  ((workers > 0)) && state=running
  ((remaining == 0)) && state=done
  ((arts > 0)) && state="$state+ARTIFACTS"
  printf '%s: %s workers=%d acc=%ds rem=%ds execs=%d %s corpus=%s(%s) artifacts=%d\n' \
    "$t" "$state" "$workers" "$acc" "$remaining" "$execs" "${cov:-}" \
    "${corpus_n:-0}" "${corpus_sz:-?}" "$arts"
  total_execs=$((total_execs + execs))
done
echo "total_execs=$total_execs"
free -g | sed -n '2p'
