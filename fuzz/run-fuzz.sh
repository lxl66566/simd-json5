#!/usr/bin/env bash
# Run one fuzzing generation for a single target.
#
# Usage: run-fuzz.sh TARGET [WORKERS] [BUDGET_SECONDS]
#
# Each invocation runs for the target's remaining budget (BUDGET minus the
# wall time already journaled in fuzz/.runs/TARGET/accumulated), so a campaign
# survives stops and restarts — after a crash fix, for example — without
# exceeding the total. Worker output goes to fuzz/logs/TARGET-gGEN-wN.log;
# wall time is appended to fuzz/journal.tsv. Stop a generation with
# `kill -TERM <pid>`; SIGTERM is forwarded to the workers and the elapsed
# time is still journaled.
set -euo pipefail

target=${1:?usage: run-fuzz.sh TARGET [WORKERS] [BUDGET_SECONDS]}
workers=${2:-4}
budget=${3:-18000}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

runs_dir=$root/fuzz/.runs/$target
mkdir -p "$runs_dir" "$root/fuzz/logs" "$root/fuzz/artifacts/$target"

acc_file=$runs_dir/accumulated
gen_file=$runs_dir/generation
acc=$(cat "$acc_file" 2>/dev/null || echo 0)
gen=$(cat "$gen_file" 2>/dev/null || echo 0)
remaining=$((budget - acc))
if (( remaining <= 0 )); then
  echo "run-fuzz: budget exhausted for $target (${acc}s/${budget}s)" >&2
  exit 0
fi
gen=$((gen + 1))
echo "$gen" >"$gen_file"

start=$(date +%s)
pids=()
finished=0
finish() {
  (( finished )) && return 0
  finished=1
  trap - TERM INT
  if ((${#pids[@]})); then
    kill -TERM -- "${pids[@]/#/-}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  local end
  end=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' "$start" "$end" "$target" "g$gen" >>"$root/fuzz/journal.tsv"
  echo $((acc + end - start)) >"$acc_file"
}
trap 'finish; exit 143' TERM
trap 'finish; exit 130' INT

# Each worker runs in its own process group (via setsid) so `finish` can take
# down the whole tree: killing only the `cargo fuzz run` wrapper would orphan
# the fuzz binary, which keeps running until its own budget expires.
# The small ASan quarantine keeps per-worker RSS low enough to run several
# targets' workers side by side; `-rss_limit_mb` caps bloated workers.
for i in $(seq 1 "$workers"); do
  ASAN_OPTIONS=quarantine_size_mb=32 setsid cargo fuzz run "$target" -- \
    "-dict=$root/fuzz/dict/json5.dict" \
    "-artifact_prefix=$root/fuzz/artifacts/$target/" \
    "-max_total_time=$remaining" -max_len=8192 -timeout=5 -rss_limit_mb=1600 \
    -use_value_profile=1 -print_final_stats=1 "-seed=$((RANDOM * 1000 + i))" \
    >"$root/fuzz/logs/$target-g$gen-w$i.log" 2>&1 &
  pids+=($!)
done
wait || true
finish
