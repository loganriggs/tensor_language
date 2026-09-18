#!/bin/bash
# BQGATE: LIBRARY -- check -> enqueue -> wait -> show, failing fast (review 27). Usage: dod_run_wait.sh <runner.py> [max_wait_s] [FORCE=1 env honoured]
set -u
runner="${1:?usage: dod_run_wait.sh <runner.py> [max_wait_s]}"; maxw="${2:-600}"
ops=/workspace/tensor_language/basis_aligned/bilinear_quotient/ops; base=$(basename "$runner" .py); root=$(dirname "$ops")
/venv/main/bin/python "$ops/dod_check_runner.py" "$runner" || { echo "CHECK FAILED: $runner"; exit 1; }
out=$(bash "$ops/enqueue.sh" "$(readlink -f "$runner")" 2>&1) || { echo "$out"; echo "ENQUEUE FAILED"; exit 1; }
echo "$out" | grep -q QUEUED || { echo "$out"; echo "NOT QUEUED"; exit 1; }
receipt=$(grep -o 'circuits/followups/[A-Za-z0-9_]*_result\.json' "$runner" | head -1); [ -n "$receipt" ] || { echo "no receipt path found in runner"; exit 1; }
log="$root/runlogs/$base.log"; t0=$(date +%s); rm -f "$root/$receipt.__stale" 2>/dev/null
while :; do
  [ -f "$root/$receipt" ] && [ "$root/$receipt" -nt "$runner" ] && break
  if [ -f "$log" ] && [ "$log" -nt "$runner" ] && grep -q "Traceback\|price exceeded" "$log"; then echo "RUN FAILED ($log):"; tail -4 "$log" | cut -c1-300; exit 2; fi
  [ $(( $(date +%s) - t0 )) -ge "$maxw" ] && { echo "TIMEOUT waiting for $receipt"; exit 3; }
  sleep 10
done
/venv/main/bin/python "$ops/dod_show.py" "$root/$receipt" 3
