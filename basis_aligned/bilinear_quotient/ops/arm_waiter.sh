#!/bin/bash
# Standard canary-filtered landing waiter (ops lane, 14:20 review).
# Reads the CURRENT last non-canary line of _completed.txt as the sentinel
# itself, so arming needs zero copy-paste (the manual nohup loop was retyped
# ~15 times today; each retype risks a stale or mistyped sentinel).
# Usage: bash /workspace/tensor_language/basis_aligned/bilinear_quotient/ops/arm_waiter.sh [max_checks=240] [interval_s=30]
# INVOKE BY ABSOLUTE PATH: two 127s on 2026-09-02 came from relative invocation
# in a shell whose cwd had drifted; the failure is silent in a background task.
BQ="$(cd "$(dirname "$0")/.." && pwd)"
MAX="${1:-240}"; IV="${2:-30}"
OTHERS=$(pgrep -f "arm_waiter.sh" | grep -vw "$$" | tr '\n' ' ')
if [ -n "$OTHERS" ]; then
  echo "ALREADY-ARMED: another arm_waiter is live (pid(s): $OTHERS); refusing duplicate arm"
  exit 3
fi
SENTINEL=$(grep -v canary "$BQ/runlogs/_completed.txt" | tail -1)
echo "ARMED pid=$$ interval=${IV}s max=${MAX} sentinel=[$SENTINEL]"
for i in $(seq 1 "$MAX"); do
  sleep "$IV"
  L=$(grep -v canary "$BQ/runlogs/_completed.txt" | tail -1)
  if [ "$L" != "$SENTINEL" ]; then echo "REAL-RUN-LANDED:"; echo "$L"; exit 0; fi
done
echo "TIMEOUT after $((MAX*IV/60)) min (sentinel was: $SENTINEL)"
