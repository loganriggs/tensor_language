#!/bin/bash
# lane_depth.sh -- is a lane about to go idle?
#
# WHY. Measured over the last 48 h (2026-08-29): of a 48.0 h span, the GPU was busy 22.7 h (47%),
# agent turnaround took 17.1 h (36%), and LONG IDLES took 17.4 h (36%) -- only two of them, but they
# are the single largest bucket. An idle lane is the cheapest possible loss: nothing is being computed
# and nothing is being written.
#
# The rule this enforces: MY lane (queue.txt) should hold >= 2 registered experiments, so the runner
# never waits on me to write one. Queue the NEXT experiment BEFORE consolidating the LAST.
#
# Exit 0 = fine. Exit 1 = actionable: my lane is short and the GPU is not busy.
# Codex's lane (queue2.txt) is REPORTED ONLY and never touched.
set -u
BQ=/workspace/tensor_language/basis_aligned/bilinear_quotient
WANT=${1:-2}

# `grep -c` PRINTS 0 and EXITS 1 on no match, so `grep -c ... || echo 0` printed "0\n0" and every
# integer test below died with "integer expression expected". Caught by running the watchdog against
# its own degraded state -- which is the only reason it was found before it went silent in production.
depth() {
    [ -f "$1" ] || { echo 0; return; }
    local n
    n=$(grep -cve '^[[:space:]]*$' "$1" 2>/dev/null)
    echo "${n:-0}"
}
Q1=$(depth "$BQ/queue.txt")
Q2=$(depth "$BQ/queue2.txt")

if bash "$BQ/ops/gpu_free.sh" >/dev/null 2>&1; then GPU=free; else GPU=busy; fi

LAST=$(ls -t "$BQ"/runlogs/*.log 2>/dev/null | head -1)
if [ -n "${LAST:-}" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$LAST") ))
    AGES="${AGE}s since $(basename "$LAST") last wrote"
else
    AGES="no run logs"
fi

echo "lane 1 (mine, queue.txt):   $Q1 queued   [want >= $WANT]"
echo "lane 2 (Codex, queue2.txt): $Q2 queued   [reported only, never edited]"
echo "GPU: $GPU   |   $AGES"

if [ "$Q1" -lt "$WANT" ] && [ "$GPU" = free ]; then
    echo "ALARM: lane 1 has $Q1 < $WANT queued and the GPU is FREE -- this is the 36% idle bucket."
    echo "       Queue the next experiment now, before writing up the last one."
    exit 1
fi
if [ "$Q1" -lt "$WANT" ]; then
    echo "WARN: lane 1 has $Q1 < $WANT queued; the GPU is busy, so there is time -- but queue now."
fi
exit 0
