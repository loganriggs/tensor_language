#!/bin/bash
# bqrunner2 -- CPU-ONLY second lane (approved by Codex 2026-09-03T16:56Z, fail-closed).
#
# WHY. 2026-09-03: a 32-min CPU probe on lane 1 held a 13-s GPU rung for 32 min. CPU probes and GPU
# rungs do not contend for the GPU, so they get their own serial lane.
#
# Contract (differences from lane 1 are the whole point):
#   * pops queue2.txt ONLY; a popped script MUST contain the literal header `# BQLANE: cpu`,
#     otherwise it is DROPPED unrun (fail-closed) with a note in runner2.log.
#   * CUDA_VISIBLE_DEVICES="" is forced: a script that needs CUDA fails loudly; this lane must
#     never be used for model code that silently falls back from CUDA to CPU.
#   * four CPU threads (OMP/MKL/torch) and nice 10, so lane 1's GPU feeding is not starved.
#   * separate state: queue2.txt, runlogs/_completed2.txt, runlogs/runner2.log, runlogs/<name>.2.log.
#     It never reads or writes queue.txt, _completed.txt, runner.log, and runs NO canary/watchdog.
utils=/opt/supervisor-scripts/utils
[ -f "${utils}/logging.sh" ] && . "${utils}/logging.sh"
[ -f "${utils}/environment.sh" ] && . "${utils}/environment.sh"

BQ=/workspace/tensor_language/basis_aligned/bilinear_quotient
QUEUE="$BQ/queue2.txt"
RUNLOGS="$BQ/runlogs"
LOG="$RUNLOGS/runner2.log"
DONE="$RUNLOGS/_completed2.txt"
THREADS=4

source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS TORCH_NUM_THREADS=$THREADS
mkdir -p "$RUNLOGS"
cd "$BQ" || exit 1
echo "[bqrunner2] started $(date -u +%H:%M:%S) (cpu-only, ${THREADS} threads, nice 10)" >> "$LOG"

while true; do
    line=""
    if [ -s "$QUEUE" ]; then
        line=$(head -n 1 "$QUEUE")
        tail -n +2 "$QUEUE" > "$QUEUE.tmp2" && mv "$QUEUE.tmp2" "$QUEUE"
    fi
    if [ -n "$line" ]; then
        name=$(basename "$line" .py)
        if [ ! -f "$line" ]; then
            echo "[bqrunner2] $(date -u +%H:%M:%S) DROPPED (not a file): $line" >> "$LOG"
        elif ! grep -q '^# BQLANE: cpu' "$line"; then
            echo "[bqrunner2] $(date -u +%H:%M:%S) DROPPED (no '# BQLANE: cpu' header, fail-closed): $line" >> "$LOG"
            echo "$(date -u +%H:%M) $name exit=DROPPED-no-cpu-header [lane2]" >> "$DONE"
        else
            echo "[bqrunner2] $(date -u +%H:%M:%S) running $name" >> "$LOG"
            nice -n 10 python "$line" > "$RUNLOGS/$name.2.log" 2>&1
            rc=$?
            echo "$(date -u +%H:%M) $name exit=$rc [lane2]" >> "$DONE"
            echo "[bqrunner2] $(date -u +%H:%M:%S) $name exit=$rc" >> "$LOG"
        fi
        continue
    fi
    sleep 15
done
