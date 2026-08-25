#!/bin/bash
# bqrunner2 -- second GPU lane (S1424 ops). Same contract as bqrunner.sh but pops
# queue2.txt, logs with a .2 suffix, and runs NO canary (lane 1 owns it).
utils=/opt/supervisor-scripts/utils
[ -f "${utils}/logging.sh" ] && . "${utils}/logging.sh"
[ -f "${utils}/environment.sh" ] && . "${utils}/environment.sh"

BQ=/workspace/tensor_language/basis_aligned/bilinear_quotient
QUEUE="$BQ/queue2.txt"
RUNLOGS="$BQ/runlogs"

source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p "$RUNLOGS"
cd "$BQ" || exit 1
echo "[bqrunner2] started $(date -u +%H:%M:%S)" >> "$RUNLOGS/runner.log"

while true; do
    line=""
    if [ -s "$QUEUE" ]; then
        line=$(head -n 1 "$QUEUE")
        tail -n +2 "$QUEUE" > "$QUEUE.tmp2" && mv "$QUEUE.tmp2" "$QUEUE"
    fi
    if [ -n "$line" ]; then
        if [ -f "$line" ]; then
            name=$(basename "$line" .py)
            echo "[bqrunner2] $(date -u +%H:%M:%S) running $name" >> "$RUNLOGS/runner.log"
            python "$line" > "$RUNLOGS/$name.2.log" 2>&1
            rc=$?
            echo "$(date -u +%H:%M) $name exit=$rc [lane2]" >> "$RUNLOGS/_completed.txt"
            echo "[bqrunner2] $(date -u +%H:%M:%S) $name exit=$rc" >> "$RUNLOGS/runner.log"
        else
            echo "[bqrunner2] $(date -u +%H:%M:%S) DROPPED (not a file): $line" >> "$RUNLOGS/runner.log"
        fi
        continue
    fi
    sleep 20
done
