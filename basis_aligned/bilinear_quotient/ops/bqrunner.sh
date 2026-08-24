#!/bin/bash
# bqrunner -- the bilinear-quotient queue runner.
#
# Pops the FIRST line of queue.txt, runs it, appends the exit status to
# runlogs/_completed.txt.  Idle cycles run the canary so the model+data
# path is continuously regression-tested and the box never sits silent.
#
# Contract (do not change without updating LESSONS.md):
#   * queue.txt holds ABSOLUTE paths, one per line.  A line that is not an
#     existing file is popped and dropped (with a note in runner.log).
#   * per-script log:      runlogs/<basename>.log   (truncated per run)
#   * completion ledger:   runlogs/_completed.txt   "HH:MM <name> exit=N"
#   * scripts run with cwd = BQ so `import bilin18_joint_removal` resolves.
utils=/opt/supervisor-scripts/utils
[ -f "${utils}/logging.sh" ] && . "${utils}/logging.sh"
[ -f "${utils}/environment.sh" ] && . "${utils}/environment.sh"

BQ=/workspace/tensor_language/basis_aligned/bilinear_quotient
QUEUE="$BQ/queue.txt"
RUNLOGS="$BQ/runlogs"
CANARY="$BQ/bilin18_canary2.py"
IDLE_CANARY_SECS=1800

source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p "$RUNLOGS"
cd "$BQ" || exit 1

last_canary=0
echo "[bqrunner] started $(date -u +%H:%M:%S)" >> "$RUNLOGS/runner.log"

run_one () {
    local path="$1"
    local name
    name=$(basename "$path" .py)
    echo "[bqrunner] $(date -u +%H:%M:%S) running $name" >> "$RUNLOGS/runner.log"
    python "$path" > "$RUNLOGS/$name.log" 2>&1
    local rc=$?
    echo "$(date -u +%H:%M) $name exit=$rc" >> "$RUNLOGS/_completed.txt"
    echo "[bqrunner] $(date -u +%H:%M:%S) $name exit=$rc" >> "$RUNLOGS/runner.log"
}

while true; do
    line=""
    if [ -s "$QUEUE" ]; then
        # pop the first non-blank line atomically-ish
        line=$(head -n 1 "$QUEUE")
        tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    fi

    if [ -n "$line" ]; then
        if [ -f "$line" ]; then
            run_one "$line"
        else
            echo "[bqrunner] $(date -u +%H:%M:%S) DROPPED (not a file): $line" \
                >> "$RUNLOGS/runner.log"
        fi
        continue
    fi

    now=$(date +%s)
    if [ $((now - last_canary)) -ge $IDLE_CANARY_SECS ]; then
        run_one "$CANARY"
        last_canary=$(date +%s)
    fi
    sleep 20
done
