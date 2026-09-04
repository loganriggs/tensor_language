#!/bin/bash
# bqrunner -- the bilinear-quotient queue runner.
#
# Pops the FIRST line of queue.txt, runs it, appends the exit status to
# runlogs/_completed.txt.  Idle cycles run the canary so the model+data
# path is continuously regression-tested and the box never sits silent.
#
# Contract (do not change without updating LESSONS.md):
#   * New queue.txt records are "<sha256><TAB><ABSOLUTE path>". The runner
#     safely captures and verifies those bytes before Python parses them.
#     Legacy bare absolute paths remain accepted for already-queued work.
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

# --- GPU watchdog (S1601 ops) -------------------------------------------
# This host loses the GPU intermittently (NVML "Unknown Error"; 3x on
# 2026-08-26). Only a container reboot recovers it, and an interactive
# session that issues the reboot dies with the container — so the runner
# (supervisor restarts it automatically after every boot) does it instead.
# Every GPU_CHECK_SECS: nvidia-smi; after GPU_FAILS_NEEDED consecutive
# failures, issue `vastai reboot` — unless GPU_MAX_REBOOTS_6H prior
# watchdog reboots in the last 6h (host likely beyond saving; leave it for
# a human and keep serving CPU-side).
GPU_CHECK_SECS=300
GPU_FAILS_NEEDED=3
GPU_MAX_REBOOTS_6H=4
GPU_REBOOT_LOG="$RUNLOGS/watchdog_reboots.log"
gpu_fails=0
last_gpu_check=$(date +%s)

gpu_watchdog_tick () {
    local now
    now=$(date +%s)
    [ $((now - last_gpu_check)) -lt $GPU_CHECK_SECS ] && return
    last_gpu_check=$now
    if nvidia-smi > /dev/null 2>&1; then
        gpu_fails=0
        return
    fi
    gpu_fails=$((gpu_fails + 1))
    echo "[bqrunner] $(date -u +%H:%M:%S) watchdog: nvidia-smi failed (${gpu_fails}/${GPU_FAILS_NEEDED})" \
        >> "$RUNLOGS/runner.log"
    [ "$gpu_fails" -lt "$GPU_FAILS_NEEDED" ] && return
    local recent=0 ts rest
    if [ -f "$GPU_REBOOT_LOG" ]; then
        while read -r ts rest; do
            [ -n "$ts" ] && [ $((now - ts)) -lt 21600 ] && recent=$((recent + 1))
        done < "$GPU_REBOOT_LOG"
    fi
    if [ "$recent" -ge "$GPU_MAX_REBOOTS_6H" ]; then
        echo "[bqrunner] $(date -u +%H:%M:%S) watchdog: ${recent} reboots in 6h — NOT rebooting, human needed" \
            >> "$RUNLOGS/runner.log"
        gpu_fails=0
        return
    fi
    echo "${now} $(date -u '+%Y-%m-%d %H:%M:%S') reboot issued (${recent} prior in 6h)" >> "$GPU_REBOOT_LOG"
    echo "[bqrunner] $(date -u +%H:%M:%S) watchdog: issuing vastai reboot" >> "$RUNLOGS/runner.log"
    vastai reboot instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" \
        >> "$RUNLOGS/runner.log" 2>&1
    sleep 900
    gpu_fails=0
}
# ------------------------------------------------------------------------

run_one () {
    local path="$1"
    local expected_sha="${2:-}"
    local name
    name=$(basename "$path" .py)
    echo "[bqrunner] $(date -u +%H:%M:%S) running $name" >> "$RUNLOGS/runner.log"
    if [ -n "$expected_sha" ]; then
        python - "$expected_sha" "$path" > "$RUNLOGS/$name.log" 2>&1 <<'PY'
# BEGIN HASH_BOUND_PYTHON
import hashlib
import os
from pathlib import Path
import stat
import sys
import types

expected, raw_path = sys.argv[1], sys.argv[2]
if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
    raise SystemExit("invalid expected SHA-256 in managed queue record")
path = Path(raw_path)
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(path, flags)
except OSError as error:
    raise SystemExit(f"cannot safely open queued script: {path}: {error}") from error
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"queued script is not a regular file: {path}")
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
)
if identity(before) != identity(after):
    raise SystemExit(f"queued script changed during safe capture: {path}")
payload = b"".join(chunks)
observed = hashlib.sha256(payload).hexdigest()
if observed != expected:
    raise SystemExit(
        f"queued script SHA-256 changed: expected={expected} observed={observed} path={path}"
    )

# Execute only the captured bytes.  Match ordinary `python path.py` import and
# __main__ behavior closely enough for repository scripts without reopening path.
sys.argv = [str(path)]
sys.path[0] = os.path.dirname(os.path.abspath(raw_path))
module = types.ModuleType("__main__")
module.__file__ = str(path)
module.__package__ = None
module.__cached__ = None
sys.modules["__main__"] = module
exec(compile(payload, str(path), "exec"), module.__dict__, module.__dict__)
# END HASH_BOUND_PYTHON
PY
    else
        python "$path" > "$RUNLOGS/$name.log" 2>&1
    fi
    local rc=$?
    echo "$(date -u +%H:%M) $name exit=$rc" >> "$RUNLOGS/_completed.txt"
    echo "[bqrunner] $(date -u +%H:%M:%S) $name exit=$rc" >> "$RUNLOGS/runner.log"
}

while true; do
    gpu_watchdog_tick
    # never pop queue entries onto a dead GPU — the run would just fail and
    # silently consume the entry (S1601 ops)
    if ! nvidia-smi > /dev/null 2>&1; then
        sleep 30
        continue
    fi
    line=""
    if [ -s "$QUEUE" ]; then
        # pop the first non-blank line atomically-ish
        line=$(head -n 1 "$QUEUE")
        tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    fi

    if [ -n "$line" ]; then
        expected_sha=""
        path="$line"
        if [[ "$line" == *$'\t'* ]]; then
            expected_sha="${line%%$'\t'*}"
            path="${line#*$'\t'}"
            if [[ ! "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || [[ "$path" != /* ]]; then
                echo "[bqrunner] $(date -u +%H:%M:%S) DROPPED (malformed hash-bound record): $line" \
                    >> "$RUNLOGS/runner.log"
                continue
            fi
        fi
        if [ -f "$path" ]; then
            run_one "$path" "$expected_sha"
        else
            echo "[bqrunner] $(date -u +%H:%M:%S) DROPPED (not a file): $path" \
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
