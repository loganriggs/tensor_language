"""failed_log_watch -- capture a failed run's log before its retry overwrites it. OPS LANE, NON-INVASIVE.

The rerun tax is the largest avoidable cost in the loop right now: 8 of the last 60 executions exited nonzero
(13.3%), 7 failure->retry pairs, 35 minutes of wall-clock re-running -- against a 7.6-minute median screen.
It cannot be reduced because the failure CLASSES are invisible: `bqrunner` writes each run to
`runlogs/<name>.log`, and the retry (median gap 4 min) overwrites it, so by the time anyone looks the failure
that mattered is gone.

The clean fix is one line inside the runner, which is runner-owned and not mine to edit. This achieves the
same end from outside and touches nothing that belongs to another lane: it watches the append-only
`runlogs/_completed.txt`, and when a nonzero exit appears it copies that run's log aside to
`runlogs/failed/<name>.<utc>.log` before the retry lands. It only ever CREATES files under `runlogs/failed/`;
it never writes to `_completed.txt`, `runner.log`, or any existing log.

Usage:
  python ops/failed_log_watch.py --once        # sweep any failures already visible, then exit
  python ops/failed_log_watch.py               # watch (poll every 10 s) until killed
"""
import os
import re
import shutil
import sys
import time
import datetime

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNLOGS = os.path.join(BQ, "runlogs")
COMPLETED = os.path.join(RUNLOGS, "_completed.txt")
FAILED_DIR = os.path.join(RUNLOGS, "failed")
STATE = os.path.join(FAILED_DIR, ".offset")
LINE = re.compile(r"^(\d\d:\d\d)\s+(\S+)\s+exit=(\d+)\s*$")
POLL_SECONDS = 10


def _read_offset():
    try:
        return int(open(STATE).read().strip())
    except (OSError, ValueError):
        return None


def _write_offset(n):
    os.makedirs(FAILED_DIR, exist_ok=True)
    with open(STATE, "w") as fh:
        fh.write(str(n))


def sweep(offset):
    """Capture logs for nonzero exits appearing after `offset` bytes. Returns (new_offset, captured)."""
    if not os.path.exists(COMPLETED):
        return offset or 0, []
    size = os.path.getsize(COMPLETED)
    if offset is None:                       # first run: start at the end, do not re-mine history
        return size, []
    if size < offset:                        # truncated/rotated
        offset = 0
    captured = []
    with open(COMPLETED, errors="ignore") as fh:
        fh.seek(offset)
        for raw in fh:
            m = LINE.match(raw.strip())
            if not m or m.group(3) == "0":
                continue
            name, code = m.group(2), m.group(3)
            src = os.path.join(RUNLOGS, f"{name}.log")
            if not os.path.isfile(src):
                continue
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dst = os.path.join(FAILED_DIR, f"{name}.{stamp}.exit{code}.log")
            os.makedirs(FAILED_DIR, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                captured.append(os.path.basename(dst))
            except OSError as exc:
                print(f"could not capture {name}: {exc}", file=sys.stderr)
        new_offset = fh.tell()
    return new_offset, captured


if __name__ == "__main__":
    once = "--once" in sys.argv
    offset = _read_offset()
    if offset is None:
        offset, _ = sweep(None)
        _write_offset(offset)
        print(f"initialised at offset {offset} (history not re-mined; captures start from the next failure)")
        if once:
            raise SystemExit(0)
    while True:
        offset, captured = sweep(offset)
        _write_offset(offset)
        for c in captured:
            print(f"captured {c}", flush=True)
        if once:
            print(f"{len(captured)} log(s) captured")
            break
        time.sleep(POLL_SECONDS)
