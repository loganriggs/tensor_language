#!/bin/bash

# --- dedup guard (added 2026-09-01 after two symmetric double-enqueue races) ---
BASE=$(basename "${1%.py}")
QDIR=$(dirname "$0")/..
# LANE=2 -> the CPU-ONLY lane (ops/bqrunner2.sh; approved by Codex 2026-09-03T16:56Z). Same parse / fast-test /
# gate / dry-run / dedup checks as lane 1, plus: the script MUST carry `# BQLANE: cpu`, and it lands in queue2.txt.
LANE="${LANE:-1}"
case "$LANE" in 1) QFILE=queue.txt; LSUF="";; 2) QFILE=queue2.txt; LSUF=".2";; *) echo "REFUSED: LANE must be 1 or 2" >&2; exit 1;; esac
if [ "${FORCE:-0}" != "1" ]; then
  if grep -q "/$BASE\.py$" "$QDIR/$QFILE" 2>/dev/null; then
    echo "DEDUP: $BASE already queued on lane $LANE (FORCE=1 to override)"; exit 3
  fi
  if [ -f "$QDIR/runlogs/$BASE$LSUF.log" ] && find "$QDIR/runlogs/$BASE$LSUF.log" -newermt '-10 minutes' | grep -q .; then
    echo "DEDUP: $BASE ran/running within 10 min (FORCE=1 to override)"; exit 3
  fi
fi
# --- end dedup guard ---

# Append a script to lane 1 ONLY if it exists, parses and gates.  Lane-1 queue
# records are content-bound as "<sha256><TAB><absolute path>"; the runner safely
# captures and verifies those exact bytes before Python parses them.  Lane 2
# retains its legacy path-only format until its separate runner is upgraded.
#
# 2026-08-29, MEASURED: this script used to also REFUSE when the GPU was busy. That guard came from the
# rule "never launch onto a busy GPU" -- but enqueue does not launch anything. ops/bqrunner.sh pops one
# line at a time in a single loop and IS the serialization point, so appending while a run is in flight
# is exactly what a queue is for. The guard's only effect was to make it impossible to queue ahead, which
# is the direct cause of the largest measured loss on this box: of a 48 h window, 17.4 h (36%) was a lane
# sitting empty while the agent wrote the next script. GPU state is now REPORTED, never a refusal.
# LESSONS 61: twice today I chained `bash ops/gpu_free.sh && echo path >> queue.txt` after a build
# step that had FAILED, so the queue got a path to a file that did not exist. Separate commands do
# not inherit each other's failure; this puts all four checks behind one exit code.
set -u
f="${1:?usage: enqueue.sh <absolute-script-path>}"
case "$f" in /*) ;; *) echo "REFUSED: path is not absolute: $f" >&2; exit 1;; esac
[ -f "$f" ] || { echo "REFUSED: no such file: $f" >&2; exit 1; }
[ ! -L "$f" ] || { echo "REFUSED: queued script may not be a symlink: $f" >&2; exit 1; }
case "$f" in *$'\t'*|*$'\n'*) echo "REFUSED: queued path contains a tab or newline" >&2; exit 1;; esac
D="$(dirname "$f")"
# scripts may live in <proj>/ops/ or at <proj>/ top level; walk up until ops/test_fast.py is found
[ -f "$D/ops/test_fast.py" ] || D="$(dirname "$D")"

# Lane 1 executes the candidate during its model-free preflight, so the reviewed
# hash must be checked before that first execution. Capture one private snapshot
# with O_NOFOLLOW and run every candidate-specific check on those bytes. The
# original path is never parsed or executed by lane-1 enqueue after this point.
check_path="$f"
sha=""
snapshot=""
if [ "$LANE" = "1" ]; then
  capture=$(python3 - "$f" "$(dirname "$f")" "$BASE" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

source = Path(sys.argv[1])
directory = Path(sys.argv[2])
prefix = "." + sys.argv[3] + ".enqueue-snapshot-"
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(source, flags)
snapshot_descriptor = None
snapshot_name = None
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("queued script is not a regular file")
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )
    if identity(before) != identity(after):
        raise SystemExit("queued script changed during safe capture")
    payload = b"".join(chunks)
    snapshot_descriptor, snapshot_name = tempfile.mkstemp(
        prefix=prefix, suffix=".py", dir=directory
    )
    written = 0
    while written < len(payload):
        written += os.write(snapshot_descriptor, payload[written:])
    os.fsync(snapshot_descriptor)
finally:
    os.close(descriptor)
    if snapshot_descriptor is not None:
        os.close(snapshot_descriptor)
if snapshot_name is None:
    raise SystemExit("failed to create private enqueue snapshot")
print(hashlib.sha256(payload).hexdigest() + "\t" + snapshot_name)
PY
  ) || { echo "REFUSED: could not safely capture queued script: $f" >&2; exit 1; }
  sha="${capture%%$'\t'*}"
  snapshot="${capture#*$'\t'}"
  if [[ ! "$sha" =~ ^[0-9a-f]{64}$ ]] || [ -z "$snapshot" ] || [ "$snapshot" = "$capture" ]; then
    [ -n "$snapshot" ] && rm -f -- "$snapshot"
    echo "REFUSED: invalid safe-capture receipt" >&2
    exit 1
  fi
  trap 'rm -f -- "$snapshot"' EXIT
  if [ -n "${EXPECTED_SHA256:-}" ] && [ "$sha" != "$EXPECTED_SHA256" ]; then
    echo "REFUSED: reviewed script SHA-256 changed: expected=$EXPECTED_SHA256 observed=$sha" >&2
    exit 1
  fi
  check_path="$snapshot"
fi

python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$check_path" \
  || { echo "REFUSED: does not parse: $f" >&2; exit 1; }
# The fast suite (~0.4s, no GPU) runs before every enqueue: a broken bqlib or a regressed gate cannot
# reach the GPU. It encodes the mistakes that actually cost runs -- see ops/test_fast.py.
python3 "$D/ops/test_fast.py" >/tmp/bq_test_fast.out 2>&1 || {
  echo "REFUSED: ops/test_fast.py is failing -- fix the library before queueing:" >&2
  tail -12 /tmp/bq_test_fast.out >&2; exit 1; }
python3 "$D/ops/gate.py" "$check_path" >/dev/null 2>&1 || {
  echo "REFUSED: gate FAILED:" >&2; python3 "$D/ops/gate.py" "$check_path" >&2; exit 1; }

# PRE-FLIGHT the PLAN itself, GPU-free, in about two seconds. For lane 1, do not
# reopen/execute even the private snapshot by path: gate.py may have changed it.
# Safely capture it again, require the reviewed digest, and compile only those
# captured bytes in the same Python process, with the original path semantics.
if [ "$LANE" = "1" ]; then
  BQLIB_DRYRUN=1 BQLIB_NO_MODEL=1 python3 - "$sha" "$check_path" "$f" \
      >/tmp/bq_dryrun.out 2>&1 <<'PY'
# BEGIN ENQUEUE_HASH_BOUND_PYTHON
import hashlib
import os
from pathlib import Path
import stat
import sys
import types

expected, capture_name, logical_name = sys.argv[1], sys.argv[2], sys.argv[3]
if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
    raise SystemExit("invalid expected SHA-256 for enqueue preflight")
capture_path = Path(capture_name)
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(capture_path, flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("enqueue snapshot is not a regular file")
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
    raise SystemExit("enqueue snapshot changed during safe capture")
payload = b"".join(chunks)
observed = hashlib.sha256(payload).hexdigest()
if observed != expected:
    raise SystemExit(
        f"enqueue snapshot SHA-256 changed: expected={expected} observed={observed}"
    )
sys.argv = [logical_name]
sys.path[0] = os.path.dirname(os.path.abspath(logical_name))
module = types.ModuleType("__main__")
module.__file__ = logical_name
module.__package__ = None
module.__cached__ = None
sys.modules["__main__"] = module
exec(compile(payload, logical_name, "exec"), module.__dict__, module.__dict__)
# END ENQUEUE_HASH_BOUND_PYTHON
PY
  preflight_rc=$?
else
  BQLIB_DRYRUN=1 BQLIB_NO_MODEL=1 python3 "$check_path" >/tmp/bq_dryrun.out 2>&1
  preflight_rc=$?
fi
if [ "$preflight_rc" -ne 0 ]; then
  echo "REFUSED: plan pre-flight FAILED (no GPU work was done):" >&2
  tail -6 /tmp/bq_dryrun.out >&2
  exit 1
fi
if [ "$LANE" = "2" ]; then
  grep -q '^# BQLANE: cpu' "$f" || { echo "REFUSED: LANE=2 needs the literal header '# BQLANE: cpu' in $f (lane 2 is CPU-only, fail-closed)" >&2; exit 1; }
  echo "$f" >> "$D/queue2.txt"
  n=$(grep -cve '^[[:space:]]*$' "$D/queue2.txt" 2>/dev/null); n=${n:-0}
  echo "QUEUED (lane 2, CPU-only) $f"
  echo "  lane 2 depth now $n   (CUDA_VISIBLE_DEVICES='', 4 threads, nice 10; log runlogs/<name>.2.log, ledger runlogs/_completed2.txt)"
else
  current_sha=$(python3 - "$f" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys

path = Path(sys.argv[1])
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("queued script is not a regular file")
    payload = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        payload.extend(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
)
if identity(before) != identity(after):
    raise SystemExit("queued script changed during hash capture")
print(hashlib.sha256(payload).hexdigest())
PY
  ) || { echo "REFUSED: could not revalidate queued script: $f" >&2; exit 1; }
  if [ "$current_sha" != "$sha" ]; then
    echo "REFUSED: queued script changed after reviewed preflight: captured=$sha observed=$current_sha" >&2
    exit 1
  fi
  printf '%s\t%s\n' "$sha" "$f" >> "$D/queue.txt"
  n=$(grep -cve '^[[:space:]]*$' "$D/queue.txt" 2>/dev/null); n=${n:-0}
  if bash "$D/ops/gpu_free.sh" >/dev/null 2>&1; then g="GPU free"; else g="GPU busy"; fi
  echo "QUEUED $f"
  echo "  script sha256 $sha (runner verifies captured bytes before parse/exec)"
  echo "  lane 1 depth now $n   ($g -- the runner serializes; depth >= 2 keeps it fed)"
fi

# Advisory lint (never blocking): show instrument-clause warnings for the same
# captured bytes that passed the blocking checks.
/venv/main/bin/python "$(dirname "$0")/preflight.py" "$check_path" 2>/dev/null || true
