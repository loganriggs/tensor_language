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

# Append a script to lane 1 ONLY if it exists, parses and gates.
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
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" \
  || { echo "REFUSED: does not parse: $f" >&2; exit 1; }
D="$(dirname "$f")"
# scripts may live in <proj>/ops/ or at <proj>/ top level; walk up until ops/test_fast.py is found
[ -f "$D/ops/test_fast.py" ] || D="$(dirname "$D")"
# The fast suite (~0.4s, no GPU) runs before every enqueue: a broken bqlib or a regressed gate cannot
# reach the GPU. It encodes the mistakes that actually cost runs -- see ops/test_fast.py.
python3 "$D/ops/test_fast.py" >/tmp/bq_test_fast.out 2>&1 || {
  echo "REFUSED: ops/test_fast.py is failing -- fix the library before queueing:" >&2
  tail -12 /tmp/bq_test_fast.out >&2; exit 1; }
python3 "$D/ops/gate.py" "$f" >/dev/null 2>&1 || {
  echo "REFUSED: gate FAILED:" >&2; python3 "$D/ops/gate.py" "$f" >&2; exit 1; }

# PRE-FLIGHT the PLAN itself, GPU-free, in about two seconds. MEASURED 2026-08-30: six GPU runs this
# session were thrown away because a plan's covered-input control was ill-formed, and every one was
# decidable from the plan alone before the first GPU call. bqlib's run() now refuses such a plan;
# BQLIB_DRYRUN makes it do so here instead of after a 450-second run.
BQLIB_DRYRUN=1 BQLIB_NO_MODEL=1 python3 "$f" >/tmp/bq_dryrun.out 2>&1 || {
  echo "REFUSED: plan pre-flight FAILED (no GPU work was done):" >&2
  tail -6 /tmp/bq_dryrun.out >&2; exit 1; }
if [ "$LANE" = "2" ]; then
  grep -q '^# BQLANE: cpu' "$f" || { echo "REFUSED: LANE=2 needs the literal header '# BQLANE: cpu' in $f (lane 2 is CPU-only, fail-closed)" >&2; exit 1; }
  echo "$f" >> "$D/queue2.txt"
  n=$(grep -cve '^[[:space:]]*$' "$D/queue2.txt" 2>/dev/null); n=${n:-0}
  echo "QUEUED (lane 2, CPU-only) $f"
  echo "  lane 2 depth now $n   (CUDA_VISIBLE_DEVICES='', 4 threads, nice 10; log runlogs/<name>.2.log, ledger runlogs/_completed2.txt)"
else
  echo "$f" >> "$D/queue.txt"
  n=$(grep -cve '^[[:space:]]*$' "$D/queue.txt" 2>/dev/null); n=${n:-0}
  if bash "$D/ops/gpu_free.sh" >/dev/null 2>&1; then g="GPU free"; else g="GPU busy"; fi
  echo "QUEUED $f"
  echo "  lane 1 depth now $n   ($g -- the runner serializes; depth >= 2 keeps it fed)"
fi

# Advisory lint (never blocking): show instrument-clause warnings at enqueue time.
/venv/main/bin/python "$(dirname "$0")/preflight.py" "$1" 2>/dev/null || true
