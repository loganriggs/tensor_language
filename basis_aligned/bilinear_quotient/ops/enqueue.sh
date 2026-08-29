#!/bin/bash
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
D="$(dirname "$(dirname "$f")")"
# The fast suite (~0.4s, no GPU) runs before every enqueue: a broken bqlib or a regressed gate cannot
# reach the GPU. It encodes the mistakes that actually cost runs -- see ops/test_fast.py.
python3 "$D/ops/test_fast.py" >/tmp/bq_test_fast.out 2>&1 || {
  echo "REFUSED: ops/test_fast.py is failing -- fix the library before queueing:" >&2
  tail -12 /tmp/bq_test_fast.out >&2; exit 1; }
python3 "$D/ops/gate.py" "$f" >/dev/null 2>&1 || {
  echo "REFUSED: gate FAILED:" >&2; python3 "$D/ops/gate.py" "$f" >&2; exit 1; }
echo "$f" >> "$D/queue.txt"
n=$(grep -cve '^[[:space:]]*$' "$D/queue.txt" 2>/dev/null); n=${n:-0}
if bash "$D/ops/gpu_free.sh" >/dev/null 2>&1; then g="GPU free"; else g="GPU busy"; fi
echo "QUEUED $f"
echo "  lane 1 depth now $n   ($g -- the runner serializes; depth >= 2 keeps it fed)"
