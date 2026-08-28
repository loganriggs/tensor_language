#!/bin/bash
# Append a script to lane 1 ONLY if it exists, parses, gates, and the GPU is free.
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
python3 "$D/ops/gate.py" "$f" >/dev/null 2>&1 || {
  echo "REFUSED: gate FAILED:" >&2; python3 "$D/ops/gate.py" "$f" >&2; exit 1; }
bash "$D/ops/gpu_free.sh" >/dev/null 2>&1 || { echo "REFUSED: GPU busy" >&2;
  bash "$D/ops/gpu_free.sh" >&2; exit 1; }
echo "$f" >> "$D/queue.txt"
echo "QUEUED $f"
