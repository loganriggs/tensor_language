#!/bin/bash
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
for stem in "$@"; do
  echo "=== $(date -u +%H:%M:%S) $stem" >> tf_interp3.log
  python tf_interp3.py --stem "$stem" >> tf_interp3.log 2>&1 || echo "!!! FAILED $stem" >> tf_interp3.log
done
echo "=== $(date -u +%H:%M:%S) INTERP3 BATCH DONE" >> tf_interp3.log
