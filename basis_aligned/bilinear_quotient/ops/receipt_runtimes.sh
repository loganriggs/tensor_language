#!/bin/bash
# List recent *_results.json receipts with their runtime_s, newest-modified first.
# WHY: the hourly ops-efficiency review scanned receipts with `find -newermt '-72 minutes'`,
# but this box's find is bfs, which REJECTS the relative "-72 minutes" form ("Invalid timestamp")
# — and a 2>/dev/null in the review swallowed the error, so the scan silently returned nothing for
# 2+ hours (2026-09-03). This uses the absolute-timestamp form (bfs-compatible) instead.
# Usage: bash ops/receipt_runtimes.sh [MINUTES]   (default 75)
set -u
MIN="${1:-75}"
BQ="$(cd "$(dirname "$0")/.." && pwd)"
since="$(date -u -d "${MIN} minutes ago" '+%Y-%m-%d %H:%M:%S')"
find "$BQ" -maxdepth 1 -name "*_results.json" -newermt "$since" 2>/dev/null | while read -r f; do
  # 2026-09-04: this printed -1.0s for 16 of 18 receipts in one review window. It searched only
  # `runtime_s`, `execution_price.runtime_seconds` and `elapsed_seconds`, but the whole frontier family
  # stores its wall-clock under `price.gpu_seconds` (the same field the ledger's `Price:` line quotes),
  # and CPU-only analyses under `price.cpu_seconds`. The hourly ops review's primary instrument was
  # therefore blind to every rung in that family -- the same "signal that cannot report the state it
  # names" defect gpu_free.sh was written to kill. The source key is now printed so a reader can tell
  # wall-clock from GPU-seconds instead of guessing.
  read -r rt src <<<"$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
p=d.get('price') or {}
e=d.get('execution_price') or {}
for k,v in (('runtime_s',d.get('runtime_s')),
            ('execution_price.runtime_seconds',e.get('runtime_seconds')),
            ('elapsed_seconds',d.get('elapsed_seconds')),
            ('price.gpu_seconds',p.get('gpu_seconds')),
            ('price.cpu_seconds',p.get('cpu_seconds')),
            ('arms.runtime_s',(d.get('arms') or {}).get('runtime_s') if isinstance(d.get('arms'),dict) else None)):
    if isinstance(v,(int,float)) and v > 0:
        print(round(float(v),2), k); break
else:
    print('?', 'no-runtime-key')
" "$f" 2>/dev/null)"
  [ -n "${rt:-}" ] || { rt='?'; src='unreadable'; }
  m=$(stat -c '%y' "$f" 2>/dev/null | cut -d. -f1 | cut -d' ' -f2)
  printf '%s\t%8ss\t%-24s\t%s\n' "$m" "$rt" "$src" "$(basename "$f")"
done | sort -r
