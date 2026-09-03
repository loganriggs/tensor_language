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
  rt=$(python3 -c "import json,sys;print(round(json.load(open(sys.argv[1])).get('runtime_s',-1),2))" "$f" 2>/dev/null)
  m=$(stat -c '%y' "$f" 2>/dev/null | cut -d. -f1 | cut -d' ' -f2)
  printf '%s\t%ss\t%s\n' "$m" "$rt" "$(basename "$f")"
done | sort -r
