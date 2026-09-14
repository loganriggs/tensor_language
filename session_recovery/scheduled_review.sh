#!/bin/bash
# User-requested active Codex reviews; serialized independently of GPU execution.
set -euo pipefail
kind="${1:?hourly or mathematical}"
case "$kind" in hourly|mathematical) ;; *) exit 2;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
state=/home/loganriggs/.local/share/bilin18
export PATH=/home/loganriggs/.nvm/versions/node/v22.20.0/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$state/logs"
exec 9>"$state/review.lock"
flock -w 3300 9
stamp=$(date -u +%Y%m%dT%H%M%SZ)
codex exec -C "$repo" -s danger-full-access \
  -c 'approval_policy="never"' \
  -c 'web_search="live"' \
  -o "$state/logs/${kind}-${stamp}-last.md" \
  - < "$repo/session_recovery/${kind}_review_prompt.md" \
  > "$state/logs/${kind}-${stamp}.log" 2>&1
# A textual refusal may still exit zero. Require a fresh substantive receipt
# before systemd reports this scheduled review as successful.
"$state/venv/bin/python" - "$repo" "$kind" <<'PY'
import datetime as dt
from pathlib import Path
import sys

root, kind = Path(sys.argv[1]), sys.argv[2]
prefix, hours = ('HOURLY_STRATEGIC_REVIEW_', 1) if kind == 'hourly' else ('THREE_HOURLY_MATHEMATICAL_REVIEW_', 3)
records = sorted((root/'basis_aligned/polynomial_causal').glob(prefix+'*.md'))
if not records:
    raise SystemExit('Scheduled review failed: no review receipt')
latest = records[-1]
stamp = dt.datetime.strptime(latest.stem[len(prefix):], '%Y-%m-%d_%H%M').replace(tzinfo=dt.timezone.utc)
age = dt.datetime.now(dt.timezone.utc)-stamp
if age < dt.timedelta(minutes=-2) or age > dt.timedelta(hours=hours, minutes=5):
    raise SystemExit(f'Scheduled review failed: stale or future receipt {latest.name}')
if latest.stat().st_size < 500:
    raise SystemExit(f'Scheduled review failed: unexpectedly short receipt {latest.name}')
print(f'Verified scheduled review: {latest}')
PY
