#!/bin/bash
# User-requested active Codex reviews; serialized independently of GPU execution.
set -euo pipefail
kind="${1:?hourly or mathematical}"
case "$kind" in hourly|mathematical) ;; *) exit 2;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
state="${BILIN18_REVIEW_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/bilin18}"
export PATH="/opt/nvm/versions/node/v24.20.0/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$state/logs"
# Wait before acquiring the shared lock, so an early cron slot cannot skip a due review.
python3 "$repo/session_recovery/review_due.py" "$repo" "$kind"
exec 9>"$state/review.lock"
flock -w 3300 9
stamp=$(date -u +%Y%m%dT%H%M%SZ)
timeout --signal=TERM --kill-after=30s 2400 codex exec -C "$repo" -s danger-full-access \
  -c 'approval_policy="never"' \
  -c 'web_search="live"' \
  -o "$state/logs/${kind}-${stamp}-last.md" \
  - < "$repo/session_recovery/${kind}_review_prompt.md" \
  > "$state/logs/${kind}-${stamp}.log" 2>&1
# A textual refusal may still exit zero. Require a fresh substantive receipt
# before systemd reports this scheduled review as successful.
python3 - "$repo" "$kind" <<'PY'
import datetime as dt
from pathlib import Path
import re
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
body = latest.read_text()
if kind == 'hourly':
    match = re.search(r'ACTIVE_TRACK\s*:\s*(CIRCUIT|WEIGHT_FOLDING)', body, re.I)
    if not match:
        raise SystemExit(f'Scheduled review failed: missing ACTIVE_TRACK in {latest.name}')
    track = match.group(1).upper()
    prior_tracks = []
    for record in records[:-1]:
        found = re.search(r'ACTIVE_TRACK\s*:\s*(CIRCUIT|WEIGHT_FOLDING)', record.read_text(), re.I)
        if found:
            prior_tracks.append(found.group(1).upper())
    expected = ('WEIGHT_FOLDING' if not prior_tracks else
                ('CIRCUIT' if prior_tracks[-1] == 'WEIGHT_FOLDING' else 'WEIGHT_FOLDING'))
    if track != expected:
        raise SystemExit(f'Scheduled review failed: ACTIVE_TRACK {track}, expected {expected}')
    required = ['TRACK_ALTERNATION', 'TRACK_PROGRESS', 'CEREMONY_BUDGET', 'NOVELTY_LESSON_GATE', 'PAST_HOUR_TIMING', 'PROCESS_IMPROVEMENT']
    missing = [key for key in required if key not in body]
    if missing:
        raise SystemExit(f'Scheduled review failed: missing verdicts {missing}')
else:
    required_topics = ['organization', 'efficiency', 'CIRCUIT', 'WEIGHT_FOLDING']
    missing = [term for term in required_topics if term.lower() not in body.lower()]
    if missing:
        raise SystemExit(f'Scheduled review failed: missing three-hour topics {missing}')
print(f'Verified scheduled review: {latest}')
PY
