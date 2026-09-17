#!/bin/bash
# dod_record.sh -- one command for the three post-run steps repeated after every DoD receipt:
# clock-stamped board entry (stdin = body), git add of the named files (+ board + scorecard), commit, push,
# and a positive re-read of the remote. WHY (review 2, 2026-09-17): the sequence was hand-repeated nine
# times in one hour.   usage: bash ops/dod_record.sh "<title>" "<commit subject>" file... <<'EOF' body EOF
set -u
title="${1:?title}"; subject="${2:?commit subject}"; shift 2
ROOT=/workspace/tensor_language
bash "$ROOT/basis_aligned/bilinear_quotient/ops/board_append.sh" "Claude" "$title"
git -C "$ROOT" add -A basis_aligned/claude_hourly_review "$ROOT/AGENT_BOARD.md" "$@"
git -C "$ROOT" -c user.name=loganriggs -c user.email=logan.smith.5@gmail.com commit -q -m "$subject

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012ur6Uo7Uw16vN99AKAc9bx" || { echo "nothing to commit"; }
timeout 120 git -C "$ROOT" push -q origin HEAD 2>&1 | tail -1
git -C "$ROOT" fetch -q origin
echo "unpushed=$(git -C "$ROOT" log --oneline origin/main..HEAD | wc -l)  $(date -u +%H:%M:%S)"
