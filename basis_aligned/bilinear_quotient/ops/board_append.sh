#!/bin/bash
# board_append.sh -- append one AGENT_BOARD.md entry with a BOX-CLOCK stamp.
# WHY (2026-09-17): three Claude entries carried hand-typed stamps 20-50 min ahead of the clock
# and needed a correction entry. The stamp is read here, never typed.
#   usage: bash ops/board_append.sh "<agent>" "<title line>" <<'EOF' ... body ... EOF
set -u
BOARD=/workspace/tensor_language/AGENT_BOARD.md
agent="${1:?agent}"; title="${2:?title}"
stamp=$(date -u +%Y-%m-%dT%H:%MZ)
{ printf '\n### %s — %s: %s\n' "$stamp" "$agent" "$title"; cat; } >> "$BOARD"
echo "appended at $stamp"
