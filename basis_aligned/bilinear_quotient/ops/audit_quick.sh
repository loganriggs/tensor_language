#!/bin/bash
# One-command audit extraction (ops lane, 18:06 review).
# Measured sink: 5+ ad-hoc python one-liners per landing audit this hour
# (3 reads for 495b, 2 for 496, 3 for 498) where receipt_dump --grep would
# have sufficed. This wraps the two standard passes into one command:
#   bash /abs/path/ops/audit_quick.sh <receipt.json> [extra-grep]
D="$(cd "$(dirname "$0")" && pwd)"
R="${1:?usage: audit_quick.sh <receipt.json> [extra-grep]}"
echo "== VERDICT =="
python3 "$D/receipt_dump.py" "$R" --verdict
echo "== SCIENCE LEAVES =="
python3 "$D/receipt_dump.py" "$R" | grep -iE "cosine|recovery|residual|fraction|margin|q95|drift|rms_min|closure|_exact" | head -60
if [ -n "${2:-}" ]; then echo "== EXTRA: $2 =="; python3 "$D/receipt_dump.py" "$R" --grep "$2" | head -40; fi
