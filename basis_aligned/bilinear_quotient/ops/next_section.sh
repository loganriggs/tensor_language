#!/bin/bash
# Print the next free ledger section number (ops lane, 23:06 review).
# Grep-and-think for the next free paragraph number happened before every ledger
# write (~10/day, with out-of-order § coordination each time). This prints the
# max existing section +1; CLAIMS in board prose still override — check the
# board tail for "owns section NNNN"-style reservations before taking it.
L="$(dirname "$0")/../BILIN18_CONNECTION.md"
MAX=$(grep -oE '^## §[0-9]+' "$L" | grep -oE '[0-9]+' | sort -n | tail -1)
echo "next free ledger section: $((MAX + 1)) (max existing: $MAX; check board for prose claims)"
