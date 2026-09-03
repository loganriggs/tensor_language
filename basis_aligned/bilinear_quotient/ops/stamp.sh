#!/bin/bash
# Print the preregistration stamp line from the BOX CLOCK, never from memory.
# Added 2026-09-03 22:16Z after five hand-written stamps in one hour drifted 1-3 min AHEAD of `date -u` and each had to be
# sed-fixed before the prereg hash could be frozen (the hash covers the stamp, so a late fix = a re-hash = a re-derived script).
# Usage:  bash ops/stamp.sh            -> "Registered 2026-09-03 22:16Z (box clock)"
#         bash ops/stamp.sh >> file.md  (or command-substitute it into a heredoc)
echo "Registered $(date -u +%Y-%m-%d\ %H:%MZ) (box clock)"
