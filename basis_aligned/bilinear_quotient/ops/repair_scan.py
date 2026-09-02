#!/usr/bin/env python3
"""Partial-repair scanner (ops lane, advisory).

2026-09-01 evening pattern: a bug is fixed at one occurrence while a
second occurrence of the SAME expression survives (rung 449: the
`base["native_counts"]` KeyError fixed at line 343, crashed again at
line 372; rung 452 needed two create-only attempts for promoted
arithmetic).  Before re-gating a repaired script, run

    python ops/repair_scan.py <script.py> <token> [<token> ...]

It prints EVERY line containing each token so the repair covers all of
them.  Exit 0 always; advisory only.
"""
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    text = Path(sys.argv[1]).read_text().splitlines()
    for token in sys.argv[2:]:
        hits = [(i + 1, line.strip()) for i, line in enumerate(text) if token in line]
        print(f"{sys.argv[1]}: {len(hits)} occurrence(s) of {token!r}")
        for number, line in hits:
            print(f"  {number}: {line[:120]}")


if __name__ == "__main__":
    main()
