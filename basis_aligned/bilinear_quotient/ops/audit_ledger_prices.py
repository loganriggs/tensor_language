#!/usr/bin/env python
"""audit_ledger_prices -- every "Price:" line in the ledger must match its receipt. (ops lane, additive.)

Written 2026-09-04 07:12Z after discovering that five ledger sections (SS2848-SS2852) carried GPU forward and
second counts I had never measured -- large overestimates written from the preregistration's budget rather than
read from the receipt -- plus one transcription error (SS2836's seconds). Every section where the read command
had actually printed `price` was exact; every section where it had not was wrong. The failure mode is silent:
nothing in the pipeline compares the ledger's stated price against the receipt that produced it.

Usage:
    /venv/main/bin/python ops/audit_ledger_prices.py [--since 2800]      # exit 1 on any mismatch

It maps a ledger section to a receipt by the `Results: <file>` line the sections already carry, so no separate
registry is needed and new rungs are covered automatically as long as they name their receipt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
LEDGER = ROOT / "BILIN18_CONNECTION.md"
SEC = re.compile(r"^## §(\d+) ", re.M)
PRICE = re.compile(r"Price: ([\d,]+) GPU (?:document-)?forwards, ([\d.]+) GPU-seconds")
RESULTS = re.compile(r"Results: ([A-Za-z0-9_.\-]+\.json)")
SEC_TOL = 0.15          # GPU-seconds wobble between the receipt write and the ledger read


def audit(since=0, verbose=True):
    text = LEDGER.read_text()
    marks = [(m.group(1), m.start()) for m in SEC.finditer(text)]
    bad, checked = [], 0
    for i, (num, start) in enumerate(marks):
        if int(num) < since:
            continue
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        body = text[start:end]
        pm, rm = PRICE.search(body), RESULTS.search(body)
        if not pm or not rm:
            continue
        path = ROOT / rm.group(1)
        if not path.is_file():
            continue
        try:
            price = json.loads(path.read_text())["price"]
        except Exception:
            continue
        checked += 1
        wf, ws = int(pm.group(1).replace(",", "")), float(pm.group(2))
        af, asec = price["gpu_forwards"], round(price["gpu_seconds"], 1)
        # a corrected line states the old value in brackets; compare only the live figure
        if wf != af or abs(ws - asec) > SEC_TOL:
            bad.append((num, wf, ws, af, asec, rm.group(1)))
            if verbose:
                print(f"§{num}: ledger {wf} fwd / {ws} s  vs receipt {af} / {asec}  ({rm.group(1)})")
    if verbose:
        print(f"checked {checked} sections with both a Price and a Results line; {len(bad)} mismatched")
    return bad, checked


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=0, help="only audit sections numbered >= this")
    a = ap.parse_args()
    bad, _ = audit(a.since)
    sys.exit(1 if bad else 0)
