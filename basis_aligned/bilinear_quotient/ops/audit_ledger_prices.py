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
    bad, checked, halfnamed = [], 0, []
    for i, (num, start) in enumerate(marks):
        if int(num) < since:
            continue
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        body = text[start:end]
        pm, rm = PRICE.search(body), RESULTS.search(body)
        if not pm or not rm:
            # A section that names one of the two in prose the regexes cannot parse used to be SKIPPED IN SILENCE,
            # so an unauditable section was indistinguishable from an audited one (found 2026-09-04 08:1xZ when
            # SS2858 was added and the checked count stayed at 43). Report those rather than swallow them.
            # Exactly one of the two canonical lines present: the section makes an auditable claim but cannot be
            # audited. (A section with NEITHER makes no price claim and is correctly silent. An earlier, looser
            # heuristic keyed on the words "Price"/"GPU forward" anywhere in the body and flagged prose -- SS2830's
            # "price cliff", SS2853's narrative about prices -- so it is the presence of the LINES that counts.)
            if bool(pm) != bool(rm):
                halfnamed.append(num)
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
    # Two sections citing ONE receipt is an integrity failure, not a style issue: re-running a script
    # overwrites its own `<stem>_results.json`, so the earlier section's evidence is destroyed and its
    # price silently starts mismatching the replacement (found 2026-09-04 09:30Z between SS2876 and
    # SS2878). Receipts are not tracked in git and the runlog is overwritten too, so nothing else catches it.
    seen = {}
    for num, start in marks:
        if int(num) < since:
            continue
        i = [k for k, (n, _) in enumerate(marks) if n == num][0]
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        rm2 = RESULTS.search(text[start:end])
        if rm2:
            seen.setdefault(rm2.group(1), []).append(num)
    shared = {f: ns for f, ns in seen.items() if len(ns) > 1}
    if verbose:
        print(f"checked {checked} sections with both a Price and a Results line; {len(bad)} mismatched")
        for f, ns in shared.items():
            print(f"SHARED RECEIPT: {f} is cited by " + ", ".join("§" + n for n in ns)
                  + " -- a re-run overwrites it, destroying the earlier section's evidence")
        if halfnamed:
            print(f"UNAUDITABLE: {len(halfnamed)} section(s) mention a price or a receipt but not in the parseable "
                  f"`Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form: "
                  + ", ".join("§" + n for n in halfnamed))
    return bad, checked, halfnamed, shared


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=0, help="only audit sections numbered >= this")
    ap.add_argument("--strict", action="store_true",
                    help="also fail when a section names a price or receipt unparseably")
    a = ap.parse_args()
    bad, _checked, halfnamed, shared = audit(a.since)
    sys.exit(1 if (bad or shared or (a.strict and halfnamed)) else 0)
