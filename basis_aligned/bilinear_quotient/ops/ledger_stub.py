#!/usr/bin/env python3
"""ledger_stub.py RESULTS.json [--sec N] -- print a ledger-§ skeleton for a landed receipt so the write-up is interpretation only.

MEASURED 2026-09-04 (ops review 01:06-02:06): with rungs landing in ~25 s, the hour's largest lane-1 gaps (10-13 min) were WRITE-UP time
between landing and the next enqueue -- transcribing preds/nulls/arms/price/sha by hand into BILIN18_CONNECTION.md and BENCHMARK_BACKLOG.md.
This prints that transcription (scored exactly as the receipt says; FALSE preds and met nulls are marked, never dropped) as markdown; the
author adds the title claim and the 'What it says' paragraph. Never edits any file. Sign convention line is copied from the receipt.
"""
import json, sys, hashlib
from pathlib import Path


def fmt(v):
    if isinstance(v, bool): return "TRUE" if v else "FALSE"
    if isinstance(v, float): return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sec = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--sec"), "NNNN")
    p = Path(args[0]); r = json.load(open(p)); sha = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    preds = r.get("preds", {}); nulls = r.get("nulls", {}); price = r.get("price", {})
    t = price.get("gpu_seconds") or price.get("cpu_seconds") or r.get("runtime_s") or 0
    fw = price.get("gpu_doc_forwards") or price.get("gpu_forwards") or price.get("cpu_full_forwards_docs") or "?"
    ok = [k for k, v in preds.items() if v]; bad = [k for k, v in preds.items() if not v]; met = [k for k, v in nulls.items() if v]
    letters = lambda ks: ", ".join(k.split("_")[1] for k in ks) or "none"
    print(f"\n## §{sec} — <TITLE CLAIM> ({r.get('rung')}; <lane>, {t:.0f} s, {fw} document-forwards): {letters(ok)} TRUE; "
          f"{letters(bad)} FALSE; null(s) met: {letters(met)}. Preserved.\n")
    print(f"Sign convention (§2135): {r.get('sign_convention', 'CE ADDED above the real model — LOWER IS BETTER')}. Preregistration "
          f"<PREREG.md> (<stamp>); script ops/{r.get('rung')}.py; results {p.name} (sha {sha}…).")
    if "instrument" in r:
        print("Instrument: " + "; ".join(f"{k} {fmt(v)}" for k, v in r["instrument"].items() if not isinstance(v, dict)) + ".")
    print("\n| pred | scored | null | met |\n|---|---|---|---|")
    nl = list(nulls.items())
    for i, (k, v) in enumerate(preds.items()):
        n = nl[i - 1] if 0 < i <= len(nl) else ("", "")
        print(f"| {k} | {fmt(v)} | {n[0]} | {fmt(n[1]) if n[0] else ''} |")
    print(f"\nBars {r.get('bars', {})}; null bars {r.get('null_bars', {})}.")
    ce = r.get("ce_added") or {s["site"]: s.get("ce_added_k32") for s in r.get("sites", [])} if isinstance(r.get("sites"), list) else r.get("ce_added")
    if isinstance(ce, dict) and ce:
        ks = list(ce)
        print("\n| arm | " + " | ".join(ks) + " |\n|---|" + "---|" * len(ks))
        print("| CE added | " + " | ".join(fmt(ce[k]) for k in ks) + " |")
    summ = r.get("summary", {})
    scal = {k: v for k, v in summ.items() if not isinstance(v, (dict, list))}
    if scal:
        print("\nSummary scalars: " + "; ".join(f"{k} {fmt(v)}" for k, v in scal.items()) + ".")
    print("\nWhat it says. <interpretation; state the arm each number belongs to; corrections to prior §§ go in a SEPARATE §>.")
    print(f"\nBACKLOG ROW:\n- §{sec} {r.get('rung')} (<lane>, {t:.0f} s, {fw} forwards): {letters(ok)} TRUE; {letters(bad)} FALSE; nulls met: {letters(met)}. <one line>.")


if __name__ == "__main__":
    main()
