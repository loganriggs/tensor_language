#!/usr/bin/env python3
# BQGATE: LIBRARY -- CPU summary of the readout-atlas per-line receipts (interim and final tables).
"""Reads circuits/followups/atlas_*_v68_result.json and prints: counts (lines, capable, live, selective), the recurring heads
among live non-auxiliary lines, recurring head triples, and a markdown table (family | contrast | top-4 | fraction | live |
selective | family overlap). Usage: python atlas_summary.py [--md out.md]"""
from __future__ import annotations
import collections
FAMILIES = {"temporal": {"attn9_h1", "attn9_h4", "attn15_h5"}, "number": {"attn5_h7", "attn7_h8", "attn9_h7"}, "pronoun": {"attn9_h6", "attn12_h4", "attn15_h1"},
            "selection": {"attn13_h8", "attn7_h8", "attn8_h8"}, "person": {"attn8_h1", "attn13_h1", "attn10_h5", "attn15_h1"}}


def family_of(top4):
    """A live line belongs to a family when >= 2 of the family core heads are in its top-4; ties -> the larger overlap, then name order."""
    best = max(FAMILIES, key=lambda f: (len(FAMILIES[f] & set(top4)), -list(FAMILIES).index(f)))
    return best if len(FAMILIES[best] & set(top4)) >= 2 else "other"


import glob, itertools, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUX = {" was", " were", " is", " has", " had", " have", " will", " would"}
FAMILY = ("attn8_h1", "attn9_h1", "attn9_h4", "attn11_h3", "attn15_h5")


def load():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "circuits/followups/atlas_*_v68_result.json"))):
        r = json.load(open(f))
        if r.get("status") == "no_rows":
            continue
        cap = all(v is not None and v >= 0.85 for v in r["capability"].values())
        live = r["joint"]["target_damage_fraction"] >= 0.10 and r["joint"]["target_damage_positive_fraction"] >= 0.75 and r["joint"]["target_damage_mean"] > r["null_damage_max"]
        rows.append({"family": r["candidate_id"].split(".")[0], "task": r["candidate_id"].rsplit(".atlas", 1)[0], "capable": cap, "live": live, "fraction": r["joint"]["target_damage_fraction"],
                     "selective": all(r["gates"].values()), "top4": r["set"], "top4_damages": [d for _, d in r["sweep_top20"][:4]], "overlap": r["family_overlap"], "instrument": r["instrument_max_abs_error"]})
    return rows


def main():
    rows = load()
    cap = [r for r in rows if r["capable"]]; live = [r for r in cap if r["live"]]; sel = [r for r in live if r["selective"]]
    print(f"lines {len(rows)}  capable {len(cap)}  live {len(live)}  selective {len(sel)}  instrument max {max((r['instrument'] for r in rows), default=0):.1e}")
    heads = collections.Counter(h for r in live for h in r["top4"]); print("recurring heads (live lines):", heads.most_common(15))
    triples = collections.Counter(t for r in live for t in itertools.combinations(sorted(r["top4"]), 3)); print("recurring triples:", [(list(t), c) for t, c in triples.most_common(8) if c >= 2])
    assigned = collections.Counter(family_of(r["top4"]) for r in live); print("family assignment (live lines, >= 2 core heads in top-4):", dict(assigned))
    other = [r for r in live if family_of(r["top4"]) == "other"]
    other_triples = collections.Counter(t for r in other for t in itertools.combinations(sorted(r["top4"]), 3)); print("recurring triples among 'other':", [(list(t), c) for t, c in other_triples.most_common(6) if c >= 2])
    if "--md" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--md") + 1])
        lines = [f"Lines {len(rows)}, capable {len(cap)}, live {len(live)}, selective {len(sel)}. Family = >= 2 core heads in the top-4, largest overlap wins (temporal 9.1/9.4/15.5, number 5.7/7.8/9.7, pronoun 9.6/12.4/15.1, selection 13.8/7.8/8.8, person 8.1/13.1/10.5/15.1): " + ", ".join(f"{k} {v}" for k, v in assigned.most_common()) + ".", "",
                 "| line | top-4 heads (logits) | set fraction | live | selective | family | aux overlap |", "|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -r["fraction"]):
            hs = ", ".join(f"{h.replace('attn','').replace('_h','.')} {d:.2f}" for h, d in zip(r["top4"], r["top4_damages"]))
            lines.append(f"| {r['family']}{'' if r['capable'] else ' (incapable)'} | {hs} | {r['fraction']:.2f} | {'yes' if r['live'] else 'no'} | {'yes' if r['selective'] else 'no'} | {family_of(r['top4']) if r['live'] else '—'} | {', '.join(h.replace('attn','').replace('_h','.') for h in r['overlap']) or '—'} |")
        out.write_text("\n".join(lines) + "\n"); print("wrote", out)


if __name__ == "__main__":
    main()
