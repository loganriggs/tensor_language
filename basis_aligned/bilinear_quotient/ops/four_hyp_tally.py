"""Tally tasks passing all four hypotheses, from receipts on disk only.

Counts ONLY per_target receipts, where `passes` is computed by the runner against registered bars.
Older single-task receipts use other shapes and are NOT counted here, so this is a LOWER BOUND
on the corpus total and is labelled as one wherever it is quoted.
"""
import json, glob, collections
tasks = {}
for p in sorted(glob.glob("circuits/followups/*_result.json")):
    try: d = json.load(open(p))
    except Exception: continue
    pt = d.get("per_target") or (d.get("predictions") or {}).get("per_target")
    if not isinstance(pt, dict): continue
    for k, v in pt.items():
        if isinstance(v, dict) and v.get("passes") is True:
            tasks[k] = p.split("/")[-1]
fam = collections.Counter()
for k in tasks:
    fam["verb_preposition" if k in ("at_to","by_from","from_for","about_for") else "?"] += 1
print(f"per_target receipts with passes=True: {len(tasks)} distinct task keys (LOWER BOUND)")
for k, v in sorted(tasks.items()): print(f"  {k:28s} {v}")
