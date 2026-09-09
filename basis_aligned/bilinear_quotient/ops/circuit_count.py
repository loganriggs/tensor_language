#!/usr/bin/env python3
"""count.py — compute DISTINCT CIRCUITS from receipts under the registered count rules, with a rule-by-rule breakdown.

Rules, as registered across the ledger:
 R1 a cell counts only if it has a four-row pass (authority: a `four_row_passes` entry in some receipt).
    finiteness_selection is the INSTRUMENT and never counts (v249 disclosure).
 R2 a cell separable at its LATEST separability rung counts +1.
 R3 a cell that fused into a SEPARABLE incumbent counts +0 (the incumbent holds the group's count) -- v242.
 R4 fused cells whose partners are also fused form one connected group, counting +1 for the group -- v242 group clause.
 R5 a four-row pass with NO family (never separability-evaluated and no family exists for it) counts +1 as a singleton.
    A GROUP of such cells known to be one fused circuit counts once (the correlative trio, v114).
 R6 a four-row pass awaiting a separability rung in an EXISTING family counts 0 until that rung runs.
"""
import json, glob, re, sys

INSTRUMENT = {"finiteness_selection"}
# R5: family-less passes counted immediately (ledger), and known fused groups among them
SINGLETONS = ["numbered_list_choice", "numeric_sequence_choice", "narrative_tense", "degree_frame",
              "interrogative_licensing", "possessive_argument", "preposition_selection", "numeral_dual_both_all"]
SINGLETON_GROUPS = [["correlative_both_either", "correlative_both_neither", "correlative_either_neither"]]

passes = set()
for f in glob.glob("circuits/followups/*_result.json"):
    try: R = json.load(open(f))
    except Exception: continue
    for n in R.get("four_row_passes", []) or []:
        if n not in INSTRUMENT: passes.add(n)

last, leak = {}, {}
for f in sorted(glob.glob("circuits/followups/unit_family_separability_spec_v2*_result.json"),
                key=lambda p: int(re.search(r"spec_v(\d+)_", p).group(1))):
    for n, m in json.load(open(f)).get("members", {}).items():
        if "separable" in m and "error" not in m:   # an errored member (e.g. CUDA OOM at v289) is NOT a verdict
            last[n] = m["separable"]
            if not m["separable"] and "arms" in m:
                leak[n] = max(m["arms"]["fam"]["siblings"].items(), key=lambda kv: abs(kv[1]))[0]

sep_all = {n for n, s in last.items() if s}
sep = {n for n in sep_all if n in passes}                      # R2
fused = [n for n, s in last.items() if not s and n in passes]
attached = [n for n in fused if leak.get(n) in sep_all]        # R3
adj = {}
for n in fused:
    p = leak.get(n)
    if p and p not in sep_all:
        adj.setdefault(n, set()).add(p); adj.setdefault(p, set()).add(n)
seen, groups = set(), []
for n in list(adj):
    if n in seen: continue
    stack, c = [n], set()
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x); c.add(x); stack += list(adj.get(x, ()))
    if c & passes: groups.append(sorted(c))                     # R4
grouped = {x for g in SINGLETON_GROUPS for x in g}
singles = [n for n in SINGLETONS if n in passes and n not in last]                       # R5
sgroups = [g for g in SINGLETON_GROUPS if any(x in passes and x not in last for x in g)]  # R5 group
pending = sorted(n for n in passes if n not in last and n not in SINGLETONS and n not in grouped)  # R6

total = len(sep) + len(groups) + len(singles) + len(sgroups)
print(f"R1 four-row passes (instrument excluded)          {len(passes)}")
print(f"R2 separable at latest rung                    +  {len(sep)}")
print(f"R3 fused into a separable incumbent            +  0   ({len(attached)} cells)")
print(f"R4 fused groups with no separable member       +  {len(groups)}")
print(f"R5 family-less singletons                      +  {len(singles)}   {singles}")
print(f"R5 family-less fused groups                    +  {len(sgroups)}   {[g[0]+'...' for g in sgroups]}")
print(f"R6 awaiting a separability rung                +  0   ({len(pending)} cells)")
print(f"\nDISTINCT CIRCUITS = {total}")
print(f"pending: {pending}")
