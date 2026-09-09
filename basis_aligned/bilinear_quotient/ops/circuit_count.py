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
 R7 (2026-09-09, v371/v373/v375) FRAME COPIES OF ONE MAPPING PAIR COUNT ONCE. A cell that reuses both of another
    cell's (cue word -> readout token) mappings in a different sentence frame is separable from it at rank 1, and I
    counted such cells separately until v371 showed that ONE unit set and ONE rank-1 direction fitted on POOLED rows
    covers every shape at the per-shape specialists' level (0.993-1.008 against 0.991-0.998), with generality costing
    nothing. v373 replicated it in the preposition class and v375 audited all ten remaining counted frame-copy groups:
    every one merged, joint per-shape extraction 0.986-1.018, minimum pooled-unit recovery 0.844, maximum own-C
    UB975 0.0106. So rank-1 separability shows that a SPECIALISED direction exists which spares the others, not that
    the mechanism differs, and each group below is ONE circuit. Receipts: unit_broad_circuit_v371/v373/v375.
"""
import json, glob, re, sys

INSTRUMENT = {"finiteness_selection"}
# R5: family-less passes counted immediately (ledger), and known fused groups among them
SINGLETONS = ["numbered_list_choice", "numeric_sequence_choice", "narrative_tense", "degree_frame",
              "interrogative_licensing", "possessive_argument", "preposition_selection", "numeral_dual_both_all"]
SINGLETON_GROUPS = [["correlative_both_either", "correlative_both_neither", "correlative_either_neither"]]

# R7: frame copies verified to merge (unit_broad_circuit_v371, v373, v375). Each list counts ONCE.
MERGED_FRAME_GROUPS = [
    ["verb_particle_up_down", "verb_particle_fb_up_down", "verb_particle_fc_up_down", "verb_particle_fd_up_down",
     "verb_particle_fe_up_down", "verb_particle_fg_up_down", "verb_particle_fh_up_down"],          # v371 + v363/v367
    ["verb_preposition_about_for", "verb_preposition_fb_about_for", "verb_preposition_fc_about_for"],   # v373
    ["verb_particle_out_down", "verb_particle_fb_out_down", "verb_particle_fc_out_down"],               # v375
    ["verb_particle_out_up", "verb_particle_fc_out_up"],
    ["verb_preposition_at_to", "verb_preposition_fb_at_to"],
    ["verb_preposition_from_about", "verb_preposition_fb_from_about"],
    ["verb_preposition_from_for", "verb_preposition_fb_from_for"],
    ["verb_preposition_into_with", "verb_preposition_fb_into_with"],
    ["verb_preposition_by_from", "verb_preposition_fc_by_from"],
    ["verb_preposition_for_at", "verb_preposition_fc_for_at"],
    ["verb_preposition_of_into", "verb_preposition_fc_of_into"],
    ["verb_preposition_through_with", "verb_preposition_fc_through_with"],
]
CANON = {n: grp[0] for grp in MERGED_FRAME_GROUPS for n in grp}

passes = set()
for f in glob.glob("circuits/followups/*_result.json"):
    try: R = json.load(open(f))
    except Exception: continue
    for n in R.get("four_row_passes", []) or []:
        if n not in INSTRUMENT: passes.add(n)

last, leak = {}, {}
for f in sorted(glob.glob("circuits/followups/unit_family_separability_spec_v*_result.json"),
                key=lambda p: int(re.search(r"spec_v(\d+)_", p).group(1))):
    for n, m in json.load(open(f)).get("members", {}).items():
        if "separable" in m and "error" not in m:   # an errored member (e.g. CUDA OOM at v289) is NOT a verdict
            last[n] = m["separable"]
            if not m["separable"] and "arms" in m:
                leak[n] = max(m["arms"]["fam"]["siblings"].items(), key=lambda kv: abs(kv[1]))[0]

sep_all = {n for n, s in last.items() if s}
sep_raw = {n for n in sep_all if n in passes}                  # R2 before the frame-copy merge
sep = {CANON.get(n, n) for n in sep_raw}                       # R2 + R7: frame copies collapse onto their group
merged_saved = len(sep_raw) - len(sep)
fused = [n for n, s in last.items() if not s and n in passes]
sep_all_canon = {CANON.get(n, n) for n in sep_all}
# R3 + R7: a fused cell is "attached" when its leak partner is separable, OR when it is itself a frame copy of a
# merged group that already counts (its canonical member holds the count) -- otherwise the merge would be undone by
# the group clause below.
attached = [n for n in fused
            if CANON.get(leak.get(n), leak.get(n)) in sep_all_canon or CANON.get(n, n) in sep]
adj = {}
for n in fused:
    if n in attached:
        continue
    p = leak.get(n)
    if p and CANON.get(p, p) not in sep_all_canon:
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
print(f"R2 separable at latest rung                    +  {len(sep)}   ({len(sep_raw)} cells, {merged_saved} folded by R7)")
print(f"R3 fused into a separable incumbent            +  0   ({len(attached)} cells)")
print(f"R4 fused groups with no separable member       +  {len(groups)}")
print(f"R5 family-less singletons                      +  {len(singles)}   {singles}")
print(f"R5 family-less fused groups                    +  {len(sgroups)}   {[g[0]+'...' for g in sgroups]}")
print(f"R6 awaiting a separability rung                +  0   ({len(pending)} cells)")
print(f"\nDISTINCT CIRCUITS = {total}")
print(f"pending: {pending}")
