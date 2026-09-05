"""screen_null_scope -- state precisely what a `no_selective_causal_site` null does and does not say.

I reported two such nulls as "no single site carries the state". The 01:46Z positive control showed that is
wrong: the engine's fixed bars are only reachable by HIGH-recovery sites. `resid:08` -- the site R538
selected for the bracket behaviour under full-state interchange -- reaches target recovery 0.282 under this
instrument and fails. A carrier of that kind is invisible here.

So the honest statement is always "the best site recovered X against a bar of Y", never "nothing carries it".
Producing that sentence by hand is how I got it wrong twice, so this produces it from the receipt.

Usage:
  python ops/screen_null_scope.py <receipt.json> [more_receipts.json ...]
"""
import json
import os
import sys

# The bars that actually gate site selection. An earlier version of this tool guessed by taking the largest
# numeric fixed bar and printed `minimum_a1_capability_accuracy` (0.85) as if it gated recovery -- it does
# not; it gates NATIVE CAPABILITY. Reporting the wrong bar is how a null gets misread.
GATING_BARS = ("minimum_target_family_recovery", "minimum_target_direction_fraction",
               "maximum_c_absolute_recovery", "maximum_p_invariance_effect")


def scope(path):
    doc = json.load(open(path))
    sites = doc.get("run", {}).get("site_results") or []
    rec = [(s["site"]["site_id"], s.get("target_recovery") or 0.0) for s in sites]
    rec.sort(key=lambda kv: -kv[1])
    passing = [s["site"]["site_id"] for s in sites if not (s.get("reasons") or [])]
    fixed = doc.get("fixed_bars") or {}
    best = sites[0] if sites else None
    for s in sites:                       # the best site by target recovery, for its binding reasons
        if s.get("target_recovery", 0) >= (best.get("target_recovery", 0) if best else 0):
            best = s
    import collections
    why = collections.Counter()
    for s in sites:
        for r in (s.get("reasons") or []):
            why[str(r)] += 1
    return {"path": path, "reason": doc.get("reason"), "selected": doc.get("selected_site_id"),
            "n_sites": len(sites), "top": rec[:5], "passing": passing,
            "gating_bars": {k: fixed[k] for k in GATING_BARS if k in fixed},
            "best_site": best, "why": why}



# A screen can stop before any site is scored (`head_stage == "capability_stop"`): the engine refuses to
# screen sites when the model cannot natively perform the cells. Then `run.site_results` is EMPTY, and the
# site-oriented report above prints 0.000 for every field -- which reads as "measured, and it was zero"
# rather than "never measured". That silent misreport is what this handles.
TARGET_FAMILIES = ("A1", "A2")   # the behaviour under study; P and C are its edit and its control


def capability_scope(doc):
    """Which families failed the native-capability gate, and how strongly the model did each."""
    run = doc.get("run") or {}
    cells = run.get("capability_cells") or []
    rows = run.get("native_logits") or []
    families = {}
    for cell in cells:
        families.setdefault(cell["family"], []).append(cell)
    margins = {}
    for row in rows:
        margins.setdefault(row["family"], []).append(row["answer_logit"] - row["foil_logit"])
    failed = {f for f, cs in families.items() if any(not c.get("passed") for c in cs)}
    return {"families": families, "margins": margins, "failed": failed,
            "target_failed": failed & set(TARGET_FAMILIES), "head_stage": run.get("head_stage")}


def print_capability_stop(doc):
    cap = capability_scope(doc)
    print(f"  STOPPED AT: {cap['head_stage']} -- no site was ever screened, so every site-level "
          f"number is absent, not zero.")
    for family in sorted(cap["families"]):
        cells = cap["families"][family]
        bad = [c for c in cells if not c.get("passed")]
        margin = cap["margins"].get(family) or [0.0]
        mean = sum(margin) / len(margin)
        right = sum(1 for v in margin if v > 0)
        mark = "FAIL" if bad else "ok  "
        print(f"   {mark} {family:3s} {len(cells) - len(bad)}/{len(cells)} cells pass   "
              f"native answer-vs-foil margin {mean:+.2f}  ({right}/{len(margin)} rows correct)")
        for cell in bad:
            print(f"        failing cell {cell['cell_id']}  base {cell['base_accuracy']:.2f} "
                  f"donor {cell['donor_accuracy']:.2f} (bar {cell['minimum_accuracy']})")
    if cap["target_failed"]:
        print(f"  READ AS: TARGET INCAPABLE. {sorted(cap['target_failed'])} failed, so the model cannot "
              "perform the behaviour under study. The null is about the model and it stands; no control "
              "redesign changes it.")
    elif cap["failed"]:
        print(f"  READ AS: CONTROL INCAPABLE, NOT A NULL ABOUT THE BEHAVIOUR. {sorted(TARGET_FAMILIES)} "
              f"pass natively -- the model DOES the behaviour -- and only {sorted(cap['failed'])} failed. "
              "The verdict `native_behavior_incapable` names the wrong cause: what the model cannot do is "
              "the CONTROL that was chosen. This is an authoring fix (pick a control the model can "
              "perform), and the behaviour has never actually been screened.")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        s = scope(path)
        print(f"\n{os.path.basename(s['path'])}")
        print(f"  verdict {s['reason']}   selected {s['selected']}   sites {s['n_sites']}")
        print(f"  gating bars: {s['gating_bars']}")
        if s["top"]:
            best_id, best = s["top"][0]
            print(f"  best target recovery: {best:.3f} at {best_id}")
            print("  top sites: " + ", ".join(f"{i}={v:.3f}" for i, v in s["top"]))
        b = s["best_site"] or {}
        if s["n_sites"]:
            a1, a2 = b.get("a1") or {}, b.get("a2") or {}
            print(f"  best site detail: A1 recovery {a1.get('mean_absolute_effect', 0):.3f} "
                  f"dir {a1.get('direction_fraction', 0):.2f} | A2 {a2.get('mean_absolute_effect', 0):.3f} "
                  f"dir {a2.get('direction_fraction', 0):.2f} | C {b.get('c_absolute_recovery', 0):.3f} "
                  f"| P {b.get('p_invariance_effect', 0):.3f}")
            print(f"  its rejection reasons: {b.get('reasons')}")
            print("  rejection reasons across all sites: "
                  + ", ".join(f"{k} x{v}" for k, v in s["why"].most_common(4)))
        if s["n_sites"] == 0:
            print_capability_stop(json.load(open(s["path"])))
        elif s["reason"] == "no_selective_causal_site":
            print("  READ AS: the binding constraint is whichever reason above dominates -- state it "
                  "explicitly. If it is C_absolute_recovery, the target WAS recovered and the failure is "
                  "SELECTIVITY, not recovery.")
        else:
            print(f"  passing sites: {s['passing']}")
