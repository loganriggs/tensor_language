"""screen_leak_compare -- compare the control leaks of two fast-screen receipts. READ-ONLY.

A single screen that terminates `no_selective_causal_site` says only "nothing passed". Two such screens on
different behaviours say considerably more, because each carries an unrelated C control: if the SAME sites
recover the control in both, that is evidence of a shared route rather than two independent failures.

Finding this by hand across the first two nulls took three ticks and one wrong claim (I first reported the
leak as unique to the second screen; it was in both). It is a mechanical comparison, so it should cost one
command:

  * are the ranked site sets identical (otherwise overlap means nothing);
  * how many sites leak in each, how many in both, and how many would overlap by chance;
  * which NON-residual sites leak -- late residual sites move any endpoint when patched near the output,
    so they are expected and uninformative; an attention or MLP block leaking is not.

Usage:
  python ops/screen_leak_compare.py <receipt_a.json> <receipt_b.json>
"""
import json
import os
import sys

LEAK_REASON = "C_absolute_recovery_above"
INFORMATIVE_FLOOR = 0.5


def load(path):
    doc = json.load(open(path))
    sites, leaks, recovery, kinds = [], set(), {}, {}
    for entry in doc.get("run", {}).get("site_results") or []:
        site = entry.get("site") or {}
        sid = site.get("site_id")
        if not sid:
            continue
        sites.append(sid)
        recovery[sid] = entry.get("c_absolute_recovery")
        kinds[sid] = site.get("evidence_kind")
        if any(LEAK_REASON in str(r) for r in (entry.get("reasons") or [])):
            leaks.add(sid)
    return {"sites": sites, "leaks": leaks, "recovery": recovery, "kinds": kinds,
            "reason": doc.get("reason"), "path": path}


if __name__ == "__main__":
    a, b = load(sys.argv[1]), load(sys.argv[2])
    na, nb = os.path.basename(a["path"])[:38], os.path.basename(b["path"])[:38]
    same = set(a["sites"]) == set(b["sites"])
    print(f"A {na}  terminal={a['reason']}")
    print(f"B {nb}  terminal={b['reason']}")
    print(f"\nranked sites: A {len(a['sites'])}, B {len(b['sites'])}, identical sets: {same}")
    if not same:
        print("site sets differ -- overlap below is not interpretable")
    inter = a["leaks"] & b["leaks"]
    n = len(set(a["sites"]) | set(b["sites"]))
    expected = len(a["leaks"]) * len(b["leaks"]) / n if n else 0.0
    print(f"control leaks: A {len(a['leaks'])}, B {len(b['leaks'])}, "
          f"overlap {len(inter)} (expected {expected:.1f} if independent)")
    print(f"  A only: {sorted(a['leaks'] - b['leaks'])}")
    print(f"  B only: {sorted(b['leaks'] - a['leaks'])}")
    informative = sorted(
        sid for sid in inter
        if not str(sid).startswith("resid")
        and (a["recovery"].get(sid) or 0) >= INFORMATIVE_FLOOR)
    print(f"\nNON-residual sites leaking in both (residual sites near the output move any endpoint, "
          f"so they are expected):")
    if not informative:
        print("  none -- the shared leak is entirely late-residual and therefore uninformative")
    for sid in informative:
        print(f"  {sid:<14} kind={a['kinds'].get(sid):<8} A {a['recovery'][sid]:.3f}  B {b['recovery'][sid]:.3f}")
