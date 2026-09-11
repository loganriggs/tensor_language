"""Tally, from receipts on disk, the tasks that pass all four hypotheses (A1, A2, P, C).

Receipts come in two shapes, so this reports the UNION of two counts and labels each:
  per_target  -- the runner computed a `passes` flag per task against its registered bars;
  verdict     -- the receipt is a four-hypothesis battery whose OWN registered predicates were all True,
                 in which case every group carrying a `hypotheses` dict passed under those bars.
KNOWN OMISSION, BY DESIGN. correlative_pair passes all four but NO SINGLE RECEIPT says so: A1 and A2 come
from v503, whose P and C predicates were misregistered as removal damage <= 0.01 (a bar P rows cannot meet,
since P rows carry the same answer), and P and C were re-measured correctly in v505 via same_answer_effect.
This script counts per receipt, so it excludes it; quote the tally as N from single receipts, plus
correlative_pair split across two.
Neither is inferred from raw numbers by this script: both read the runner's own verdict. Census receipts
(reach/inertness) are excluded, since their predicates are about partner availability, not the four rows.
Task names are resolved to the FULL cell name (family prefix included) via the receipt's own
groups[key]["cells"], because bare keys collide across families -- `in_to` exists in both
verb_preposition and adjective_preposition, and stripping the prefix would merge them into one.
"""
import json, glob, os, collections

FAMILIES = ("verb_preposition_", "adjective_preposition_", "noun_preposition_", "possessive_",
            "correlative_", "verb_particle_")


def _bare(name):
    for f in FAMILIES:
        if name.startswith(f):
            return f[:-1], name[len(f):]
    return "?", name


def tally(root="circuits/followups"):
    per_target, verdict = {}, {}
    for p in sorted(glob.glob(os.path.join(root, "*_result.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        base = os.path.basename(p)
        pt = d.get("per_target") or (d.get("predictions") or {}).get("per_target")
        if isinstance(pt, dict):
            groups = d.get("groups") or {}
            for k, v in pt.items():
                if isinstance(v, dict) and v.get("passes") is True:
                    cells = ((groups.get(k) or {}).get("cells")) or [k]
                    for c in cells:
                        per_target.setdefault(c, base)
        pr = {k: v for k, v in (d.get("predictions") or {}).items() if k.startswith("pred_")}
        if not pr or not all(v is True for v in pr.values()):
            continue
        names = " ".join(pr).lower()
        if not (("a1" in names) and ("a2" in names) and ("p" in names) and ("c" in names)):
            continue
        if "reach" in names or "reachable" in names:      # census, not a battery
            continue
        for k, v in (d.get("groups") or {}).items():
            if not (v or {}).get("hypotheses"):
                continue
            for c in ((v or {}).get("cells") or [k]):
                verdict.setdefault(c, base)
    union = dict(verdict)
    union.update({k: v for k, v in per_target.items() if k not in union})
    return per_target, verdict, union


if __name__ == "__main__":
    pt, vd, un = tally()
    print("per_target passes : %d" % len(pt))
    print("runner-verdict    : %d" % len(vd))
    print("UNION             : %d distinct tasks passing all four hypotheses" % len(un))
    fam = collections.Counter(_bare(k)[0] for k in un)
    for f, n in sorted(fam.items(), key=lambda x: -x[1]):
        print("   %-22s %d" % (f, n))
    for k, v in sorted(un.items()):
        print("     %-24s %s" % (k, v))
