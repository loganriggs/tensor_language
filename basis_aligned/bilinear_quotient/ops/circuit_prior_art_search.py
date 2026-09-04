"""circuit_prior_art_search -- "has this been done, or already failed?" in one command. READ-ONLY.

The directive requires searching the dossier AND the ledger before starting any candidate, so we neither
duplicate a known result nor rediscover a known failed target. Two things make that slow by hand:

  1. The event authority is `circuits/task_*.json` (`evidence_events`), not the rendered `circuits/DOSSIER.md`,
     and the dossier goes stale -- Codex flagged a bracket-task "next" entry whose R546/R548/R549/R551/R556/
     R560/R561 results already existed in the task JSON. Reading the dossier alone can send you to redo
     finished work.
  2. Known FAILED targets are the expensive ones to rediscover, and they are spread across four verdicts:
     `invalid` (method voided), `null` (measured and negative), plus `inconclusive` and `held`.

This searches the authority, reports verdicts, and separately reports which authority events the dossier
does not mention -- so "the dossier does not say so" is never mistaken for "it has not been done".

Usage:
  python ops/circuit_prior_art_search.py head11.3 agreement      # free-text terms
  python ops/circuit_prior_art_search.py --stale                 # dossier-vs-authority drift only
"""
import glob
import json
import os
import sys
import collections

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIRCUITS = os.path.join(BQ, "circuits")
DOSSIER = os.path.join(CIRCUITS, "DOSSIER.md")
FAILED_VERDICTS = ("invalid", "null")


def load_events():
    out = []
    for path in sorted(glob.glob(os.path.join(CIRCUITS, "task_*.json"))):
        try:
            doc = json.load(open(path))
        except (OSError, ValueError):
            continue
        events = doc.get("evidence_events") or []
        if isinstance(events, dict):
            events = list(events.values())
        for e in events:
            if isinstance(e, dict):
                out.append((os.path.basename(path), doc.get("tag") or "", e))
    return out


def search(terms):
    hits = []
    for task, tag, e in load_events():
        blob = json.dumps(e).lower() + " " + task.lower() + " " + str(tag).lower()
        if all(t.lower() in blob for t in terms):
            hits.append((task, e))
    return hits


def stale_dossier():
    """Authority events the rendered dossier never mentions."""
    try:
        text = open(DOSSIER, errors="ignore").read()
    except OSError:
        return {}
    missing = collections.defaultdict(list)
    for task, _tag, e in load_events():
        eid = e.get("event_id")
        if eid and eid not in text:
            missing[task].append((eid, e.get("verdict")))
    return missing


def _fmt(task, e):
    v = e.get("verdict")
    mark = "FAILED-TARGET" if v in FAILED_VERDICTS else "prior"
    bits = [f"{mark:<13}", f"{v or '?':<12}", f"{e.get('event_id', '?')}"]
    extra = [f"{k}={e[k]}" for k in ("stage", "site_id", "failure_kind") if e.get(k)]
    return f"  {' '.join(bits)}\n      {task}  " + "  ".join(extra)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--stale" in sys.argv or not args:
        missing = stale_dossier()
        total = sum(len(v) for v in missing.values())
        print(f"authority events NOT mentioned in DOSSIER.md: {total}")
        for task, items in sorted(missing.items()):
            print(f"  {task}: {len(items)}")
            for eid, v in items[:6]:
                print(f"      {v or '?':<12} {eid}")
        if not args:
            raise SystemExit(0)
        print()
    hits = search(args)
    failed = [h for h in hits if h[1].get("verdict") in FAILED_VERDICTS]
    print(f"search {args}: {len(hits)} prior event(s), {len(failed)} already-failed target(s)")
    for task, e in hits:
        print(_fmt(task, e))
    if failed:
        print(f"\n{len(failed)} FAILED TARGET(S) above: rediscovering these is not a result.")
    raise SystemExit(1 if failed else 0)
