"""circuit_prior_art_search -- "has this been done, or already failed?" in one command. READ-ONLY.

The directive requires searching the dossier AND the ledgers before starting any candidate, so we neither
duplicate a known result nor rediscover a known failed target. Two things make that slow by hand:

  1. The event authority is `circuits/task_*.json` (`evidence_events`) plus the append-only fast-screen ledger,
     not the rendered `circuits/DOSSIER.md`,
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
import re
import json
import os
import sys
import collections

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIRCUITS = os.path.join(BQ, "circuits")
DOSSIER = os.path.join(CIRCUITS, "DOSSIER.md")
FAILED_VERDICTS = ("invalid", "null")
FAST_SCREEN_LEDGER = os.path.join(CIRCUITS, "fast_screen_ledger.jsonl")


def _load_fast_screen_events():
    """Normalize terminal fast-screen receipts into the event search vocabulary."""
    out = []
    try:
        lines = open(FAST_SCREEN_LEDGER, errors="strict").read().splitlines()
    except OSError:
        return out
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            receipt = json.loads(line)
        except ValueError as error:
            raise RuntimeError(f"invalid fast-screen ledger line {number}") from error
        if not isinstance(receipt, dict):
            raise RuntimeError(f"fast-screen ledger line {number} is not an object")
        terminal = receipt.get("terminal")
        event = dict(receipt)
        event.update({
            "event_id": receipt.get("request_id"),
            "verdict": terminal,
            "stage": "complete" if terminal in {"screen", "null"} else "invalid",
            "site_id": receipt.get("selected_site_id"),
        })
        out.append(("fast_screen_ledger.jsonl", receipt.get("candidate_id") or "", event))
    return out


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
    out.extend(_load_fast_screen_events())
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


def coverage():
    """Per behaviour: what verdicts exist, and is there anything positive to build on?

    Written after the mandatory search stopped me twice in one tick: both cheap candidates I could have
    revived (`pronoun-antecedent-gender-reference-v1`, `quote-parity-pending-close-v1`) are recorded NULLS,
    so authoring either would have rediscovered a known failed target. That is the gate working -- but it
    answers "is this one dead?" one candidate at a time, and the expensive question when the fast loop
    stalls is "where is there anything left to build on?". A behaviour with `held` results has a positive
    foothold; one with only nulls/invalids has been probed and has not yielded.
    """
    out = {}
    for task, tag, e in load_events():
        rec = out.setdefault(task, collections.Counter())
        rec[e.get("verdict") or "?"] += 1
    return out


CAP_RE = re.compile(r"capability|native", re.I)
LOC_RE = re.compile(r"localization|localisation|site|factor|projector|das|interchange|removal|reuse|transfer", re.I)


def localisation():
    """Per behaviour: does capability hold, and has any LOCALISATION attempt ever held?

    `--coverage` says where footholds exist; this says what KIND. A behaviour whose capability holds but
    whose every localisation attempt is null is a different proposition from one that already localises.
    Added after I hypothesised the former was the general case, checked, and found it false: localisation
    holds in 6 of 7 behaviours. The check is kept because the hypothesis was worth refuting cheaply.
    """
    out = {}
    for task, _tag, e in load_events():
        rec = out.setdefault(task, {"capability": None, "localisation": collections.Counter()})
        if e.get("stage") != "complete":
            continue
        eid = str(e.get("event_id", ""))
        if CAP_RE.search(eid):
            if e.get("verdict") == "held" or rec["capability"] is None:
                rec["capability"] = e.get("verdict")
        elif LOC_RE.search(eid):
            rec["localisation"][e.get("verdict")] += 1
    return out


if __name__ == "__main__":
    if "--localisation" in sys.argv:
        for task, rec in sorted(localisation().items()):
            loc = rec["localisation"]
            held = loc.get("held", 0)
            note = ("localises" if held else
                    ("capability only -- no localisation has ever held" if rec["capability"] == "held"
                     else "no capability result"))
            print(f"{task[:44]:<44} capability={str(rec['capability']):<8} "
                  f"localisation={dict(loc)}  {note}")
        raise SystemExit(0)
    if "--coverage" in sys.argv:
        cov = coverage()
        print(f"{'behaviour record':<44} {'held':>5} {'null':>5} {'inval':>6} {'incon':>6}  foothold")
        for task, c in sorted(cov.items(), key=lambda kv: -kv[1].get("held", 0)):
            foot = "YES" if c.get("held") else "none -- probed, nothing positive"
            print(f"{task[:44]:<44} {c.get('held',0):>5} {c.get('null',0):>5} "
                  f"{c.get('invalid',0):>6} {c.get('inconclusive',0):>6}  {foot}")
        raise SystemExit(0)
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
