"""circuit_latency -- serial minutes per circuit, from repository timestamps. READ-ONLY.

The standing directive requires, at a safe boundary every UTC hour: measure the serial path from prior-art
check through candidate, managed runner and terminal receipt; target one meaningful screen or honest null per
TEN serial minutes; and if the median exceeds that, make one bounded systems change before opening another
bespoke experiment. Doing that measurement by hand costs several shell round-trips every hour, which is the
same repeated-step waste the directive is aimed at -- so it is automated here.

Stages, joined on the candidate id in `circuits/fast_screen_ledger.jsonl`:

    prior_art   circuits/fast_screen_<slug>_prior_art.json      mtime
    candidate   ops/circuit_fast_screen_candidate_<slug>.py     mtime
    runner      ops/run_circuit_fast_screen_<slug>.py           mtime
    terminal    ledger `finished_utc` for that candidate

HONEST LIMIT, because it changes how the numbers should be read: an mtime is the LAST write, not the first.
A file edited after its screen ran reports a later stage time, which can compress or invert a stage. Rows
where that happens are marked `?`. The terminal time comes from the ledger and is exact; `serial_seconds`
(the compute itself) is exact too. Treat stage splits as indicative and the total as a lower bound on
elapsed design time.

Usage:  python ops/circuit_latency.py [--since HH:MM]
"""
import datetime
import glob
import json
import os
import re
import statistics
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(BQ, "circuits", "fast_screen_ledger.jsonl")
TARGET_MIN = 10.0


def _mtime(path):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).replace(tzinfo=None)
    except OSError:
        return None


def _slugs(candidate_id):
    """Candidate ids and file slugs do not match one-to-one; try the plausible spellings."""
    base = candidate_id.split(".")[0]
    tail = candidate_id.split(".")[-1]
    out = {base, tail, candidate_id.replace(".", "_"), base.replace("_", "")}
    if base == "subject_verb":
        out |= {"task14_agreement", "task14_cross_syntax"}
    if base == "sentence_terminal":
        out.add("sentence_terminal")
    if base == "pronoun_antecedent":
        out.add("pronoun")
    if base == "quote_parity":
        out.add("quote_parity")
    return out


def _first_existing(patterns):
    best = None
    for pat in patterns:
        for p in glob.glob(pat):
            t = _mtime(p)
            if t and (best is None or t < best[0]):
                best = (t, p)
    return best


def rows():
    out = []
    if not os.path.exists(LEDGER):
        return out
    for ln in open(LEDGER, errors="ignore"):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        cid = d.get("candidate_id")
        fin = d.get("finished_utc")
        if not cid or not fin:
            continue
        terminal = datetime.datetime.strptime(fin[:19], "%Y-%m-%dT%H:%M:%S")
        stages = {}
        for name, pats in (
            ("prior_art", [os.path.join(BQ, "circuits", f"*{s}*prior_art*.json") for s in _slugs(cid)]),
            ("candidate", [os.path.join(BQ, "ops", f"*candidate*{s}*.py") for s in _slugs(cid)]),
            ("runner", [os.path.join(BQ, "ops", f"run_*{s}*.py") for s in _slugs(cid)]),
        ):
            hit = _first_existing(pats)
            stages[name] = hit[0] if hit else None
        out.append({"candidate_id": cid, "terminal": terminal, "stages": stages,
                    "serial_seconds": d.get("serial_seconds")})
    return out


if __name__ == "__main__":
    data = rows()
    if not data:
        print("no ledger rows"); raise SystemExit(0)
    since = None
    if "--since" in sys.argv:
        hh, mm = sys.argv[sys.argv.index("--since") + 1].split(":")
        since = data[0]["terminal"].replace(hour=int(hh), minute=int(mm), second=0)
    totals = []
    # A REPEAT screen of the same candidate does not start from the family's first file -- that would charge
    # it for every earlier screen in the family (head11.3_complement read 77.7 min that way, which is an
    # artifact of file matching, not a design cycle). Its serial clock starts at the PREVIOUS terminal.
    prev_terminal = {}
    print(f"{'candidate':<46} {'start':>8} {'terminal':>9} {'serial_min':>11} {'compute_s':>10}")
    for r in data:
        if since and r["terminal"] < since:
            continue
        starts = [t for t in r["stages"].values() if t]
        if not starts:
            print(f"{r['candidate_id'][:46]:<46} {'?':>8} {r['terminal']:%H:%M:%S}  (no stage files matched)")
            continue
        start = min(starts)
        fam = r["candidate_id"].split(".")[0]
        if fam in prev_terminal and prev_terminal[fam] > start:
            start = prev_terminal[fam]
        prev_terminal[fam] = r["terminal"]
        mins = (r["terminal"] - start).total_seconds() / 60.0
        mark = "" if mins >= 0 else " ?"
        if mins >= 0:
            totals.append(mins)
        cs = r["serial_seconds"]
        print(f"{r['candidate_id'][:46]:<46} {start:%H:%M:%S} {r['terminal']:%H:%M:%S} "
              f"{mins:>10.1f}{mark} {float(cs) if cs else 0:>10.2f}")
    if totals:
        med = statistics.median(totals)
        print(f"\nmedian serial {med:.1f} min vs {TARGET_MIN:.0f} min target "
              f"({'OVER' if med > TARGET_MIN else 'within'}); n={len(totals)}")
        # The directive permits a deep follow-up to exceed ten minutes, but requires it to run IN PARALLEL
        # without stopping the fast loop. So report the fast loop separately: if dropping the single longest
        # row brings the median inside target, the fast loop is healthy and the deep work is serialising it.
        if len(totals) > 2:
            trimmed = sorted(totals)[:-1]
            tmed = statistics.median(trimmed)
            print(f"excluding the longest row ({max(totals):.1f} min): median {tmed:.1f} min "
                  f"({'OVER' if tmed > TARGET_MIN else 'within'} target) -- if this is within target while "
                  f"the full median is over, the fast loop is healthy and a deep follow-up is serialising it")
        print(f"total compute across these screens: "
              f"{sum(float(r['serial_seconds'] or 0) for r in data):.1f} s")
        raise SystemExit(1 if med > TARGET_MIN else 0)
