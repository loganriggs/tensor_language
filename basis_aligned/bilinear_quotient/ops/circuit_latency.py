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
import subprocess
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


def followup_rows():
    """The deep-arc receipts under circuits/followups/, which never reach the fast-screen ledger.

    Closing the coverage gap found at 18:07 (23 runner executions, 13 ledger rows). These carry timestamps
    INSIDE the receipt -- `checked_utc` on a prior-art file, `created_utc` on a result/audit -- which is
    better evidence than an mtime, so these rows are exact where the ledger rows are indicative.
    """
    by_candidate = {}
    for path in glob.glob(os.path.join(BQ, "circuits", "followups", "*.json")):
        try:
            doc = json.load(open(path, errors="ignore"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        cid = doc.get("candidate_id")
        if not cid:
            continue
        stamp = doc.get("created_utc") or doc.get("checked_utc")
        if not stamp:
            continue
        try:
            when = datetime.datetime.strptime(str(stamp)[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        lo, hi = by_candidate.get(cid, (when, when))
        by_candidate[cid] = (min(lo, when), max(hi, when))
    out = []
    for cid, (lo, hi) in by_candidate.items():
        if hi <= lo:
            continue                      # a single-receipt candidate gives no interval
        out.append({"candidate_id": cid, "terminal": hi,
                    "stages": {"receipt_start": lo}, "serial_seconds": None})
    return out


def _completed_events(path, end_day):
    """Parse append-order HH:MM records, reconstructing dates from the tail.

    `_completed.txt` deliberately has no date column.  Treating every clock time
    as `start_day` silently dropped every execution after the first midnight.
    The file can also begin days before our first timestamped receipt, so a
    forward anchor duplicates old history on the wrong dates.  Anchor the final
    record to the known final day and walk backwards: a backwards traversal from
    an early clock to a late clock crosses into the preceding day.
    """
    parsed = []
    for raw in open(path, errors="ignore"):
        match = re.match(r"^(\d\d):(\d\d)\s+(\S+)\s+exit=(\d+)", raw.strip())
        if not match:
            continue
        parsed.append((datetime.time(int(match.group(1)), int(match.group(2))),
                       match.group(3), match.group(4)))
    day = end_day
    following = None
    dated = []
    for clock, name, exit_code in reversed(parsed):
        when = datetime.datetime.combine(day, clock)
        if following is not None and when > following + datetime.timedelta(hours=12):
            day -= datetime.timedelta(days=1)
            when = datetime.datetime.combine(day, clock)
        following = min(following, when) if following is not None else when
        dated.append((when, name, exit_code))
    yield from reversed(dated)


def runner_rows(known):
    """Executions the receipt sources cannot explain, taken from the runner's own completion log.

    The deep arc's receipts carry NO time field at all -- `..._causal_projector_program_a_v1_receipt.json`
    has `experiment_id` but no `*_utc` and no `candidate_id` -- so those screens are unjoinable from
    receipts by anyone, not just by this tool. `runlogs/_completed.txt` is append-only and records the
    runner's own exit time, so lane occupancy is recoverable there without asking another lane to change a
    schema. Minute resolution; dated from the receipt rows.
    """
    completed = os.path.join(BQ, "runlogs", "_completed.txt")
    if not os.path.exists(completed) or not known:
        return []
    day = max(r["terminal"] for r in known).date()
    first = min(r["terminal"] for r in known)
    out = []
    for when, name, exit_code in _completed_events(completed, day):
        if not name.startswith("run_"):
            continue
        if when < first:
            continue
        # The stage marker is deliberately datetime.min, NOT `when`: a runner row has no authoring
        # timestamp of its own, so its honest serial cost is LANE OCCUPANCY -- terminal minus the previous
        # terminal -- which the single-lane chaining below produces once the stage cannot win the max().
        # Setting the stage to `when` made start == terminal and every such row read 0.0 min, dragging the
        # median to zero across 35 rows.
        out.append({"candidate_id": f"[runner] {name}" + ("" if exit_code == "0" else " FAILED"),
                    "terminal": when, "stages": {"runner": datetime.datetime.min},
                    "serial_seconds": None})
    return out


def merge_receipts_and_runner_rows(data):
    """Remove only as many same-minute runner rows as receipts can explain.

    A minute can contain two fast runs.  The old set-of-minutes de-duplication
    discarded both runner rows when one receipt existed, hiding the second
    experiment.  Receipts remain the preferred rows; any excess runner events
    in that minute are retained as otherwise-unrepresented terminals.
    """
    receipt_counts = {}
    for row in data:
        if row["candidate_id"].startswith("[runner]"):
            continue
        minute = row["terminal"].replace(second=0)
        receipt_counts[minute] = receipt_counts.get(minute, 0) + 1
    runner_seen = {}
    merged = []
    for row in sorted(data, key=lambda item: (item["terminal"],
                                               item["candidate_id"].startswith("[runner]"))):
        if row["candidate_id"].startswith("[runner]"):
            minute = row["terminal"].replace(second=0)
            runner_seen[minute] = runner_seen.get(minute, 0) + 1
            if runner_seen[minute] <= receipt_counts.get(minute, 0):
                continue
        merged.append(row)
    return merged


if __name__ == "__main__":
    data = rows() + followup_rows()
    data += runner_rows(data)
    data = merge_receipts_and_runner_rows(data)
    if not data:
        print("no ledger rows"); raise SystemExit(0)
    since = None
    if "--since" in sys.argv:
        hh, mm = sys.argv[sys.argv.index("--since") + 1].split(":")
        since = data[0]["terminal"].replace(hour=int(hh), minute=int(mm), second=0)
    totals = []
    rows_seen = []
    # The runner is a SINGLE SERIAL LANE, so the honest cost of screen N is how long it occupied that lane:
    # terminal_N - max(its earliest stage file, terminal_{N-1}). Clocking from the family's first file charged
    # a repeat screen for every earlier screen in its family (77.7 min for head11.3_complement). Keying by
    # candidate-id family then broke again when the SAME circuit was renamed across ids -- `subject_verb.*`
    # vs `task14.head11_3.*` -- charging the reader screen 101.5 min. Chaining on the previous terminal
    # regardless of name removes the dependence on id spelling altogether.
    prev_terminal = None
    print(f"{'candidate':<46} {'start':>8} {'terminal':>9} {'serial_min':>11} {'compute_s':>10}")
    for r in data:
        if since and r["terminal"] < since:
            continue
        starts = [t for t in r["stages"].values() if t]
        if not starts:
            print(f"{r['candidate_id'][:46]:<46} {'?':>8} {r['terminal']:%H:%M:%S}  (no stage files matched)")
            continue
        start = min(starts)
        if prev_terminal is not None and prev_terminal > start:
            start = prev_terminal
        prev_terminal = r["terminal"]
        mins = (r["terminal"] - start).total_seconds() / 60.0
        mark = "" if mins >= 0 else " ?"
        if mins >= 0:
            totals.append(mins)
            rows_seen.append((r["candidate_id"][:46], mins))
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

        # The hourly duty is: measure the path, NAME the largest avoidable delay, and account for the rerun
        # tax. Doing that as three tool invocations plus interpretation was my own largest repeated step, so
        # it is one command now.
        # THROUGHPUT, not just latency. The directive's target is a RATE -- one meaningful screen or honest
        # null per TEN serial minutes -- and every report before this one answered a different question by
        # quoting a median DURATION. A lane that runs nothing for an hour has an excellent median. This is
        # the quantity the target is actually about.
        last = max(r["terminal"] for r in data)
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
        idle_min = (now - last).total_seconds() / 60.0
        recent = [r for r in data if (now - r["terminal"]).total_seconds() <= 3600]
        target_per_hour = 60.0 / TARGET_MIN
        verdict = "ON TARGET" if len(recent) >= target_per_hour else "BELOW TARGET"
        print(f"\nTHROUGHPUT: {len(recent)} terminal(s) in the last 60 min vs {target_per_hour:.0f} needed "
              f"({verdict}); {idle_min:.0f} min since the last terminal ({last:%H:%M})")
        if idle_min > TARGET_MIN:
            print(f"   the lane has been idle longer than one whole target window -- a median computed over "
                  f"older rows does NOT describe the current hour")

        worst = max(rows_seen, key=lambda kv: kv[1])
        print(f"\nLARGEST AVOIDABLE DELAY: {worst[0]} at {worst[1]:.1f} min "
              f"({worst[1] / max(sum(t for _n, t in rows_seen), 1e-9) * 100:.0f}% of measured serial time)")
        # COVERAGE. This tool reads `fast_screen_ledger.jsonl`. When an arc writes its receipts elsewhere
        # -- as the head-11.3 projector work does -- the ledger stops growing and every number above silently
        # repeats last hour's, so the hourly review reports "healthy" while the lane is busy on something the
        # instrument cannot see. An instrument must report its own blind spot, so runner executions in the
        # measured window that no ledger row explains are counted here.
        completed = os.path.join(BQ, "runlogs", "_completed.txt")
        if os.path.exists(completed) and data:
            first_terminal = min(r["terminal"] for r in data)
            runs = []
            last_day = max(r["terminal"] for r in data).date()
            for when, name, exit_code in _completed_events(completed, last_day):
                if when >= first_terminal and name.startswith("run_"):
                    runs.append((when.strftime("%m-%d %H:%M"), name, exit_code))
            # Count, do NOT name-match: an earlier version token-matched run names against candidate ids and
            # called `run_task14_head11_3_projector_discovery` "covered" because it shares the token task14
            # with a ledger row. Counting cannot produce that false negative.
            gap = len(runs) - len(data)
            if gap > 0:
                print(f"\nCOVERAGE GAP: {len(runs)} runner executions since {first_terminal:%m-%d %H:%M} but only {len(data)} "
                      f"ledger rows -- {gap} execution(s) are NOT described by the numbers above:")
                for t, n, c in runs[-6:]:
                    print(f"   {t} {n} exit={c}")

        census = os.path.join(BQ, "ops", "failure_census.py")
        if os.path.exists(census):
            out = subprocess.run([sys.executable, census], capture_output=True, text=True).stdout
            for line in out.splitlines():
                if line.startswith("last ") or "failure->retry" in line:
                    print("RERUN TAX: " + line.strip())
        raise SystemExit(1 if med > TARGET_MIN else 0)
