"""hour_review -- what happened in the last hour, how long each thing took, and what it was for.

Reconstructed from evidence already on disk rather than from self-reported effort:
  * git commits (this repo)          -- each commit ends a work span; the gap since the previous commit is
                                        that span's wall-clock cost
  * runlogs/_completed.txt           -- runner executions and their wall-clock
  * circuits/fast_screen_ledger.jsonl -- compute seconds per screen

Activity is inferred from the commit-message prefix, so the categories are only as good as the messages:
  circuits:  screen, candidate, receipt, claim, result
  ops:       tooling, tests, measurement
  board:     coordination and write-ups
  other:     anything not prefixed

The point is the RATIO. A basic screen costs ~11 s of compute; if an hour spent four times as long on
tooling as on circuits, that is the CEREMONY_BUDGET check failing, and it should be visible without
anyone's recollection.

Usage:  python ops/hour_review.py [--hours N]   (default 1)
"""
import collections
import datetime
import os
import re
import subprocess
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = os.path.dirname(os.path.dirname(BQ))
CATEGORIES = (("circuits", "circuits:"), ("ops", "ops:"), ("board", "board:"))


def category(subject):
    low = subject.lower()
    for name, prefix in CATEGORIES:
        if low.startswith(prefix):
            return name
    return "other"


def commits(hours):
    out = subprocess.run(
        ["git", "-C", TL, "log", f"--since={hours} hours ago", "--format=%at\t%s"],
        capture_output=True, text=True).stdout.strip()
    rows = []
    for line in out.splitlines():
        ts, _, subject = line.partition("\t")
        try:
            rows.append((int(ts), subject))
        except ValueError:
            continue
    return sorted(rows)


def screens(hours):
    path = os.path.join(BQ, "runlogs", "_completed.txt")
    if not os.path.exists(path):
        return []
    # `_completed.txt` carries HH:MM with NO DATE, so a naive time filter matches every prior day too --
    # it reported 2825 screens in two hours before this was fixed. Walk the tail BACKWARDS and stop at the
    # first backwards step, which is the day boundary.
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    cutoff_minutes = now.hour * 60 + now.minute - hours * 60
    out, prev = [], None
    for raw in reversed(open(path, errors="ignore").readlines()[-400:]):
        m = re.match(r"^(\d\d):(\d\d)\s+(\S+)\s+exit=(\d+)", raw.strip())
        if not m:
            continue
        minutes = int(m.group(1)) * 60 + int(m.group(2))
        if prev is not None and minutes > prev:
            break                      # time went backwards walking up the file: previous day
        prev = minutes
        if minutes < cutoff_minutes:
            break
        if "canary" not in m.group(3):
            out.append((m.group(1) + ":" + m.group(2), m.group(3), m.group(4)))
    return list(reversed(out))


if __name__ == "__main__":
    hours = int(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 1
    rows = commits(hours)
    if not rows:
        print(f"no commits in the last {hours}h"); raise SystemExit(0)
    spans, prev = [], None
    for ts, subject in rows:
        gap = (ts - prev) / 60.0 if prev else None
        spans.append((ts, subject, gap))
        prev = ts
    print(f"{'when':>6}  {'min':>6}  {'category':<9} subject")
    totals = collections.Counter()
    for ts, subject, gap in spans:
        when = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%H:%M")
        cat = category(subject)
        if gap is not None:
            totals[cat] += gap
        shown = "  --" if gap is None else f"{gap:6.1f}"
        print(f"{when:>6}  {shown}  {cat:<9} {subject[:74]}")
    total = sum(totals.values())
    print(f"\ntime by category over {hours}h (first commit has no prior gap, so it is unattributed):")
    for cat, mins in totals.most_common():
        share = 100 * mins / total if total else 0
        print(f"   {cat:<9} {mins:>6.1f} min  {share:>4.0f}%")
    runs = screens(hours)
    print(f"\nscreens run: {len(runs)}" + ("  " + ", ".join(f"{t} {n}" for t, n, _c in runs) if runs else ""))
    if totals.get("circuits", 0) and totals.get("ops", 0):
        ratio = totals["ops"] / totals["circuits"]
        verdict = "OK" if ratio <= 1.0 else "CEREMONY_BUDGET: tooling outweighed circuits this hour"
        print(f"ops/circuits time ratio: {ratio:.2f}   {verdict}")
