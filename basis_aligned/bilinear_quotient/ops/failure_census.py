"""failure_census -- how often does a queued run fail, and what does the retry cost? READ-ONLY.

Measured 2026-09-04: 5 of the last 40 runner executions exited nonzero (12.5%), each followed by a retry
2-3 minutes later, in a regime where the science itself takes 0.4-7.1 s. A failed execution is therefore
~30x the cost of the computation it was trying to do, and it is invisible in receipt-based reporting because
a failed run writes no receipt.

It is also currently unfixable-by-class: `runlogs/<name>.log` is overwritten by the retry, so by the time
anyone looks, the failure that mattered is gone. This tool measures the rate and the retry cost from
`runlogs/_completed.txt`, which is append-only and survives; naming the CLASS needs the log-preservation
change proposed on the board (bqrunner is runner-owned, so this lane does not touch it).

Usage:  python ops/failure_census.py [--last N]     (default: whole file, plus a recent-N window)
"""
import os
import re
import sys
import collections
import datetime

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPLETED = os.path.join(BQ, 'runlogs', '_completed.txt')
LINE = re.compile(r'^(\d\d):(\d\d)\s+(\S+)\s+exit=(\d+)\s*$')


def parse(path=COMPLETED):
    out = []
    for ln in open(path, errors='ignore'):
        m = LINE.match(ln.strip())
        if m:
            hh, mm, name, code = m.groups()
            out.append((int(hh) * 60 + int(mm), name, int(code)))
    return out


def retry_pairs(events, window_minutes=15):
    """A failure followed by the same name succeeding soon after: the retry and its wall-clock cost."""
    pairs = []
    for i, (t, name, code) in enumerate(events):
        if code == 0:
            continue
        for t2, name2, code2 in events[i + 1:]:
            if name2 != name:
                continue
            gap = t2 - t
            if 0 <= gap <= window_minutes:
                pairs.append((name, gap, code2))
            break
    return pairs


def report(events, label):
    if not events:
        print(f'{label}: no parseable entries')
        return
    fails = [e for e in events if e[2] != 0]
    rate = 100 * len(fails) / len(events)
    pairs = retry_pairs(events)
    cost = sum(g for _, g, _ in pairs)
    print(f'{label}: {len(events)} executions, {len(fails)} nonzero ({rate:.1f}%)')
    if pairs:
        print(f'   {len(pairs)} failure->retry pairs, {cost} min of wall-clock spent re-running '
              f'(median gap {sorted(g for _, g, _ in pairs)[len(pairs)//2]} min)')
    worst = collections.Counter(name for _, name, code in events if code != 0).most_common(5)
    if worst:
        print('   most-failed scripts: ' + ', '.join(f'{n} x{c}' for n, c in worst))


if __name__ == '__main__':
    events = parse()
    n = int(sys.argv[sys.argv.index('--last') + 1]) if '--last' in sys.argv else 60
    report(events, 'all time')
    report(events[-n:], f'last {n}')
    surviving = 0
    for _, name, code in events[-n:]:
        if code != 0 and os.path.exists(os.path.join(BQ, 'runlogs', f'{name}.log')):
            surviving += 1
    print(f'\nfailed runs in the last {n} whose log still exists: {surviving} '
          f'(a retry overwrites it, so the failure CLASS is usually unrecoverable)')
