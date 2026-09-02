"""Hourly runtime/gap report (ops-efficiency lane, 2026-09-01).
Usage: python3 ops/runtime_report.py [minutes_back=70]
Parses receipts' runtime_s (mtime within window) and landing gaps from the
TAIL of runlogs/_completed.txt only (avoids whole-history HH:MM pollution)."""
import json, glob, os, sys, time, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
back = int(sys.argv[1]) if len(sys.argv) > 1 else 70
now = time.time()
rows = []
for f in glob.glob(os.path.join(ROOT, '*_results.json')):
    if now - os.path.getmtime(f) < back * 60:
        try:
            rt = json.load(open(f)).get('runtime_s')
            if rt:
                # clip to window overlap: a 2.5h run landing 5 min ago must
                # not claim 2.5h of THIS window (11:07 bug: busy ~224%)
                end = os.path.getmtime(f)
                overlap = min(end, now) - max(end - rt, now - back * 60)
                rows.append((os.path.basename(f)[:44], round(rt, 1),
                             round(max(0.0, overlap), 1)))
        except Exception: pass
rows.sort(key=lambda r: -r[1])
heavy = [r[1] for r in rows if r[1] > 100]; light = [r[1] for r in rows if r[1] <= 100]
busy = sum(r[2] for r in rows)   # window-clipped
print(f"receipts<{back}m: {len(rows)} | heavy {len(heavy)} median {statistics.median(heavy) if heavy else 0} | "
      f"light {len(light)} median {statistics.median(light) if light else 0} | busy {busy/60:.1f} min")
tail = [l.split() for l in open(os.path.join(ROOT, 'runlogs/_completed.txt')).read().splitlines()[-40:] if l.strip()]
mins = []
for l in tail:
    try:
        h, m = l[0].split(':'); mins.append(int(h) * 60 + int(m))
    except Exception: pass
# keep only entries within the window of the last entry, handling wrap
if mins:
    lt = time.localtime(now)
    end = lt.tm_hour * 60 + lt.tm_min          # anchor to NOW (10:06 bug:
    windowed = [m if m <= end else m - 1440 for m in mins]  # stale 08:24
    windowed = [m for m in windowed if end - m <= back]     # rows counted)
    gaps = [windowed[i+1] - windowed[i] for i in range(len(windowed) - 1) if windowed[i+1] >= windowed[i]]
    if gaps:
        print(f"landings in window: {len(windowed)} | mean gap {sum(gaps)/len(gaps):.1f} min | "
              f"gaps>5min: {sum(1 for g in gaps if g > 5)} | busy-fraction ~{busy/60/max(back,1)*100:.0f}%")

# in-flight run (added 09:06 review): receipts-only busy accounting reported
# ~3% while rung480 had occupied the GPU 39 min -- count the running job too.
_rl = os.path.join(ROOT, 'runlogs/runner.log')
try:
    _lines = open(_rl).read().splitlines()[-6:]
    _last = _lines[-1] if _lines else ''
    if ' running ' in _last:
        _ts = _last.split(']', 1)[1].split()[0]
        _h, _m, _s = (int(x) for x in _ts.split(':'))
        _lt = time.localtime(now)
        _started = _h * 3600 + _m * 60 + _s
        _nowsec = _lt.tm_hour * 3600 + _lt.tm_min * 60 + _lt.tm_sec
        _elapsed = (_nowsec - _started) % 86400
        print(f"IN-FLIGHT: {_last.split(' running ')[-1]} for {_elapsed/60:.0f} min "
              f"(true busy-fraction ~{min(100.0, (busy + _elapsed)/60/max(back,1)*100):.0f}%)")
except Exception:
    pass

# idle-cause context (added 02:06 review): queue depth + last commit age
import subprocess as _sp
from pathlib import Path as _P
_q = _P(__file__).resolve().parent.parent / "queue.txt"
_depth = sum(1 for l in _q.read_text().splitlines() if l.strip() and not l.startswith("#")) if _q.exists() else 0
try:
    _age = _sp.run(["git", "-C", "/workspace/tensor_language", "log", "-1", "--format=%cr"],
                   capture_output=True, text=True, timeout=10).stdout.strip()
except Exception:
    _age = "?"
print(f"queue depth: {_depth} | last commit: {_age}")
