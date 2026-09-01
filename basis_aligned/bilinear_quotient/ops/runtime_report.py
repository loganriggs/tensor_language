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
            if rt: rows.append((os.path.basename(f)[:44], round(rt, 1)))
        except Exception: pass
rows.sort(key=lambda r: -r[1])
heavy = [r[1] for r in rows if r[1] > 100]; light = [r[1] for r in rows if r[1] <= 100]
busy = sum(r[1] for r in rows)
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
    end = mins[-1]
    windowed = [m if m <= end else m - 1440 for m in mins]
    windowed = [m for m in windowed if end - m <= back]
    gaps = [windowed[i+1] - windowed[i] for i in range(len(windowed) - 1) if windowed[i+1] >= windowed[i]]
    if gaps:
        print(f"landings in window: {len(windowed)} | mean gap {sum(gaps)/len(gaps):.1f} min | "
              f"gaps>5min: {sum(1 for g in gaps if g > 5)} | busy-fraction ~{busy/60/max(back,1)*100:.0f}%")
