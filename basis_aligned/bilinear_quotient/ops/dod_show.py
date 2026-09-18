#!/usr/bin/env python3
# BQGATE: LIBRARY -- CPU receipt printer (review 26): predictions, forwards, top-level numbers, list heads; schema-agnostic.
"""Usage: python dod_show.py <receipt.json> [depth=2]. Prints predictions as held/FAILED, the forward count, then every numeric field up to
`depth` levels deep (rounded) and the first three entries of every list; long dicts are truncated at 12 keys."""
import json, sys
r = json.load(open(sys.argv[1])); depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
p = r.get("predictions", {}); print(f"{sum(bool(v) for v in p.values())}/{len(p)} held; forwards {r.get('forwards')}")
for k, v in p.items(): print(f"  {'held  ' if v else 'FAILED'} {k}")
def show(obj, prefix, d):
    if isinstance(obj, (int, float)) and not isinstance(obj, bool): print(f"{prefix}: {round(obj, 4) if isinstance(obj, float) else obj}")
    elif isinstance(obj, dict) and d > 0:
        for i, (k, v) in enumerate(obj.items()):
            if i >= 12: print(f"{prefix}.…({len(obj) - 12} more)"); break
            show(v, f"{prefix}.{k}", d - 1)
    elif isinstance(obj, list) and obj: print(f"{prefix}: [{len(obj)}] {json.dumps(obj[:3])[:160]}")
try:
  for k, v in r.items():
    if k in ("predictions", "plan", "schema", "candidate_id", "finished_utc", "per_row", "nulls"): continue
    show(v, k, depth)
except BrokenPipeError: pass
