"""The FINDING 21 interaction-share interval, as a saved script instead of an inline snippet.

Today's methodological correction (RESULTS.md 11:55) happened because this computation lived in
throwaway snippets and nobody could re-read what the interval actually measured. This file is the
single source of truth from now on.

Decomposition, on the cap-off 2x2 at one (depth, width) cell, main effects at the our-model corner.
The decomposed field is the INDUCTION SCORE (probe-seed mean, raw nats) from the induction battery,
NOT held cross-entropy — FINDING 21 decomposes the copying capability's move, and reconstructing
this script against held CE failed its own gate by 54 points before this was pinned down:
    total       = ind(conventional) - ind(ours)            [sg - bb]
    attention   = ind(softmax+our-ff) - ind(ours)          [sb - bb]
    feedforward = ind(our-attn+GELU) - ind(ours)           [bg - bb]
    interaction = total - attention - feedforward          [sg - sb - bg + bb]
Shares are each term over total, from CORNER MEANS over seeds; they sum to 100% by construction.

Interval = quantity (2) of the 11:55 correction: 95% percentile bootstrap on the MEAN share,
resampling seeds with replacement WITHIN each corner (10,000 reps, numpy seed 0). It is NOT the
spread of one-seed-per-corner estimates. Caveat from 13:45: with three seeds per corner this
bootstrap can understate its own uncertainty; six-seed intervals are the trustworthy ones.

Gate: rerunning with --gate must reproduce the published point estimates exactly and the published
intervals to within bootstrap-RNG jitter (the points are deterministic; intervals depend on RNG).
"""
import argparse
import glob
import json
import re
import numpy as np

CORNERS = {  # corner key -> filename stem prefix (cap-off arms)
    'bb': 'tff_bilin_bilin_d{d}_w{w}_b8192_s{s}_noqknorm',
    'bg': 'tff_bilin_gelu_d{d}_w{w}_b8192_s{s}_noqknorm',
    'sb': 'tff_softmax_bilin_d{d}_w{w}_b8192_s{s}_noqknorm',
    'sg': 'tfb_std7_d{d}_w{w}_b8192_s{s}_noqknorm',
}


def corner_ces(depth, width, max_seed=None):
    out = {}
    for key, pat in CORNERS.items():
        ces = []
        for f in sorted(glob.glob(pat.format(d=depth, w=width, s='*') + '_induction.json')):
            s = int(re.search(r'_s(\d+)_', f).group(1))
            if max_seed is not None and s > max_seed:
                continue
            ces.append(json.load(open(f))['induction_score_mean'])
        if not ces:
            raise SystemExit(f'no cells for corner {key} at d{depth} w{width}')
        out[key] = np.array(ces)
    return out


def shares(m):
    total = m['sg'] - m['bb']
    return {'attention': (m['sb'] - m['bb']) / total * 100,
            'feedforward': (m['bg'] - m['bb']) / total * 100,
            'interaction': (m['sg'] - m['sb'] - m['bg'] + m['bb']) / total * 100}


def cell(depth, width, max_seed=None, reps=10000):
    ces = corner_ces(depth, width, max_seed)
    point = shares({k: v.mean() for k, v in ces.items()})
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(reps):
        m = {k: rng.choice(v, size=len(v), replace=True).mean() for k, v in ces.items()}
        boot.append(shares(m)['interaction'])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'depth': depth, 'width': width,
            'seeds_per_corner': {k: len(v) for k, v in ces.items()},
            'shares_pct': {k: round(v, 1) for k, v in point.items()},
            'interaction_interval_95': [round(lo, 1), round(hi, 1)],
            'interval_width': round(hi - lo, 1)}


PUBLISHED = [  # (depth, width, max_seed, point, interval) from RESULTS.md 11:55 / 13:45
    (2, 128, None, 87.9, (81.4, 92.2)),
    (3, 64, 2, 50.8, (38.5, 60.4)),
    (3, 64, None, 61.2, (48.0, 75.6)),
    (3, 128, 2, 9.6, (1.1, 17.8)),
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int)
    ap.add_argument('--width', type=int)
    ap.add_argument('--max-seed', type=int, default=None)
    ap.add_argument('--gate', action='store_true')
    a = ap.parse_args()
    if a.gate:
        ok = True
        for d, w, ms, pt, iv in PUBLISHED:
            r = cell(d, w, ms)
            dpt = abs(r['shares_pct']['interaction'] - pt)
            div = max(abs(r['interaction_interval_95'][0] - iv[0]),
                      abs(r['interaction_interval_95'][1] - iv[1]))
            verdict = 'PASS' if dpt < 0.05 and div < 1.5 else 'FAIL'
            ok &= verdict == 'PASS'
            print(f'gate d{d} w{w} seeds<={ms}: point {r["shares_pct"]["interaction"]} '
                  f'(pub {pt}, diff {dpt:.2f}), interval {r["interaction_interval_95"]} '
                  f'(pub {list(iv)}, max diff {div:.1f}) -> {verdict}')
        print('GATE', 'PASS' if ok else 'FAIL')
    if a.depth:
        print(json.dumps(cell(a.depth, a.width, a.max_seed), indent=1))
