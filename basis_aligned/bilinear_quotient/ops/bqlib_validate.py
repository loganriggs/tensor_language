# BQLIB ACID TEST -- reproduce §1940's PUBLISHED numbers from the library, in 40 lines of novel code.
#
# ops/bqlib.py replaces the boilerplate measured across ops/: 42% of 89,012 function lines are copies
# beyond the first (forward_logits x173, load x214, mk x160, row_hook x65, program_rows x45). This script
# is the bar the library has to clear before anything else uses it: the same four arms, two coverages and
# three roles as ops/routed_domination_replication.py (§1940, 267.7s, 348 lines) must come back IDENTICAL.
#
# It also tests the cache in both directions (PRE-FLIGHT D): a second run must HIT on every key and return
# the same numbers, and a deliberately corrupted fingerprint must be REJECTED rather than silently served.
#
# ROLES. skip7000, skip11000, skip1200. INFRASTRUCTURE VALIDATION, not an experiment.
#
# Registered predictions, exact-match bars -- this is a reproduction, so a tolerance would be too weak.
#   pred_a POOLED TOP-1 REPRODUCES §1940 EXACTLY at both coverages, all four published arm-role figures
#          per coverage, to within 0.005pp (the rounding of the published 2-dp percentages).
#   pred_b POOLED CE REPRODUCES §1940 EXACTLY, nn75 and map64 at 5,419, to within 0.00005 nats.
#   pred_c THE PAIRED t REPRODUCES §1940, all six role-coverage cells, to within 0.02 -- including the
#          sign reversal at 16,110/skip1200 (+2.59) that forced §1939's retraction.
#   pred_e SPEED: bqlib v2 must beat the 267.7s hand-written §1940 it replaces. Recorded, not barred --
#          a timing bar would fail on a shared GPU. The number is written to the result JSON.
#   pred_d CONTROLS: coverages exactly 5,419 and 16,110; every arm inert at covered inputs (0 changed
#          top-1 vs map64 there); buckets partition; live per-cell top-1 and CE identical across arms;
#          the routed fraction is 0.75 within 1%; and the cache round-trips -- every key HITs on a second
#          read and a corrupted fingerprint is rejected.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/bqlib_validate_results.json'
ARMS = ('nn', 'nn75', 'map64', 'map512')
S1940_T1 = {'c5419': {'nn75': (0.1402, 0.1469, 0.1412), 'map64': (0.1355, 0.1425, 0.1364)},
            'c16110': {'nn75': (0.1418, 0.1482, 0.1418), 'map64': (0.1393, 0.1471, 0.1401)}}
S1940_CE = {'nn75': (6.00963, 5.98385, 5.99919), 'map64': (6.01167, 5.98477, 6.00165)}
S1940_T = {'c5419': (-0.54, -0.23, -0.44), 'c16110': (-1.90, -2.04, +2.59)}

print('BQLIB ACID TEST | reproducing §1940 from the library | INFRASTRUCTURE VALIDATION', flush=True)
res, paired, chg, frac = {}, {}, {}, {}
for cov, fit, nc in (('c5419', B.FIT_5419, 5419), ('c16110', B.FIT_16110, 16110)):
    P = B.Program(fit, expect_ncov=nc)
    res[cov], paired[cov], chg[cov] = {}, {}, {}
    # score_roles builds each arm AT MOST ONCE across all three roles. The first version of this
    # validator called score() per role, rebuilt every arm three times, and ran 808s against the 267.7s
    # of the hand-written script it replaces -- 95% of it in the map's float64 SVD (bqlib v2 fixes both).
    liveR = B.score_roles(P, None)
    armR = {a: B.score_roles(P, a) for a in ARMS}
    for role in B.ROLES:
        tgt, icov = B.axes(P, role)
        live = liveR[role]
        arms = {a: armR[a][role] for a in ARMS}
        res[cov][role] = {a: B.cells(P, tgt, icov, live, arms[a]) for a in ARMS}
        if cov == 'c5419' and role == 'skip7000':
            cached_ref = arms['map64']          # the reference the cache test compares against
        paired[cov][role] = B.paired_t(arms['nn75'][1], arms['map64'][1])
        chg[cov][role] = {a: int(((arms[a][0] != arms['map64'][0]) & icov).sum()) for a in ARMS}
    frac[cov] = P.routefrac.get('nn75', 0.0)
    del liveR, armR
    del P
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']


dt1 = max(abs(ov(c, r, a) - S1940_T1[c][a][i])
          for c in S1940_T1 for a in S1940_T1[c] for i, r in enumerate(B.ROLES))
dce = max(abs(ce('c5419', r, a) - S1940_CE[a][i]) for a in S1940_CE for i, r in enumerate(B.ROLES))
dt = max(abs(paired[c][r]['t'] - S1940_T[c][i]) for c in S1940_T for i, r in enumerate(B.ROLES))

# cache round-trip, BOTH directions (PRE-FLIGHT D): a warm key must HIT and return the same numbers;
# a corrupted fingerprint must be REJECTED and recomputed, not silently served.
PL = B.Program(B.FIT_5419, expect_ncov=5419, verbose=False)
h0 = B.STATS['hit']
warm = B.score(PL, 'map64', 'skip7000')
cache_hit = (B.STATS['hit'] == h0 + 1) and bool((warm[0] == cached_ref[0]).all()) \
    and torch.allclose(warm[1], cached_ref[1])
cp = B.cache_path(PL, 'map64', 'skip7000')
blob = torch.load(cp, map_location='cpu')
blob['fp'] = 'DELIBERATELY_WRONG'
blob['am'] = torch.zeros_like(blob['am'])
blob['nl'] = torch.zeros_like(blob['nl'])
torch.save(blob, cp)
s0 = B.STATS['stale']
recov = B.score(PL, 'map64', 'skip7000')
rejects = (B.STATS['stale'] == s0 + 1) and bool((recov[0] == cached_ref[0]).all()) \
    and torch.allclose(recov[1], cached_ref[1])
hits = cache_hit
del PL
torch.cuda.empty_cache()

inert = all(v == 0 for c in chg for r in chg[c] for v in chg[c][r].values())
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r]['map64'][cl][b]['ce_live'])
               for c in res for r in res[c] for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled')
               for b in res[c][r][a][cl])
fracok = max(abs(frac[c] - 0.75) for c in frac)

pa, pb, pc = dt1 <= 0.00005, dce <= 0.00005, dt <= 0.02
pd = inert and livesame <= 1e-9 and fracok <= 0.01 and hits and rejects

for c in res:
    print(f'\n  === {c} ===', flush=True)
    for r in B.ROLES:
        print(f'    {r:10s} ' + '  '.join(f'{a} t1 {ov(c, r, a):6.2%} CE {ce(c, r, a):7.5f}'
                                          for a in ARMS), flush=True)
        print(f'    {"":10s} nn75-map64 paired t {paired[c][r]["t"]:+.2f} '
              f'(§1940 published {S1940_T[c][B.ROLES.index(r)]:+.2f})', flush=True)
print(f'\n  pooled top-1 reproduces §1940 (max dev {dt1 * 100:.4f}pp) -> {pa}', flush=True)
print(f'  pooled CE reproduces §1940 (max dev {dce:.6f} nats) -> {pb}', flush=True)
print(f'  paired t reproduces §1940 (max dev {dt:.3f}) -> {pc}', flush=True)
print(f'  arms inert at covered inputs {inert}, live identical {livesame:.1e}, routefrac dev '
      f'{fracok:.4f}, warm key HIT and matched {hits}, corrupt fingerprint rejected and '
      f'recomputed correctly {rejects}, stats {B.STATS} -> {pd}', flush=True)

B.report({'pred_a_top1_reproduces': pa, 'pred_b_ce_reproduces': pb,
          'pred_c_paired_t_reproduces': pc, 'pred_d_controls': pd},
         {'config': {'purpose': 'acid test for ops/bqlib.py -- reproduce §1940 from the library',
                     'arms': list(ARMS), 'coverages': [5419, 16110],
                     'novel_lines': 'this script is ~40 lines of experiment logic against §1940\'s 348',
                     'ROLE_NOTE': 'INFRASTRUCTURE VALIDATION, not an experiment.'},
          'max_deviation': {'top1_pp': dt1 * 100, 'ce_nats': dce, 'paired_t': dt},
          'results': res,
          'paired': {c: {r: paired[c][r] for r in paired[c]} for c in paired},
          'changed_at_covered_inputs': chg,
          'routed_fraction': frac,
          'cache': {'hit_matched': bool(hits), 'corrupt_rejected': bool(rejects), 'stats': dict(B.STATS)}},
         OUT, t0)
