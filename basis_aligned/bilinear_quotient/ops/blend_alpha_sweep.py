# WHERE IS THE BLEND OPTIMUM, AND DOES IT SURVIVE AT 16,110?
#
# §1942 closed the §1937/§1938 objective fork: a convex blend of the output-NN neighbour row and a
# rank-512 map row beats BOTH former champions, and it is strongly superadditive -- every blended arm
# beats the linear interpolation of its endpoints, 9/9 cells, by -0.030 to -0.047 nats. But alpha was
# swept at three points only and the best CE (mix25m512) sat at the EDGE of that grid, so the optimum is
# unmeasured; and every figure was at 5,419, where the fallback arm is ~24% of positions against ~10% at
# 16,110 (§1936). §1940 is the precedent: margins roughly halve at the higher coverage.
#
# ARMS. mix10/mix25/mix40/mix50/mix60/mix75/mix90 m512, plus the endpoints -- map512 is alpha=0 and nn is
# alpha=100 -- and map64 as the deployed anchor. BOTH coverages, full table rank.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1942's open question.
#
# Registered predictions, SIGNED per LESSON 72, every CE claim with a PAIRED t up front (LESSON 78).
#   pred_a THE CE OPTIMUM IS INTERIOR: at 5,419 some alpha strictly between 0 and 100 has pooled CE BELOW
#          both endpoints -- map512 (alpha=0) and nn (alpha=100) -- on 3 of 3 roles. §1942 already shows
#          this at alpha=25, so a failure here would mean §1942 does not reproduce and something is
#          wrong; it is registered as the reproduction leg, not as news.
#   pred_b AND THE ARGMIN SITS BELOW 50: the CE-minimising alpha on this grid is <= 40 at 5,419 on at
#          least 2 of 3 roles. §1942's three points ran 5.94165 / 5.94962 / 5.97602 for 25 / 50 / 75, so
#          CE is increasing in alpha across the measured range and the true minimum should be at or below
#          the bottom of that grid. If FALSE the curve turns over inside 25-50 and §1942 read the trend
#          backwards.
#   pred_c AND IT SURVIVES AT 16,110: the arm that minimises CE at 16,110 beats map512 there on pooled
#          top-1 AND has a paired CE t <= -2.0 against map512, on at least 2 of 3 roles. If FALSE the
#          blend is a 5,419-only result and §1942 is scoped exactly as §1939 was.
#   pred_d CONTROLS: coverages exactly 5,419 and 16,110; every arm inert at covered inputs at both;
#          buckets partition; live per-cell top-1 and CE identical across arms; and at 5,419 the
#          mix25m512 / mix50m512 / mix75m512 / map512 arms reproduce §1942's PUBLISHED pooled CE
#          (5.94165/5.91021/5.93277, 5.94962/5.92175/5.93841, 5.97602/5.95255/5.96351,
#          5.96702/5.93645/5.96095) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/blend_alpha_sweep_results.json'
MIX = ('mix10m512', 'mix25m512', 'mix40m512', 'mix50m512', 'mix60m512', 'mix75m512', 'mix90m512')
ARMS = MIX + ('nn', 'map512', 'map64')
ALPHA = {a: int(a[3:-4]) / 100.0 for a in MIX}
COVS = (('c5419', B.FIT_5419, 5419), ('c16110', B.FIT_16110, 16110))
S1942_CE = {'mix25m512': (5.94165, 5.91021, 5.93277), 'mix50m512': (5.94962, 5.92175, 5.93841),
            'mix75m512': (5.97602, 5.95255, 5.96351), 'map512': (5.96702, 5.93645, 5.96095)}

print('BLEND ALPHA SWEEP | 7 alphas x 2 coverages | paired t on every CE margin | DISCOVERY ONLY',
      flush=True)
res, pt, chg, ncov = {}, {}, {}, {}
for cov, fit, nc in COVS:
    print(f'\n########## COVERAGE {nc} ##########', flush=True)
    P = B.Program(fit, expect_ncov=nc)
    liveR = B.score_roles(P, None)
    armR = {a: B.score_roles(P, a) for a in ARMS}
    res[cov], pt[cov], chg[cov] = {}, {}, {}
    for role in B.ROLES:
        tgt, icov = B.axes(P, role)
        res[cov][role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR['map512'][role][1]) for a in ARMS}
        chg[cov][role] = {a: int(((armR[a][role][0] != armR['map64'][role][0]) & icov).sum())
                          for a in ARMS}
    ncov[cov] = P.ncov
    COST = {a: P.cost(a) / 1e6 for a in ARMS}
    del P, liveR, armR
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']


def argmin_ce(c, r):
    return min(MIX, key=lambda a: ce(c, r, a))


interior = sum(1 for r in B.ROLES
               if min(ce('c5419', r, a) for a in MIX) < min(ce('c5419', r, 'map512'),
                                                            ce('c5419', r, 'nn')))
pa = interior == 3
low = sum(1 for r in B.ROLES if ALPHA[argmin_ce('c5419', r)] <= 0.40)
pb = low >= 2
survive = sum(1 for r in B.ROLES
              for a in [argmin_ce('c16110', r)]
              if ov('c16110', r, a) > ov('c16110', r, 'map512') and pt['c16110'][r][a]['t'] <= -2.0)
pc = survive >= 2

inert = all(v == 0 for c in chg for r in chg[c] for v in chg[c][r].values())
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r]['map64'][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(abs(ce('c5419', r, a) - S1942_CE[a][i])
            for a in S1942_CE for i, r in enumerate(B.ROLES))
pd = (ncov['c5419'] == 5419 and ncov['c16110'] == 16110 and inert and livesame <= 1e-9
      and repro <= 0.0002)

for c, _f, nc in COVS:
    print(f'\n  === coverage {nc} ===', flush=True)
    for r in B.ROLES:
        print(f'    {r}  (CE argmin: {argmin_ce(c, r)})', flush=True)
        for a in ARMS:
            d1 = (ov(c, r, a) - ov(c, r, 'map512')) * 100
            dc = ce(c, r, a) - ce(c, r, 'map512')
            print(f'      {a:10s} top1 {ov(c, r, a):6.2%} ({d1:+.2f}pp)  CE {ce(c, r, a):7.5f} '
                  f'({dc:+.5f}, t {pt[c][r][a]["t"]:+.2f})', flush=True)
print(f'\n  the CE optimum is INTERIOR at 5,419 (3/3) -> {pa}  {interior}/3', flush=True)
print(f'  and the CE argmin alpha is <= 0.40 (>=2 roles) -> {pb}  {low}/3  '
      f'(argmins {[argmin_ce("c5419", r) for r in B.ROLES]})', flush=True)
print(f'  and the 16,110 CE argmin still beats map512 on top-1 with t<=-2 (>=2 roles) -> {pc}  '
      f'{survive}/3  (argmins {[argmin_ce("c16110", r) for r in B.ROLES]})', flush=True)
print(f'  coverages {ncov}, arms inert {inert}, live identical {livesame:.1e}, §1942 CE '
      f'reproduced within {repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_optimum_is_interior': pa, 'pred_b_argmin_below_40': pb,
          'pred_c_survives_at_16110': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [5419, 16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1942 open question.',
                     'alpha': ALPHA},
          'results': res,
          'paired_vs_map512': pt,
          'ce_argmin': {c: {r: argmin_ce(c, r) for r in B.ROLES} for c in res},
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
