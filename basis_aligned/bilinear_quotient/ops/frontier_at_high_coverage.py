# DOES THE CHEAPER FRONTIER HOLD AT 16,110?
#
# §1944 put mix25m256 (246.102M) strictly ahead of map512 (267.246M) on BOTH instruments at 5,419 -- top-1
# +0.20/+0.16/+0.21pp and CE -0.0171/-0.0176/-0.0210 nats at paired t -9.13/-8.55/-7.02 -- for 21.1M LESS.
# Every figure in that section is at 5,419. §1943 measured what the higher coverage does to a fallback
# margin: it runs a little over half, because the uncovered arm falls from ~24% to ~10% of scored
# positions (§1936). §1939 is the cautionary case -- a frontier claim published at one coverage and
# retracted when the instrument and the second coverage arrived.
#
# ARMS. the §1944 frontier -- mix25m256, mix40m128, mix25m512 -- against map512, map64 (DEPLOYED) and nn.
# 16,110 coverage, full table rank.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- second-class confirmation of §1944 at a
# different coverage.
#
# Registered predictions, SIGNED per LESSON 72, paired t up front (LESSON 78).
#   pred_a THE CHEAPER BUILD STILL WINS: mix25m256's pooled top-1 is ABOVE map512's AND its paired CE t
#          against map512 is <= -2.0, on at least 2 of 3 roles. If FALSE, §1944's frontier is a 5,419
#          result and I scope it the way §1940 scoped §1939.
#   pred_b AND THE MARGIN SHRINKS BY THE PREDICTED FACTOR: the 16,110 CE margin (mix25m256 - map512) is
#          between 30% and 80% of the 5,419 margin (-0.01714 / -0.01759 / -0.02100), on at least 2 of 3
#          roles. A TWO-SIDED band, because this is a reproduction check on a known scaling and not a
#          direction test -- pred_a already carries the sign. §1943 measured "a little over half".
#   pred_c AND THE FRONTIER ORDER SURVIVES: at 16,110, CE(mix25m512) < CE(mix25m256) < CE(map512) on at
#          least 2 of 3 roles -- i.e. rank still buys CE and the blend still beats the pure map. If FALSE
#          the ordering §1944 established is coverage-dependent and the frontier must be redrawn.
#   pred_d CONTROLS: coverage exactly 16,110; every arm inert at covered inputs; buckets partition; live
#          per-cell top-1 and CE identical across arms; and map512 / map64 / mix25m512 reproduce §1943's
#          PUBLISHED 16,110 pooled CE (5.88338/5.82928/5.86044, 5.90522/5.85230/5.88575,
#          5.87265/5.81812/5.85260) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/frontier_at_high_coverage_results.json'
MIX = ('mix25m256', 'mix40m128', 'mix25m512')
ARMS = MIX + ('nn', 'map512', 'map64')
ALPHA = {'mix25m256': 0.25, 'mix40m128': 0.40, 'mix25m512': 0.25}
RANK = {'mix25m256': 256, 'mix40m128': 128, 'mix25m512': 512}
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'
S1944_D5419 = (-0.01714, -0.01759, -0.02100)      # mix25m256 - map512 CE margin at 5,419
S1943_C16 = {'map512': (5.88338, 5.82928, 5.86044), 'map64': (5.90522, 5.85230, 5.88575),
             'mix25m512': (5.87265, 5.81812, 5.85260)}

print(f'FRONTIER AT 16,110 | arms {ARMS} | rung 2 second-class confirmation of §1944 | '
      f'DISCOVERY ONLY', flush=True)
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


wins = sum(1 for r in B.ROLES
           if ov(C, r, 'mix25m256') > ov(C, r, 'map512') and pt[C][r]['mix25m256']['t'] <= -2.0)
pa = wins >= 2

ratio = {r: (ce(C, r, 'mix25m256') - ce(C, r, 'map512')) / S1944_D5419[i]
         for i, r in enumerate(B.ROLES)}
inband = sum(1 for r in B.ROLES if 0.30 <= ratio[r] <= 0.80)
pb = inband >= 2

order = sum(1 for r in B.ROLES
            if ce(C, r, 'mix25m512') < ce(C, r, 'mix25m256') < ce(C, r, 'map512'))
pc = order >= 2

inert = all(v == 0 for c in chg for r in chg[c] for v in chg[c][r].values())
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r]['map64'][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(abs(ce(C, r, a) - S1943_C16[a][i])
            for a in S1943_C16 for i, r in enumerate(B.ROLES))
pd = ncov[C] == 16110 and inert and livesame <= 1e-9 and repro <= 0.0002

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:10s} [{COST[a]:8.3f}M]  top1 {ov(C, r, a):6.2%} '
              f'({(ov(C, r, a) - ov(C, r, "map512")) * 100:+.2f}pp)  CE {ce(C, r, a):7.5f} '
              f'({ce(C, r, a) - ce(C, r, "map512"):+.5f}, t {pt[C][r][a]["t"]:+.2f})', flush=True)
    print(f'    mix25m256 CE margin is {ratio[r]:.0%} of its 5,419 value '
          f'({S1944_D5419[B.ROLES.index(r)]:+.5f})', flush=True)
print(f'\n  mix25m256 still beats map512 on both, with the t bar (>=2 roles) -> {pa}  {wins}/3',
      flush=True)
print(f'  and the CE margin is 30-80% of its 5,419 value (>=2 roles) -> {pb}  {inband}/3  '
      f'(ratios {[f"{ratio[r]:.0%}" for r in B.ROLES]})', flush=True)
print(f'  and the frontier ORDER mix25m512 < mix25m256 < map512 holds (>=2 roles) -> {pc}  {order}/3',
      flush=True)
print(f'  coverage {ncov}, arms inert {inert}, live identical {livesame:.1e}, §1943 16,110 CE '
      f'reproduced within {repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_cheaper_build_still_wins': pa, 'pred_b_margin_shrinks_as_predicted': pb,
          'pred_c_frontier_order_survives': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 2 -- second-class confirmation of §1944 at 16,110.',
                     'alpha': ALPHA},
          'results': res,
          'paired_vs_map512': pt,
          'margin_ratio_vs_5419': ratio,
          'alpha': ALPHA, 'rank': RANK,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
