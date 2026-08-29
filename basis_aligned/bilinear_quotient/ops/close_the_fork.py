# CAN THE OBJECTIVE FORK BE CLOSED? -- routing the remainder to a rank-512 map instead of rank-64.
#
# §1938 found CE and top-1 rank the fallback forms in exactly reversed order. §1939 claimed a routed
# hybrid closed the fork; §1940's paired t retracted that -- nn75 is a top-1 win and CE-NEUTRAL (t
# -0.54/-0.23/-0.44 at 5,419), so the fork is open: nn75 wins top-1, map512 wins CE by 0.038-0.047 nats.
#
# nn75 routes its non-neighbour quarter to a rank-64 map, because it was built as a cheap variant of the
# DEPLOYED design. That is an accident of lineage, not a choice: §1938 showed the map's CE strength lives
# almost entirely in the unseen-target bucket, and rank-512 is where that strength is. If the neighbour
# supplies the top-1 on the three quarters where it is close, and a rank-512 map supplies the CE on the
# quarter where it is not, nn75m512 should take both -- for 42.56M against map512's 42.47M, i.e. free.
#
# ARMS. nn75m512 (the candidate), and nn75 / map64 / map512 as the published anchors. 5,419 coverage,
# full table rank. Written against ops/bqlib.py: ~50 lines of experiment against §1940's 348.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1940's open question.
#
# Registered predictions, SIGNED per LESSON 72, and every CE claim carries a PAIRED t up front rather
# than as a caveat (LESSON 78 -- that omission is exactly what cost §1939 its headline).
#   pred_a IT KEEPS THE TOP-1 WIN: nn75m512's pooled top-1 is ABOVE map512's on at least 2 of 3 roles.
#          If FALSE the neighbour's top-1 advantage does not survive a stronger map on the remainder and
#          the two effects are not separable by routing.
#   pred_b AND IT DOES NOT GIVE UP map512's CE: the paired per-position CE difference
#          (nn75m512 - map512) has t >= -2 is NOT the bar -- the bar is that it does not get WORSE by a
#          significant margin, i.e. t <= +2.0, on at least 2 of 3 roles. One-sided: I am testing that the
#          routing does not COST CE, not that it gains any.
#   pred_c AND THE FORK CLOSES AGAINST THE DEPLOYED DESIGN: nn75m512's pooled top-1 is ABOVE map64's AND
#          its paired CE t against map64 is <= -2.0, on at least 2 of 3 roles. This is §1939's retracted
#          claim, restated with the significance bar it should have carried the first time. If FALSE the
#          deployed design is still not strictly beaten and I say so plainly.
#   pred_d CONTROLS: coverage exactly 5,419; every arm inert at covered inputs; buckets partition; live
#          per-cell top-1 and CE identical across arms; and the nn75 / map64 / map512 anchors reproduce
#          §1940's PUBLISHED pooled top-1 (14.02/14.69/14.12, 13.55/14.25/13.64, 13.77/14.37/13.72%)
#          within 0.01pp.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/close_the_fork_results.json'
ARMS = ('nn75m512', 'nn75', 'map64', 'map512')
S1940_T1 = {'nn75': (0.1402, 0.1469, 0.1412), 'map64': (0.1355, 0.1425, 0.1364),
            'map512': (0.1377, 0.1437, 0.1372)}

print('CLOSE THE FORK | nn75m512 vs the §1940 anchors | 5,419 | DISCOVERY ONLY', flush=True)
P = B.Program(B.FIT_5419, expect_ncov=5419)
liveR = B.score_roles(P, None)
armR = {a: B.score_roles(P, a) for a in ARMS}
res, pt, chg = {}, {}, {}
for role in B.ROLES:
    tgt, icov = B.axes(P, role)
    res[role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
    pt[role] = {b: B.paired_t(armR['nn75m512'][role][1], armR[b][role][1]) for b in ('map512', 'map64')}
    chg[role] = {a: int(((armR[a][role][0] != armR['map64'][role][0]) & icov).sum()) for a in ARMS}


def ov(r, a):
    return res[r][a]['pooled']['overall']['top1_acc_prog']


def ce(r, a):
    return res[r][a]['pooled']['overall']['ce_prog']


keeps = sum(1 for r in B.ROLES if ov(r, 'nn75m512') > ov(r, 'map512'))
noloss = sum(1 for r in B.ROLES if pt[r]['map512']['t'] <= 2.0)
closes = sum(1 for r in B.ROLES if ov(r, 'nn75m512') > ov(r, 'map64') and pt[r]['map64']['t'] <= -2.0)
pa, pb, pc = keeps >= 2, noloss >= 2, closes >= 2

inert = all(v == 0 for r in chg for v in chg[r].values())
livesame = max(abs(res[r][a][cl][b]['ce_live'] - res[r]['map64'][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[r][a][cl])
repro = max(abs(ov(r, a) - S1940_T1[a][i]) for a in S1940_T1 for i, r in enumerate(B.ROLES))
pd = P.ncov == 5419 and inert and livesame <= 1e-9 and repro <= 0.0001

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:10s} top1 {ov(r, a):6.2%}  CE {ce(r, a):7.5f}  cost {P.cost(a) / 1e6:8.3f}M',
              flush=True)
    print(f'    nn75m512 - map512: top1 {(ov(r, "nn75m512") - ov(r, "map512")) * 100:+.2f}pp  '
          f'CE {ce(r, "nn75m512") - ce(r, "map512"):+.5f} (paired t {pt[r]["map512"]["t"]:+.2f})',
          flush=True)
    print(f'    nn75m512 - map64 : top1 {(ov(r, "nn75m512") - ov(r, "map64")) * 100:+.2f}pp  '
          f'CE {ce(r, "nn75m512") - ce(r, "map64"):+.5f} (paired t {pt[r]["map64"]["t"]:+.2f})',
          flush=True)
print(f'\n  keeps the top-1 win over map512 (>=2 roles) -> {pa}  {keeps}/3', flush=True)
print(f'  and does not significantly cost CE vs map512 (>=2 roles) -> {pb}  {noloss}/3', flush=True)
print(f'  and CLOSES the fork against the deployed design, with the t bar (>=2 roles) -> {pc}  '
      f'{closes}/3', flush=True)
print(f'  coverage {P.ncov}, arms inert {inert}, live identical {livesame:.1e}, anchors reproduce '
      f'§1940 within {repro * 100:.3f}pp -> control {pd}', flush=True)

B.report({'pred_a_keeps_top1': pa, 'pred_b_no_ce_loss': pb, 'pred_c_closes_fork': pc,
          'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverage': 5419, 'table_rank': 'FULL',
                     'costs_M': {a: P.cost(a) / 1e6 for a in ARMS},
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1940 open question.'},
          'results': res,
          'paired': {r: {b: pt[r][b] for b in pt[r]} for r in B.ROLES},
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
