# ROUTE OR BLEND? -- are the neighbour's top-1 and the map's CE additive in ROW space?
#
# §1941 left the frontier at ~267M as exactly two points: nn75m512 wins top-1 by +0.34/+0.37/+0.41pp,
# map512 wins CE by +0.0243/+0.0275/+0.0164 nats at paired t +5.93/+6.30/+2.67. Routing gives each
# uncovered token ONE form or the other, so it gives up CE in proportion to what it routes away -- and
# §1939 showed the cosine router does not find the unseen-target case that the CE penalty lives in.
#
# BLENDING is a different mechanism with the same two ingredients: every uncovered type gets
# A% neighbour row + (100-A)% rank-512 map row. If the neighbour's top-1 advantage and the map's CE
# advantage are carried by different DIRECTIONS in row space they may partly survive a convex mix;
# if they are the same direction traded off, the mix just interpolates and there is nothing here.
# Cost is identical to map512 plus one index -- 267.335M -- so this is free if it works.
#
# ARMS. mix25m512 / mix50m512 / mix75m512, against nn75m512, map512, map64 (deployed). 5,419, full
# table rank. Written against ops/bqlib.py.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1941's open question.
#
# Registered predictions, SIGNED per LESSON 72, every CE claim with a PAIRED t up front (LESSON 78).
#   pred_a A BLEND BEATS map512 ON TOP-1: some mix arm's pooled top-1 exceeds map512's on >=2 of 3 roles.
#          If FALSE the neighbour contributes nothing to top-1 once averaged and blending is dead.
#   pred_b AND IT COSTS LESS CE THAN ROUTING DOES: that same arm's paired CE t against map512 is strictly
#          SMALLER than nn75m512's (+5.93/+6.30/+2.67) on >=2 of 3 roles -- i.e. blending gives up less
#          of map512's CE than routing did for a comparable top-1 gain. This is the actual question.
#   pred_c AND THE MIX IS NOT MERE INTERPOLATION: some mix arm's pooled CE is BELOW the linear
#          interpolation of the nn and map512 endpoint CEs at its own alpha, on >=2 of 3 roles. §1939
#          found exactly this for routing (12/12 arms beat interpolation); if blending does NOT, the two
#          mechanisms are not equivalent and the row-space average destroys what the router kept.
#   pred_d CONTROLS: coverage exactly 5,419; every arm inert at covered inputs; buckets partition; live
#          per-cell top-1 and CE identical across arms; and nn75m512 / map512 / map64 reproduce §1941's
#          PUBLISHED pooled top-1 (14.12/14.74/14.13, 13.77/14.37/13.72, 13.55/14.25/13.64%) within
#          0.01pp.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/route_or_blend_results.json'
ARMS = ('mix25m512', 'mix50m512', 'mix75m512', 'nn', 'nn75m512', 'map64', 'map512')
MIX = ('mix25m512', 'mix50m512', 'mix75m512')
ALPHA = {'mix25m512': 0.25, 'mix50m512': 0.50, 'mix75m512': 0.75}
S1941_T1 = {'nn75m512': (0.1412, 0.1474, 0.1413), 'map64': (0.1355, 0.1425, 0.1364),
            'map512': (0.1377, 0.1437, 0.1372)}
S1941_T = {'skip7000': 5.93, 'skip11000': 6.30, 'skip1200': 2.67}   # nn75m512 vs map512 paired t

print('ROUTE OR BLEND | row-space mixes vs §1941 routing | 5,419 | DISCOVERY ONLY', flush=True)
P = B.Program(B.FIT_5419, expect_ncov=5419)
liveR = B.score_roles(P, None)
armR = {a: B.score_roles(P, a) for a in ARMS}
res, pt, chg = {}, {}, {}
for role in B.ROLES:
    tgt, icov = B.axes(P, role)
    res[role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
    pt[role] = {a: B.paired_t(armR[a][role][1], armR['map512'][role][1]) for a in ARMS}
    chg[role] = {a: int(((armR[a][role][0] != armR['map64'][role][0]) & icov).sum()) for a in ARMS}


def ov(r, a):
    return res[r][a]['pooled']['overall']['top1_acc_prog']


def ce(r, a):
    return res[r][a]['pooled']['overall']['ce_prog']


beats = {a: sum(1 for r in B.ROLES if ov(r, a) > ov(r, 'map512')) for a in MIX}
pa_arm = max(MIX, key=lambda a: beats[a])
pa = beats[pa_arm] >= 2
cheaper = {a: sum(1 for r in B.ROLES if pt[r][a]['t'] < pt[r]['nn75m512']['t']) for a in MIX}
pb = cheaper[pa_arm] >= 2


def interp(r, a):
    al = ALPHA[a]
    return al * ce(r, 'nn') + (1 - al) * ce(r, 'map512')


beatint = {a: sum(1 for r in B.ROLES if ce(r, a) < interp(r, a)) for a in MIX}
pc_arm = max(MIX, key=lambda a: beatint[a])
pc = beatint[pc_arm] >= 2

inert = all(v == 0 for r in chg for v in chg[r].values())
livesame = max(abs(res[r][a][cl][b]['ce_live'] - res[r]['map64'][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[r][a][cl])
repro = max(abs(ov(r, a) - S1941_T1[a][i]) for a in S1941_T1 for i, r in enumerate(B.ROLES))
pd = P.ncov == 5419 and inert and livesame <= 1e-9 and repro <= 0.0001

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:10s} top1 {ov(r, a):6.2%}  CE {ce(r, a):7.5f}  cost {P.cost(a) / 1e6:8.3f}M',
              flush=True)
    for a in MIX:
        print(f'    {a} - map512: top1 {(ov(r, a) - ov(r, "map512")) * 100:+.2f}pp  '
              f'CE {ce(r, a) - ce(r, "map512"):+.5f} (paired t {pt[r][a]["t"]:+.2f}; nn75m512 was '
              f'{pt[r]["nn75m512"]["t"]:+.2f}) | vs linear interp {ce(r, a) - interp(r, a):+.5f}',
              flush=True)
print(f'\n  a BLEND beats map512 on top-1 (>=2 roles) -> {pa}  best {pa_arm} {beats[pa_arm]}/3', flush=True)
print(f'  and gives up LESS CE than routing did (>=2 roles) -> {pb}  {cheaper[pa_arm]}/3', flush=True)
print(f'  and beats linear interpolation (>=2 roles) -> {pc}  best {pc_arm} {beatint[pc_arm]}/3', flush=True)
print(f'  coverage {P.ncov}, arms inert {inert}, live identical {livesame:.1e}, anchors reproduce '
      f'§1940 within {repro * 100:.3f}pp -> control {pd}', flush=True)

B.report({'pred_a_blend_beats_map512': pa, 'pred_b_less_ce_given_up': pb,
          'pred_c_beats_interpolation': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverage': 5419, 'table_rank': 'FULL',
                     'costs_M': {a: P.cost(a) / 1e6 for a in ARMS},
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1941 open question.',
                     'alpha': ALPHA},
          'results': res,
          'paired_vs_map512': {r: {a: pt[r][a] for a in pt[r]} for r in B.ROLES},
          'vs_linear_interp': {r: {a: ce(r, a) - interp(r, a) for a in MIX} for r in B.ROLES},
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
