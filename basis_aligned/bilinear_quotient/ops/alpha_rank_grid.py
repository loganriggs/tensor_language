# ALPHA AND MAP RANK TOGETHER -- is rank 512 still right once the blend does part of the work?
#
# §1941-§1943 pinned the map rank at 512 because §1938 located the map's CE strength there. But §1942's
# finding is that the neighbour and the map contribute along DIFFERENT directions in row space, and
# §1943 found the CE optimum at alpha=0.25 -- the blend leans mostly on the map. If the two ingredients
# are orthogonal, the best (alpha, rank) pair need not have rank 512, and rank is where the cost is:
# 36*R*2*D is 5.308M at R=64 and 42.467M at R=512, on a 224.778M table.
#
# ARMS. alpha in {10, 25, 40, 60} x map rank in {64, 128, 256, 512}, plus map512 / map64 / nn as anchors.
# 5,419 coverage, full table rank. Two nested loops over arms ops/bqlib.py already builds.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1943's open question.
#
# Registered predictions, SIGNED per LESSON 72, every CE claim with a PAIRED t up front (LESSON 78).
#   pred_a A CHEAPER RANK MATCHES: some pair with map rank <= 256 has pooled CE within 0.005 nats of the
#          best rank-512 blend, on at least 2 of 3 roles. That is a >= 21.2M saving for a margin smaller
#          than the one §1943 measured between alpha=25 and alpha=40. If FALSE the CE genuinely needs
#          rank 512 and the blend does not substitute for map capacity.
#   pred_b THE OPTIMAL ALPHA RISES AS RANK FALLS: the CE-minimising alpha at map rank 64 is strictly
#          GREATER than the CE-minimising alpha at rank 512, on at least 2 of 3 roles. A weaker map
#          should be leaned on less, so the blend should shift toward the neighbour. If FALSE the alpha
#          optimum is a property of the neighbour alone and does not know what it is mixed with.
#   pred_c AND A CHEAPER PAIR STILL BEATS map512: some pair whose TOTAL cost is BELOW map512's 267.246M
#          beats map512 on pooled top-1 AND has a paired CE t <= -2.0 against it, on at least 2 of 3
#          roles. This is the deployable claim -- strictly better on both instruments AND cheaper. If
#          FALSE every improvement over map512 costs at least as much as map512 does.
#   pred_d CONTROLS: coverage exactly 5,419; every arm inert at covered inputs; buckets partition; live
#          per-cell top-1 and CE identical across arms; and the alpha=25 rank=512 pair reproduces
#          §1943's PUBLISHED pooled CE (5.94165 / 5.91021 / 5.93277) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/alpha_rank_grid_results.json'
AL = (10, 25, 40, 60)
RK = (64, 128, 256, 512)
MIX = tuple(f'mix{a}m{r}' for r in RK for a in AL)
ARMS = MIX + ('nn', 'map512', 'map64')
ALPHA = {f'mix{a}m{r}': a / 100.0 for r in RK for a in AL}
RANK = {f'mix{a}m{r}': r for r in RK for a in AL}
COVS = (('c5419', B.FIT_5419, 5419),)
S1943_CE = {'mix25m512': (5.94165, 5.91021, 5.93277), 'map512': (5.96702, 5.93645, 5.96095)}

print(f'ALPHA x MAP RANK GRID | alphas {AL} x ranks {RK} | 5,419 | DISCOVERY ONLY', flush=True)
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


def argmin_ce_at(c, r, rank):
    cand = [a for a in MIX if RANK[a] == rank]
    return min(cand, key=lambda a: ce(c, r, a))


C = 'c5419'
best512 = {r: min((a for a in MIX if RANK[a] == 512), key=lambda a: ce(C, r, a)) for r in B.ROLES}
cheapmatch = sum(1 for r in B.ROLES
                 if min(ce(C, r, a) for a in MIX if RANK[a] <= 256) - ce(C, r, best512[r]) <= 0.005)
pa = cheapmatch >= 2
rises = sum(1 for r in B.ROLES
            if ALPHA[argmin_ce_at(C, r, 64)] > ALPHA[argmin_ce_at(C, r, 512)])
pb = rises >= 2
CHEAP = [a for a in MIX if COST[a] < COST['map512']]
cheapwin = sum(1 for r in B.ROLES
               if any(ov(C, r, a) > ov(C, r, 'map512') and pt[C][r][a]['t'] <= -2.0 for a in CHEAP))
pc = cheapwin >= 2

inert = all(v == 0 for c in chg for r in chg[c] for v in chg[c][r].values())
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r]['map64'][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(abs(ce('c5419', r, a) - S1943_CE[a][i])
            for a in S1943_CE for i, r in enumerate(B.ROLES))
pd = ncov['c5419'] == 5419 and inert and livesame <= 1e-9 and repro <= 0.0002

for r in B.ROLES:
    print(f'\n  {r}   (best rank-512 blend: {best512[r]})', flush=True)
    for rank in RK:
        row = '  '.join(f'a{a:02d} CE {ce(C, r, f"mix{a}m{rank}"):7.5f} '
                        f'({ce(C, r, f"mix{a}m{rank}") - ce(C, r, "map512"):+.4f},'
                        f't{pt[C][r][f"mix{a}m{rank}"]["t"]:+6.2f}) '
                        f't1 {ov(C, r, f"mix{a}m{rank}"):5.2%}' for a in AL)
        print(f'    rank {rank:4d} [{COST[f"mix{AL[0]}m{rank}"]:7.3f}M]  {row}', flush=True)
    print(f'    anchors: map512 CE {ce(C, r, "map512"):7.5f} t1 {ov(C, r, "map512"):5.2%} '
          f'[{COST["map512"]:7.3f}M] | map64 CE {ce(C, r, "map64"):7.5f} t1 '
          f'{ov(C, r, "map64"):5.2%} [{COST["map64"]:7.3f}M]', flush=True)
    print(f'    CE argmin at rank 64: {argmin_ce_at(C, r, 64)}   at rank 512: '
          f'{argmin_ce_at(C, r, 512)}', flush=True)
print(f'\n  a rank <= 256 pair matches the best rank-512 blend within 0.005 nats (>=2 roles) -> {pa}  '
      f'{cheapmatch}/3', flush=True)
print(f'  and the CE-optimal alpha RISES as rank falls (>=2 roles) -> {pb}  {rises}/3', flush=True)
print(f'  and a pair CHEAPER than map512 still beats it on both instruments (>=2 roles) -> {pc}  '
      f'{cheapwin}/3', flush=True)
print(f'  coverage {ncov}, arms inert {inert}, live identical {livesame:.1e}, §1943 CE '
      f'reproduced within {repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_cheaper_rank_matches': pa, 'pred_b_alpha_rises_as_rank_falls': pb,
          'pred_c_cheaper_pair_beats_map512': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [5419], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1943 open question.',
                     'alpha': ALPHA},
          'results': res,
          'paired_vs_map512': pt,
          'ce_argmin_by_rank': {r: {rk: argmin_ce_at(C, r, rk) for rk in RK} for r in B.ROLES},
          'alpha': ALPHA, 'rank': RANK,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
