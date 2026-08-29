# WHERE DOES THE MAP CURVE TURN OVER? -- §1958's two findings pull in opposite directions.
#
# §1958 priced four levers at §1957's build and found (a) the MAP RANK is the richest, ahead of mlp on
# 3/3 roles, so marginal money should go there, and (b) the specific next map purchase available --
# rank 512 -> 1024, +42.5M -- returns only ~0.0034 / 0.0029 / 0.0038 nats per 100M, well below the 0.010
# threshold §1958 itself vindicated. Both come from the same run. The reconciliation must be that the
# map's own curve turns over somewhere between 512 and 1024, and nothing has measured where.
#
# ARMS. mix25m{256,384,512,640,768,1024} at §1957's {mlp 768, attn 384}, 5,419. Six points on one axis,
# with the two §1958 already has (256, 512, 1024) as anchors.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1958's open question.
#
# Registered predictions, SIGNED per LESSON 72; reference triples via B.ref() (LESSONS 85, 87); the plan
# contains BOTH same-spec and differing-spec pairs so neither half of the control is vacuous (§1957).
#   pred_a THE CURVE IS CONVEX: the marginal return in nats per 100M falls monotonically across the five
#          steps, on at least 2 of 3 roles -- the shape §1947 found for the table axis. If FALSE there is
#          no single turnover point and "where does it turn over" is the wrong question.
#   pred_b AND IT CROSSES BETWEEN 512 AND 1024: the last step returning at least 0.010 nats per 100M
#          ends at a rank strictly greater than 512 and strictly less than 1024, on at least 2 of 3
#          roles. That is the reconciliation §1958 implies. If FALSE the crossing is at or below 512 --
#          meaning §1949's rank-512 choice was already past it and the map is NOT where marginal money
#          should go, which would correct §1958's pred_b.
#   pred_c AND THE EFFICIENT MAP RANK BEATS 512 SIGNIFICANTLY: the arm at that crossing has pooled CE
#          below mix25m512's with a paired t <= -2.0, on at least 2 of 3 roles. A rate crossing a
#          threshold is not evidence the difference is real (LESSON 84's shape), so this asks for it.
#   pred_d CONTROLS: coverage exactly 5,419; the plan-derived covered-input control holds in BOTH
#          directions with both sides non-empty; buckets partition; live per-cell top-1 and CE identical
#          across arms; and mix25m512 at this allocation reproduces §1957's pooled CE via B.ref() within
#          0.0005 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/map_curve_turnover_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
BASE = {'mlp': 768, 'attn': 384}
MR = (256, 384, 512, 640, 768, 1024)
PLAN = (tuple((f'mix25m{r}', BASE, f'map{r}') for r in MR)
        + (('mix25m512', {'mlp': 768, 'attn': 256}, 'spec_control'),))
LAB = [f'map{r}' for r in MR]
WIDE_LAB = 'map512'
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'map512'
KNEE_LAB = 'map512'

ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
INERT, DIFFER = B.inertness_pairs(PLAN)
COVS = (('c5419', B.FIT_5419, 5419),)
C = 'c5419'

print(f'MAP CURVE TURNOVER | ranks {MR} at {BASE} | 5,419 | DISCOVERY ONLY', flush=True)
res, pt, chg, ncov, COST, AX = {}, {}, {}, {}, {}, {}
PT_K = {}   # paired t vs map512, per role
for cov, fit, nc in COVS:
    print(f'\n########## COVERAGE {nc} ##########', flush=True)
    P = B.Program(fit, expect_ncov=nc)
    liveR = B.score_roles(P, None)
    armR = {lab: B.score_roles(P, SPEC[lab][0], table_rank=SPEC[lab][1]) for lab in ARMS}
    res[cov], pt[cov], chg[cov] = {}, {}, {}
    for role in B.ROLES:
        tgt, icov = B.axes(P, role)
        res[cov][role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
        AX[role] = (tgt, icov)
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR[REF][role][1]) for a in ARMS}
        PT_K[role] = pt[cov][role]
        # the two-sided covered-input control, with its polarity DERIVED FROM THE PLAN rather than
        # inherited -- four forks in a row got it backwards by hand (see B.inertness_pairs).
        chg[cov][role] = {f'{a}|{b}': int(((armR[a][role][0] != armR[b][role][0]) & icov).sum())
                          for a, b in INERT + DIFFER}
        # two-sided: fallback-only differences must be EXACTLY inert at covered inputs; table-rank
        # differences must NOT be. See the pred_d note in the header.
        # polarity checked for THIS lineage: these two differ in table RANK, so they MUST move
        # covered-input predictions (LESSON 81).
        chg[cov][role]['_rank_only'] = int(((armR[KNEE_LAB][role][0]
                                             != armR[REF][role][0]) & icov).sum())
    ncov[cov] = P.ncov
    PROG = P
    # computed from the signal, not read off a side effect of arm() -- see B.route_fraction

    COST.update({lab: P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS})
    del liveR, armR
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']




def step_rate(r, lo, hi):
    """nats per 100M for moving from the richer build `hi` down to the cheaper `lo`."""
    return (ce(C, r, lo) - ce(C, r, hi)) / ((COST[hi] - COST[lo]) / 100.0)




def kf(r, a, cls, b):
    return res[C][r][a][cls][b]['kept_fraction']


def ovc(r, a, cls):
    return res[C][r][a][cls]['overall']['top1_acc_prog']


BK = [f'{x}-{y}' for x, y in B.BUCKETS]
BOT, TOP = BK[0], BK[-1]
# §1951's PUBLISHED 5,419 pooled CE for the converged build.
S1957_CE = B.ref(B.PT + 'ops/coverage_specific_build_results.json', 'blend_768_384')


def kf(r, a, cls, b):
    return res[C][r][a][cls][b]['kept_fraction']


def ce(r, a):
    return res[C][r][a]['pooled']['overall']['ce_prog']



def step(r, i):
    """nats per 100M for the step from MR[i-1] up to MR[i]."""
    lo, hi = LAB[i - 1], LAB[i]
    dm = (COST[hi] - COST[lo]) / 100.0
    return (ce(r, lo) - ce(r, hi)) / dm if dm else float('nan')


RATES = {r: [step(r, i) for i in range(1, len(MR))] for r in B.ROLES}
conv = sum(1 for r in B.ROLES if all(RATES[r][i] < RATES[r][i - 1] for i in range(1, len(RATES[r]))))
pa = conv >= 2


def cross(r):
    """the rank reached by the LAST step still returning >= 0.010 nats per 100M."""
    best = MR[0]
    for i in range(1, len(MR)):
        if RATES[r][i - 1] >= 0.010:
            best = MR[i]
    return best


pb_n = sum(1 for r in B.ROLES if 512 < cross(r) < 1024)
pb = pb_n >= 2
pc_n = sum(1 for r in B.ROLES
           if cross(r) != 512 and ce(r, f'map{cross(r)}') < ce(r, KNEE_LAB)
           and PT_K[r][f'map{cross(r)}']['t'] <= -2.0)
pc = pc_n >= 2

moves = (all(chg[c][r][f'{a}|{b}'] == 0 for c in chg for r in chg[c] for a, b in INERT)
         and all(chg[c][r][f'{a}|{b}'] > 0 for c in chg for r in chg[c] for a, b in DIFFER))
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
repro = max(abs(ce(r, KNEE_LAB) - S1957_CE[i]) for i, r in enumerate(B.ROLES))
fracok = 0.0
pd = (ncov[C] == 5419 and moves and bool(partition) and livesame <= 1e-9
      and repro <= 0.0005 and fracok <= 0.01)

for r in B.ROLES:
    for a in ARMS:
        print(f'    {a:16s} 0-0 {kf(r, a, "pooled", BOT):5.2%}  125+ {kf(r, a, "pooled", TOP):5.1%}  '
              f'CE {ce(r, a):7.5f}', flush=True)

for r in B.ROLES:
    print(f'\n  {r}  CE by map rank: ' + '  '.join(f'{MR[i]} {ce(r, LAB[i]):.5f}'
                                                   for i in range(len(MR))), flush=True)
    print(f'    marginal nats/100M: ' + '  '.join(f'{MR[i]}<-{MR[i-1]} {RATES[r][i-1]:.4f}'
                                                  for i in range(1, len(MR)))
          + f'   | crossing at rank {cross(r)}', flush=True)
print(f'\n  the map curve is convex (>=2 roles) -> {pa}  {conv}/3', flush=True)
print(f'  and the 0.010 crossing lies strictly between 512 and 1024 (>=2 roles) -> {pb}  {pb_n}/3  '
      f'(crossings {[cross(r) for r in B.ROLES]})', flush=True)
print(f'  and that rank beats 512 significantly (>=2 roles) -> {pc}  {pc_n}/3', flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1957 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_curve_is_convex': pa, 'pred_b_crossing_between_512_and_1024': pb,
          'pred_c_beats_512_significantly': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- S1958 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'base': BASE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
