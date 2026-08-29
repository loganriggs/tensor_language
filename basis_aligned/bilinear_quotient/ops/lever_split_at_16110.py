# WHICH LEVER CARRIES IT AT 16,110? -- the same decomposition, the other coverage.
#
# §1953 decomposed §1951's win at 5,419 and found the FALLBACK carries essentially all of it: the
# uncovered-input arm gains +1.59/+1.08/+1.26pp of top-1 against the covered arm's +0.04/+0.08/+0.04pp,
# a ratio of 44.4x/12.9x/29.3x, so the 51%-cheaper tables are very nearly free rather than beneficial.
# §1947 concluded the opposite emphasis at 16,110 -- that the TABLE axis was the larger lever, and
# §1946 measured it at ~30x cheaper per nat than the fallback axis there.
#
# Both can be true: the fallback touches ~24% of scored positions at 5,419 and ~10% at 16,110 (§1936).
# But the two claims have never been put on the SAME instrument, and §1953's covered/uncovered
# decomposition has never been run at 16,110. Until it is, "which lever matters" rests on a comparison
# between a per-nat cost rate and a per-position accuracy split, which are not the same measurement.
#
# ARMS. the 16,110 deployed design (full-rank tables, rank-64 map) and the converged build
# (mlp 768 / attn 256, fallback mix25m512). Reference labels bound as NAMES (LESSON 83).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 16,110. Rung 3 -- §1953's open question.
#
# Registered predictions, SIGNED per LESSON 72; control polarity derived for THIS lineage (LESSON 81);
# every reproduction bar >= the precision its reference was PUBLISHED at (LESSON 84).
#   pred_a THE REDISTRIBUTION SHAPE PERSISTS: at 16,110 the converged build LOSES the unseen (0-0)
#          bucket pooled AND wins both the 5-24 and 25-124 buckets, on at least 2 of 3 roles -- the
#          shape §1953 found at 5,419. If FALSE the redistribution is a low-coverage phenomenon and
#          §1953's scoping of §1951 applies only there.
#   pred_b BUT THE CONCENTRATION FALLS: the ratio of uncovered-input to covered-input overall top-1 gain
#          at 16,110 is SMALLER than the 44.4 / 12.9 / 29.3x §1953 measured at 5,419, on at least 2 of 3
#          roles. The uncovered arm is ~10% of positions here against ~24% there, so a fallback-carried
#          win must dilute. If FALSE the ratio is coverage-independent, which would mean it is not
#          measuring what §1953 took it to measure.
#   pred_c AND THE TABLES CARRY MORE HERE: the COVERED-input overall top-1 gain at 16,110 EXCEEDS the
#          +0.04 / +0.08 / +0.04pp §1953 measured at 5,419, on at least 2 of 3 roles. This is the
#          question the section exists for -- it puts §1947's "the table axis is the larger lever at
#          16,110" and §1953's "the fallback carries it at 5,419" on one instrument. If FALSE the tables
#          are nearly free at BOTH coverages and §1946/§1947's per-nat framing, not §1953's, is the one
#          that needs re-reading.
#   pred_d CONTROLS: coverage exactly 16,110; the two arms differ in table rank so they MUST move
#          covered-input predictions; buckets partition; live per-cell top-1 and CE identical across
#          arms; and the deployed arm reproduces §1934's PUBLISHED 16,110 kept-fractions -- 125+ 53.6 /
#          54.1 / 53.9% and unseen 2.6 / 4.9 / 3.5% -- within 0.1pp, a bar set at twice the 0.05pp
#          rounding envelope of a 1-dp published figure (LESSON 84).
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/lever_split_at_16110_results.json'
GRID = ((768, 256),)
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = ((('map64', None, 'deployed'),)
        + tuple((FB, {'mlp': m, 'attn': a}, f'blend_{m}_{a}') for m, a in GRID))
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'deployed'                # §1789's deployed design, the anchor
KNEE_LAB = 'blend_768_256'      # §1949's build

ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'

print(f'LEVER SPLIT AT 16,110 | buckets x input coverage | vs the deployed design | '
      f'DISCOVERY ONLY', flush=True)
res, pt, chg, ncov, COST = {}, {}, {}, {}, {}
for cov, fit, nc in COVS:
    print(f'\n########## COVERAGE {nc} ##########', flush=True)
    P = B.Program(fit, expect_ncov=nc)
    liveR = B.score_roles(P, None)
    armR = {lab: B.score_roles(P, SPEC[lab][0], table_rank=SPEC[lab][1]) for lab in ARMS}
    res[cov], pt[cov], chg[cov] = {}, {}, {}
    for role in B.ROLES:
        tgt, icov = B.axes(P, role)
        res[cov][role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR[REF][role][1])
                         for a in ARMS}
        chg[cov][role] = {a: int(((armR[a][role][0] != armR[REF][role][0]) & icov).sum())
                          for a in ARMS}
        # two-sided: fallback-only differences must be EXACTLY inert at covered inputs; table-rank
        # differences must NOT be. See the pred_d note in the header.
        # polarity checked for THIS lineage: these two differ in table RANK, so they MUST move
        # covered-input predictions (LESSON 81).
        chg[cov][role]['_rank_only'] = int(((armR[KNEE_LAB][role][0]
                                             != armR[REF][role][0]) & icov).sum())
    ncov[cov] = P.ncov
    COST.update({lab: P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS})
    del P, liveR, armR
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']


LAB = [f'blend_{m}_{a}' for m, a in GRID]


def step_rate(r, lo, hi):
    """nats per 100M for moving from the richer build `hi` down to the cheaper `lo`."""
    return (ce(C, r, lo) - ce(C, r, hi)) / ((COST[hi] - COST[lo]) / 100.0)


BK = [f'{x}-{y}' for x, y in B.BUCKETS]
TOP = BK[-1]


def kf(r, a, cls, b):
    return res[C][r][a][cls][b]['kept_fraction']


def ovc(r, a, cls):
    return res[C][r][a][cls]['overall']['top1_acc_prog']


S1953_RATIO = (44.37, 12.89, 29.25)          # §1953's uncovered/covered gain ratio at 5,419
S1953_COV = (0.0004, 0.0008, 0.0004)        # §1953's covered-input overall top-1 gain at 5,419
MID = [BK[2], BK[3]]                        # the 5-24 and 25-124 buckets

shape = sum(1 for r in B.ROLES
            if kf(r, KNEE_LAB, 'pooled', BK[0]) < kf(r, REF, 'pooled', BK[0])
            and all(kf(r, KNEE_LAB, 'pooled', b) > kf(r, REF, 'pooled', b) for b in MID))
pa = shape >= 2

gu = {r: (ovc(r, KNEE_LAB, 'uncovered_input') - ovc(r, REF, 'uncovered_input')) for r in B.ROLES}
gc = {r: (ovc(r, KNEE_LAB, 'covered_input') - ovc(r, REF, 'covered_input')) for r in B.ROLES}
ratio = {r: (gu[r] / gc[r] if gc[r] else float('inf')) for r in B.ROLES}
dilute = sum(1 for i, r in enumerate(B.ROLES) if ratio[r] < S1953_RATIO[i])
pb = dilute >= 2

tables = sum(1 for i, r in enumerate(B.ROLES) if gc[r] > S1953_COV[i])
pc = tables >= 2

# controls: the arms differ in TABLE RANK, so they MUST move covered-input predictions (LESSON 81).
moves = all(chg[c][r][KNEE_LAB] > 0 for c in chg for r in chg[c])
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
# §1934's PUBLISHED 16,110 kept-fractions for the deployed design, and a bar at 0.1pp -- twice the
# 0.05pp rounding envelope of a 1-dp figure. §1953's pred_d failed on a 0.02pp bar against exactly this
# kind of reference, which no run could have met (LESSON 84).
S1934_TOP = (0.536, 0.541, 0.539)
S1934_BOT = (0.026, 0.049, 0.035)
BOT = BK[0]
repro = max(max(abs(kf(r, REF, 'pooled', TOP) - S1934_TOP[i]),
                abs(kf(r, REF, 'pooled', BOT) - S1934_BOT[i]))
            for i, r in enumerate(B.ROLES))
pd = ncov[C] == 16110 and moves and bool(partition) and livesame <= 1e-9 and repro <= 0.001

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for cls in ('pooled', 'uncovered_input', 'covered_input'):
        print(f'    {cls:16s} ' + '  '.join(
            f'{b:>7s} {kf(r, REF, cls, b):5.1%}->{kf(r, KNEE_LAB, cls, b):5.1%} '
            f'({(kf(r, KNEE_LAB, cls, b) - kf(r, REF, cls, b)) * 100:+.2f})' for b in BK), flush=True)
    print(f'    overall top-1 gain: uncovered {gu[r] * 100:+.2f}pp  covered {gc[r] * 100:+.2f}pp  '
          f'ratio {gu[r] / gc[r] if gc[r] else float("inf"):.2f}x', flush=True)

print(f'\n  the redistribution SHAPE persists at 16,110 (>=2 roles) -> {pa}  {shape}/3', flush=True)
print(f'  and the uncovered/covered concentration FALLS vs §1953 (>=2 roles) -> {pb}  {dilute}/3  '
      f'(ratios ' + ' '.join(f'{ratio[r]:.1f}x' for r in B.ROLES) + f' vs {S1953_RATIO})', flush=True)
print(f'  and the COVERED-input gain EXCEEDS §1953\'s (>=2 roles) -> {pc}  {tables}/3  '
      f'(covered ' + ' '.join(f'{gc[r] * 100:+.3f}pp' for r in B.ROLES) + ')', flush=True)
print(f'  coverage {ncov}, rank-differing arms move covered inputs {moves}, buckets partition, live '
      f'identical {livesame:.1e}, §1934 reproduced within {repro * 100:.3f}pp -> control {pd}',
      flush=True)

B.report({'pred_a_shape_persists': pa, 'pred_b_concentration_falls': pb,
          'pred_c_tables_carry_more': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1953 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'grid': [list(g) for g in GRID], 'fallback': FB,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
