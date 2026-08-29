# IS THE CONVERGED BUILD A UNIFORM WIN, OR A REDISTRIBUTION?
#
# §1951 established that the converged build (36 tables at mlp 768 / attn 256, fallback = 25% output-NN
# neighbour + 75% rank-512 map, 163.757M) beats §1789's deployed design (230.087M) at 5,419 by +0.41pp of
# top-1 and -0.0638 nats at paired t = -28.60. That is a POOLED claim on two aggregate instruments.
#
# §1932 is the cautionary precedent from this thread's own history: an earlier build looked like a strict
# win on aggregates and turned out to be a REDISTRIBUTION -- moving accuracy off frequent targets onto
# rare ones -- which §1933 then showed was itself a superposition of two separate levers. The converged
# build differs from the deployed one on BOTH axes at once (table rank AND fallback form), so it has
# every opportunity to be doing the same thing.
#
# This scores both builds on the two structural instruments the thread interprets with: §1789's buckets
# on the TRUE TARGET's fit-row frequency, and §1936's INPUT-token coverage axis. Rung 2 -- a
# second-class confirmation of §1951 with different instruments, not a replication of the same one.
#
# ARMS. deployed (§1789: full-rank tables, rank-64 map) and blend_768_256 (§1950's converged build).
# 5,419 coverage. Reference labels bound as NAMES (LESSON 83).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY.
#
# Registered predictions, SIGNED per LESSON 72, control polarity derived for THIS lineage (LESSON 81 --
# three polarity errors this session, every one caught by pred_d).
#   pred_a IT IS NOT A REDISTRIBUTION: the converged build's kept-fraction is ABOVE the deployed design's
#          in ALL FIVE target buckets, on at least 2 of 3 roles. If FALSE it is trading buckets like
#          §1932's build did, and §1951's pooled headline needs the same scoping §1932 got.
#   pred_b AND THE WIN IS CONCENTRATED WHERE THE FALLBACK ACTS: the uncovered-input arm's overall top-1
#          gain is at least 2x the covered-input arm's, on at least 2 of 3 roles. The fallback change
#          only touches uncovered inputs (§1936) while the table truncation touches everything, so this
#          measures which of the two levers is actually carrying §1951's margin.
#   pred_c BUT THE TRUNCATION IS VISIBLE ON COMMON TARGETS: restricted to COVERED inputs, the converged
#          build's 125+ kept-fraction is BELOW the deployed design's, on at least 2 of 3 roles. Rank 768
#          against full rank has to cost something somewhere, and the most-frequent bucket at covered
#          inputs is where the tables do their best work. If FALSE the truncation is free even there,
#          which would be worth stating plainly rather than assuming.
#   pred_d CONTROLS: coverage exactly 5,419; the two arms differ in table rank so they MUST move
#          covered-input predictions; buckets partition; live per-cell top-1 and CE identical across
#          arms; and the deployed arm reproduces §1932's PUBLISHED pooled top-1 (13.55 / 14.25 / 13.64%)
#          and 125+ kept-fraction (63.5 / 62.9 / 63.4%) within 0.02pp.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/converged_build_structure_results.json'
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
COVS = (('c5419', B.FIT_5419, 5419),)
C = 'c5419'

print(f'CONVERGED BUILD STRUCTURE | buckets x input coverage | vs §1789 deployed | 5,419 | '
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


allbuckets = sum(1 for r in B.ROLES
                 if all(kf(r, KNEE_LAB, 'pooled', b) > kf(r, REF, 'pooled', b) for b in BK))
pa = allbuckets >= 2

gu = {r: (ovc(r, KNEE_LAB, 'uncovered_input') - ovc(r, REF, 'uncovered_input')) for r in B.ROLES}
gc = {r: (ovc(r, KNEE_LAB, 'covered_input') - ovc(r, REF, 'covered_input')) for r in B.ROLES}
conc = sum(1 for r in B.ROLES if gu[r] >= 2.0 * gc[r])
pb = conc >= 2

trunc = sum(1 for r in B.ROLES
            if kf(r, KNEE_LAB, 'covered_input', TOP) < kf(r, REF, 'covered_input', TOP))
pc = trunc >= 2

# controls: the arms differ in TABLE RANK, so they MUST move covered-input predictions (LESSON 81).
moves = all(chg[c][r][KNEE_LAB] > 0 for c in chg for r in chg[c])
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
S1932_T1 = (0.1355, 0.1425, 0.1364)
S1932_TOP = (0.635, 0.629, 0.634)
repro = max(max(abs(ovc(r, REF, 'pooled') - S1932_T1[i]),
                abs(kf(r, REF, 'pooled', TOP) - S1932_TOP[i]))
            for i, r in enumerate(B.ROLES))
pd = ncov[C] == 5419 and moves and bool(partition) and livesame <= 1e-9 and repro <= 0.0002

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for cls in ('pooled', 'uncovered_input', 'covered_input'):
        print(f'    {cls:16s} ' + '  '.join(
            f'{b:>7s} {kf(r, REF, cls, b):5.1%}->{kf(r, KNEE_LAB, cls, b):5.1%} '
            f'({(kf(r, KNEE_LAB, cls, b) - kf(r, REF, cls, b)) * 100:+.2f})' for b in BK), flush=True)
    print(f'    overall top-1 gain: uncovered {gu[r] * 100:+.2f}pp  covered {gc[r] * 100:+.2f}pp  '
          f'ratio {gu[r] / gc[r] if gc[r] else float("inf"):.2f}x', flush=True)

print(f'\n  the converged build wins ALL FIVE buckets (>=2 roles) -> {pa}  {allbuckets}/3', flush=True)
print(f'  and the win is >=2x concentrated on uncovered inputs (>=2 roles) -> {pb}  {conc}/3', flush=True)
print(f'  but the truncation costs the 125+ bucket at COVERED inputs (>=2 roles) -> {pc}  {trunc}/3',
      flush=True)
print(f'  coverage {ncov}, rank-differing arms move covered inputs {moves}, buckets partition, live '
      f'identical {livesame:.1e}, §1932 reproduced within {repro * 100:.3f}pp -> control {pd}',
      flush=True)

B.report({'pred_a_wins_all_buckets': pa, 'pred_b_win_is_on_uncovered': pb,
          'pred_c_truncation_costs_common': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [5419], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 2 -- §1951 on structural instruments.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'grid': [list(g) for g in GRID], 'fallback': FB,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
