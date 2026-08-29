# THE ATTENTION SWEEP WHERE ATTENTION IS WORTH MORE.
#
# §1948 swept the attention share at 16,110 and found CE strictly MONOTONE decreasing in it -- no interior
# optimum -- which is why §1928's 12.5-25% band was corrected from an allocation optimum to an efficiency
# rule. §1951 then found that the attention share's WORTH is not coverage-invariant: removing capacity
# (384 -> 256) costs 0.0113/0.0135/0.0091 nats per 100M at 5,419 against 0.0047/0.0051/0.0034 at 16,110,
# two to three times more. §1948 was never repeated at 5,419 -- and that is exactly where a genuine
# interior optimum could exist, because the site classes are worth different amounts there.
#
# ARMS. mlp 768 with attention at 64 / 128 / 192 / 256 / 384 / 576, fallback fixed at §1949's mix25m512;
# plus §1789's deployed design as the anchor. 5,419 coverage.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §1948 at the other coverage.
#
# Registered predictions, SIGNED per LESSON 72, reference labels bound as NAMES (LESSON 83), control
# polarity derived for THIS lineage (LESSON 81 -- three polarity errors this session, all caught by
# pred_d).
#   pred_a MONOTONICITY SURVIVES: pooled CE is strictly decreasing in the attention rank across the whole
#          sweep, on at least 2 of 3 roles, as it was at 16,110. If FALSE there IS an interior optimum at
#          the deployed coverage and §1948's shape claim is a 16,110 statement, not a general one.
#   pred_b BUT THE EFFICIENT POINT MOVES UP: the cheapest attention rank whose step DOWN costs at least
#          0.010 nats per 100M is STRICTLY GREATER than 256, on at least 2 of 3 roles -- i.e. at 5,419
#          the efficient stopping point is above §1947's {768,256}. §1951 measured the 384 -> 256 step at
#          0.0113/0.0135/0.0091, already at or over the threshold on two roles, so this is the direct
#          consequence. If FALSE §1951's rates do not imply what they appear to.
#   pred_c AND THE BUILD SHOULD CHANGE: the best attention rank by that criterion beats {768,256} on
#          pooled CE with a paired t <= -2.0, on at least 2 of 3 roles. A rate crossing a threshold is
#          not by itself evidence the difference is real; this asks for significance before I would
#          recommend moving the deployed-coverage build off §1947's allocation.
#   pred_d CONTROLS: coverage exactly 5,419; arms differing in attention RANK DO move covered-input
#          predictions; the deployed anchor differs from the {768,256} arm in BOTH table rank and
#          fallback so it must also differ there; buckets partition; live per-cell top-1 and CE identical
#          across arms; and the deployed anchor reproduces §1932's PUBLISHED pooled top-1 (13.55 / 14.25
#          / 13.64%) within 0.02pp.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/attention_sweep_at_5419_results.json'
GRID = ((768, 64), (768, 128), (768, 192), (768, 256), (768, 384), (768, 576))
FB = 'mix25m512'   # §1949's fallback
ATTN_G = [a for _m, a in GRID]
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = ((('map64', None, 'deployed'), (FB, None, 'blend_full'))
        + tuple((FB, {'mlp': m, 'attn': a}, f'blend_{m}_{a}') for m, a in GRID))
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'deployed'                # §1789's deployed design, the anchor
KNEE_LAB = 'blend_768_256'      # §1949's build
FULL_LAB = 'blend_full'
ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
COVS = (('c5419', B.FIT_5419, 5419),)
C = 'c5419'
S1932_T1 = {'deployed': (0.1355, 0.1425, 0.1364)}   # §1932's PUBLISHED deployed pooled top-1

print(f'ATTENTION SWEEP AT 5,419 | mlp 768, attn {ATTN_G} | fallback {FB} | DISCOVERY ONLY',
      flush=True)
res, pt, chg, ncov, COST, PTK = {}, {}, {}, {}, {}, {}
for cov, fit, nc in COVS:
    print(f'\n########## COVERAGE {nc} ##########', flush=True)
    P = B.Program(fit, expect_ncov=nc)
    liveR = B.score_roles(P, None)
    armR = {lab: B.score_roles(P, SPEC[lab][0], table_rank=SPEC[lab][1]) for lab in ARMS}
    res[cov], pt[cov], chg[cov] = {}, {}, {}
    for role in B.ROLES:
        tgt, icov = B.axes(P, role)
        res[cov][role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR[REF][role][1]) for a in ARMS}
        PTK[role] = {a: B.paired_t(armR[a][role][1], armR[KNEE_LAB][role][1]) for a in ARMS}
        chg[cov][role] = {a: int(((armR[a][role][0] != armR[FULL_LAB][role][0]) & icov).sum())
                          for a in ARMS}
        # two-sided: fallback-only differences must be EXACTLY inert at covered inputs; table-rank
        # differences must NOT be. See the pred_d note in the header.
        # polarity checked for THIS lineage: these two differ in table RANK, so they MUST move
        # covered-input predictions (LESSON 81).
        chg[cov][role]['_rank_only'] = int(((armR[KNEE_LAB][role][0]
                                             != armR[FULL_LAB][role][0]) & icov).sum())
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


ATTN = [a for _m, a in GRID]
ORDER = [f'blend_768_{a}' for a in ATTN]


def mono(r):
    return all(ce(C, r, ORDER[i]) < ce(C, r, ORDER[i - 1]) for i in range(1, len(ORDER)))


pa = sum(1 for r in B.ROLES if mono(r)) >= 2
monon = sum(1 for r in B.ROLES if mono(r))


def eff_rank(r):
    """cheapest attention rank whose step DOWN to the next one costs >= 0.010 nats per 100M."""
    for i in range(1, len(ORDER)):
        rate = ((ce(C, r, ORDER[i - 1]) - ce(C, r, ORDER[i]))
                / ((COST[ORDER[i]] - COST[ORDER[i - 1]]) / 100.0))
        if rate >= 0.010:
            return ATTN[i]
    return ATTN[-1]


up = sum(1 for r in B.ROLES if eff_rank(r) > 256)
pb = up >= 2
sig = sum(1 for r in B.ROLES
          if ce(C, r, f'blend_768_{eff_rank(r)}') < ce(C, r, KNEE_LAB)
          and PTK[r][f'blend_768_{eff_rank(r)}']['t'] <= -2.0)
pc = sig >= 2

inert_fb = all(chg[c][r]['_rank_only'] > 0 for c in chg for r in chg[c])
# POLARITY, third correction of the session (LESSON 81). REF here is §1789's DEPLOYED design --
# map64 at FULL table rank -- and chg measures differences against blend_full, which is also at full
# table rank. They differ ONLY in the fallback, so by §1765/§1936 they are EXACTLY INERT at covered
# inputs. The inherited clause asserted they differ, which is true only when the table RANK varies.
# the deployed anchor differs from blend_full in the FALLBACK only (both full rank) -> inert.
moves_rank = all(chg[c][r][REF] == 0 for c in chg for r in chg[c])
inert = inert_fb and moves_rank
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r][ARMS[0]][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(abs(ov(C, r, a) - S1932_T1[a][i])
            for a in S1932_T1 for i, r in enumerate(B.ROLES))
pd = ncov[C] == 5419 and inert and livesame <= 1e-9 and repro <= 0.0002

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:18s} [{COST[a]:8.3f}M]  top1 {ov(C, r, a):6.2%} '
              f'({(ov(C, r, a) - ov(C, r, REF)) * 100:+.2f}pp)  CE {ce(C, r, a):7.5f} '
              f'({ce(C, r, a) - ce(C, r, REF):+.5f}, t {pt[C][r][a]["t"]:+.2f})',
              flush=True)
    print(f'    monotone in attention rank: {mono(r)} | efficient rank by the 0.010 rule: '
          f'{eff_rank(r)} | vs {{768,256}} CE {ce(C, r, f"blend_768_{eff_rank(r)}") - ce(C, r, KNEE_LAB):+.5f} '
          f'(t {PTK[r][f"blend_768_{eff_rank(r)}"]["t"]:+.2f})', flush=True)
print(f'\n  CE is monotone in the attention share at 5,419 too (>=2 roles) -> {pa}  {monon}/3',
      flush=True)
print(f'  and the efficient attention rank is above 256 (>=2 roles) -> {pb}  {up}/3  '
      f'(ranks {[eff_rank(r) for r in B.ROLES]})', flush=True)
print(f'  and it beats {{768,256}} significantly (>=2 roles) -> {pc}  {sig}/3', flush=True)
print(f'  coverage {ncov}, rank-differing arm DOES move covered inputs {inert_fb}, identical-spec '
      f'fallback-only anchor is INERT there {moves_rank}, live identical {livesame:.1e}, §1932 top-1 reproduced within '
      f'{repro * 100:.3f}pp -> control {pd}', flush=True)

B.report({'pred_a_monotone_here_too': pa, 'pred_b_efficient_rank_above_256': pb,
          'pred_c_beats_knee_significantly': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [5419], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 2 -- §1948 at the deployed coverage.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'grid': [list(g) for g in GRID], 'fallback': FB,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
