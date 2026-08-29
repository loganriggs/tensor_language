# DOES THE CONVERGED BUILD HOLD AT THE DEPLOYED COVERAGE?
#
# §1950 closed the cost arc at 16,110 on (tables {mlp 768, attn 256}, fallback mix25m512), a fixed point
# of the alternation between the two axes. Everything from §1946 on is at 16,110. §1789's DEPLOYED
# coverage is 5,419, where the fallback touches ~24% of positions instead of ~10% (§1936) and the tables
# are a much smaller share of the build -- so both knees could sit elsewhere.
#
# §1945 is the cautionary precedent: a frontier that held at the second coverage but whose margin
# scaling was NOT uniform across roles (45/46/18%).
#
# ARMS. the converged fallback mix25m512 at FULL, {1024,256}, {768,384}, {768,256}, {640,160}, {512,128};
# plus §1789's deployed design (map64 at full rank) as the anchor everything is measured against.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 2 -- second-class confirmation of
# §1950 at the other coverage.
#
# Registered predictions, SIGNED per LESSON 72, reference labels bound as NAMES (LESSON 83), controls
# polarity-checked for this lineage (LESSON 81).
#   pred_a THE CONVERGED BUILD STILL BEATS THE DEPLOYED DESIGN: at 5,419, the {768,256} build with the
#          mix25m512 fallback beats §1789's deployed design on pooled top-1 AND has a paired CE t <= -2.0
#          against it, on at least 2 of 3 roles. If FALSE the converged build is a high-coverage result
#          and §1950 is scoped to 16,110.
#   pred_b BUT THE TABLE KNEE MOVES: at 5,419 the 0.010 nats per 100M threshold is crossed at a
#          DIFFERENT step than {768,256} -> {640,160}, on at least 2 of 3 roles. The tables are a smaller
#          fraction of the build here, so truncating them should buy less per M and the efficient
#          stopping point should sit at a HIGHER rank. If FALSE the knee is coverage-invariant, which
#          would be a stronger and more surprising result than the one I am predicting.
#   pred_c AND THE ATTENTION SHARE DOES NOT: going from attn 384 down to 256 still costs less than 0.010
#          nats per 100M at 5,419, on at least 2 of 3 roles -- §1948/§1950's finding, which is about the
#          relative worth of the two site classes and should not care about coverage. If FALSE both knees
#          move and nothing about §1946-§1950 transfers.
#   pred_d CONTROLS, polarity derived for THIS lineage (LESSON 81, third time): coverage exactly 5,419;
#          arms differing in table RANK DO move covered-input predictions, while the deployed anchor --
#          which differs from blend_full only in the FALLBACK, both at full table rank -- is EXACTLY
#          INERT at covered inputs; buckets
#          partition; live per-cell top-1 and CE identical across arms; and the deployed anchor
#          reproduces §1932's PUBLISHED pooled top-1 (13.55 / 14.25 / 13.64%) within 0.02pp.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/converged_at_deployed_coverage_results.json'
GRID = ((1024, 256), (768, 384), (768, 256), (640, 160), (512, 128))
FB = 'mix25m512'   # §1949's fallback
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

print(f'CONVERGED BUILD AT 5,419 | fallback {FB} | vs §1789 deployed | DISCOVERY ONLY',
      flush=True)
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


beats = sum(1 for r in B.ROLES
            if ov(C, r, KNEE_LAB) > ov(C, r, REF) and pt[C][r][KNEE_LAB]['t'] <= -2.0)
pa = beats >= 2
crossed = sum(1 for r in B.ROLES
              if not (step_rate(r, KNEE_LAB, 'blend_1024_256') < 0.010
                      <= step_rate(r, 'blend_640_160', KNEE_LAB)))
pb = crossed >= 2
attn = sum(1 for r in B.ROLES if step_rate(r, KNEE_LAB, 'blend_768_384') < 0.010)
pc = attn >= 2

inert_fb = all(chg[c][r]['_rank_only'] > 0 for c in chg for r in chg[c])
# POLARITY, third correction of the session (LESSON 81). REF here is §1789's DEPLOYED design --
# map64 at FULL table rank -- and chg measures differences against blend_full, which is also at full
# table rank. They differ ONLY in the fallback, so by §1765/§1936 they are EXACTLY INERT at covered
# inputs. The inherited clause asserted they differ, which is true only when the table RANK varies.
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
    print(f'    1024->768 {step_rate(r, "blend_768_256", "blend_1024_256"):.4f} | '
          f'768->640 {step_rate(r, "blend_640_160", "blend_768_256"):.4f} | '
          f'attn 384->256 {step_rate(r, "blend_768_256", "blend_768_384"):.4f}  nats/100M', flush=True)
print(f'\n  the converged build still beats §1789 deployed, with the t bar (>=2 roles) -> {pa}  '
      f'{beats}/3', flush=True)
print(f'  and the table knee MOVES at 5,419 (>=2 roles) -> {pb}  {crossed}/3', flush=True)
print(f'  and the attention share does NOT (>=2 roles) -> {pc}  {attn}/3', flush=True)
print(f'  coverage {ncov}, rank-differing arm DOES move covered inputs {inert_fb}, identical-spec '
      f'fallback-only anchor is INERT there {moves_rank}, live identical {livesame:.1e}, §1932 top-1 reproduced within '
      f'{repro * 100:.3f}pp -> control {pd}', flush=True)

B.report({'pred_a_beats_deployed': pa, 'pred_b_table_knee_moves': pb,
          'pred_c_attention_share_holds': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [5419], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 2 -- §1950 at the deployed coverage.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'grid': [list(g) for g in GRID], 'fallback': FB,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
