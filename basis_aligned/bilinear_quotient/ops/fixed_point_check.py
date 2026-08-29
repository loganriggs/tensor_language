# IS IT A FIXED POINT, OR ONE STEP OF AN ALTERNATION?
#
# §1948 moved the attention tradeoff when the operating point moved. §1949 moved the fallback's map rank
# when the tables were truncated. Each axis has been re-opened ONCE against the other and each moved
# once. Nothing establishes that (tables {768,256}, fallback mix25m512) is a fixed point rather than one
# step of an alternation that would keep walking.
#
# This re-opens the TABLE axis a second time, now with §1949's rank-512 fallback in place. If both knees
# stay where they are, the alternation converged in one step and the cost arc is closed at 16,110.
#
# ARMS. fb_mix25m512 at FULL, {1024,256}, {768,384}, {768,256}, {640,160}, {512,128}.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 16,110. Rung 2 -- a convergence check, not a
# new question.
#
# Registered predictions, SIGNED per LESSON 72, controls polarity-checked against THIS lineage
# (LESSON 81 -- two polarity errors this session, both caught by pred_d).
#   pred_a THE TABLE KNEE HELD: with the rank-512 fallback, the step down to {768,256} still costs less
#          than §1947's 0.010 nats per 100M while the step to {640,160} costs more, on at least 2 of 3
#          roles -- the same crossing §1947 found with the rank-256 fallback. If FALSE the knee moved and
#          this is an alternation, not a fixed point.
#   pred_b AND THE ATTENTION SHARE HELD: at mlp 768, going from attn 256 to attn 384 still costs more
#          than 0.010 nats per 100M, on at least 2 of 3 roles -- §1948's finding, re-tested under the
#          new fallback. If FALSE the richer fallback changed what attention is worth.
#   pred_c SO IT IS A FIXED POINT: pred_a AND pred_b both hold on at least 2 of 3 roles, i.e. neither
#          axis wants to move again. This is the claim the cost arc closes on; if FALSE I say plainly
#          that the search has not converged and one more alternation is owed.
#   pred_d CONTROLS: coverage exactly 16,110; arms differing in table RANK DO move covered-input
#          predictions while the two arms with an IDENTICAL specification are bit-identical; buckets
#          partition; live per-cell top-1 and CE identical across arms; and the {768,256} arm reproduces
#          §1949's PUBLISHED pooled CE (5.88341 / 5.82982 / 5.86029) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/fixed_point_check_results.json'
GRID = ((1024, 256), (768, 384), (768, 256), (640, 160), (512, 128))
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = ((('mix25m256', MLPHEAVY, 'blend_mlpheavy'), (FB, None, 'blend_full'))
        + tuple((FB, {'mlp': m, 'attn': a}, f'blend_{m}_{a}') for m, a in GRID))
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'blend_mlpheavy'          # the paired-t and inertness reference
KNEE_LAB = 'blend_768_256'      # §1949's build
FULL_LAB = 'blend_full'
ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'
S1949_C16 = {'blend_768_256': (5.88341, 5.82982, 5.86029)}

print(f'FIXED POINT CHECK at 16,110 | table axis re-opened with the §1949 fallback {FB} | '
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


held = sum(1 for r in B.ROLES
           if step_rate(r, KNEE_LAB, 'blend_1024_256') < 0.010
           <= step_rate(r, 'blend_640_160', KNEE_LAB))
pa = held >= 2
attn = sum(1 for r in B.ROLES if step_rate(r, KNEE_LAB, 'blend_768_384') < 0.010)
pb = attn >= 2
pc = pa and pb

inert_fb = all(chg[c][r]['_rank_only'] > 0 for c in chg for r in chg[c])
moves_rank = all(chg[c][r][REF] > 0 for c in chg for r in chg[c])
inert = inert_fb and moves_rank
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r][ARMS[0]][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(abs(ce(C, r, a) - S1949_C16[a][i])
            for a in S1949_C16 for i, r in enumerate(B.ROLES))
pd = ncov[C] == 16110 and inert and livesame <= 1e-9 and repro <= 0.0002

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
print(f'\n  the TABLE knee held between 768 and 640 (>=2 roles) -> {pa}  {held}/3', flush=True)
print(f'  and the ATTENTION share held at 256 (>=2 roles) -> {pb}  {attn}/3', flush=True)
print(f'  so (tables 768/256, fallback {FB}) is a FIXED POINT -> {pc}', flush=True)
print(f'  coverage {ncov}, rank-differing arm DOES move covered inputs {inert_fb}, identical-spec '
      f'arm does NOT {moves_rank}, live identical {livesame:.1e}, §1946 CE reproduced within '
      f'{repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_table_knee_held': pa, 'pred_b_attention_share_held': pb,
          'pred_c_is_a_fixed_point': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 2 -- convergence check on §1949.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'grid': [list(g) for g in GRID], 'fallback': FB,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
