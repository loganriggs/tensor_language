# THE BLEND MEETS THE ALLOCATION LEVER -- two independent savings, never combined.
#
# §1937-§1945 improved the FALLBACK and held the 36 tables at FULL rank throughout. At 16,110 the tables
# are 667.9M of a 689.5M build -- 97% -- so every nat won since §1937 was won on 3% of the cost.
# §1928-§1935 worked the other axis: per-site rank allocation is worth ~0.015-0.019 nats for free, with
# the scale-free rule that the 18 attention sites want 12.5-25% of the per-site table budget. The two
# levers have never been applied together, and §1942's finding -- that the fallback's two ingredients are
# orthogonal in row space -- says nothing about whether the fallback advantage survives table truncation.
#
# The allocation arms are MATCHED ON COST by construction: {mlp 768, attn 256} and uniform 512 both cost
# 318.3M of table at 16,110, so §1928's rule is tested against its own control rather than against a
# cheaper build.
#
# ARMS. mix25m256 and map512, each at FULL rank, at {mlp 768, attn 256} (attention at 25%), and at
# uniform 512; plus mix25m256 at {mlp 512, attn 128}. 16,110 coverage.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1945's open question.
#
# Registered predictions, SIGNED per LESSON 72, paired t up front (LESSON 78).
#   pred_a THE LEVERS COMPOSE: under the {mlp 768, attn 256} allocation, mix25m256 still beats map512 at
#          the SAME allocation on pooled top-1 AND with a paired CE t <= -2.0, on at least 2 of 3 roles.
#          If FALSE the fallback advantage does not survive table truncation and §1937-§1945 only apply
#          to full-rank builds.
#   pred_b MLP-HEAVY BEATS UNIFORM AT MATCHED COST: with the fallback fixed at mix25m256, the
#          {mlp 768, attn 256} allocation has pooled CE BELOW uniform rank 512, on at least 2 of 3 roles.
#          Both cost 318.3M of table. This re-tests §1928's rule under a fallback it was never measured
#          with. If FALSE the rule was an artefact of the rank-64-map builds it was found on.
#   pred_c AND THE TRUNCATION IS CHEAP RELATIVE TO WHAT IT SAVES: moving mix25m256 from FULL rank to
#          {mlp 768, attn 256} -- a saving of ~371M, more than half the build -- costs LESS than 0.10
#          nats of pooled CE, on at least 2 of 3 roles. If FALSE the table rank is not a cheap lever at
#          this coverage and the cost arc should stay at full rank.
#   pred_d CONTROLS: coverage exactly 16,110; every arm inert at covered inputs; buckets partition; live
#          per-cell top-1 and CE identical across arms; the two matched-cost allocations really do cost
#          the same to within 0.1M; and the FULL-rank mix25m256 and map512 arms reproduce §1945's
#          PUBLISHED pooled CE (5.87567/5.82113/5.85661 and 5.88338/5.82928/5.86044) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/blend_meets_allocation_results.json'
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's best-known allocation: attention at 25%
LEAN = {'mlp': 512, 'attn': 128}
# (arm, table_rank spec, label)
PLAN = (('mix25m256', None, 'blend_full'), ('map512', None, 'map512_full'),
        ('mix25m256', MLPHEAVY, 'blend_mlpheavy'), ('map512', MLPHEAVY, 'map512_mlpheavy'),
        ('mix25m256', 512, 'blend_uniform512'), ('mix25m256', LEAN, 'blend_lean'))
ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'
S1945_C16 = {'blend_full': (5.87567, 5.82113, 5.85661), 'map512_full': (5.88338, 5.82928, 5.86044)}

print(f'BLEND x ALLOCATION at 16,110 | {ARMS} | matched-cost allocation control | '
      f'DISCOVERY ONLY', flush=True)
res, pt, chg, ncov = {}, {}, {}, {}
for cov, fit, nc in COVS:
    print(f'\n########## COVERAGE {nc} ##########', flush=True)
    P = B.Program(fit, expect_ncov=nc)
    liveR = B.score_roles(P, None)
    armR = {lab: B.score_roles(P, SPEC[lab][0], table_rank=SPEC[lab][1]) for lab in ARMS}
    res[cov], pt[cov], chg[cov] = {}, {}, {}
    for role in B.ROLES:
        tgt, icov = B.axes(P, role)
        res[cov][role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR['map512_mlpheavy'][role][1])
                         for a in ARMS}
        chg[cov][role] = {a: int(((armR[a][role][0] != armR['blend_full'][role][0]) & icov).sum())
                          for a in ARMS}
    ncov[cov] = P.ncov
    COST = {lab: P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS}
    del P, liveR, armR
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']


compose = sum(1 for r in B.ROLES
              if ov(C, r, 'blend_mlpheavy') > ov(C, r, 'map512_mlpheavy')
              and pt[C][r]['blend_mlpheavy']['t'] <= -2.0)
pa = compose >= 2
mlpwin = sum(1 for r in B.ROLES if ce(C, r, 'blend_mlpheavy') < ce(C, r, 'blend_uniform512'))
pb = mlpwin >= 2
trunc = {r: ce(C, r, 'blend_mlpheavy') - ce(C, r, 'blend_full') for r in B.ROLES}
cheap = sum(1 for r in B.ROLES if trunc[r] < 0.10)
pc = cheap >= 2
matched = abs(COST['blend_mlpheavy'] - COST['blend_uniform512'])

inert = all(v == 0 for c in chg for r in chg[c] for v in chg[c][r].values())
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r]['map64'][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(abs(ce(C, r, a) - S1945_C16[a][i])
            for a in S1945_C16 for i, r in enumerate(B.ROLES))
pd = (ncov[C] == 16110 and inert and livesame <= 1e-9 and repro <= 0.0002 and matched <= 0.1)

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:18s} [{COST[a]:8.3f}M]  top1 {ov(C, r, a):6.2%}  CE {ce(C, r, a):7.5f}  '
              f'(vs map512_mlpheavy {ce(C, r, a) - ce(C, r, "map512_mlpheavy"):+.5f}, '
              f't {pt[C][r][a]["t"]:+.2f})', flush=True)
    print(f'    FULL -> mlp-heavy costs {trunc[r]:+.5f} nats and saves '
          f'{COST["blend_full"] - COST["blend_mlpheavy"]:.1f}M', flush=True)
print(f'\n  the blend still beats the map under mlp-heavy truncation (>=2 roles) -> {pa}  '
      f'{compose}/3', flush=True)
print(f'  and mlp-heavy beats uniform-512 at MATCHED cost (>=2 roles) -> {pb}  {mlpwin}/3  '
      f'(cost gap {matched:.4f}M)', flush=True)
print(f'  and the truncation costs < 0.10 nats for a {COST["blend_full"] - COST["blend_mlpheavy"]:.0f}M '
      f'saving (>=2 roles) -> {pc}  {cheap}/3  (costs {[f"{trunc[r]:+.4f}" for r in B.ROLES]})',
      flush=True)
print(f'  coverage {ncov}, arms inert {inert}, live identical {livesame:.1e}, matched-cost gap '
      f'{matched:.4f}M, §1945 CE reproduced within {repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_levers_compose': pa, 'pred_b_mlpheavy_beats_uniform': pb,
          'pred_c_truncation_is_cheap': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1945 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'alpha': ALPHA},
          'results': res,
          'paired_vs_map512': pt,
          'truncation_cost_nats': trunc, 'matched_cost_gap_M': matched,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
