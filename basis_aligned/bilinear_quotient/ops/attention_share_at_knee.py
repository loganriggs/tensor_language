# IS §1928'S RAY STILL RIGHT AT THE KNEE?
#
# §1947 put the efficient build at {mlp 768, attn 256} -- attention at 25% of the MLP rank, the TOP of the
# 12.5-25% band §1928-§1935 established. But that band was measured where the tables were ~97% of the
# build and the fallback was a rank-64 map. At §1947's knee the tables are 78% of a 339.558M build and
# the fallback is §1944's blend. Every allocation since §1941 has moved along §1928's ray with the share
# pinned; the share itself has never been re-opened here.
#
# ARMS. mlp 768 with attention at 64 / 128 / 192 / 256 / 384 / 576 (8.3% to 75% of the MLP rank), blend
# fallback fixed at mix25m256; plus §1946's blend_768_256 and §1931's map512_mlpheavy as anchors. Costs
# differ across the sweep, so the comparison is reported per-M as well as absolutely.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 16,110. Rung 3 -- §1947's open question.
#
# Registered predictions, SIGNED per LESSON 72, paired t up front (LESSON 78), controls TWO-SIDED
# (LESSON 81).
#   pred_a THE OPTIMUM IS INTERIOR IN THE SHARE: at fixed mlp 768, some attention rank strictly between
#          64 and 576 has pooled CE BELOW both endpoints, on at least 2 of 3 roles. If FALSE the CE is
#          monotone in the attention share and the right answer is a corner, which would contradict
#          §1928's finding that attention wants a SMALL but nonzero slice.
#   pred_b AND IT IS ABOVE §1928'S BAND: the CE-minimising attention rank at mlp 768 is STRICTLY GREATER
#          than 192 (i.e. above 25% of the MLP rank), on at least 2 of 3 roles. §1928's band was found
#          with a rank-64 map on near-full tables; §1942 showed the blend carries information the map
#          does not, and if attention sites are where that lands they should now want MORE. If FALSE the
#          12.5-25% rule survives the change of fallback and of operating point, which is the more
#          conservative outcome and worth stating either way.
#   pred_c AND THE BEST SHARE BEATS §1946 PER PARAMETER: some arm has a better CE-per-M than
#          blend_768_256 -- specifically, it beats blend_768_256 on pooled CE by enough that the gain
#          per additional 100M exceeds §1947's 0.010 nats/100M threshold, on at least 2 of 3 roles. If
#          FALSE the share is already efficient at 256 and §1947's build stands unchanged.
#   pred_d CONTROLS, TWO-SIDED: coverage exactly 16,110; a rank-differing arm DOES move covered-input
#          predictions while an identical-spec arm does NOT; buckets partition; live per-cell top-1 and
#          CE identical across arms; and blend_768_256 and map512_mlpheavy reproduce §1946's PUBLISHED
#          pooled CE (5.88609/5.83249/5.86357 and 5.89445/5.84120/5.86873) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/attention_share_at_knee_results.json'
GRID = ((768, 64), (768, 128), (768, 192), (768, 256), (768, 384), (768, 576))
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = ((('mix25m256', MLPHEAVY, 'blend_mlpheavy_anchor'),
         ('map512', MLPHEAVY, 'map512_mlpheavy'))
        + tuple(('mix25m256', {'mlp': m, 'attn': a}, f'blend_{m}_{a}') for m, a in GRID))
ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'
S1946_C16 = {'blend_768_256': (5.88609, 5.83249, 5.86357)}
S1946_ANCHOR = (5.89445, 5.84120, 5.86873)     # map512_mlpheavy, §1931's best-known

print(f'ATTENTION SHARE AT THE KNEE | mlp 768, attn {[g[1] for g in GRID]} | 16,110 | '
      f'DISCOVERY ONLY', flush=True)
res, pt, chg, ncov, COST = {}, {}, {}, {}, {}
for cov, fit, nc in COVS:
    print(f'\n########## COVERAGE {nc} ##########', flush=True)
    P = B.Program(fit, expect_ncov=nc)
    liveR = B.score_roles(P, None)
    armR = {}
    for lab in ARMS:
        armR[lab] = B.score_roles(P, SPEC[lab][0], table_rank=SPEC[lab][1])
        torch.cuda.empty_cache()
    res[cov], pt[cov], chg[cov] = {}, {}, {}
    for role in B.ROLES:
        tgt, icov = B.axes(P, role)
        res[cov][role] = {a: B.cells(P, tgt, icov, liveR[role], armR[a][role]) for a in ARMS}
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR['map512_mlpheavy'][role][1])
                         for a in ARMS}
        chg[cov][role] = {a: int(((armR[a][role][0] != armR['blend_mlpheavy_anchor'][role][0]) & icov).sum())
                          for a in ARMS}
        # two-sided: fallback-only differences must be EXACTLY inert at covered inputs; table-rank
        # differences must NOT be. See the pred_d note in the header.
        chg[cov][role]['_rank_only'] = int(((armR['blend_768_64'][role][0]
                                             != armR['blend_768_576'][role][0]) & icov).sum())
    ncov[cov] = P.ncov
    COST.update({lab: P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS})
    del P, liveR, armR
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']


LAB = [f'blend_{m}_{a}' for m, a in GRID]
ATTN = [a for _m, a in GRID]


def argmin_share(r):
    return min(range(len(LAB)), key=lambda i: ce(C, r, LAB[i]))


interior = sum(1 for r in B.ROLES
               if min(ce(C, r, LAB[i]) for i in range(1, len(LAB) - 1))
               < min(ce(C, r, LAB[0]), ce(C, r, LAB[-1])))
pa = interior >= 2
above = sum(1 for r in B.ROLES if ATTN[argmin_share(r)] > 192)
pb = above >= 2
base = 'blend_768_256'
gains = {}
for r in B.ROLES:
    best = None
    for lab in LAB:
        if lab == base or COST[lab] <= COST[base]:
            continue
        dn = ce(C, r, base) - ce(C, r, lab)
        rate = dn / ((COST[lab] - COST[base]) / 100.0)
        if best is None or rate > best[1]:
            best = (lab, rate)
    gains[r] = best if best else (None, float('-inf'))
worth = sum(1 for r in B.ROLES if gains[r][1] > 0.010)
pc = worth >= 2

inert_fb = all(chg[c][r]['_rank_only'] > 0 for c in chg for r in chg[c])
moves_rank = all(chg[c][r]['blend_768_256'] == 0 for c in chg for r in chg[c])
inert = inert_fb and moves_rank
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r][ARMS[0]][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(max(max(abs(ce(C, r, a) - S1946_C16[a][i]) for a in S1946_C16),
                abs(ce(C, r, 'map512_mlpheavy') - S1946_ANCHOR[i]))
            for i, r in enumerate(B.ROLES))
pd = ncov[C] == 16110 and inert and livesame <= 1e-9 and repro <= 0.0002

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:18s} [{COST[a]:8.3f}M]  top1 {ov(C, r, a):6.2%} '
              f'({(ov(C, r, a) - ov(C, r, "map512_mlpheavy")) * 100:+.2f}pp)  CE {ce(C, r, a):7.5f} '
              f'({ce(C, r, a) - ce(C, r, "map512_mlpheavy"):+.5f}, t {pt[C][r][a]["t"]:+.2f})',
              flush=True)
    print(f'    CE argmin attention rank: {ATTN[argmin_share(r)]} '
          f'({ATTN[argmin_share(r)] / 768:.0%} of mlp) | best rate above the anchor: '
          f'{gains[r][0]} at {gains[r][1]:.4f} nats/100M', flush=True)
print(f'\n  the CE optimum is INTERIOR in the attention share (>=2 roles) -> {pa}  {interior}/3',
      flush=True)
print(f'  and the argmin attention rank is above 192 = 25% of mlp (>=2 roles) -> {pb}  {above}/3  '
      f'(argmins {[ATTN[argmin_share(r)] for r in B.ROLES]})', flush=True)
print(f'  and a larger share beats §1946 at better than 0.010 nats/100M (>=2 roles) -> {pc}  '
      f'{worth}/3', flush=True)
print(f'  coverage {ncov}, rank-differing arm DOES move covered inputs {inert_fb}, identical-spec '
      f'arm does NOT {moves_rank}, live identical {livesame:.1e}, §1946 CE reproduced within '
      f'{repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_share_optimum_interior': pa, 'pred_b_argmin_above_25pc': pb,
          'pred_c_larger_share_worth_it': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1947 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'attn_ranks': ATTN, 'grid': [list(g) for g in GRID],
          'ce_argmin_attn': {r: ATTN[argmin_share(r)] for r in B.ROLES},
          'best_rate_above_anchor': {r: [gains[r][0], gains[r][1]] for r in B.ROLES},
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
