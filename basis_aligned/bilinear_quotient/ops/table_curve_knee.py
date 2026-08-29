# WHERE DOES THE TABLE CURVE TURN OVER? -- the cheapest remaining lever, measured.
#
# §1946 repriced the arc. Truncating the 36 tables from FULL rank to {mlp 768, attn 256} saves 349.9M --
# more than half the build at 16,110 -- for +0.0104/+0.0114/+0.0070 nats. Going on to {mlp 512, attn 128}
# saves a further 119.3M but costs 0.027/0.028/0.022 nats, a much worse rate. So the table curve turns
# over somewhere between, and nobody has looked. At 16,110 the tables are 94-97% of every build in this
# lineage, so this is where the remaining cost is.
#
# The fallback is FIXED at §1944's blend (mix25m256) in every arm, so only the allocation varies. Every
# allocation keeps §1928's shape -- attention at 25% of the MLP rank -- so the sweep moves along that
# ray rather than re-opening the allocation question §1946 just confirmed.
#
# ARMS. blend fallback at FULL rank and at {mlp, attn} = (1024,256), (768,256), (640,160), (512,128),
# (384,96), (256,64); plus map512_mlpheavy, which IS §1931's best-known build, as the anchor to beat.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 16,110. Rung 3 -- §1946's open question.
#
# Registered predictions, SIGNED per LESSON 72, paired t up front (LESSON 78), controls TWO-SIDED
# (LESSON 81 -- §1946 registered an inertness clause that was false by construction in this lineage).
#   pred_a DIMINISHING RETURNS ARE MONOTONE: the marginal cost in nats per 100M of each successive
#          truncation step STRICTLY INCREASES as the rank falls, across the whole grid, on at least 2 of
#          3 roles. If FALSE the curve is not convex and there is no single knee to find.
#   pred_b THE KNEE IS AT OR BELOW mlp 640: the step down to {640,160} still costs less than 0.010 nats
#          per 100M saved, while the step to {512,128} costs more, on at least 2 of 3 roles. That would
#          put the efficient stopping point below §1946's build rather than at it.
#   pred_c AND COST CAN FALL BELOW 300M AND STILL BEAT §1931: some allocation costing under 300M beats
#          map512_mlpheavy (360.723M) on pooled top-1 on at least 2 of 3 roles. Its CE will be worse --
#          this asks only how far cost can fall while the top-1 win survives. If FALSE the top-1
#          advantage is spent by 300M and §1946's 339.558M is close to the floor.
#   pred_d CONTROLS, TWO-SIDED per LESSON 81: coverage exactly 16,110; arms differing ONLY in the
#          fallback are EXACTLY inert at covered inputs while arms differing in table rank are NOT;
#          buckets partition; live per-cell top-1 and CE identical across arms; and the {768,256} and
#          {512,128} arms reproduce §1946's PUBLISHED pooled CE (5.88609/5.83249/5.86357 and
#          5.91309/5.86085/5.88599) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/table_curve_knee_results.json'
GRID = ((1024, 256), (768, 256), (640, 160), (512, 128), (384, 96), (256, 64))
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = ((('mix25m256', None, 'blend_full'), ('map512', MLPHEAVY, 'map512_mlpheavy'))
        + tuple(('mix25m256', {'mlp': m, 'attn': a}, f'blend_{m}_{a}') for m, a in GRID))
ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'
S1946_C16 = {'blend_768_256': (5.88609, 5.83249, 5.86357),
             'blend_512_128': (5.91309, 5.86085, 5.88599)}

print(f'TABLE CURVE KNEE at 16,110 | allocations {GRID} | blend fallback fixed | '
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
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR['map512_mlpheavy'][role][1])
                         for a in ARMS}
        chg[cov][role] = {a: int(((armR[a][role][0] != armR['blend_full'][role][0]) & icov).sum())
                          for a in ARMS}
        # two-sided: fallback-only differences must be EXACTLY inert at covered inputs; table-rank
        # differences must NOT be. See the pred_d note in the header.
        chg[cov][role]['_rank_only'] = int(((armR['blend_768_256'][role][0]
                                             != armR['blend_full'][role][0]) & icov).sum())
    ncov[cov] = P.ncov
    COST.update({lab: P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS})
    del P, liveR, armR
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']


LAB = [f'blend_{m}_{a}' for m, a in GRID]
COSTL = {lab: COST[lab] for lab in LAB}


def rate(r, i):
    """nats per 100M for the step from grid point i-1 (or FULL) down to i."""
    prev = 'blend_full' if i == 0 else LAB[i - 1]
    dn = ce(C, r, LAB[i]) - ce(C, r, prev)
    dm = (COST['blend_full'] if i == 0 else COSTL[LAB[i - 1]]) - COSTL[LAB[i]]
    return dn / (dm / 100.0)


rates = {r: [rate(r, i) for i in range(len(LAB))] for r in B.ROLES}
mono = sum(1 for r in B.ROLES if all(rates[r][i] > rates[r][i - 1] for i in range(1, len(LAB))))
pa = mono >= 2
i640, i512 = LAB.index('blend_640_160'), LAB.index('blend_512_128')
knee = sum(1 for r in B.ROLES if rates[r][i640] < 0.010 <= rates[r][i512])
pb = knee >= 2
CHEAP = [lab for lab in LAB if COSTL[lab] < 300.0]
under300 = sum(1 for r in B.ROLES
               if any(ov(C, r, lab) > ov(C, r, 'map512_mlpheavy') for lab in CHEAP))
pc = under300 >= 2

inert_fb = all(chg[c][r]['_rank_only'] > 0 for c in chg for r in chg[c])
moves_rank = all(chg[c][r]['blend_full'] == 0 for c in chg for r in chg[c])
inert = inert_fb and moves_rank
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r][ARMS[0]][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(abs(ce(C, r, a) - S1946_C16[a][i])
            for a in S1946_C16 for i, r in enumerate(B.ROLES))
pd = ncov[C] == 16110 and inert and livesame <= 1e-9 and repro <= 0.0002

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:18s} [{COST[a]:8.3f}M]  top1 {ov(C, r, a):6.2%} '
              f'({(ov(C, r, a) - ov(C, r, "map512_mlpheavy")) * 100:+.2f}pp)  CE {ce(C, r, a):7.5f} '
              f'({ce(C, r, a) - ce(C, r, "map512_mlpheavy"):+.5f}, t {pt[C][r][a]["t"]:+.2f})',
              flush=True)
    print(f'    marginal nats per 100M by step: ' + '  '.join(
        f'{LAB[i][6:]} {rates[r][i]:.4f}' for i in range(len(LAB))), flush=True)
print(f'\n  the marginal rate rises monotonically as rank falls (>=2 roles) -> {pa}  {mono}/3',
      flush=True)
print(f'  and the knee is at or below mlp 640 (>=2 roles) -> {pb}  {knee}/3', flush=True)
print(f'  and some build under 300M still beats §1931 on top-1 (>=2 roles) -> {pc}  {under300}/3',
      flush=True)
print(f'  coverage {ncov}, rank-differing arm DOES move covered inputs {inert_fb}, identical-spec '
      f'arm does NOT {moves_rank}, live identical {livesame:.1e}, §1946 CE reproduced within '
      f'{repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_marginal_rate_monotone': pa, 'pred_b_knee_at_or_below_640': pb,
          'pred_c_under_300M_beats_s1931': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1946 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'marginal_nats_per_100M': rates, 'grid': [list(g) for g in GRID],
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
