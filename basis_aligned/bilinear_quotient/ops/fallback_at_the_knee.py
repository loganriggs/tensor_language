# RE-OPEN THE FALLBACK AT THE KNEE -- its two parameters were chosen at a different operating point.
#
# §1946-§1948 converged the TABLE axis on {mlp 768, attn 256} at 16,110, with the fallback held fixed at
# §1944's mix25m256. But §1944 chose alpha = 0.25 and map rank 256 on FULL-RANK tables at 5,419, where
# the fallback touched ~24% of positions and the tables were ~97% of the build. At the knee the tables
# are 78% of a 339.558M build and the fallback touches ~10% of positions (§1936). §1948 is the precedent
# that matters: the MLP/attention tradeoff MOVED when the operating point moved, and §1928's rule needed
# correcting as a result. The fallback's own two parameters have never been re-opened here.
#
# ARMS. mix{10,25,40}m{128,256,512} at the knee allocation, plus map512 and the §1948 anchor
# blend_768_256 (= mix25m256 at the knee). All eleven share ONE table spec, so this is cheap.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 16,110. Rung 3 -- §1948's open question.
#
# Registered predictions, SIGNED per LESSON 72, paired t up front (LESSON 78), controls TWO-SIDED
# (LESSON 81).
#   pred_a THE FALLBACK OPTIMUM MOVED: the (alpha, map rank) pair minimising pooled CE at the knee is NOT
#          (25, 256), on at least 2 of 3 roles. If FALSE, §1944's choice transfers unchanged across a
#          coverage change AND a 51% table truncation, which would be a stronger transfer result than
#          anything in the fallback arc so far and worth banking as such.
#   pred_b AND IT WANTS AT LEAST AS MUCH MAP: the CE-optimal map rank at the knee is >= 256, on at least
#          2 of 3 roles. With the tables truncated the fallback carries relatively more of the build, so
#          it should not want LESS capacity. If FALSE the truncation makes the map less useful, not more,
#          and the two axes interact in the opposite direction to the one I expect.
#   pred_c AND THE MOVE IS WORTH BUYING: the best pair beats blend_768_256 at better than §1947's 0.010
#          nats per 100M threshold, on at least 2 of 3 roles. If FALSE the fallback optimum may have
#          moved but the converged build stands, which is the more likely outcome and the one that would
#          close the cost arc.
#   pred_d CONTROLS, TWO-SIDED with the polarity checked against THIS lineage (LESSON 81): every arm
#          here shares ONE table spec and differs only in the FALLBACK, so by §1936 they must be EXACTLY
#          inert at COVERED inputs and must DIFFER at uncovered ones; the two arms with an IDENTICAL
#          specification must be bit-identical everywhere;
#          buckets partition; live per-cell top-1 and CE identical across arms; and blend_768_256
#          reproduces §1948's PUBLISHED pooled CE (5.88609 / 5.83249 / 5.86357) within 0.0002 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/fallback_at_the_knee_results.json'
KNEE = {'mlp': 768, 'attn': 256}          # §1947's efficient allocation
FB = tuple(f'mix{a}m{r}' for r in (128, 256, 512) for a in (10, 25, 40))
MLPHEAVY = KNEE
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = ((('mix25m256', KNEE, 'blend_768_256'), ('map512', KNEE, 'map512_mlpheavy'))
        + tuple((fb, KNEE, f'fb_{fb}') for fb in FB))
ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'
S1948_C16 = {'blend_768_256': (5.88609, 5.83249, 5.86357)}
S1948_ANCHOR = (5.89445, 5.84120, 5.86873)     # map512 at the knee

print(f'FALLBACK RE-OPENED AT THE KNEE | {FB} at {KNEE} | 16,110 | DISCOVERY ONLY',
      flush=True)
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
        chg[cov][role] = {a: int(((armR[a][role][0] != armR['blend_768_256'][role][0]) & icov).sum())
                          for a in ARMS}
        # two-sided: fallback-only differences must be EXACTLY inert at covered inputs; table-rank
        # differences must NOT be. See the pred_d note in the header.
        # POLARITY, corrected. Every arm here shares ONE table spec and differs only in the FALLBACK,
        # so by §1936 they must be EXACTLY inert at covered inputs and must differ at uncovered ones.
        # The first version of this control asserted the opposite -- inherited from §1947, where the
        # arms differed in table RANK. Second polarity error of the session; see LESSON 81.
        d = (armR['fb_mix40m512'][role][0] != armR['fb_mix10m128'][role][0])
        chg[cov][role]['_fb_at_covered'] = int((d & icov).sum())
        chg[cov][role]['_fb_at_uncovered'] = int((d & ~icov).sum())
        chg[cov][role]['_identical_spec'] = int((armR['fb_mix25m256'][role][0]
                                                 != armR['blend_768_256'][role][0]).sum())
    ncov[cov] = P.ncov
    COST.update({lab: P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS})
    del P, liveR, armR
    torch.cuda.empty_cache()


def ov(c, r, a):
    return res[c][r][a]['pooled']['overall']['top1_acc_prog']


def ce(c, r, a):
    return res[c][r][a]['pooled']['overall']['ce_prog']


LAB = [f'fb_{fb}' for fb in FB]
AR = {f'fb_{fb}': (int(fb[3:].split('m')[0]), int(fb.rsplit('m', 1)[1])) for fb in FB}
base = 'blend_768_256'


def argmin_fb(r):
    return min(LAB, key=lambda lab: ce(C, r, lab))


moved = sum(1 for r in B.ROLES if AR[argmin_fb(r)] != (25, 256))
pa = moved >= 2
morerank = sum(1 for r in B.ROLES if AR[argmin_fb(r)][1] >= 256)
pb = morerank >= 2
gains = {}
for r in B.ROLES:
    best = None
    for lab in LAB:
        if COST[lab] <= COST[base]:
            continue
        rate = (ce(C, r, base) - ce(C, r, lab)) / ((COST[lab] - COST[base]) / 100.0)
        if best is None or rate > best[1]:
            best = (lab, rate)
    gains[r] = best if best else (None, float('-inf'))
worth = sum(1 for r in B.ROLES if gains[r][1] > 0.010)
pc = worth >= 2

inert_fb = all(chg[c][r]['_fb_at_covered'] == 0 and chg[c][r]['_fb_at_uncovered'] > 0
               for c in chg for r in chg[c])
moves_rank = all(chg[c][r]['_identical_spec'] == 0 for c in chg for r in chg[c])
inert = inert_fb and moves_rank
livesame = max(abs(res[c][r][a][cl][b]['ce_live'] - res[c][r][ARMS[0]][cl][b]['ce_live'])
               for c in res for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in res[c][r][a][cl])
repro = max(max(max(abs(ce(C, r, a) - S1948_C16[a][i]) for a in S1948_C16),
                abs(ce(C, r, 'map512_mlpheavy') - S1948_ANCHOR[i]))
            for i, r in enumerate(B.ROLES))
pd = ncov[C] == 16110 and inert and livesame <= 1e-9 and repro <= 0.0002

for r in B.ROLES:
    print(f'\n  {r}', flush=True)
    for a in ARMS:
        print(f'    {a:18s} [{COST[a]:8.3f}M]  top1 {ov(C, r, a):6.2%} '
              f'({(ov(C, r, a) - ov(C, r, "map512_mlpheavy")) * 100:+.2f}pp)  CE {ce(C, r, a):7.5f} '
              f'({ce(C, r, a) - ce(C, r, "map512_mlpheavy"):+.5f}, t {pt[C][r][a]["t"]:+.2f})',
              flush=True)
    print(f'    CE argmin fallback: {argmin_fb(r)} = alpha {AR[argmin_fb(r)][0]}%, map rank '
          f'{AR[argmin_fb(r)][1]} | best rate above the anchor: {gains[r][0]} at '
          f'{gains[r][1]:.4f} nats/100M', flush=True)
print(f'\n  the fallback optimum MOVED off (25, 256) (>=2 roles) -> {pa}  {moved}/3  '
      f'(argmins {[AR[argmin_fb(r)] for r in B.ROLES]})', flush=True)
print(f'  and it wants map rank >= 256 (>=2 roles) -> {pb}  {morerank}/3', flush=True)
print(f'  and the move beats §1947\'s 0.010 nats/100M threshold (>=2 roles) -> {pc}  {worth}/3',
      flush=True)
print(f'  coverage {ncov}, fallback-differing arms inert at COVERED and differing at UNCOVERED '
      f'{inert_fb}, identical-spec arms are '
      f'bit-identical {moves_rank}, live identical {livesame:.1e}, §1948 CE reproduced within '
      f'{repro:.6f} nats -> control {pd}', flush=True)

B.report({'pred_a_fallback_optimum_moved': pa, 'pred_b_wants_more_map': pb,
          'pred_c_move_worth_buying': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1948 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'fallbacks': list(FB), 'knee': KNEE,
          'ce_argmin_fallback': {r: list(AR[argmin_fb(r)]) for r in B.ROLES},
          'best_rate_above_anchor': {r: [gains[r][0], gains[r][1]] for r in B.ROLES},
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
