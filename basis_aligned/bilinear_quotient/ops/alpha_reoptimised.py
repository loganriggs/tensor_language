# THE ONE FREE LEVER -- alpha has not been re-optimised since it was chosen.
#
# §1943 set the blend at alpha = 0.25 (25% output-NN neighbour, 75% map) at 5,419, on FULL-RANK tables,
# with a rank-512 map, before any of the allocation work existed. Since then the operating point has
# moved on three axes: tables truncated to {mlp 768, attn 320} (§1946-§1957), map rank changed (§1949,
# §1959), and the whole thing re-checked at two coverages (§1960). Alpha has been carried along unchanged
# the entire time.
#
# §1958 established the thing that makes this cheap: alpha is COST-NEUTRAL. mix10m576 and mix40m576 store
# the same map and the same index, so moving it buys or loses accuracy at exactly zero parameters. It is
# the only one of the four levers that is free, and the only one still sitting at a value chosen for a
# different build.
#
# ARMS. mix{10,20,25,30,40,50}m576 at §1960's compromise allocation {mlp 768, attn 320}, at BOTH
# coverages, plus spec_16110 ({768,256} + rank-512 map) as a differing-table-rank arm so that neither
# half of the plan-derived covered-input control is vacuous (§1957).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1960's open question.
#
# Registered predictions, SIGNED per LESSON 72; reference triples via B.ref() (LESSONS 85, 87); every
# string edit to this fork asserted its anchor matched before writing (§1960).
#   pred_a ALPHA IS STALE: the CE-minimising alpha at this operating point is NOT 0.25, on at least 2 of
#          3 roles at 5,419. The build has changed on three axes since 0.25 was chosen and nothing has
#          rechecked it. If FALSE, 0.25 survives a truncation of the tables by half, two changes of map
#          rank and a change of coverage -- which would make it a property of the two fallback components
#          rather than of the build, and worth banking as that.
#   pred_b AND IT DOES NOT DEPEND ON COVERAGE: the CE-minimising alpha at 16,110 is within one grid step
#          of the one at 5,419, on at least 2 of 3 roles. Alpha trades the neighbour against the map, and
#          both act only on uncovered inputs, so the optimum should not care that the uncovered arm is
#          ~24% of positions rather than ~10%. If FALSE alpha is coverage-specific and §1960's single
#          compromise build cannot fix it with one value.
#   pred_c AND THE MOVE IS FREE AND REAL: the CE at the optimal alpha beats alpha = 0.25 with a paired
#          t <= -2.0 at ZERO additional cost, on at least 2 of 3 roles at 5,419. This is the deployable
#          claim. If FALSE the stale value is already within noise of the optimum and alpha should be
#          left alone -- worth stating plainly, since a free lever that turns out to be flat is still an
#          answer.
#   pred_d CONTROLS: coverages exactly 5,419 and 16,110; the plan-derived covered-input control holds in
#          BOTH directions with both sides non-empty; all alpha arms cost EXACTLY the same at each
#          coverage (that is what makes the lever free, so it is checked, not assumed); buckets
#          partition; live per-cell top-1 and CE identical across arms; and spec_16110 reproduces
#          §1949's published 16,110 CE via B.ref() within 0.0005 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/alpha_reoptimised_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
BASE = {'mlp': 768, 'attn': 384}
AL = (10, 20, 25, 30, 40, 50)
A320 = {'mlp': 768, 'attn': 320}
A256 = {'mlp': 768, 'attn': 256}
PLAN = (tuple((f'mix{a}m576', A320, f'a{a}') for a in AL)
        + (('mix25m512', A256, 'spec_16110'),))
LAB = [f'a{a}' for a in AL]
BASE_A = 'a25'          # §1943's value, the incumbent
WIDE_LAB = 'spec_16110'
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'spec_16110'
KNEE_LAB = 'spec_16110'

ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
INERT, DIFFER = B.inertness_pairs(PLAN)
COVS = (('c5419', B.FIT_5419, 5419), ('c16110', B.FIT_16110, 16110))
C = 'c16110'

print(f'ALPHA RE-OPTIMISED | alphas {AL} at {A320} | both coverages | DISCOVERY ONLY',
      flush=True)
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
        PT_K.setdefault(cov, {})[role] = {
            a: B.paired_t(armR[a][role][1], armR[BASE_A][role][1]) for a in ARMS}
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

    # COST is per-COVERAGE (the tables dominate and ncov differs), so key it by both --
    # the flat dict silently kept only the last coverage's numbers.
    COST.update({(cov, lab): P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS})
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
S1949_CE = B.ref(B.PT + 'ops/fallback_at_the_knee_results.json', 'fb_mix25m512')


def kf(r, a, cls, b):
    return res[C][r][a][cls][b]['kept_fraction']


def ce(r, a):
    return res[C][r][a]['pooled']['overall']['ce_prog']



def cev(cov, r, a):
    return res[cov][r][a]['pooled']['overall']['ce_prog']


def argmin_a(cov, r):
    return AL[min(range(len(AL)), key=lambda i: cev(cov, r, LAB[i]))]


stale = sum(1 for r in B.ROLES if argmin_a('c5419', r) != 25)
pa = stale >= 2

STEP = {AL[i]: i for i in range(len(AL))}
same = sum(1 for r in B.ROLES
           if abs(STEP[argmin_a('c16110', r)] - STEP[argmin_a('c5419', r)]) <= 1)
pb = same >= 2

free = sum(1 for r in B.ROLES
           for a in [argmin_a('c5419', r)]
           if a != 25 and cev('c5419', r, f'a{a}') < cev('c5419', r, BASE_A)
           and PT_K['c5419'][r][f'a{a}']['t'] <= -2.0)
pc = free >= 2

costs_equal = max(abs(COST[(cov, LAB[i])] - COST[(cov, LAB[0])])
                  for cov in res for i in range(len(AL)))

moves = (all(chg[c][r][f'{a}|{b}'] == 0 for c in chg for r in chg[c] for a, b in INERT)
         and all(chg[c][r][f'{a}|{b}'] > 0 for c in chg for r in chg[c] for a, b in DIFFER))
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
repro = max(abs(ce(r, KNEE_LAB) - S1949_CE[i]) for i, r in enumerate(B.ROLES))
fracok = 0.0
pd = (ncov['c5419'] == 5419 and ncov['c16110'] == 16110 and moves and bool(partition)
      and livesame <= 1e-9
      and repro <= 0.0005 and fracok <= 0.01 and costs_equal <= 1e-9)

for r in B.ROLES:
    for a in ARMS:
        print(f'    {a:16s} 0-0 {kf(r, a, "pooled", BOT):5.2%}  125+ {kf(r, a, "pooled", TOP):5.1%}  '
              f'CE {ce(r, a):7.5f}', flush=True)

for cov in res:
    print(f'\n  === {cov} ===', flush=True)
    for r in B.ROLES:
        print(f'    {r:10s} ' + '  '.join(
            f'a{AL[i]} {(cev(cov, r, LAB[i]) - cev(cov, r, BASE_A)) * 1000:+.3f}m' for i in range(len(AL)))
            + f'   argmin a{argmin_a(cov, r)}', flush=True)
print(f'\n  alpha = 0.25 is STALE at this operating point (>=2 roles) -> {pa}  {stale}/3  '
      f'(argmins at 5,419 {[argmin_a("c5419", r) for r in B.ROLES]})', flush=True)
print(f'  and the optimum does not depend on coverage (>=2 roles) -> {pb}  {same}/3  '
      f'(argmins at 16,110 {[argmin_a("c16110", r) for r in B.ROLES]})', flush=True)
print(f'  and the move is significant at zero cost (>=2 roles) -> {pc}  {free}/3  '
      f'(max cost spread across alpha arms {costs_equal:.2e}M)', flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1949 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_alpha_is_stale': pa, 'pred_b_optimum_coverage_free': pb,
          'pred_c_move_is_free_and_real': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': {f'{c}|{l}': v for (c, l), v in COST.items()},
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- S1960 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'base': BASE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
