# DOES S1963 HOLD ON THE BUILD WE WOULD ACTUALLY SHIP?
#
# §1963 tested the per-token alpha at §1960's COMPROMISE allocation ({mlp 768, attn 320}, rank-576 map),
# because that section was about one build versus two. The deployed-coverage build §1957/§1959 actually
# arrived at is {mlp 768, attn 384} with a rank-640 map, and §1963's conclusion -- that the tilt raises
# the unseen bucket by the intended mechanism and still pays more CE than it is worth -- has never been
# checked there. A conclusion measured on a build nobody ships is a weaker claim than it reads as.
#
# Rung 2: a second-class confirmation of §1963 at the operating point that matters, not a replication of
# the same arms.
#
# ARMS. pat10_40m640 and pat20_30m640 (widest and narrowest tilt) against mix30m640 and mix25m640, all
# at {mlp 768, attn 384}; plus mix25m512 at {768,256} as a differing-table-rank arm so neither half of
# the plan-derived covered-input control is vacuous (§1957). 5,419 coverage.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §1963 at the shipped operating point.
#
# Registered predictions, SIGNED per LESSON 72; every per-role figure read from the result JSON
# (LESSON 85); reference triples via B.ref() with an EXPLICIT coverage (§1963).
#   pred_a THE TILT STILL RAISES THE UNSEEN BUCKET: some pat arm's pooled 0-0 kept-fraction exceeds
#          mix30m640's, on at least 2 of 3 roles. §1963 got 3/3 at 5,419 on the compromise build. If
#          FALSE the effect was specific to the rank-576 map and §1963's mechanism claim does not
#          transfer to the build we would ship.
#   pred_b AND IT STILL COSTS MORE THAN IT IS WORTH: that arm's pooled CE exceeds mix30m640's by more
#          than 0.002 nats, on at least 2 of 3 roles -- reproducing §1963's pred_b failure, which is the
#          finding. If FALSE the tilt is affordable on this build and §1963's negative was an artefact of
#          the compromise allocation, which would REOPEN a line I recorded as closed.
#   pred_c AND THE WIDEST TILT IS STILL THE STRONGEST: pat10_40m640 gives a higher 0-0 kept-fraction than
#          pat20_30m640, on at least 2 of 3 roles -- §1963's pred_c, which is what shows the arm acts
#          through its mechanism rather than by accident.
#   pred_d CONTROLS: coverage exactly 5,419; the plan-derived covered-input control holds in BOTH
#          directions with both sides non-empty; buckets partition; live per-cell top-1 and CE identical
#          across arms; and mix25m640 reproduces §1959's published pooled CE for map640 via B.ref()
#          within 0.0005 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/tilt_on_the_shipped_build_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
BASE = {'mlp': 768, 'attn': 384}
A320 = {'mlp': 768, 'attn': 320}
A256 = {'mlp': 768, 'attn': 256}
A384 = {'mlp': 768, 'attn': 384}
PAT = ('pat10_40m640', 'pat20_30m640')
PLAN = ((('mix30m640', A384, 'flat30'), ('mix25m640', A384, 'flat25'),
         ('mix25m512', A256, 'spec_control'))
        + tuple((a, A384, a) for a in PAT))
LAB = ['flat30', 'flat25'] + list(PAT)
BASE_A = 'flat30'
WIDE_LAB = 'flat30'
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'flat30'
KNEE_LAB = 'flat30'

ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
INERT, DIFFER = B.inertness_pairs(PLAN)
COVS = (('c5419', B.FIT_5419, 5419),)
C = 'c5419'

print(f'TILT ON THE SHIPPED BUILD | {PAT} at {A384} | 5,419 | DISCOVERY ONLY', flush=True)
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


# §1951's PUBLISHED 5,419 pooled CE for the converged build.
S1959_CE = B.ref(B.PT + 'ops/map_curve_turnover_results.json', 'map640')


def kf(r, a, cls, b):
    return res[C][r][a][cls][b]['kept_fraction']


def ce(r, a):
    return res[C][r][a]['pooled']['overall']['ce_prog']



BK = [f'{x}-{y}' for x, y in B.BUCKETS]
BOT, TOP = BK[0], BK[-1]


def kf(cov, r, a, b):
    return res[cov][r][a]['pooled'][b]['kept_fraction']


def cev(cov, r, a):
    return res[cov][r][a]['pooled']['overall']['ce_prog']


helps = {cov: sum(1 for r in B.ROLES
                  if max(kf(cov, r, a, BOT) for a in PAT) > kf(cov, r, BASE_A, BOT))
         for cov in res}
pa = all(v >= 2 for v in helps.values())


def best_pat(cov, r):
    return max(PAT, key=lambda a: kf(cov, r, a, BOT))


cheap = {cov: sum(1 for r in B.ROLES
                  if cev(cov, r, best_pat(cov, r)) - cev(cov, r, BASE_A) > 0.002) for cov in res}
pb = all(v >= 2 for v in cheap.values())

widest = sum(1 for r in B.ROLES
             if kf('c5419', r, 'pat10_40m640', BOT) > kf('c5419', r, 'pat20_30m640', BOT))
pc = widest >= 2

costs_equal = 0.0
repro = max(abs(cev('c5419', r, 'flat25') - S1959_CE[i]) for i, r in enumerate(B.ROLES))

moves = (all(chg[c][r][f'{a}|{b}'] == 0 for c in chg for r in chg[c] for a, b in INERT)
         and all(chg[c][r][f'{a}|{b}'] > 0 for c in chg for r in chg[c] for a, b in DIFFER))
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
fracok = 0.0
pd = (ncov['c5419'] == 5419 and ncov['c16110'] == 16110 and moves and bool(partition)
      and livesame <= 1e-9
      and repro <= 0.0005 and fracok <= 0.01 and costs_equal <= 1e-9)

for cov in res:
    print(f'\n  === {cov} ===', flush=True)
    for r in B.ROLES:
        print(f'    {r:10s} ' + '  '.join(
            f'{a.replace("m576", ""):9s} 0-0 {kf(cov, r, a, BOT):5.2%} dCE '
            f'{(cev(cov, r, a) - cev(cov, r, BASE_A)) * 1000:+.2f}m' for a in LAB), flush=True)
print(f'\n  a per-token tilt raises the unseen bucket (>=2 roles each) -> {pa}  {helps}', flush=True)
print(f'  and STILL costs >0.002 nats of pooled CE (>=2 roles) -> {pb}  {cheap}', flush=True)
print(f'  and the widest tilt still beats the narrowest (>=2 roles) -> {pc}  {widest}/3',
      flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1959 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_tilt_still_helps': pa, 'pred_b_still_costs_too_much': pb,
          'pred_c_widest_still_strongest': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': {f'{c}|{l}': v for (c, l), v in COST.items()},
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 2 -- S1963 at the shipped operating point.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'base': BASE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
