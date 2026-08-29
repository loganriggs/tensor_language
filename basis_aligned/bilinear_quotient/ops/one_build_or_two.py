# ONE BUILD OR TWO? -- the coverages now disagree and nobody has priced the compromise.
#
# §1957 put the 5,419 attention rank at 384 against §1947's 256 at 16,110; §1959 put the 5,419 map rank
# at 640 against §1949's 512. So "the build" is now two builds, and every section since §1946 has
# implicitly assumed you would ship whichever matches your coverage. That is a real operational cost --
# two sets of 36 tables, two maps -- and nobody has asked what a single compromise build gives up.
#
# ARMS. the two coverage-specific builds and three compromises, all scored at BOTH coverages:
#   spec_5419   {mlp 768, attn 384} + rank-640 map    (§1959)
#   spec_16110  {mlp 768, attn 256} + rank-512 map    (§1949/§1950)
#   mid_a       {mlp 768, attn 320} + rank-576 map    (midpoint of both)
#   mid_b       {mlp 768, attn 384} + rank-512 map    (5,419 attention, 16,110 map)
#   mid_c       {mlp 768, attn 256} + rank-640 map    (16,110 attention, 5,419 map)
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1959's open question.
#
# Registered predictions, SIGNED per LESSON 72; reference triples via B.ref() (LESSONS 85, 87); the plan
# contains BOTH same-spec and differing-spec pairs so neither half of the control is vacuous (§1957).
#   pred_a THE COMPROMISE IS CHEAP: the best single build's pooled CE is within 0.002 nats of the
#          coverage-specific build at EACH coverage, on at least 2 of 3 roles per coverage. 0.002 is the
#          size of the last two marginal purchases (§1957: 0.0017/0.0021/0.0014; §1959:
#          0.0011/0.0010/0.0011), so this asks whether shipping one build costs less than one of the
#          steps that produced the disagreement. If FALSE the two builds are genuinely different objects
#          and the operational cost of maintaining both is justified.
#   pred_b AND THE MAP AXIS DOMINATES THE CHOICE: mid_b and mid_c differ from each other by MORE than
#          mid_a differs from either, in pooled CE at 16,110, on at least 2 of 3 roles -- i.e. which map
#          rank you pick matters more than which attention rank. §1958 priced map as the richest lever
#          and attention as the poorest, so the compromise should be decided on the map. If FALSE the
#          two axes contribute comparably and a midpoint on both is the right compromise.
#   pred_c AND ONE OF THEM IS NOT A COMPROMISE AT ALL: at least one arm is within 0.002 nats of the
#          coverage-specific build at BOTH coverages simultaneously, on at least 2 of 3 roles. That is
#          the deployable claim -- a single build that is effectively coverage-free. If FALSE, shipping
#          one build costs real accuracy at one coverage or the other and the choice is a genuine
#          operational tradeoff, which is the more likely outcome and worth stating plainly.
#   pred_d CONTROLS: coverages exactly 5,419 and 16,110; the plan-derived covered-input control holds in
#          both directions with both sides non-empty at each coverage; buckets partition; live per-cell
#          top-1 and CE identical across arms; and spec_16110 reproduces §1949's published 16,110 CE via
#          B.ref() within 0.0005 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/one_build_or_two_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
BASE = {'mlp': 768, 'attn': 384}
A384 = {'mlp': 768, 'attn': 384}
A320 = {'mlp': 768, 'attn': 320}
A256 = {'mlp': 768, 'attn': 256}
PLAN = (('mix25m640', A384, 'spec_5419'), ('mix25m512', A256, 'spec_16110'),
        ('mix25m576', A320, 'mid_a'), ('mix25m512', A384, 'mid_b'),
        ('mix25m640', A256, 'mid_c'))
LAB = [p[2] for p in PLAN]
COMPROMISE = ('mid_a', 'mid_b', 'mid_c')
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

print(f'ONE BUILD OR TWO | {LAB} at both coverages | DISCOVERY ONLY', flush=True)
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
        PT_K[role] = pt[cov][role]
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



COVSPEC = {'c5419': 'spec_5419', 'c16110': 'spec_16110'}


def cev(cov, r, a):
    return res[cov][r][a]['pooled']['overall']['ce_prog']


def gap(cov, r, a):
    return cev(cov, r, a) - cev(cov, r, COVSPEC[cov])


near = {a: {cov: sum(1 for r in B.ROLES if gap(cov, r, a) <= 0.002) for cov in res}
        for a in COMPROMISE}
best = max(COMPROMISE, key=lambda a: min(near[a].values()))
pa = min(near[best].values()) >= 2

sep_map = {r: abs(cev('c16110', r, 'mid_b') - cev('c16110', r, 'mid_c')) for r in B.ROLES}
sep_mid = {r: max(abs(cev('c16110', r, 'mid_a') - cev('c16110', r, 'mid_b')),
                  abs(cev('c16110', r, 'mid_a') - cev('c16110', r, 'mid_c'))) for r in B.ROLES}
pb_n = sum(1 for r in B.ROLES if sep_map[r] > sep_mid[r])
pb = pb_n >= 2

both = {a: sum(1 for r in B.ROLES if all(gap(cov, r, a) <= 0.002 for cov in res)) for a in COMPROMISE}
pc_n = max(both.values())
pc = pc_n >= 2

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
      and repro <= 0.0005 and fracok <= 0.01)

for r in B.ROLES:
    for a in ARMS:
        print(f'    {a:16s} 0-0 {kf(r, a, "pooled", BOT):5.2%}  125+ {kf(r, a, "pooled", TOP):5.1%}  '
              f'CE {ce(r, a):7.5f}', flush=True)

for cov in res:
    print(f'\n  === {cov} (coverage-specific arm: {COVSPEC[cov]}) ===', flush=True)
    for r in B.ROLES:
        print(f'    {r:10s} ' + '  '.join(f'{a} {cev(cov, r, a):.5f} ({gap(cov, r, a)*1000:+.2f}m)'
                                          for a in LAB), flush=True)
print(f'\n  the best single build is within 0.002 nats at BOTH coverages (>=2 roles each) -> {pa}  '
      f'best {best} {near[best]}', flush=True)
print(f'  and the MAP axis separates more than the attention axis (>=2 roles) -> {pb}  {pb_n}/3',
      flush=True)
print(f'  and some arm is within 0.002 at both coverages on the same role (>=2 roles) -> {pc}  '
      f'{pc_n}/3  {both}', flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1949 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_compromise_is_cheap': pa, 'pred_b_map_axis_dominates': pb,
          'pred_c_one_build_suffices': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': {f'{c}|{l}': v for (c, l), v in COST.items()},
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- S1959 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'base': BASE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
