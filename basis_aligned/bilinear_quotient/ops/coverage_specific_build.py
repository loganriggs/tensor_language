# THE COVERAGE-SPECIFIC BUILD -- {768, 384} at 5,419, scored end to end.
#
# §1947 put the table knee at {mlp 768, attn 256} and §1950 proved it a fixed point -- both at 16,110.
# §1952 then measured the efficient attention rank at 5,419 as 384 on skip7000 and skip11000, 256 on
# skip1200: one doubling higher, because §1951 found attention capacity worth 2-3x more at the deployed
# coverage. Nothing has scored {768, 384} end to end at 5,419 against the two builds it would replace.
#
# ARMS. blend_768_384 (the coverage-specific candidate), blend_768_256 (§1950's converged build),
# and the deployed design. Fallback fixed at §1949's mix25m512. 5,419 coverage.
#
# All reference triples come from B.ref(), read out of the artifacts that published them -- I mistyped
# one by hand in §1953 and again in §1956 (LESSONS 85, 87).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1952's open consequence.
#
# Registered predictions, SIGNED per LESSON 72; paired t on every CE claim up front (LESSON 78);
# control polarity DERIVED from the plan (LESSON 86).
#   pred_a THE EXTRA ATTENTION PAYS ON ITS OWN TERMS: blend_768_384's pooled CE is BELOW
#          blend_768_256's with a paired t <= -2.0, on at least 2 of 3 roles. §1952 found the step worth
#          more than 0.010 nats per 100M on two roles; this asks whether the difference is significant,
#          which a rate crossing a threshold does not establish.
#   pred_b AND IT IS WORTH ITS MONEY: the CE gained per 100M spent going 256 -> 384 exceeds §1947's
#          0.010 nats/100M threshold, on at least 2 of 3 roles -- reproducing §1952's rate finding on an
#          end-to-end build rather than inside an allocation sweep.
#   pred_c AND IT DOES NOT UNDO THE REDISTRIBUTION: blend_768_384's 0-0 kept-fraction is NOT below
#          blend_768_256's by more than 0.2pp, on at least 2 of 3 roles. §1953 showed the converged build
#          already loses the unseen bucket on 2 of 3 roles at this coverage; spending on attention should
#          not deepen that. If FALSE the coverage-specific build buys CE by making the known deficit
#          worse and should be reported that way rather than as a free upgrade.
#   pred_d CONTROLS: coverage exactly 5,419; the plan-derived covered-input control (same table_rank =>
#          exactly inert, different => must differ) holds in both directions; buckets partition; live
#          per-cell top-1 and CE identical across arms; and blend_768_256 reproduces §1951's pooled CE,
#          read via B.ref() from ops/converged_at_deployed_coverage_results.json, within 0.0005 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/coverage_specific_build_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = (('map64', None, 'deployed'), ('mix25m512', KNEE, 'blend_768_256'),
        ('mix25m512', WIDE, 'blend_768_384'))
WIDE_LAB = 'blend_768_384'
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'deployed'                # §1789's deployed design, the anchor
KNEE_LAB = 'blend_768_256'      # §1949's build

ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
INERT, DIFFER = B.inertness_pairs(PLAN)
COVS = (('c5419', B.FIT_5419, 5419),)
C = 'c5419'

print(f'COVERAGE-SPECIFIC BUILD | {WIDE} vs {KNEE} vs deployed | 5,419 | DISCOVERY ONLY',
      flush=True)
res, pt, chg, ncov, COST, AX = {}, {}, {}, {}, {}, {}
PT_K = {}   # paired t of the wide build against the converged one, per role
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
        pt[cov][role] = {a: B.paired_t(armR[a][role][1], armR[REF][role][1])
                         for a in ARMS}
        PT_K[role] = B.paired_t(armR[WIDE_LAB][role][1], armR[KNEE_LAB][role][1])
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

    COST.update({lab: P.cost(SPEC[lab][0], SPEC[lab][1]) / 1e6 for lab in ARMS})
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
S1951_CE = B.ref(B.PT + 'ops/converged_at_deployed_coverage_results.json', 'blend_768_256')


def kf(r, a, cls, b):
    return res[C][r][a][cls][b]['kept_fraction']


def ce(r, a):
    return res[C][r][a]['pooled']['overall']['ce_prog']


# pred_a: validate the signal itself against the eval data before anything is built on it.
# Among UNCOVERED-input positions, split by the INPUT token's unc_mass quartile and compare how often
# the true target is genuinely unseen (fit-row count 0). If unc_mass means what it was constructed to
# mean, the top quartile must be higher.
sep = {}
for r in B.ROLES:
    tgt, icov = AX[r]
    iid = B.input_ids(PROG, r)
    freq = PROG.freq.cpu()[tgt.long()]
    unseen = (freq == 0)
    um = PROG.unc_mass.cpu()[iid.long()]
    u = ~icov
    q = torch.quantile(um[u].double(), torch.tensor([0.25, 0.75], dtype=torch.float64))
    lo = u & (um <= q[0].float())
    hi = u & (um >= q[1].float())
    sep[r] = float(unseen[hi].float().mean()) - float(unseen[lo].float().mean())

pa_n = sum(1 for r in B.ROLES if sep[r] > 0)
pa = pa_n >= 2

sig = sum(1 for r in B.ROLES
          if ce(r, WIDE_LAB) < ce(r, KNEE_LAB) and PT_K[r]['t'] <= -2.0)
pa = sig >= 2

rate = {r: (ce(r, KNEE_LAB) - ce(r, WIDE_LAB)) / ((COST[WIDE_LAB] - COST[KNEE_LAB]) / 100.0)
        for r in B.ROLES}
worth = sum(1 for r in B.ROLES if rate[r] > 0.010)
pb = worth >= 2

deficit = sum(1 for r in B.ROLES
              if kf(r, WIDE_LAB, 'pooled', BOT) >= kf(r, KNEE_LAB, 'pooled', BOT) - 0.002)
pc = deficit >= 2

moves = (all(chg[c][r][f'{a}|{b}'] == 0 for c in chg for r in chg[c] for a, b in INERT)
         and all(chg[c][r][f'{a}|{b}'] > 0 for c in chg for r in chg[c] for a, b in DIFFER))
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
repro = max(abs(ce(r, KNEE_LAB) - S1951_CE[i]) for i, r in enumerate(B.ROLES))
fracok = 0.0
pd = (ncov[C] == 5419 and moves and bool(partition) and livesame <= 1e-9
      and repro <= 0.0005 and fracok <= 0.01)

for r in B.ROLES:
    print(f'\n  {r}   unseen-rate top vs bottom unc_mass quartile: {sep[r]:+.3f}', flush=True)
    for a in ARMS:
        print(f'    {a:16s} 0-0 {kf(r, a, "pooled", BOT):5.2%}  125+ {kf(r, a, "pooled", TOP):5.1%}  '
              f'CE {ce(r, a):7.5f}', flush=True)

print(f'\n  the extra attention lowers CE significantly (>=2 roles) -> {pa}  {sig}/3  '
      + ' '.join(f'{PT_K[r]["t"]:+.2f}' for r in B.ROLES), flush=True)
print(f'  and it clears §1947\'s 0.010 nats/100M threshold (>=2 roles) -> {pb}  {worth}/3  '
      + ' '.join(f'{rate[r]:.4f}' for r in B.ROLES), flush=True)
print(f'  and it does not deepen the unseen-bucket deficit (>=2 roles) -> {pc}  {deficit}/3',
      flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1951 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_attention_lowers_ce': pa, 'pred_b_clears_the_threshold': pb,
          'pred_c_no_deeper_deficit': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- S1952 open consequence.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'knee': KNEE, 'wide': WIDE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
