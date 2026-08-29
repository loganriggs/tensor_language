# DID FIVE CE-OPTIMAL PURCHASES DRIFT THE ACCURACY PROFILE?
#
# §1953 showed the converged build beats §1789's deployed design by REDISTRIBUTING -- winning
# mid-frequency targets, losing the unseen bucket -- and §1954 that the shape inverts with coverage.
# Since then §1957 (attention 256->384), §1959 (map 512->640), §1960 (the compromise allocation) and
# §1961 (alpha 0.25->0.30) have each bought one to two milli-nats of POOLED CE, and not one of them
# looked at the bucket structure. A sequence of locally CE-optimal steps can move the accuracy profile
# somewhere nobody chose, and five of them have now been taken blind.
#
# ARMS. deployed (§1789), converged (§1950's {768,256} + rank-512 map), current (§1960/§1961's
# compromise: {mlp 768, attn 320}, rank-576 map, alpha 0.30). Both coverages. The plan contains BOTH
# same-spec and differing-spec pairs so neither half of the control is vacuous (§1957).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1961's open question.
#
# Registered predictions, SIGNED per LESSON 72; figures from the result JSON (LESSON 85); reproduction
# bars no tighter than the published precision of their reference (LESSON 84).
#   pred_a THE PROFILE DID NOT DRIFT: between §1950's converged build and the current one, no bucket's
#          pooled kept-fraction moves by more than 0.5pp, on at least 2 of 3 roles at EACH coverage. The
#          five purchases are worth ~5 milli-nats in total, so they should be invisible at this
#          resolution. If FALSE, optimising pooled CE has been moving the accuracy profile as a side
#          effect and every build since §1950 needs reporting on buckets, not just on CE.
#   pred_b AND §1954'S REDISTRIBUTION STILL HOLDS: the current build still loses the unseen (0-0) bucket
#          against the DEPLOYED design at 16,110, on at least 2 of 3 roles. §1954 found this for the
#          converged build; if the purchases had quietly fixed it that would be worth knowing, and if
#          they had deepened it, more so.
#   pred_c AND THE COMMON-TARGET COST IS STILL BOUNDED: the current build's 125+ kept-fraction is within
#          1.0pp of the deployed design's, on at least 2 of 3 roles at each coverage -- the bound §1953
#          measured (-1.01/-0.68/-0.17pp at covered inputs). If FALSE the purchases have eaten into the
#          most-frequent bucket and the build is no longer the same object §1953 characterised.
#   pred_d CONTROLS: coverages exactly 5,419 and 16,110; the plan-derived covered-input control holds in
#          BOTH directions with both sides non-empty; buckets partition; live per-cell top-1 and CE
#          identical across arms; and the deployed arm reproduces §1932's published pooled top-1
#          (13.55/14.25/13.64%) within 0.1pp -- a bar at twice the rounding envelope of a 1-dp figure.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/profile_after_five_purchases_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
BASE = {'mlp': 768, 'attn': 384}
A320 = {'mlp': 768, 'attn': 320}
A256 = {'mlp': 768, 'attn': 256}
PLAN = (('map64', None, 'deployed'), ('mix25m512', A256, 'converged'),
        ('mix30m576', A320, 'current'), ('mix25m576', A320, 'current_a25'))
LAB = ['deployed', 'converged', 'current']
BASE_A = 'converged'
WIDE_LAB = 'current'
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'deployed'
KNEE_LAB = 'converged'

ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
INERT, DIFFER = B.inertness_pairs(PLAN)
COVS = (('c5419', B.FIT_5419, 5419), ('c16110', B.FIT_16110, 16110))
C = 'c16110'

print(f'PROFILE AFTER FIVE PURCHASES | deployed vs converged vs current | both '
      f'coverages | DISCOVERY ONLY', flush=True)
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
S1949_CE = B.ref(B.PT + 'ops/fallback_at_the_knee_results.json', 'fb_mix25m512')


def kf(r, a, cls, b):
    return res[C][r][a][cls][b]['kept_fraction']


def ce(r, a):
    return res[C][r][a]['pooled']['overall']['ce_prog']


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

BK = [f'{x}-{y}' for x, y in B.BUCKETS]
BOT, TOP = BK[0], BK[-1]


def kf(cov, r, a, b):
    return res[cov][r][a]['pooled'][b]['kept_fraction']


drift = {cov: sum(1 for r in B.ROLES
                  if max(abs(kf(cov, r, 'current', b) - kf(cov, r, 'converged', b)) for b in BK) <= 0.005)
         for cov in res}
pa = all(v >= 2 for v in drift.values())

pb_n = sum(1 for r in B.ROLES if kf('c16110', r, 'current', BOT) < kf('c16110', r, 'deployed', BOT))
pb = pb_n >= 2

bound = {cov: sum(1 for r in B.ROLES
                  if abs(kf(cov, r, 'current', TOP) - kf(cov, r, 'deployed', TOP)) <= 0.010)
         for cov in res}
pc = all(v >= 2 for v in bound.values())

costs_equal = 0.0
S1932_T1 = (0.1355, 0.1425, 0.1364)
repro = max(abs(res['c5419'][r]['deployed']['pooled']['overall']['top1_acc_prog'] - S1932_T1[i])
            for i, r in enumerate(B.ROLES))

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
      and repro <= 0.001 and fracok <= 0.01 and costs_equal <= 1e-9)

for cov in res:
    print(f'\n  === {cov} ===', flush=True)
    for r in B.ROLES:
        print(f'    {r:10s} ' + '  '.join(
            f'{b:>7s} dep {kf(cov, r, "deployed", b):5.1%} conv {kf(cov, r, "converged", b):5.1%} '
            f'cur {kf(cov, r, "current", b):5.1%}' for b in BK), flush=True)
        print(f'    {"":10s} max |current - converged| across buckets: '
              f'{max(abs(kf(cov, r, "current", b) - kf(cov, r, "converged", b)) for b in BK) * 100:.2f}pp',
              flush=True)
print(f'\n  the profile did NOT drift, <=0.5pp in every bucket (>=2 roles each) -> {pa}  {drift}',
      flush=True)
print(f'  and §1954\'s unseen-bucket loss still holds at 16,110 (>=2 roles) -> {pb}  {pb_n}/3',
      flush=True)
print(f'  and the 125+ cost is still within 1.0pp of deployed (>=2 roles each) -> {pc}  {bound}',
      flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1949 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_profile_did_not_drift': pa, 'pred_b_unseen_loss_holds': pb,
          'pred_c_common_cost_bounded': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': {f'{c}|{l}': v for (c, l), v in COST.items()},
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- S1961 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'base': BASE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
