# A PER-TOKEN ALPHA -- the one combination three converging lines never tried.
#
# Three independent findings point at the same residual. §1954/§1962: the unseen (0-0) bucket is the only
# place the build is worse than §1789's deployed design, and the only cell that drifted the wrong way
# under five CE purchases. §1955/§1956: unc_mass predicts that case well (+23.3/+24.3/+19.8pp quartile
# separation) but ROUTING on it -- giving a token one row or the other -- is unprofitable at both
# coverages. §1961: a single global alpha is flat near its optimum and effectively already right.
#
# Nothing has varied the blend WEIGHT per token. pat<LO>_<HI>m<R> gives every uncovered type its own
# alpha, linear in its unc_mass quantile: HI% neighbour for the token whose next-token distribution
# stays inside the table, falling to LO% for the token that leaves it. That is not routing -- every
# token still gets both components -- and §1962 named it as the untried combination. It costs one float
# per uncovered type on top of the index: 272.823M against mix25m576's 272.644M at 16,110, +0.18M.
#
# ARMS. pat10_40m576, pat15_35m576, pat20_30m576 (widening to narrowing tilt) against mix30m576
# (§1961's flat optimum) and mix25m576, at {mlp 768, attn 320}, both coverages.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1962's open question.
#
# Registered predictions, SIGNED per LESSON 72; figures from the result JSON (LESSON 85).
#   pred_a THE TILT HELPS THE UNSEEN BUCKET: some pat arm's pooled 0-0 kept-fraction exceeds mix30m576's
#          on at least 2 of 3 roles at EACH coverage. That is the residual all three lines converge on.
#          If FALSE, varying the weight does no better than fixing it and the unseen deficit is not
#          addressable from the input side at all -- which, after §1956, would close the question.
#   pred_b AND IT DOES NOT COST POOLED CE: that same arm's pooled CE is no worse than mix30m576's by
#          more than 0.002 nats, on at least 2 of 3 roles at each coverage. 0.002 is the size of the
#          marginal purchases §1957-§1961 were making. If FALSE the tilt buys the bucket by paying CE,
#          which is the trade §1955/§1956 already rejected, in a new costume.
#   pred_c AND THE DIRECTION IS THE PREDICTED ONE: among the three pat arms, the one with the WIDEST
#          tilt (pat10_40) gives the highest 0-0 kept-fraction, on at least 2 of 3 roles at 5,419. The
#          mechanism says more neighbour where the table suffices and less where it does not, so more
#          tilt should mean more of the effect. If FALSE the tilt is not acting through the mechanism I
#          built it for, whatever it is doing.
#   pred_d CONTROLS: coverages exactly 5,419 and 16,110; the plan-derived covered-input control holds in
#          BOTH directions with both sides non-empty; buckets partition; live per-cell top-1 and CE
#          identical across arms; and mix25m576 reproduces §1961's pooled CE via B.ref() within 0.0005.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/per_token_alpha_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
BASE = {'mlp': 768, 'attn': 384}
A320 = {'mlp': 768, 'attn': 320}
A256 = {'mlp': 768, 'attn': 256}
PAT = ('pat10_40m576', 'pat15_35m576', 'pat20_30m576')
PLAN = ((('mix30m576', A320, 'flat30'), ('mix25m576', A320, 'flat25'),
         ('mix25m512', A256, 'spec_control'))
        + tuple((a, A320, a) for a in PAT))
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
COVS = (('c5419', B.FIT_5419, 5419), ('c16110', B.FIT_16110, 16110))
C = 'c16110'

print(f'PER-TOKEN ALPHA | {PAT} vs flat | both coverages | DISCOVERY ONLY', flush=True)
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
S1961_CE = B.ref(B.PT + 'ops/alpha_reoptimised_results.json', 'a25')


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
                  if cev(cov, r, best_pat(cov, r)) - cev(cov, r, BASE_A) <= 0.002) for cov in res}
pb = all(v >= 2 for v in cheap.values())

widest = sum(1 for r in B.ROLES if best_pat('c5419', r) == 'pat10_40m576')
pc = widest >= 2

costs_equal = 0.0
repro = max(abs(cev('c5419', r, 'flat25') - S1961_CE[i]) for i, r in enumerate(B.ROLES))

moves = (all(chg[c][r][f'{a}|{b}'] == 0 for c in chg for r in chg[c] for a, b in INERT)
         and all(chg[c][r][f'{a}|{b}'] > 0 for c in chg for r in chg[c] for a, b in DIFFER))
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
repro = max(abs(ce(r, KNEE_LAB) - S1961_CE[i]) for i, r in enumerate(B.ROLES))
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
print(f'  and costs <=0.002 nats of pooled CE (>=2 roles each) -> {pb}  {cheap}', flush=True)
print(f'  and the WIDEST tilt is the best at 5,419 (>=2 roles) -> {pc}  {widest}/3', flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1961 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_tilt_helps_unseen': pa, 'pred_b_no_ce_cost': pb,
          'pred_c_widest_tilt_wins': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': {f'{c}|{l}': v for (c, l), v in COST.items()},
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- S1962 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'base': BASE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
