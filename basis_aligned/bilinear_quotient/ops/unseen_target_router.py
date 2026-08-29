# AN INPUT-SIDE ROUTER FOR THE UNSEEN-TARGET CASE.
#
# §1954: the converged build loses the unseen (0-0) target bucket on 5 of 6 role-coverage cells.
# §1937/§1938 give the reason -- the neighbour half of the blend emits a real covered token's
# distribution, which puts near-zero mass on a token no fit row contains, and the blend is 25% neighbour
# on EVERY uncovered type. §1939 showed the neighbour cosine does NOT find that case (recovery
# proportional to the fraction routed). LESSON 74 forbids routing on the bucket itself: it is a property
# of the TARGET and the row must be chosen per INPUT token.
#
# There is an obvious input-side predictor nobody has tried: for each uncovered token, how much
# probability the LIVE model puts on OUT-OF-TABLE vocabulary from that token alone, on a length-1
# sequence. Call it unc_mass. A token whose own next-token distribution mostly leaves the table is
# exactly the token whose targets the neighbour cannot reach. It falls out of the loop that already
# builds the neighbour index, so it costs nothing to compute.
#
# ARMS. msk<P>m512 for P = 10 / 25 / 50 -- the top P% of uncovered types BY unc_mass take the rank-512
# map row, the rest take the neighbour row -- against mix25m512 (§1949's converged fallback, no routing),
# nn75m512 (§1939's cosine router, the only alternative tried), and map512. All at the converged
# allocation {mlp 768, attn 256}, 16,110. Reference labels bound as NAMES (LESSON 83).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1954's open question.
#
# Registered predictions, SIGNED per LESSON 72; every per-role figure read from the result JSON, not a
# log tail (LESSON 85); reproduction bars >= the published precision of their reference (LESSON 84).
#   pred_a THE SIGNAL IS REAL: among uncovered-input positions, those whose input token is in the TOP
#          quartile of unc_mass have a higher rate of genuinely unseen (0-0 bucket) targets than those in
#          the bottom quartile, on at least 2 of 3 roles. This validates the signal before anything is
#          built on it. If FALSE, unc_mass does not predict the case it was constructed for and the rest
#          of the section is uninterpretable.
#   pred_b AND ROUTING ON IT BEATS THE COSINE: some msk arm's pooled 0-0 kept-fraction exceeds
#          nn75m512's, on at least 2 of 3 roles. §1939's cosine is the only router this thread has tried
#          and it demonstrably misses this case; if a signal built FOR the case does not beat it, the
#          case is not routable from the input side.
#   pred_c AND IT REMOVES THE LOSS: some msk arm's pooled 0-0 kept-fraction is at least as high as the
#          DEPLOYED design's, on at least 2 of 3 roles -- i.e. the unseen-bucket deficit §1954 identified
#          is closed, not merely reduced. This is the deployable claim. If FALSE the deficit is a real
#          price of the blend and should be stated as such rather than engineered around.
#   pred_d CONTROLS: coverage exactly 16,110; all arms share one table spec and differ only in the
#          fallback, so they must be EXACTLY inert at covered inputs and must differ at uncovered ones
#          (LESSON 81, polarity derived for THIS lineage); buckets partition; live per-cell top-1 and CE
#          identical across arms; the routed fractions match their targets within 1%; and mix25m512
#          reproduces §1949's PUBLISHED pooled CE (5.88341 / 5.82982 / 5.86029) within 0.0005 nats.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/unseen_target_router_results.json'
KNEE = {'mlp': 768, 'attn': 256}
MSK = ('msk10m512', 'msk25m512', 'msk50m512')
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
PLAN = ((('map64', None, 'deployed'), ('mix25m512', KNEE, 'blend_768_256'),
         ('nn75m512', KNEE, 'cosine_router'), ('map512', KNEE, 'pure_map'))
        + tuple((a, KNEE, a) for a in MSK))
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
COVS = (('c16110', B.FIT_16110, 16110),)
C = 'c16110'

print(f'UNSEEN-TARGET ROUTER | msk arms on unc_mass vs §1939 cosine | 16,110 | '
      f'DISCOVERY ONLY', flush=True)
res, pt, chg, ncov, COST, AX = {}, {}, {}, {}, {}, {}
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
    FRAC = {a: P.route_fraction(a) for a in MSK}
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
S1949_CE = (5.88341, 5.82982, 5.86029)


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

beats_cos = sum(1 for r in B.ROLES
                if max(kf(r, a, 'pooled', BOT) for a in MSK) > kf(r, 'cosine_router', 'pooled', BOT))
pb = beats_cos >= 2
closes = sum(1 for r in B.ROLES
             if max(kf(r, a, 'pooled', BOT) for a in MSK) >= kf(r, 'deployed', 'pooled', BOT))
pc = closes >= 2

moves = (all(chg[c][r][f'{a}|{b}'] == 0 for c in chg for r in chg[c] for a, b in INERT)
         and all(chg[c][r][f'{a}|{b}'] > 0 for c in chg for r in chg[c] for a, b in DIFFER))
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
repro = max(abs(ce(r, KNEE_LAB) - S1949_CE[i]) for i, r in enumerate(B.ROLES))
assert all(FRAC[a] is not None for a in MSK), 'route_fraction returned None for a msk arm'
fracok = max(abs(FRAC[a] - int(a[3:].split('m')[0]) / 100.0) for a in MSK)
pd = (ncov[C] == 16110 and moves and bool(partition) and livesame <= 1e-9
      and repro <= 0.0005 and fracok <= 0.01)

for r in B.ROLES:
    print(f'\n  {r}   unseen-rate top vs bottom unc_mass quartile: {sep[r]:+.3f}', flush=True)
    for a in ARMS:
        print(f'    {a:16s} 0-0 {kf(r, a, "pooled", BOT):5.2%}  125+ {kf(r, a, "pooled", TOP):5.1%}  '
              f'CE {ce(r, a):7.5f}', flush=True)

print(f'\n  unc_mass predicts the unseen-target case (>=2 roles) -> {pa}  {pa_n}/3', flush=True)
print(f'  and routing on it beats §1939\'s cosine on the 0-0 bucket (>=2 roles) -> {pb}  '
      f'{beats_cos}/3', flush=True)
print(f'  and it closes the deficit vs the DEPLOYED design (>=2 roles) -> {pc}  {closes}/3', flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1949 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_signal_is_real': pa, 'pred_b_beats_cosine': pb,
          'pred_c_closes_the_deficit': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- §1954 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'msk': list(MSK), 'knee': KNEE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
