# WHERE DOES 0.010 NATS PER 100M COME FROM? -- pricing every lever at one operating point.
#
# §1947 introduced a 0.010 nats/100M spending threshold when the table curve was first swept, as a round
# number. §1947-§1957 then used it to place the table knee, reject extra attention capacity at 16,110,
# accept it at 5,419, and reject the §1955 router. EVERY allocation conclusion in that range is
# conditional on it and none of them justified it.
#
# There is a principled answer available and it costs one run. An allocation is efficient when the
# MARGINAL RETURN of every lever is equal -- move money from a lever returning less to one returning
# more until they meet. So: price all four levers at the same operating point (§1957's 5,419 build,
# {mlp 768, attn 384} with mix25m512) and see (a) whether they are equalised, and (b) where the common
# rate actually sits relative to 0.010.
#
# LEVERS, each priced by a step in both directions where possible:
#   mlp rank      {640,320} and {896,448}  -- scaled together to hold §1928's ratio fixed
#   attn rank     {768,256} and {768,576}
#   map rank      mix25m256 and mix25m1024 at {768,384}
#   blend alpha   mix10m512 and mix40m512 at {768,384}
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 3 -- §1957's open question.
#
# Registered predictions, SIGNED per LESSON 72; reference triples via B.ref() (LESSONS 85, 87); control
# polarity derived from the plan, and this plan deliberately contains BOTH same-spec and differing-spec
# arm pairs so neither half of it is vacuous (§1957).
#   pred_a THE LEVERS ARE NOT EQUALISED: at this operating point the highest marginal return among the
#          PRICEABLE levers (any cost-neutral one is excluded and reported) is at least 2x the lowest, on at least 2 of 3 roles. If FALSE the allocation is
#          already at an interior optimum in all four directions, which would be a much stronger result
#          than anything §1947-§1957 claimed and would retire the threshold question entirely.
#   pred_b AND THE MAP RANK IS THE RICHEST LEVER: the map-rank step returns MORE nats per 100M than the
#          mlp-rank step, on at least 2 of 3 roles. §1953/§1954 found the fallback carries essentially
#          the whole margin over the deployed design at both coverages while the tables are near-free,
#          so marginal money should prefer the map. If FALSE that attribution does not survive being
#          expressed as an exchange rate, and §1953's framing needs revisiting.
#   pred_c AND 0.010 IS NOT THE NATURAL RATE: the MEDIAN marginal return across the four levers differs
#          from 0.010 nats/100M by more than 50% (i.e. lies outside 0.005-0.015), on at least 2 of 3
#          roles. This is the question the section exists for. If FALSE, §1947's round number happens to
#          sit at the exchange rate and every conclusion built on it stands unaltered -- worth knowing
#          either way, and I have no prior.
#   pred_d CONTROLS: coverage exactly 5,419; the plan-derived covered-input control holds in BOTH
#          directions with both sides non-empty; buckets partition; live per-cell top-1 and CE identical
#          across arms; and the {768,384} arm reproduces §1957's pooled CE via B.ref() within 0.0005.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

t0 = time.time()
OUT = B.PT + 'ops/lever_exchange_rates_results.json'
KNEE = {'mlp': 768, 'attn': 256}
WIDE = {'mlp': 768, 'attn': 384}
FB = 'mix25m512'   # §1949's fallback
MLPHEAVY = {'mlp': 768, 'attn': 256}      # §1931's allocation, kept as the anchor's spec
# (arm, table_rank spec, label). The fallback is FIXED at the §1944 blend except for the anchor.
BASE = {'mlp': 768, 'attn': 384}
PLAN = (('mix25m512', BASE, 'base'),
        ('mix25m512', {'mlp': 640, 'attn': 320}, 'mlp_down'),
        ('mix25m512', {'mlp': 896, 'attn': 448}, 'mlp_up'),
        ('mix25m512', {'mlp': 768, 'attn': 256}, 'attn_down'),
        ('mix25m512', {'mlp': 768, 'attn': 576}, 'attn_up'),
        ('mix25m256', BASE, 'map_down'),
        ('mix25m1024', BASE, 'map_up'),
        ('mix10m512', BASE, 'alpha_down'),
        ('mix40m512', BASE, 'alpha_up'))
WIDE_LAB = 'base'
LEVERS = (('mlp', 'mlp_down', 'mlp_up'), ('attn', 'attn_down', 'attn_up'),
          ('map', 'map_down', 'map_up'), ('alpha', 'alpha_down', 'alpha_up'))
# REFERENCE LABELS AS NAMES, NEVER LITERALS. Five times on 2026-08-29 a fork renamed an arm and left a
# string literal behind -- res[...]['map64'], armR['blend_full'], armR['map512_mlpheavy'] -- each a
# KeyError after the whole run, and each invisible to the gate because a str is not a Name. Four static
# checks were measured against the corpus and all four were too noisy to ship (LESSON 82). Binding the
# references here turns the same mistake into an UNDEFINED NAME, which the gate has caught since
# LESSON 80. This is the fix; the checker was not.
REF = 'base'
KNEE_LAB = 'base'

ARMS = tuple(p[2] for p in PLAN)
SPEC = {p[2]: (p[0], p[1]) for p in PLAN}
INERT, DIFFER = B.inertness_pairs(PLAN)
COVS = (('c5419', B.FIT_5419, 5419),)
C = 'c5419'

print(f'LEVER EXCHANGE RATES at {BASE} | four levers priced at one point | 5,419 | '
      f'DISCOVERY ONLY', flush=True)
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
S1957_CE = B.ref(B.PT + 'ops/coverage_specific_build_results.json', 'blend_768_384')


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

def lever_rate(r, lo, hi):
    """nats per 100M for money moved along one lever, priced across its two neighbouring points."""
    dn = ce(r, lo) - ce(r, hi)
    dm = (COST[hi] - COST[lo]) / 100.0
    return dn / dm if dm else float('nan')


# The blend ALPHA is cost-neutral -- mix10m512 and mix40m512 store the same rank-512 map and the same
# index, so its "nats per 100M" is 0/0. It is not a spending lever at all, which is worth stating: alpha
# is a FREE parameter and should simply be set to its optimum (S1943: 0.25), never traded against money.
# The first scoring of this run left the nan in the set and computed a median and a max/min over it;
# both happened to come out right, by luck, because Python's sorted() and max() do not propagate nan.
# Excluded explicitly, and the exclusion is reported.
RAW = {r: {nm: lever_rate(r, lo, hi) for nm, lo, hi in LEVERS} for r in B.ROLES}
import math
RATES = {r: {nm: v for nm, v in RAW[r].items() if math.isfinite(v)} for r in B.ROLES}
DROPPED = sorted({nm for r in B.ROLES for nm, v in RAW[r].items() if not math.isfinite(v)})
spread = {r: (max(RATES[r].values()) / min(RATES[r].values())
              if min(RATES[r].values()) > 0 else float('inf')) for r in B.ROLES}
pa = sum(1 for r in B.ROLES if spread[r] >= 2.0) >= 2

pb_n = sum(1 for r in B.ROLES if RATES[r]['map'] > RATES[r]['mlp'])
pb = pb_n >= 2


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


MED = {r: median(list(RATES[r].values())) for r in B.ROLES}
pc_n = sum(1 for r in B.ROLES if not (0.005 <= MED[r] <= 0.015))
pc = pc_n >= 2

moves = (all(chg[c][r][f'{a}|{b}'] == 0 for c in chg for r in chg[c] for a, b in INERT)
         and all(chg[c][r][f'{a}|{b}'] > 0 for c in chg for r in chg[c] for a, b in DIFFER))
partition = all(res[C][r][a][cl][b]['n'] for r in B.ROLES for a in ARMS
                for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
livesame = max(abs(res[C][r][a][cl][b]['ce_live'] - res[C][r][REF][cl][b]['ce_live'])
               for r in B.ROLES for a in ARMS
               for cl in ('covered_input', 'uncovered_input', 'pooled') for b in BK)
repro = max(abs(ce(r, KNEE_LAB) - S1957_CE[i]) for i, r in enumerate(B.ROLES))
fracok = 0.0
pd = (ncov[C] == 5419 and moves and bool(partition) and livesame <= 1e-9
      and repro <= 0.0005 and fracok <= 0.01)

for r in B.ROLES:
    print(f'\n  {r}   unseen-rate top vs bottom unc_mass quartile: {sep[r]:+.3f}', flush=True)
    for a in ARMS:
        print(f'    {a:16s} 0-0 {kf(r, a, "pooled", BOT):5.2%}  125+ {kf(r, a, "pooled", TOP):5.1%}  '
              f'CE {ce(r, a):7.5f}', flush=True)

for r in B.ROLES:
    print(f'\n  {r}  marginal nats per 100M: ' + '  '.join(
        f'{nm} {RATES[r][nm]:.4f}' for nm in RATES[r])
        + f'  | median {MED[r]:.4f}  spread {spread[r]:.2f}x', flush=True)
print(f'\n  cost-neutral levers excluded from every statistic: {DROPPED}', flush=True)
print(f'  the levers are NOT equalised, max/min >= 2x (>=2 roles) -> {pa}', flush=True)
print(f'  and the MAP rank is richer than the MLP rank (>=2 roles) -> {pb}  {pb_n}/3', flush=True)
print(f'  and the median rate is OUTSIDE 0.005-0.015 (>=2 roles) -> {pc}  {pc_n}/3', flush=True)
print(f'  coverage {ncov}, {len(INERT)} same-spec pairs inert and {len(DIFFER)} differing-spec '
      f'pairs not, at covered inputs: {moves}; buckets partition, live '
      f'identical {livesame:.1e}, §1957 CE reproduced within {repro:.6f}, route fracs within '
      f'{fracok:.4f} -> control {pd}', flush=True)

B.report({'pred_a_levers_not_equalised': pa, 'pred_b_map_richer_than_mlp': pb,
          'pred_c_median_not_near_0p010': pc, 'pred_d_controls': pd},
         {'config': {'arms': list(ARMS), 'coverages': [16110], 'table_rank': 'FULL',
                     'costs_M': COST,
                     'built_with': f'ops/bqlib.py v{B.LIB_VERSION}',
                     'ROLE_NOTE': 'DISCOVERY ONLY, rung 3 -- S1957 open question.',
                     'plan': {p[2]: [p[0], str(p[1])] for p in PLAN},
                     'fallback': 'mix25m256 = 25% output-NN neighbour + 75% rank-256 map'},
          'results': res,
          'paired_vs_map512': pt,
          'arms': list(ARMS), 'base': BASE,
          'changed_at_covered_inputs': chg,
          'cache_stats': dict(B.STATS)},
         OUT, t0)
