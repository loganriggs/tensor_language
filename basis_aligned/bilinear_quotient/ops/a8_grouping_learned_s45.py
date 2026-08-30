# TWO MORE SEEDS: is §2077's weak seed one bad draw, or is the effect two-out-of-three?
#
# RUNG 3: the open question named at the end of §2077.
#
# §2077 confirmed §2075's a8 grouping on the learned DAS directions at a pooled p = 0.0270 against a
# size-matched permutation null. The per-seed column qualified it hard: the statistic was 0.0842, 0.1143
# and **0.0072**, and that third value is INSIDE the null (p95 0.0574), not a weak positive. Two seeds
# cleared individually and one saw nothing, so a single-seed run drawing 20260832 would have reported a
# clean negative. §2077 recorded that the pooled p-value was doing real work and that a reader taking it
# without the per-seed column would overrate the result.
#
# The distinguishing question is cheap: seed 20260832 was the weakest on EVERY statistic it appears in
# (pooled difference 0.0072, four-cluster margin 1.04 against 1.68 and 1.89), which looks like one bad
# draw rather than a quantity-wide instability. Two more seeds decide it. The per-seed test needs no
# pooling -- each seed's within-minus-between is compared to the SAME null p95 of 0.0574 that §2077
# computed, so the two new seeds are directly comparable to the three already measured.
#
# The parent's dead inherited fit loop was removed (LESSON 112) before this was derived, so this run does
# 32 fits rather than 48.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  BOTH new seeds exceed the null p95 of 0.0574, making it 4 of 5 and supporting "one bad draw".
#           Registered in the strong direction on purpose: the weak reading (some seeds work) is what the
#           data already permits, so the risk belongs on the claim that would change how §2077 is quoted.
#           If FALSE, at most 3 of 5 seeds show the effect and §2077 should be read as "unreliable per
#           fit" rather than "real but noisy" -- a distinction I would then apply to its registry entry.
#   pred_b  The four-circuit cluster {r.2.0.1, r.2.0.2, r.2.1.1, r.2.2.1} is above the a8-wide mean in
#           both new seeds, continuing §2077's 3/3 to 5/5. §2077's third seed cleared this at a ratio of
#           1.04, an effective tie, so this asks whether that was the seed or the cluster.
#   pred_c  CONTROL: every one of the 32 fits passes the optimiser health gate (LESSON 108), and the
#           five-seed mean and sd of the statistic are reported (LESSON 110) rather than a pooled scalar.
#
# Writes circuits/A8_GROUPING_LEARNED_S45.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

# PLAN PRE-FLIGHT (LESSON 109): census_lib builds MODS from the live model at import, so enqueue's
# BQLIB_DRYRUN gate must be answered BEFORE that import or the gate runs the experiment for real.
COMPONENT = 'a8'
if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    if not os.path.exists(os.path.join(BQ, 'circuits/A8_GROUPING.json')):
        print('DRYRUN FAIL: circuits/A8_GROUPING.json absent -- S2075 must have run')
        raise SystemExit(1)
    g = json.load(open(os.path.join(BQ, 'circuits/A8_GROUPING.json')))
    b = json.load(open(os.path.join(BQ, 'circuits/BATTERY.json')))
    n = sum(1 for v in b['by_tag'].values() if v['best_mean'] == COMPONENT)
    multi = [c for c in g['clusters'] if len(c) >= 2]
    if n < 3 or not multi:
        print(f'DRYRUN FAIL: {n} circuits at {COMPONENT}, {len(multi)} multi-member clusters')
        raise SystemExit(1)
    print(f'DRYRUN OK: {n} circuits at {COMPONENT}; S2075 grouping has {len(g["clusters"])} clusters '
          f'({len(multi)} multi-member)')
    raise SystemExit(0)

import time

import torch
import torch.nn.functional as F

import census_lib as C

RANKS = (1,)
STEPS = 400
LR = 5e-2
BATCH = 4
SEED = 20260830
SEEDS = (20260833, 20260834)
TRAIN_ROWS = (0, 600)
EVAL_ROWS = (600, 1000)
C.use_state('census_state_diverse.pt')
for p in C.m.parameters():
    p.requires_grad_(False)
R = C.rows()
NP = C.T                                                    # scored positions per row

BAT = json.load(open('circuits/BATTERY.json'))
GRP = json.load(open('circuits/A8_GROUPING.json'))
TARGETS = [t for t, v in BAT['by_tag'].items() if v['best_mean'] == COMPONENT]
_RAW_CL = [[t for t in c if t in TARGETS] for c in GRP['clusters']]
CLUSTERS = [c for c in _RAW_CL if c]                 # single binding; _RAW_CL is the pre-filter form
FOUR = next((c for c in CLUSTERS if len(c) == 4), None)
print(f'DAS at {COMPONENT} on all {len(TARGETS)} circuits localised there: {TARGETS}', flush=True)
assert len(TARGETS) >= 3, 'need at least three circuits to ask whether they separate'

def leaf_masks(tag):
    lf = C.leaf(tag)
    mm = torch.zeros(len(R) * NP, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(len(R) * NP, dtype=torch.bool); sl[lf['slice']] = True
    return mm.view(len(R), NP), sl.view(len(R), NP)


def fwd(idx, tg, key=None, Q=None, donor=None, full_donor=None):
    """forward with an optional interchange at `key`; along Q if given, on the whole output if not."""
    h = None
    if key is not None:
        def fh(mo, i_, o_):
            y = o_[0] if isinstance(o_, tuple) else o_
            f = y.float()
            if full_donor is not None:
                f = full_donor.view_as(f)
            else:
                co = f.reshape(-1, C.D) @ Q
                cd = donor.reshape(-1, C.D) @ Q
                f = (f.reshape(-1, C.D) + (cd - co) @ Q.T).view_as(f)
            f = f.to(y.dtype)
            return (f, o_[1]) if isinstance(o_, tuple) else f
        h = C.MODS[key].register_forward_hook(fh)
    x = F.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
    for blk in C.m.transformer.h:
        x, v1 = blk(x, v1, x0)
    lg = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
    if h is not None:
        h.remove()
    return F.cross_entropy(lg.view(-1, lg.size(-1)), tg.reshape(-1), reduction='none').view(idx.shape)


@torch.no_grad()
def capture(key, lo, hi):
    cap = []
    h = C.MODS[key].register_forward_hook(
        lambda mo, i_, o_: cap.append(((o_[0] if isinstance(o_, tuple) else o_)
                                       .detach().float().cpu())))
    for i in range(lo, hi, BATCH):
        bb = R[i:i + BATCH, :NP + 1].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    return torch.cat(cap)


def batches(lo, hi, mm=None):
    """row batches; when mm is given, only batches containing at least one circuit member.

    A circuit's members are ~0.3% of the grid, so a blind batch of 4 rows carries about three of them and
    the member half of the objective is estimated from three positions. Skipping member-free batches puts
    every gradient step on a batch that can actually see the circuit.
    """
    for i in range(lo, hi, BATCH):
        if mm is not None and mm[i:i + BATCH].sum() == 0:
            continue
        bb = R[i:i + BATCH, :NP + 1].to(C.DEV)
        yield i, bb[:, :-1].contiguous(), bb[:, 1:].contiguous()


def closed_form_dir(acts, mm, sl, lo, hi):
    """§2056's probe: unit(mean over members - mean off slice), at this component, on the train rows."""
    f = acts.reshape(-1, C.D)
    m_ = mm[lo:hi].reshape(-1); s_ = sl[lo:hi].reshape(-1)
    if m_.sum() == 0 or (~s_).sum() == 0:
        return None
    u = f[m_].mean(0) - f[~s_].mean(0)
    return (u / u.norm()).to(C.DEV)


def evaluate(tag, key, mm, sl, Q, acts, lo, hi):
    """held-out mean dCE on members and off slice, for a Q-interchange and for the full-output one."""
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(acts.shape[0] * NP, generator=g)
    flat = acts.reshape(-1, C.D)
    dq_m, dq_o, df_m, df_o = [], [], [], []
    with torch.no_grad():
        for i, idx, tg in batches(lo, hi):
            k = (i - lo) // BATCH
            n = idx.shape[0] * NP
            dn = flat[perm[k * BATCH * NP:k * BATCH * NP + n]].to(C.DEV).view(idx.shape[0], NP, C.D)
            b0 = fwd(idx, tg)
            m_, s_ = mm[i:i + idx.shape[0]].to(C.DEV), sl[i:i + idx.shape[0]].to(C.DEV)
            if Q is not None:
                d = fwd(idx, tg, key, Q, dn) - b0
                dq_m.append(d[m_]); dq_o.append(d[~s_])
            d = fwd(idx, tg, key, full_donor=dn) - b0
            df_m.append(d[m_]); df_o.append(d[~s_])
    f = lambda L: float(torch.cat(L).abs().mean()) if L and sum(x.numel() for x in L) else float('nan')
    return f(dq_m), f(dq_o), f(df_m), f(df_o)


# THE INHERITED SINGLE-SEED FIT LOOP WAS DELETED HERE (2026-08-30, mid-run).
# It fit all 16 circuits once and printed a "rank 1:" line each, and NOTHING below reads its output --
# verified: the analysis references neither `out` nor its `QDIR`. On the run that produced this file's
# results it burned 1122s of a ~3400s run, 16 wasted fits out of 64. §2070 recorded the identical waste
# in its own parent at 357s and I then derived this script from that parent and carried it anyway.
# LESSON 112. The results already written are unaffected: the wasted loop's output was discarded either
# way, so removing it changes cost and not numbers.

t0 = time.time()                                     # was bound by the inherited loop removed
#                                                      under LESSON 112; the deletion dropped a
#                                                      LIVE binding and ops/gate.py caught it.

# ---- PER-SEED LEARNED DIRECTIONS, THEN THE S2075 GROUPING TESTED AGAINST A NULL
import random                                                             # noqa: E402

PERSEED = {}
HEALTH = []
for sd_ in SEEDS:
    QD = {}
    for tag in TARGETS:
        mm, sl = leaf_masks(tag)
        acts_tr = capture(COMPONENT, *TRAIN_ROWS)
        g = torch.Generator(device='cpu').manual_seed(sd_ + 1)
        P0 = (torch.randn(C.D, 1, generator=g) / C.D ** 0.5).to(C.DEV)
        Q_init = torch.linalg.qr(P0)[0].detach().clone()
        P = P0.clone().requires_grad_(True)
        opt = torch.optim.Adam([P], lr=LR)
        gg = torch.Generator().manual_seed(sd_)
        perm = torch.randperm(acts_tr.shape[0] * NP, generator=gg)
        flat = acts_tr.reshape(-1, C.D)
        step = 0; losses = []
        while step < STEPS:
            for i_, idx, tg in batches(*TRAIN_ROWS, mm=mm):
                if step >= STEPS:
                    break
                m_, s_ = mm[i_:i_ + idx.shape[0]].to(C.DEV), sl[i_:i_ + idx.shape[0]].to(C.DEV)
                if m_.sum() == 0:
                    continue
                k = (i_ - TRAIN_ROWS[0]) // BATCH
                n = idx.shape[0] * NP
                dn = flat[perm[k * BATCH * NP:k * BATCH * NP + n]].to(C.DEV).view(
                    idx.shape[0], NP, C.D)
                Q = torch.linalg.qr(P)[0]
                with torch.no_grad():
                    b0 = fwd(idx, tg)
                d = fwd(idx, tg, COMPONENT, Q, dn) - b0
                loss = -d[m_].mean() + d[~s_].abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.detach())); step += 1
        with torch.no_grad():
            Q = torch.linalg.qr(P)[0]
            moved = 1.0 - float((Q.T @ Q_init).pow(2).sum())
        first = sum(losses[:20]) / 20; last = sum(losses[-20:]) / 20
        HEALTH.append((sd_, tag, bool(moved > 0.02 and last < first)))
        QD[tag] = Q[:, 0].detach().clone()
    PERSEED[sd_] = {a: {b: float(abs(QD[a] @ QD[b])) for b in TARGETS} for a in TARGETS}
    print(f'  seed {sd_}: {len(TARGETS)} directions fit ({time.time()-t0:.0f}s)', flush=True)


def within_between(cos, clusters):
    of = {t: i for i, c in enumerate(clusters) for t in c}
    win, bet = [], []
    for a in TARGETS:
        for b in TARGETS:
            if a == b or a not in of or b not in of:
                continue
            (win if of[a] == of[b] else bet).append(cos[a][b])
    if not win or not bet:
        return None, None, None
    mw = sum(win) / len(win); mb = sum(bet) / len(bet)
    return mw, mb, mw - mb


POOL = {a: {b: sum(PERSEED[s_][a][b] for s_ in SEEDS) / len(SEEDS) for b in TARGETS} for a in TARGETS}
sizes = [len(c) for c in CLUSTERS]
rng = random.Random(20260830)
NDRAW = 20000


def null_dist(cos):
    out = []
    for _ in range(NDRAW):
        t = TARGETS[:]; rng.shuffle(t)
        cl = []; j = 0
        for s_ in sizes:
            cl.append(t[j:j + s_]); j += s_
        _w, _b, dd = within_between(cos, cl)
        if dd is not None:
            out.append(dd)
    out.sort()
    return out


mw, mb, obs = within_between(POOL, CLUSTERS)
nd = null_dist(POOL)
pval = sum(1 for x in nd if x >= obs) / len(nd)
per_seed_stat = []
for s_ in SEEDS:
    _w, _b, dd = within_between(PERSEED[s_], CLUSTERS)
    per_seed_stat.append(dd)
mstat = sum(per_seed_stat) / len(per_seed_stat)
sdstat = (sum((x - mstat) ** 2 for x in per_seed_stat) / max(1, len(per_seed_stat) - 1)) ** 0.5

four_ok = 0
four_detail = []
if FOUR:
    for s_ in SEEDS:
        cs = PERSEED[s_]
        wf = [cs[a][b] for a in FOUR for b in FOUR if a != b]
        allp = [cs[a][b] for a in TARGETS for b in TARGETS if a != b]
        w = sum(wf) / len(wf); g_ = sum(allp) / len(allp)
        four_detail.append({'seed': s_, 'within_four': round(w, 4), 'a8_mean': round(g_, 4),
                            'above': bool(w > g_)})
        four_ok += int(w > g_)

unhealthy = [h for h in HEALTH if not h[2]]
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude', 'component': COMPONENT,
       'seeds': list(SEEDS), 'circuits': TARGETS, 'clusters_from_S2075': CLUSTERS,
       'confirms': 'S2075 / A8_HAS_A_CAUSALLY_VALIDATED_GROUPING_S2075, on a THIRD quantity',
       'method': 'rank-1 DAS per circuit at three seeds; pairwise |cos| of the LEARNED directions pooled '
                 'over seeds; S2075 clustering tested by within-minus-between against a 20,000-draw '
                 'size-matched permutation null registered BEFORE the run (LESSON 111)',
       'pooled_within': round(mw, 4), 'pooled_between': round(mb, 4),
       'pooled_within_minus_between': round(obs, 4),
       'null_median': round(nd[len(nd) // 2], 4), 'null_p95': round(nd[int(0.95 * len(nd))], 4),
       'p_value': round(pval, 4),
       'per_seed_statistic': [round(x, 4) for x in per_seed_stat],
       'statistic_mean': round(mstat, 4), 'statistic_sd': round(sdstat, 4),
       'four_cluster': FOUR, 'four_cluster_detail': four_detail,
       'four_above_a8_mean_in_n_seeds': four_ok,
       'unhealthy_fits': unhealthy,
       'S2077_null_p95': 0.0574, 'S2077_per_seed': [0.0842, 0.1143, 0.0072],
       'new_seeds_exceeding_S2077_null_p95': sum(1 for x in per_seed_stat if x > 0.0574),
       'pred_a_both_new_seeds_clear_the_null': bool(all(x > 0.0574 for x in per_seed_stat)),
       'pred_b_four_cluster_above_in_both': bool(four_ok >= 2),
       'pred_c_controls_pass': bool(not unhealthy),
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/A8_GROUPING_LEARNED_S45.json', 'w'), indent=1)

print(f'\nwrote circuits/A8_GROUPING_LEARNED_S45.json ({time.time()-t0:.0f}s)')
print(f'pred_c  CONTROL -- all {len(HEALTH)} fits healthy : {rep["pred_c_controls_pass"]}')
if unhealthy:
    print(f'        UNHEALTHY {unhealthy} -- reporting nothing (LESSON 108)')
else:
    print(f'        statistic per seed {[round(x, 4) for x in per_seed_stat]}  '
          f'mean {mstat:.4f} +- {sdstat:.4f}   (LESSON 110)')
    n_clear = rep['new_seeds_exceeding_S2077_null_p95']
    allfive = [0.0842, 0.1143, 0.0072] + [round(x, 4) for x in per_seed_stat]
    m5 = sum(allfive) / 5
    sd5 = (sum((x - m5) ** 2 for x in allfive) / 4) ** 0.5
    print(f'pred_a  new seeds clearing S2077 null p95 0.0574: {n_clear}/2 (bar 2) : '
          f'{rep["pred_a_both_new_seeds_clear_the_null"]}')
    print(f'        all five seeds: {allfive}   mean {m5:.4f} +- {sd5:.4f}   '
          f'clearing the null: {sum(1 for x in allfive if x > 0.0574)}/5')
    print(f'        (this run pooled over its own 2 seeds: {obs:.4f}, p = {pval:.4f})')
    print(f'pred_b  the four-cluster is above the a8 mean in {four_ok}/2 new seeds (bar 2) : '
          f'{rep["pred_b_four_cluster_above_in_both"]}')
    for dd in four_detail:
        print(f'        seed {dd["seed"]}: within-four {dd["within_four"]:.4f} vs a8 mean '
              f'{dd["a8_mean"]:.4f}  above={dd["above"]}')
