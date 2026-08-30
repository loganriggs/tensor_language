# THE ONE m16 NUMBER STILL RESTING ON A SINGLE SEED: the mean pairwise |cos| of the learned directions.
#
# RUNG 3: the gap §2072 exposes, applied to my own §2067/§2068 before anyone quotes them.
#
# §2067's pred_a and §2068's pred_a both FAILED against a 0.50 bar on a single-seed mean pairwise |cos|
# of the learned directions -- 0.3976 over six circuits and 0.3896 over five. Those two failures are what
# refuted "DAS also fails to separate m16" and killed the mixture hypothesis, and both are quoted in the
# registry. **Neither has a spread.** §2070 measured spreads for MARGINS at m16 and §2071 for OVERLAPS at
# a8/a16; the pairwise cosine between learned directions has never been measured at more than one seed.
#
# §2072 is what happens when a single draw is compared to a fixed threshold without knowing the noise: a
# claim 0.6% from its threshold, on a quantity with ~6% relative noise, survived four sections before
# falling. Here the gap is 0.102 (0.3976 against 0.50) and the noise is unknown. That is a better ratio
# than §2060 had, which is a reason to check rather than a reason to assume.
#
# This fits m16's six circuits at three seeds and reports the mean pairwise |cos| per seed. It skips the
# projection-ablation tables entirely -- the expensive half -- because only the geometry is in question.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  The quantity is STABLE: sd of the mean pairwise |cos| across seeds < 0.05. Registered from
#           §2071's rule, which says noise is a property of the quantity -- a cosine is one inner product
#           between two learned vectors, not a ratio of ratios, so it should behave like the overlaps
#           (sd 0.007-0.103) rather than the margins (0.029-0.319). If FALSE the rule does not extend to
#           pairs of learned directions, and §2071's generalisation needs narrowing in turn.
#   pred_b  §2067 and §2068 STAND: the mean across seeds stays below 0.50, so DAS does separate m16's
#           circuits geometrically and both pred_a failures were real. If FALSE the geometric half of
#           §2067/§2068 joins §2060's highlight in retraction, and m16's story changes materially --
#           which is the outcome this run exists to be able to detect.
#   pred_c  CONTROL: every fit passes the optimiser health gate. LESSON 108.
#
# Writes circuits/DAS_M16_COS_SEEDS.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

# PLAN PRE-FLIGHT (LESSON 109): census_lib builds MODS from the live model at import, so enqueue's
# BQLIB_DRYRUN gate must be answered BEFORE that import or the gate runs the experiment for real.
COMPONENT = 'm16'
if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    b = json.load(open(os.path.join(BQ, 'circuits/BATTERY.json')))
    n = sum(1 for v in b['by_tag'].values() if v['best_mean'] == COMPONENT)
    if n < 3:
        print(f'DRYRUN FAIL: only {n} circuits at {COMPONENT}')
        raise SystemExit(1)
    print(f'DRYRUN OK: {n} circuits localise to {COMPONENT}')
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
SEEDS = (20260830, 20260831, 20260832)
TRAIN_ROWS = (0, 600)
EVAL_ROWS = (600, 1000)
C.use_state('census_state_diverse.pt')
for p in C.m.parameters():
    p.requires_grad_(False)
R = C.rows()
NP = C.T                                                    # scored positions per row

BAT = json.load(open('circuits/BATTERY.json'))
TARGETS = [t for t, v in BAT['by_tag'].items() if v['best_mean'] == COMPONENT]
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


out = {}
QDIR = {}
ACTIVE_SEED = SEEDS[0]
t0 = time.time()
for tag in TARGETS:
    key = BAT['by_tag'][tag]['best_mean']
    mm, sl = leaf_masks(tag)
    acts_tr = capture(key, *TRAIN_ROWS)
    acts_ev = capture(key, *EVAL_ROWS)
    rec = {'component': key, 'members': int(mm.sum()), 'ranks': {}}
    for r in RANKS:
        g = torch.Generator(device='cpu').manual_seed(ACTIVE_SEED + r)
        # unit-scale init: |P| ~ 1 so an lr 5e-2 Adam step is a real rotation, not a 1e-4 nudge
        P0 = (torch.randn(C.D, r, generator=g) / C.D ** 0.5).to(C.DEV)
        Q_init = torch.linalg.qr(P0)[0].detach().clone()
        P = P0.clone().requires_grad_(True)
        opt = torch.optim.Adam([P], lr=LR)
        gg = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(acts_tr.shape[0] * NP, generator=gg)
        flat = acts_tr.reshape(-1, C.D)
        step = 0
        losses = []
        while step < STEPS:
            for i, idx, tg in batches(*TRAIN_ROWS, mm=mm):
                if step >= STEPS:
                    break
                m_, s_ = mm[i:i + idx.shape[0]].to(C.DEV), sl[i:i + idx.shape[0]].to(C.DEV)
                if m_.sum() == 0:
                    continue
                k = (i - TRAIN_ROWS[0]) // BATCH
                n = idx.shape[0] * NP
                dn = flat[perm[k * BATCH * NP:k * BATCH * NP + n]].to(C.DEV).view(
                    idx.shape[0], NP, C.D)
                Q = torch.linalg.qr(P)[0]
                with torch.no_grad():
                    b0 = fwd(idx, tg)
                d = fwd(idx, tg, key, Q, dn) - b0
                loss = -d[m_].mean() + d[~s_].abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
                step += 1
        with torch.no_grad():
            Q = torch.linalg.qr(P)[0]
            moved = 1.0 - float((Q.T @ Q_init).pow(2).sum() / r)      # 0 = never moved, 1 = orthogonal
        first, last = (sum(losses[:20]) / 20, sum(losses[-20:]) / 20) if len(losses) >= 40 else (0., 0.)
        healthy = moved > 0.02 and last < first
        qm, qo, fm, fo = evaluate(tag, key, mm, sl, Q, acts_ev, *EVAL_ROWS)
        ent = {'das_dce_members': round(qm, 4), 'das_dce_offslice': round(qo, 4),
               'das_concentration': round(qm / qo, 3) if qo > 0 else None,
               'full_dce_members': round(fm, 4), 'full_dce_offslice': round(fo, 4),
               'full_concentration': round(fm / fo, 3) if fo > 0 else None,
               'fraction_of_full_recovered': round(qm / fm, 3) if fm > 0 else None,
               'subspace_moved_from_init': round(moved, 4),
               'loss_first20': round(first, 6), 'loss_last20': round(last, 6),
               'optimiser_healthy': bool(healthy)}
        if r == 1:
            u = closed_form_dir(acts_tr, mm, sl, *TRAIN_ROWS)
            if u is not None:
                ent['overlap_with_closed_form'] = round(float((Q[:, 0] @ u) ** 2), 3)
                cm, co, _fm, _fo = evaluate(tag, key, mm, sl, u.unsqueeze(1), acts_ev, *EVAL_ROWS)
                ent['closed_form_dce_members'] = round(cm, 4)
                ent['closed_form_concentration'] = round(cm / co, 3) if co > 0 else None
                ent['das_beats_closed_form'] = bool(qm > cm)
        rec['ranks'][r] = ent
        if r == 1:
            QDIR[tag] = Q[:, 0].detach().clone()
        print(f'  {tag:12s} {key:4s} rank {r}: members {qm:.4f} off {qo:.4f} conc '
              f'{ent["das_concentration"]} recovered {ent["fraction_of_full_recovered"]} | '
              f'moved {moved:.3f} loss {first:+.5f}->{last:+.5f} healthy {healthy} '
              f'({time.time()-t0:.0f}s)', flush=True)
    out[tag] = rec



# ---- PER-SEED PAIRWISE COSINES (no ablation tables: only the geometry is in question)
COS = []
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
    pr = [float(abs(QD[a] @ QD[b])) for x, a in enumerate(TARGETS) for b in TARGETS[x + 1:]]
    mc = sum(pr) / len(pr)
    COS.append(mc)
    PERSEED[sd_] = {'mean_pairwise_abs_cos': round(mc, 4),
                    'min': round(min(pr), 4), 'max': round(max(pr), 4)}
    print(f'  seed {sd_}: mean pairwise |cos| {mc:.4f}  (min {min(pr):.4f} max {max(pr):.4f})  '
          f'({time.time()-t0:.0f}s)', flush=True)

m = sum(COS) / len(COS)
sdv = (sum((x - m) ** 2 for x in COS) / max(1, len(COS) - 1)) ** 0.5
unhealthy = [h for h in HEALTH if not h[2]]
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude', 'component': COMPONENT,
       'seeds': list(SEEDS), 'circuits': TARGETS,
       'checks': "S2067 pred_a (0.3976) and S2068 pred_a (0.3896), both single-seed against a 0.50 bar",
       'per_seed': PERSEED, 'values': [round(x, 4) for x in COS],
       'mean': round(m, 4), 'sd': round(sdv, 4),
       'S2067_single_seed': 0.3976, 'bar': 0.50,
       'unhealthy_fits': unhealthy,
       'pred_a_cosine_is_stable': bool(sdv < 0.05),
       'pred_b_S2067_S2068_stand': bool(m < 0.50),
       'pred_c_optimiser_healthy': bool(not unhealthy),
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/DAS_M16_COS_SEEDS.json', 'w'), indent=1)
print(f'\nwrote circuits/DAS_M16_COS_SEEDS.json ({time.time()-t0:.0f}s)')
if unhealthy:
    print(f'  UNHEALTHY FITS {unhealthy} -- reporting nothing (LESSON 108)')
else:
    print(f'pred_a  sd of mean pairwise |cos| {sdv:.4f} (bar <0.05) : {rep["pred_a_cosine_is_stable"]}')
    print(f'pred_b  mean {m:.4f} stays below the 0.50 bar (S2067 single seed 0.3976) : '
          f'{rep["pred_b_S2067_S2068_stand"]}')
    print(f'pred_c  every fit healthy : {rep["pred_c_optimiser_healthy"]}')
    print(f'        per-seed: {[round(x, 4) for x in COS]}')
