# HOW MUCH OF r.1.1.1'S MARGIN IS THE SEED? THE VARIANCE NOBODY HAS EVER MEASURED FOR THESE FITS.
#
# RUNG 3: the open question named at the end of §2069.
#
# §2069 measured r.1.1.1's held-out selectivity margin at 1.1010 against a >1.10 bar -- clearing by
# 0.0010 -- and said plainly that the number carries no conclusion. The reason is not the row split, which
# §2069 already did correctly: it is that every DAS margin in §2067, §2068 and §2069 comes from a SINGLE
# initialisation, and the seed-to-seed spread of these fits has never been measured. A 0.001 margin is
# meaningless against an unknown spread, and quoting it either way would be quoting noise.
#
# Every circuit is re-fit at three seeds and each fit is scored on the held-out rows exactly as §2069
# scored its one. The output is a margin with a spread attached, which is what §2067-§2069 should have had
# from the start.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  The spread is LARGE relative to the thing it was asked to decide: the standard deviation of
#           r.1.1.1's held-out margin across seeds is >= 0.02, i.e. at least twenty times the 0.0010 by
#           which §2069's single fit cleared the bar. If TRUE, §2069's pass was inside noise and this
#           ledger should stop treating any single-seed DAS margin as decisive. If FALSE these fits are
#           far more reproducible than I expect and the 0.001 meant more than I credited.
#   pred_b  r.1.1.1's MEAN margin across the three seeds still exceeds 1.10 -- the effect survives
#           averaging even if any one seed does not. Registered because it is the question §2068 and
#           §2069 were actually trying to answer, and a spread measurement that does not also give a
#           better point estimate would leave it exactly where it was.
#   pred_c  The five others stay non-selective on average: none of their mean held-out margins exceeds
#           1.10. §2069 found all five below 1.0 with the sign against selectivity; if a second circuit
#           crosses once seeds are averaged, "m16 is one mechanism" needs more than r.1.1.1 explained.
#
# Writes circuits/DAS_M16_SEEDS.json. DISCOVERY ONLY. No circuit file is modified.
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



# ---- PER-SEED HELD-OUT MARGINS
@torch.no_grad()
def project_out_dce(key, u, lo, hi):
    R_ = R; u = u.to(C.DEV); ces = []
    base_full = C.base_ce()
    for i in range(lo, hi, 4):
        bb = R_[i:i + 4, :NP + 1].to(C.DEV)

        def fh(mo, i_, o_):
            y = o_[0] if isinstance(o_, tuple) else o_
            f = y.float().reshape(-1, C.D)
            f = f - (f @ u).unsqueeze(1) * u.unsqueeze(0)
            f = f.view_as(y).to(y.dtype)
            return (f, o_[1]) if isinstance(o_, tuple) else f

        hh = C.MODS[key].register_forward_hook(fh)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
        for blkm in C.m.transformer.h:
            x, v1 = blkm(x, v1, x0)
        lg = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
        ces.append(F.cross_entropy(lg.view(-1, lg.size(-1)), tg, reduction='none').cpu())
        hh.remove()
    v = torch.cat(ces).float()
    return v - base_full[lo * NP:lo * NP + v.numel()]


FLAT = {}
for t in TARGETS:
    mm, sl = leaf_masks(t)
    FLAT[t] = (mm.reshape(-1), sl.reshape(-1))

LO, HI = EVAL_ROWS
MARG = {t: [] for t in TARGETS}
HEALTH = []
for si, sd in enumerate(SEEDS):
    ACTIVE_SEED = sd
    QD = {}
    for tag in TARGETS:
        key = COMPONENT
        mm, sl = leaf_masks(tag)
        acts_tr = capture(key, *TRAIN_ROWS)
        acts_ev = capture(key, *EVAL_ROWS)
        g = torch.Generator(device='cpu').manual_seed(sd + 1)
        P0 = (torch.randn(C.D, 1, generator=g) / C.D ** 0.5).to(C.DEV)
        Q_init = torch.linalg.qr(P0)[0].detach().clone()
        P = P0.clone().requires_grad_(True)
        opt = torch.optim.Adam([P], lr=LR)
        gg = torch.Generator().manual_seed(sd)
        perm = torch.randperm(acts_tr.shape[0] * NP, generator=gg)
        flat = acts_tr.reshape(-1, C.D)
        step = 0; losses = []
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
                losses.append(float(loss.detach())); step += 1
        with torch.no_grad():
            Q = torch.linalg.qr(P)[0]
            moved = 1.0 - float((Q.T @ Q_init).pow(2).sum())
        first = sum(losses[:20]) / 20; last = sum(losses[-20:]) / 20
        HEALTH.append((sd, tag, moved > 0.02 and last < first))
        QD[tag] = Q[:, 0].detach().clone()
    for src in TARGETS:
        d = project_out_dce(COMPONENT, QD[src], LO, HI)
        row = {}
        for tgt in TARGETS:
            mm2, sl2 = FLAT[tgt]
            m2 = mm2[LO * NP:HI * NP]; s2 = sl2[LO * NP:HI * NP]
            am = float(d[m2].abs().mean()); ag = float(d[~s2].abs().mean())
            row[tgt] = am / ag if ag > 0 else None
        oth = [v for k, v in row.items() if k != src and v is not None]
        MARG[src].append(row[src] / (sum(oth) / len(oth)))
    print(f'  seed {sd}: ' + ', '.join(f'{t} {MARG[t][-1]:.4f}' for t in TARGETS)
          + f'  ({time.time()-t0:.0f}s)', flush=True)


def mean(v):
    return sum(v) / len(v)


def sd_(v):
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5


R111 = 'r.1.1.1'
unhealthy = [h for h in HEALTH if not h[2]]
sd111 = sd_(MARG[R111]); mean111 = mean(MARG[R111])
others_over = [t for t in TARGETS if t != R111 and mean(MARG[t]) > 1.10]
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude', 'component': COMPONENT,
       'seeds': list(SEEDS), 'circuits': TARGETS,
       'question': "S2069's open question: how large is the seed-to-seed spread of these DAS margins, "
                   "against the 0.0010 by which r.1.1.1 cleared its bar on a single fit?",
       'margins_by_seed': {t: [round(x, 4) for x in MARG[t]] for t in TARGETS},
       'mean_margin': {t: round(mean(MARG[t]), 4) for t in TARGETS},
       'sd_margin': {t: round(sd_(MARG[t]), 4) for t in TARGETS},
       'S2069_single_seed_r111': 1.1010, 'S2069_margin_over_bar': 0.0010,
       'unhealthy_fits': unhealthy,
       'pred_a_spread_swamps_the_margin': bool(sd111 >= 0.02),
       'pred_b_r111_mean_still_selective': bool(mean111 > 1.10),
       'pred_c_others_still_non_selective': bool(len(others_over) == 0),
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/DAS_M16_SEEDS.json', 'w'), indent=1)
print(f'\nwrote circuits/DAS_M16_SEEDS.json ({time.time()-t0:.0f}s)')
if unhealthy:
    print(f'  UNHEALTHY FITS {unhealthy} -- reporting nothing (LESSON 108)')
else:
    print(f'pred_a  sd of r.1.1.1 margin across seeds {sd111:.4f} (bar >=0.02; S2069 cleared its bar by '
          f'0.0010) : {rep["pred_a_spread_swamps_the_margin"]}')
    print(f'pred_b  mean r.1.1.1 margin {mean111:.4f} (bar >1.10) : '
          f'{rep["pred_b_r111_mean_still_selective"]}')
    print(f'pred_c  other circuits with mean margin >1.10: {others_over} (bar none) : '
          f'{rep["pred_c_others_still_non_selective"]}')
    print('        mean+-sd: ' + ', '.join(f'{t} {mean(MARG[t]):.3f}+-{sd_(MARG[t]):.3f}'
                                           for t in TARGETS))
