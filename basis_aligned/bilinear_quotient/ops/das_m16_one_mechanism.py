# IS m16 GENUINELY ONE MECHANISM? THE TEST THAT SETTLED IT FOR a8, RUN AT m16.
#
# RUNG 3: the open question named at the end of §2064.
#
# §2064 found m16 is the first component where "one mechanism seen several ways" is LIVE rather than
# retracted: its six circuits have a shared direction carrying 0.9567 of their directional variance, their
# full directions are non-selective (2/6 at a 10% margin), and -- unlike a8 -- projecting the shared
# direction out reveals nothing (residual |cos| stays 0.5185, selectivity stays 2/6). §2065 confirmed
# across six components that a8's reversal is unique, so m16 is not a failed a8; it is a different thing.
#
# That is exactly the claim §2058 made for a8 and had to RETRACT, and what retracted it was not the
# closed-form probe -- it was noticing that a large shared component can mask selective structure
# underneath. At a8 the closed-form residual found that structure. At m16 the closed-form residual finds
# nothing, and the honest question is whether there is nothing to find or whether the closed form is the
# wrong instrument. §2060 established that gradient-descent DAS and the closed-form mean-difference
# direction find overlapping but genuinely DIFFERENT directions (overlap 0.006-0.336, up to 390x random
# but never near identity), so DAS can see what the closed form misses. It has never been run at m16.
#
# This is therefore the strongest available test of "m16 is one mechanism": if a learned subspace cannot
# separate the six circuits either, the claim survives its best challenge instead of merely surviving a
# weak one.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  DAS FAILS to separate them, as the closed form did: the six learned rank-1 directions have
#           mean pairwise |cos| >= 0.50, i.e. they stay entangled at the same level the closed-form
#           residuals did (0.5185). Registered in the direction that CONFIRMS §2064 rather than the one I
#           would prefer to find, because a negative here is the load-bearing claim and it should be the
#           one carrying the risk. If FALSE, gradient descent separates what the mean difference could
#           not, and "m16 is one mechanism" is retracted the same way §2058's a8 version was.
#   pred_b  And they are not individually selective: fewer than 4 of the 6 learned directions concentrate
#           on their own circuit above the mean of the others at a >= 10% margin -- the same bar and the
#           same margin discipline §2062/§2065 use.
#   pred_c  CONTROL, and the one that decides whether pred_a and pred_b mean anything: every fit passes
#           the optimiser health gate (moved from init, loss decreased). LESSON 108 exists because a DAS
#           run whose subspace never moved produced ten circuits of clean-looking numbers; a NULL result
#           here is worthless unless the optimiser demonstrably worked, and a null is what pred_a
#           predicts. If FALSE the run reports nothing at all.
#
# Writes circuits/DAS_M16.json. DISCOVERY ONLY. No circuit file is modified.
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
t0 = time.time()
for tag in TARGETS:
    key = BAT['by_tag'][tag]['best_mean']
    mm, sl = leaf_masks(tag)
    acts_tr = capture(key, *TRAIN_ROWS)
    acts_ev = capture(key, *EVAL_ROWS)
    rec = {'component': key, 'members': int(mm.sum()), 'ranks': {}}
    for r in RANKS:
        g = torch.Generator(device='cpu').manual_seed(SEED + r)
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



# ---- CROSS-CIRCUIT ANALYSIS: do the six learned directions separate the six circuits?
# Same instrument as §2062/§2065 (rank-1 projection ablation, >=10% selectivity margin) so the numbers
# sit beside the closed-form ones rather than needing a translation.
@torch.no_grad()
def project_out_dce(key, u):
    R_ = R; u = u.to(C.DEV); ces = []
    base_full = C.base_ce()
    for i in range(0, R_.shape[0], 4):
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
    return torch.cat(ces).float() - base_full


TG = [t for t in TARGETS if t in QDIR]
FLAT = {}
for t in TG:
    mm, sl = leaf_masks(t)
    FLAT[t] = (mm.reshape(-1), sl.reshape(-1))

pairs = [float(abs(QDIR[a] @ QDIR[b])) for i, a in enumerate(TG) for b in TG[i + 1:]]
mean_cos = sum(pairs) / len(pairs) if pairs else float('nan')

TAB = {}
for src in TG:
    d = project_out_dce(COMPONENT, QDIR[src])
    row = {}
    for tgt in TG:
        mm, sl = FLAT[tgt]
        am = float(d[mm].abs().mean()); ag = float(d[~sl].abs().mean())
        row[tgt] = round(am / ag, 3) if ag > 0 else None
    TAB[src] = row
    oth = [v for k, v in row.items() if k != src and v is not None]
    print(f'  DAS dir of {src:10s}: own {row[src]:6.3f}  mean(others) {sum(oth)/len(oth):6.3f}  '
          f'selective(10%)={row[src] > 1.10 * (sum(oth)/len(oth))}', flush=True)


def n_sel(tab, margin=1.10):
    n = 0
    for src in TG:
        oth = [v for k, v in tab[src].items() if k != src and v is not None]
        if tab[src][src] is not None and oth and tab[src][src] > margin * (sum(oth) / len(oth)):
            n += 1
    return n


nsel = n_sel(TAB)
unhealthy = [(t, r) for t, v in out.items() for r, e in v['ranks'].items()
             if not e.get('optimiser_healthy')]
healthy_all = not unhealthy

CLOSED = {'residual_mean_abs_cos': 0.5185, 'residual_selective_10pct': '2/6',
          'shared_variance_explained': 0.9567}
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude',
       'component': COMPONENT, 'circuits': TG,
       'question': "S2064's open question: are m16's circuits genuinely ONE mechanism, or does the "
                   "closed-form residual simply fail to see the structure that gradient descent can?",
       'method': 'rank-1 DAS per circuit at m16 (interchange restricted to the learned direction, trained '
                 'on rows 0-600); the learned directions are then compared pairwise and used for rank-1 '
                 'projection ablation over the full grid, the same instrument as S2062/S2065',
       'closed_form_reference_S2064': CLOSED,
       'das_mean_pairwise_abs_cos': round(mean_cos, 4),
       'das_selective_10pct': f'{nsel}/{len(TG)}',
       'concentration_das_directions': TAB,
       'optimiser_unhealthy_fits': unhealthy,
       'pred_a_das_also_fails_to_separate': bool(mean_cos >= 0.50),
       'pred_b_das_directions_not_selective': bool(nsel < 4),
       'pred_c_optimiser_healthy': bool(healthy_all),
       'per_circuit': out,
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/DAS_M16.json', 'w'), indent=1)

print(f'\nwrote circuits/DAS_M16.json ({time.time()-t0:.0f}s)')
print(f'pred_c  CONTROL -- every fit passed the optimiser health gate: {healthy_all}'
      f'{"" if healthy_all else "  UNHEALTHY: " + str(unhealthy)}')
if not healthy_all:
    print('        A null result from an optimiser that did not work is worthless. Reporting nothing.')
else:
    print(f'pred_a  DAS directions stay entangled: mean pairwise |cos| {mean_cos:.4f} '
          f'(bar >=0.50; closed-form residual 0.5185) : {mean_cos >= 0.50}')
    print(f'pred_b  and are not individually selective: {nsel}/{len(TG)} at a 10% margin '
          f'(bar <4; closed-form residual 2/6) : {nsel < 4}')
    print(f'        -> m16 is ONE MECHANISM survives its best challenge: '
          f'{mean_cos >= 0.50 and nsel < 4}')
