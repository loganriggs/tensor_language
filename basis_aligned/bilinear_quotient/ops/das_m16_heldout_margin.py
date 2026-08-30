# DOES r.1.1.1'S SELECTIVITY SURVIVE ROWS ITS DIRECTION WAS NOT TRAINED ON?
#
# RUNG 3: the open question named at the end of §2068.
#
# §2067 and §2068 leave exactly one loose end at m16. Five of its six circuits are causally
# indistinguishable (0 of 5 selective at a 10% margin) while sitting at the same angles as each other;
# `r.1.1.1` alone is selective, at own 2.015 against a mean-of-others 1.776 -- a 13.5% margin, above the
# bar but not far above it, and the single fact standing between m16 and a clean "one mechanism".
#
# There is a specific reason to doubt it that has nothing to do with the circuit. Every DAS direction in
# §2067/§2068 was TRAINED on rows 0-600, and the cross-circuit concentration table was then computed over
# the WHOLE grid -- training rows included. That is the shape LESSON 106 and §2061 were written about, and
# §2067's conclusion turns on the one number most exposed to it.
#
# This recomputes the table on rows 600-1000 ONLY, which no direction was trained on, and reports the
# fit-row table beside it so the size of any contamination is visible rather than inferred.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  r.1.1.1 SURVIVES: its own-vs-others margin on held-out rows alone stays above 10%. Registered
#           in the direction that the effect is real, because §2061 measured the analogous census
#           selection and found NO inflation (median held-out/in-sample ratio 1.0217) -- so absent
#           contamination this should hold. If FALSE, §2067's "1 of 6 selective" was partly an artefact of
#           scoring directions on the rows that trained them, m16 goes to 0 of 6, and "one mechanism"
#           becomes exceptionless rather than nearly so.
#   pred_b  The other five stay at 0 of 5 on held-out rows. §2068 found zero on the full grid; a circuit
#           APPEARING once training rows are removed would be a new effect needing its own explanation.
#   pred_c  Contamination is measurable but small: the fit-row margin exceeds the held-out margin for at
#           least 4 of the 6 circuits, and the median held-out/fit margin ratio is >= 0.80 -- the same bar
#           §2061 used. This separates "the direction memorised its training rows" from "the direction
#           found something", and tells me which of pred_a's outcomes to believe.
#
# Writes circuits/DAS_M16_HELDOUT.json. DISCOVERY ONLY. No circuit file is modified.
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
def project_out_dce(key, u, lo=0, hi=None):
    """dCE from removing direction u, over rows [lo, hi) only -- so a direction can be scored on rows
    it was never trained on."""
    R_ = R; u = u.to(C.DEV); ces = []
    base_full = C.base_ce()
    hi = R_.shape[0] if hi is None else hi
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


TG = [t for t in TARGETS if t in QDIR]
FLAT = {}
for t in TG:
    mm, sl = leaf_masks(t)
    FLAT[t] = (mm.reshape(-1), sl.reshape(-1))

pairs = [float(abs(QDIR[a] @ QDIR[b])) for i, a in enumerate(TG) for b in TG[i + 1:]]
mean_cos = sum(pairs) / len(pairs) if pairs else float('nan')

def build_table(lo, hi, label):
    tab = {}
    for src in TG:
        d = project_out_dce(COMPONENT, QDIR[src], lo, hi)
        row = {}
        for tgt in TG:
            mm, sl = FLAT[tgt]
            m_ = mm[lo * NP:hi * NP]; s_ = sl[lo * NP:hi * NP]
            am = float(d[m_].abs().mean()); ag = float(d[~s_].abs().mean())
            row[tgt] = round(am / ag, 3) if ag > 0 else None
        tab[src] = row
        oth = [v for k, v in row.items() if k != src and v is not None]
        mo = sum(oth) / len(oth)
        print(f'  [{label}] {src:10s}: own {row[src]:6.3f}  others {mo:6.3f}  margin '
              f'{row[src]/mo:5.3f}  selective(10%)={row[src] > 1.10 * mo}', flush=True)
    return tab


TAB = build_table(EVAL_ROWS[0], EVAL_ROWS[1], 'heldout')
FITTAB = build_table(TRAIN_ROWS[0], TRAIN_ROWS[1], 'fitrows')


def n_sel(tab, margin=1.10):
    n = 0
    for src in TG:
        oth = [v for k, v in tab[src].items() if k != src and v is not None]
        if tab[src][src] is not None and oth and tab[src][src] > margin * (sum(oth) / len(oth)):
            n += 1
    return n


def margin(tab, t):
    oth = [v for k, v in tab[t].items() if k != t and v is not None]
    return tab[t][t] / (sum(oth) / len(oth))


R111 = 'r.1.1.1'
mh = {t: margin(TAB, t) for t in TG}
mf = {t: margin(FITTAB, t) for t in TG}
ratios = sorted(mh[t] / mf[t] for t in TG)
med_ratio = ratios[len(ratios) // 2]
n_fit_higher = sum(1 for t in TG if mf[t] > mh[t])
others_sel = sum(1 for t in TG if t != R111 and mh[t] > 1.10)
unhealthy = [(t, r) for t, v in out.items() for r, e in v['ranks'].items()
             if not e.get('optimiser_healthy')]

rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude', 'component': COMPONENT, 'circuits': TG,
       'question': "S2068's open question: does r.1.1.1's 13.5% margin survive rows its direction was "
                   "not trained on?",
       'method': 'DAS directions trained on rows 0-600 as in S2067/S2068; cross-circuit rank-1 '
                 'projection-ablation tables computed separately on held-out rows 600-1000 and on the fit '
                 'rows, so contamination is measured rather than assumed',
       'margins_heldout': {t: round(mh[t], 4) for t in TG},
       'margins_fitrows': {t: round(mf[t], 4) for t in TG},
       'heldout_over_fit_ratio_median': round(med_ratio, 4),
       'n_circuits_fit_margin_higher': n_fit_higher,
       'r_1_1_1_heldout_margin': round(mh.get(R111, float('nan')), 4),
       'concentration_heldout': TAB, 'concentration_fitrows': FITTAB,
       'optimiser_unhealthy_fits': unhealthy,
       'pred_a_r111_survives_heldout': bool(mh.get(R111, 0) > 1.10),
       'pred_b_others_still_zero': bool(others_sel == 0),
       'pred_c_contamination_small': bool(n_fit_higher >= 4 and med_ratio >= 0.80),
       'per_circuit': out, 'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/DAS_M16_HELDOUT.json', 'w'), indent=1)

print(f'\nwrote circuits/DAS_M16_HELDOUT.json ({time.time()-t0:.0f}s)')
if unhealthy:
    print(f'  UNHEALTHY FITS {unhealthy} -- reporting nothing (LESSON 108)')
else:
    print(f'pred_a  r.1.1.1 held-out margin {mh.get(R111, float("nan")):.4f} (bar >1.10; full grid gave '
          f'1.135) : {rep["pred_a_r111_survives_heldout"]}')
    print(f'pred_b  other circuits selective on held-out rows: {others_sel}/{len(TG)-1} (bar ==0) : '
          f'{rep["pred_b_others_still_zero"]}')
    print(f'pred_c  fit margin higher for {n_fit_higher}/{len(TG)} (bar >=4), median heldout/fit ratio '
          f'{med_ratio:.4f} (bar >=0.80) : {rep["pred_c_contamination_small"]}')
    print('        held-out margins: ' + ', '.join(f'{t} {mh[t]:.3f}' for t in TG))
