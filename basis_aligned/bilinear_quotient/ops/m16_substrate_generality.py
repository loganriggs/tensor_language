# WHICH IS TYPICAL -- a8's SHARED SUBSTRATE, OR a16's ALREADY-SEPARATE CIRCUITS?
#
# RUNG 3: the open question named at the end of §2062.
#
# §2056/§2058 found at a8 that circuits sharing a component have strongly parallel, individually
# NON-selective directions dominated by one shared direction (91.6% of the variance), separating only once
# it is projected out (1/5 selective -> 4/5). §2062 ran the identical test at a16 and found the OPPOSITE:
# no dominant shared direction (48.9%), directions not parallel (0.427), and circuits ALREADY individually
# selective before removal (11/13 raw, 7/13 at an honest 10% margin), with removal HURTING (-> 6/13).
#
# Two components, two opposite geometries, and no way yet to say which is typical of bilin18. m16 is the
# third-densest component in §2059's census (6 circuits) and is an MLP rather than an attention site,
# which makes it the informative third case: if the dichotomy tracks the attention/MLP distinction rather
# than being idiosyncratic, m16 should differ from BOTH attention sites measured so far.
#
# This is §2062's script with KEY changed and nothing else, so the comparison is instrument-identical.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  m16 follows the a16 pattern rather than the a8 one: the shared direction explains < 0.80 of
#           the variance of the per-circuit directions. Registered this way round because two of the three
#           densest components would then agree, making a8 the special case rather than the rule -- the
#           reading §2062 leaves as most likely, and the one this can falsify.
#   pred_b  Its full directions are correspondingly NOT strongly parallel: mean pairwise |cos| < 0.70.
#   pred_c  And its circuits are already individually selective BEFORE any removal, at a >= 10% margin for
#           at least half of them -- the honest bar §2062 had to retrofit onto its own headline, used here
#           as the registered one from the start. If pred_a and pred_b hold with pred_c, the a8
#           arrangement is the exception among the three densest components. If m16 instead reproduces a8,
#           two of three share a substrate, a16 is the outlier, and §2062's "single-component result"
#           framing is the thing needing revision -- which I would report as written.
#
# Writes circuits/M16.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

# PLAN PRE-FLIGHT (LESSON 109): census_lib builds MODS from the live model at import, so enqueue's
# BQLIB_DRYRUN gate must be answered BEFORE that import or the gate runs the experiment for real.
KEY = 'm16'                                          # single source of truth: the guard and the body
#                                                      must never disagree about which component this is

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    b = json.load(open(os.path.join(BQ, 'circuits/BATTERY.json')))
    n = sum(1 for v in b['by_tag'].values() if v['best_mean'] == KEY)
    if n < 3:
        print(f'DRYRUN FAIL: only {n} circuits localise to {KEY}; nothing to compare')
        raise SystemExit(1)
    print(f'DRYRUN OK: BATTERY.json present, {n} circuits localise to {KEY}')
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import census_lib as C                                                    # noqa: E402

A8_REF = {'shared_variance': 0.9161, 'full_cos': 0.8942, 'resid_cos': 0.3587,
          'full_selective': '1/5', 'resid_selective': '4/5'}

BAT = json.load(open('circuits/BATTERY.json'))
CAND = [t for t, v in BAT['by_tag'].items() if v['best_mean'] == KEY]

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()

keep, masks = [], {}
for t in CAND:
    try:
        lf = C.leaf(t)
    except Exception:
        continue
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    if mm.sum() == 0 or (~sl).sum() == 0:
        continue                                     # a root node has no off-slice; §2057's NaN trap
    keep.append(t); masks[t] = (mm, sl)
TAGS = keep                                          # single binding; CAND is the pre-filter candidate list
print(f'grid {nflat}; {len(TAGS)} of {len(CAND)} candidates usable at {KEY}: {TAGS}', flush=True)
assert len(TAGS) >= 3, 'need at least three circuits to speak of a shared direction'


@torch.no_grad()
def capture(key):
    R = C.rows(); cap = []
    h = C.MODS[key].register_forward_hook(
        lambda mo, i_, o_: cap.append(((o_[0] if isinstance(o_, tuple) else o_)
                                       .detach().float().reshape(-1, C.D).cpu())))
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    return torch.cat(cap)


@torch.no_grad()
def project_out_dce(key, u):
    R = C.rows(); u = u.to(C.DEV); ces = []
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)

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
    return torch.cat(ces).float() - base


t0 = time.time()
acts = capture(KEY)
print(f'captured {KEY} over the grid ({time.time()-t0:.0f}s)', flush=True)

dirs = {}
for t in TAGS:
    mm, sl = masks[t]
    u = acts[mm].mean(0) - acts[~sl].mean(0)
    assert torch.isfinite(u).all() and u.norm() > 0, f'{t}: degenerate direction'
    dirs[t] = u / u.norm()

M = torch.stack([dirs[t] for t in TAGS])
full_cos = {a: {b: round(float(abs(dirs[a] @ dirs[b])), 4) for b in TAGS} for a in TAGS}
pairs = [full_cos[a][b] for i, a in enumerate(TAGS) for b in TAGS[i + 1:]]
mean_full = sum(pairs) / len(pairs)

U, S, Vh = torch.linalg.svd(M, full_matrices=False)
shared = Vh[0] / Vh[0].norm()
share_var = float((S[0] ** 2) / (S ** 2).sum())

resid = {}
for t in TAGS:
    r = dirs[t] - (dirs[t] @ shared) * shared
    resid[t] = r / r.norm()
res_cos = {a: {b: round(float(abs(resid[a] @ resid[b])), 4) for b in TAGS} for a in TAGS}
rpairs = [res_cos[a][b] for i, a in enumerate(TAGS) for b in TAGS[i + 1:]]
mean_res = sum(rpairs) / len(rpairs)
print(f'shared direction explains {share_var:.4f}; full |cos| {mean_full:.4f}; '
      f'residual |cos| {mean_res:.4f} ({time.time()-t0:.0f}s)', flush=True)


def conc_table(vecs, label):
    """concentration of each circuit under rank-1 projection ablation along each circuit's direction"""
    out = {}
    for src in TAGS:
        d = project_out_dce(KEY, vecs[src])
        row = {}
        for tgt in TAGS:
            mm, sl = masks[tgt]
            am = float(d[mm].abs().mean()); ag = float(d[~sl].abs().mean())
            row[tgt] = round(am / ag, 3) if ag > 0 else None
        out[src] = row
        own = row[src]; oth = [v for k, v in row.items() if k != src and v is not None]
        print(f'  {label} along {src:10s}: own {own:6.3f}  mean(others) '
              f'{sum(oth)/len(oth):6.3f}  selective={own > sum(oth)/len(oth)} '
              f'({time.time()-t0:.0f}s)', flush=True)
    return out


full_tab = conc_table(dirs, 'full   ')
res_tab = conc_table(resid, 'residual')


def n_sel(tab, margin=1.0):
    """circuits whose own concentration exceeds the mean of the others by at least `margin`.

    §2062 reported raw selectivity (margin 1.0) and then had to retrofit a 10% bar, because 4 of its 11
    'selective' circuits were within 5% of a tie. Here the honest bar is the registered one.
    """
    n = 0
    for src in TAGS:
        oth = [v for k, v in tab[src].items() if k != src and v is not None]
        if tab[src][src] is not None and oth and tab[src][src] > margin * (sum(oth) / len(oth)):
            n += 1
    return n


nf, nr = n_sel(full_tab), n_sel(res_tab)
nf10, nr10 = n_sel(full_tab, 1.10), n_sel(res_tab, 1.10)
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude',
       'component': KEY, 'circuits': TAGS, 'a8_reference': A8_REF,
       'method': 'per-circuit direction = unit(mean over members - mean off slice) in the component '
                 'output space; shared direction = top right-singular vector of the stacked directions; '
                 'residual = direction with the shared component removed; selectivity by rank-1 '
                 'projection ablation along each direction',
       'shared_variance_explained': round(share_var, 4),
       'mean_pairwise_abs_cos_full': round(mean_full, 4),
       'mean_pairwise_abs_cos_residual': round(mean_res, 4),
       'full_selective_raw': f'{nf}/{len(TAGS)}', 'residual_selective_raw': f'{nr}/{len(TAGS)}',
       'full_selective_10pct_margin': f'{nf10}/{len(TAGS)}',
       'residual_selective_10pct_margin': f'{nr10}/{len(TAGS)}',
       'pred_a_follows_a16_not_a8': bool(share_var < 0.80),
       'pred_b_directions_not_parallel': bool(mean_full < 0.70),
       'pred_c_already_selective_before_removal': bool(nf10 >= 0.50 * len(TAGS)),
       'cos_full': full_cos, 'cos_residual': res_cos,
       'concentration_full': full_tab, 'concentration_residual': res_tab,
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/M16.json', 'w'), indent=1)

print(f'\nwrote circuits/M16.json ({time.time()-t0:.0f}s)   [{len(TAGS)} circuits at {KEY}]')
print(f'pred_a  shared direction explains {share_var:.4f} (bar <0.80 = a16-like; a8 0.9161, '
      f'a16 0.4887) : {share_var < 0.80}')
print(f'pred_b  full directions mean |cos| {mean_full:.4f} (bar <0.70 = a16-like; a8 0.8942, '
      f'a16 0.4271) : {mean_full < 0.70}')
print(f'pred_c  FULL directions already selective before removal at a >=10% margin: '
      f'{nf10}/{len(TAGS)} (bar >= half) : {rep["pred_c_already_selective_before_removal"]}')
print(f'        raw (any margin): full {nf}/{len(TAGS)}, residual {nr}/{len(TAGS)}')
print(f'        at 10% margin   : full {nf10}/{len(TAGS)}, residual {nr10}/{len(TAGS)}')
print(f'        residual |cos| {mean_res:.4f}   [a8: full 1/5 -> residual 4/5; '
      f'a16: full 7/13 -> residual 6/13, both at the 10% margin]')
