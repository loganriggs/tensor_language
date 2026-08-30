# IS §2058'S SHARED-SUBSTRATE STRUCTURE A PROPERTY OF a8, OR OF THE MODEL?
#
# RUNG 3: the open question named at the end of §2059 -- "a16 is a8's near-twin and has had no analysis
# at all". §2056 and §2058 established, at a8 alone, a specific two-level structure: the circuits sharing
# a component have (i) full directions that are strongly parallel (mean pairwise |cos| 0.894) and
# individually NON-selective (1 of 5), and (ii) once the dominant shared direction is projected out --
# 91.6% of the directional variance -- residuals that are near-ORTHOGONAL (0.359) and mostly SELECTIVE
# (4 of 5). "Five circuits on a common substrate, not one circuit seen five ways."
#
# That was measured on ONE component, at five circuits chosen because they happened to be there. §2059
# then found a8 is the densest component in the model (16 circuits) and a16 its near-twin (13) -- so the
# generality question is now answerable on a second, independent, comparably-dense component.
#
# The circuit list is taken from circuits/BATTERY.json (every curated circuit whose best mean-ablation
# component is a16), not hand-picked -- circuit_subspace_separation.py still names only three a16
# circuits, which predates the battery.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  a16's circuits also have ONE dominant shared direction, explaining >= 0.80 of the variance of
#           their per-circuit directions (a8 measured 0.9161). If FALSE, the shared substrate is a fact
#           about a8 rather than about components that carry many circuits.
#   pred_b  Their full directions are strongly parallel too: mean pairwise |cos| >= 0.70 (a8: 0.8942).
#   pred_c  And the §2058 reversal reproduces: after projecting out the shared direction, the residual
#           directions are near-orthogonal (mean pairwise |cos| < 0.50, a8: 0.3587) AND rank-1 projection
#           ablation along a residual is selective for its own circuit in >= 60% of them (a8: 4/5 = 80%,
#           against 1/5 for the full directions). If TRUE, the two-level structure is a general property
#           of circuit-dense components and §2058 generalises. If FALSE it is a8-specific and §2058 must
#           be read as a single-component result.
#
# Writes circuits/A16.json. DISCOVERY ONLY. No circuit file is modified.
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
if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    b = json.load(open(os.path.join(BQ, 'circuits/BATTERY.json')))
    n = sum(1 for v in b['by_tag'].values() if v['best_mean'] == 'a16')
    if n < 3:
        print(f'DRYRUN FAIL: only {n} circuits localise to a16; nothing to compare')
        raise SystemExit(1)
    print(f'DRYRUN OK: BATTERY.json present, {n} circuits localise to a16')
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import census_lib as C                                                    # noqa: E402

KEY = 'a16'
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


def n_sel(tab):
    n = 0
    for src in TAGS:
        oth = [v for k, v in tab[src].items() if k != src and v is not None]
        if tab[src][src] is not None and oth and tab[src][src] > sum(oth) / len(oth):
            n += 1
    return n


nf, nr = n_sel(full_tab), n_sel(res_tab)
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude',
       'component': KEY, 'circuits': TAGS, 'a8_reference': A8_REF,
       'method': 'per-circuit direction = unit(mean over members - mean off slice) in the component '
                 'output space; shared direction = top right-singular vector of the stacked directions; '
                 'residual = direction with the shared component removed; selectivity by rank-1 '
                 'projection ablation along each direction',
       'shared_variance_explained': round(share_var, 4),
       'mean_pairwise_abs_cos_full': round(mean_full, 4),
       'mean_pairwise_abs_cos_residual': round(mean_res, 4),
       'full_selective': f'{nf}/{len(TAGS)}', 'residual_selective': f'{nr}/{len(TAGS)}',
       'pred_a_shared_direction_dominates': bool(share_var >= 0.80),
       'pred_b_full_directions_parallel': bool(mean_full >= 0.70),
       'pred_c_residuals_orthogonal_and_selective': bool(mean_res < 0.50
                                                         and nr >= 0.60 * len(TAGS)),
       'cos_full': full_cos, 'cos_residual': res_cos,
       'concentration_full': full_tab, 'concentration_residual': res_tab,
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/A16.json', 'w'), indent=1)

print(f'\nwrote circuits/A16.json ({time.time()-t0:.0f}s)   [{len(TAGS)} circuits at {KEY}]')
print(f'pred_a  shared direction explains {share_var:.4f} of variance (bar >=0.80, a8 0.9161) : '
      f'{share_var >= 0.80}')
print(f'pred_b  full directions mean |cos| {mean_full:.4f} (bar >=0.70, a8 0.8942) : {mean_full >= 0.70}')
print(f'pred_c  residual |cos| {mean_res:.4f} (bar <0.50, a8 0.3587) AND residual selective '
      f'{nr}/{len(TAGS)} (bar >=60%, a8 4/5) : '
      f'{rep["pred_c_residuals_orthogonal_and_selective"]}')
print(f'        for contrast, FULL directions were selective {nf}/{len(TAGS)} (a8: 1/5)')
