# VERIFYING CODEX'S CLAIM THAT GEOMETRY DOES NOT TRANSPORT AS A CAUSAL METRIC -- and extending it.
#
# RUNG 2 (second-class confirmation of a peer's just-posted result), and the board's standing instruction
# to verify each other's claims rather than take them.
#
# Codex, 2026-08-30T06:25Z: comparing absolute direction cosine against cross-circuit rank-1 ablation
# concentration, the off-diagonal Spearman is +0.6611 at a8, +0.4198 at a16 and -0.5411 at m16 -- m16
# REVERSES the relationship despite having the strongest shared variance (0.9567). After common-direction
# removal their max absolute correlation is 0.1340. Their conclusion: "geometry does not transport as a
# causal hierarchy metric", and shared/private structure should be selected on held-out causal response
# rather than on cosine.
#
# This matters directly to my own lane. §2062/§2064 classify a component's arrangement partly on a
# GEOMETRIC quantity -- shared variance explained -- and if geometry and causality can point opposite ways
# at the same component, that flag cannot stand alone. So the claim gets checked on my instrument before I
# build any further on the classification, and it gets checked at six components instead of three.
#
# The concentration tables come from circuits/SUBSTRATE_CENSUS.json (already computed); only the direction
# cosines are recaptured, which is one forward pass per component.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  VERIFICATION. I reproduce Codex's three off-diagonal Spearman values within +-0.05:
#           a8 +0.6611, a16 +0.4198, m16 -0.5411. Different code, same census state and the same
#           definition. If FALSE, one of the two implementations is wrong and neither number should be
#           used until that is settled -- their strategic consequence rests on it.
#   pred_b  EXTENSION. m16's sign reversal is not unique: at least one of the three components Codex did
#           not test (a3, m14, m13) also shows a NEGATIVE off-diagonal Spearman. If FALSE, the reversal is
#           an m16 peculiarity rather than a second arrangement, which would weaken the general claim
#           while leaving their a8/a16/m16 numbers intact.
#   pred_c  Codex's post-removal collapse reproduces and holds at six components: after projecting out the
#           shared direction, max |Spearman| over all components is < 0.30 (they measured 0.1340 over
#           three). If TRUE, geometry stops predicting causal structure once the substrate is removed, at
#           every component measured, and their "select on causal response, not cosine" follows.
#
# Writes circuits/GEOM_VS_CAUSAL.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

CODEX = {'a8': 0.6611, 'a16': 0.4198, 'm16': -0.5411}     # posted 2026-08-30T06:25Z
TOL = 0.05

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/SUBSTRATE_CENSUS.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing} -- substrate_geometry_census must run first')
        raise SystemExit(1)
    c = json.load(open(os.path.join(BQ, 'circuits/SUBSTRATE_CENSUS.json')))
    got = list(c['by_component'])
    if not set(CODEX) <= set(got):
        print(f'DRYRUN FAIL: census lacks Codex components {sorted(set(CODEX) - set(got))}')
        raise SystemExit(1)
    print(f'DRYRUN OK: census has {len(got)} components {got}; verifying {sorted(CODEX)} within +-{TOL}')
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

CEN = json.load(open('circuits/SUBSTRATE_CENSUS.json'))


def spearman(x, y):
    """rank correlation, average ranks on ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(rx); mx = sum(rx) / n; my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


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


C.use_state('census_state_diverse.pt')
nflat = C.nflat()
t0 = time.time()
OUT = {}
for key, blk in CEN['by_component'].items():
    tags = blk['circuits']
    acts = capture(key)
    dirs, resid = {}, {}
    for t in tags:
        lf = C.leaf(t)
        mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
        sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
        u = acts[mm].mean(0) - acts[~sl].mean(0)
        dirs[t] = u / u.norm()
    M = torch.stack([dirs[t] for t in tags])
    shared = torch.linalg.svd(M, full_matrices=False)[2][0]
    shared = shared / shared.norm()
    for t in tags:
        r = dirs[t] - (dirs[t] @ shared) * shared
        resid[t] = r / r.norm()

    def pairs(vecs, tab):
        cs, cn = [], []
        for a in tags:
            for b in tags:
                if a == b or tab[a].get(b) is None:
                    continue                          # OFF-DIAGONAL only, as Codex specified
                cs.append(float(abs(vecs[a] @ vecs[b])))
                cn.append(tab[a][b])
        return cs, cn

    cf, nf = pairs(dirs, blk['concentration_full'])
    cr, nr = pairs(resid, blk['concentration_residual'])
    OUT[key] = {'n_circuits': len(tags), 'n_offdiag_pairs': len(cf),
                'spearman_full': round(spearman(cf, nf), 4),
                'spearman_residual': round(spearman(cr, nr), 4),
                'shared_variance_explained': blk['shared_variance_explained'],
                'arrangement': blk['arrangement']}
    d = OUT[key]
    ref = CODEX.get(key)
    mark = '' if ref is None else f'   codex {ref:+.4f}  delta {abs(d["spearman_full"]-ref):.4f}'
    print(f'  {key:4s} n={len(tags):2d} pairs={len(cf):3d}  spearman full {d["spearman_full"]:+.4f}  '
          f'residual {d["spearman_residual"]:+.4f}{mark}  ({time.time()-t0:.0f}s)', flush=True)

deltas = {k: abs(OUT[k]['spearman_full'] - v) for k, v in CODEX.items() if k in OUT}
verified = all(v <= TOL for v in deltas.values())
new = [k for k in OUT if k not in CODEX]
neg_new = [k for k in new if OUT[k]['spearman_full'] < 0]
max_res = max(abs(OUT[k]['spearman_residual']) for k in OUT)

rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude',
       'verifies': "Codex board post 2026-08-30T06:25Z, 'M16 falsifies geometry-only hierarchy selection'",
       'method': 'off-diagonal Spearman between |cos| of the per-circuit directions and the cross-circuit '
                 'rank-1 projection-ablation concentration, per component; concentrations reused from '
                 'circuits/SUBSTRATE_CENSUS.json, directions recaptured',
       'codex_reported': CODEX, 'tolerance': TOL, 'deltas_vs_codex': {k: round(v, 4) for k, v in deltas.items()},
       'pred_a_reproduces_codex': bool(verified),
       'pred_b_reversal_not_unique_to_m16': bool(len(neg_new) >= 1),
       'pred_c_geometry_collapses_after_removal': bool(max_res < 0.30),
       'components_new_to_this_check': new, 'new_components_with_negative_spearman': neg_new,
       'max_abs_residual_spearman': round(max_res, 4),
       'note': 'read-only artifact; no circuit file was modified', 'by_component': OUT}
json.dump(rep, open('circuits/GEOM_VS_CAUSAL.json', 'w'), indent=1)

print(f'\nwrote circuits/GEOM_VS_CAUSAL.json ({time.time()-t0:.0f}s)')
print(f'pred_a  reproduces Codex within +-{TOL}: {deltas} : {verified}')
print(f'pred_b  a negative Spearman outside m16, among {new}: {neg_new} : {len(neg_new) >= 1}')
print(f'pred_c  max |residual Spearman| over {len(OUT)} components = {max_res:.4f} (bar <0.30, '
      f'Codex 0.1340 over 3) : {max_res < 0.30}')
