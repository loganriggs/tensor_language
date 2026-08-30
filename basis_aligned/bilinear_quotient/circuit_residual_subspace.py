# DO a8's FIVE CIRCUITS DIFFER AT ALL, ONCE THE SHARED DIRECTION IS REMOVED?
#
# §2056 found a8's five circuit directions 0.894-parallel and no rank-1 direction selective for its own
# circuit: a8 runs one dominant mechanism that five census leaves each see a slice of. Its open question
# was precise -- "if the five differ at all, they differ in a subspace ORTHOGONAL to this dominant
# direction" -- and that is what this measures.
#
# Method. Take the five member-minus-offslice directions at a8, extract their shared component as the top
# principal direction of the set, project it out of all five, and re-ask §2056's two questions in the
# residual: are the residual directions near-orthogonal, and is rank-1 ablation along a residual direction
# selective for its own circuit? Then measure how much effect the residual carries at all.
#
# THIS IS NOT GRADIENT-DESCENT DAS. Proper DAS (Geiger/Wu, and das_class_learned.py in this directory)
# learns an orthonormal subspace by optimising an interchange objective and can find structure a
# closed-form residual misses. This is the cheap closed-form probe of the same question, and it is
# labelled as such: a negative here does not rule out a subspace that DAS could find.
#
# REGISTERED PREDICTIONS (before running):
#   pred_a  After removing the shared direction, the five residual directions are near-orthogonal:
#           mean pairwise |cos| < 0.5. If FALSE the five share MORE than one direction and the collapse
#           §2056 reported is even stronger than a rank-1 story.
#   pred_b  Rank-1 ablation along a residual direction becomes SELECTIVE -- own concentration exceeds the
#           mean over the other four -- for at least 3 of 5. In §2056 this held for only 1 of 5. If TRUE
#           the five circuits do differ, in a subspace the dominant direction was masking.
#   pred_c  But the residual carries much less: each residual direction's own concentration is below half
#           its §2056 whole-direction value. The shared direction should be where the effect lives.
#
# Writes circuits/RESIDUAL.json. Read-only with respect to circuit files.
import json
import time

import torch
import torch.nn.functional as F

import census_lib as C

A8 = ['r.11.1.1', 'r.11.1.2', 'r.11.3.1', 'r.23.2.1', 'r.23.2.3']
KEY = 'a8'
# §2056's own-circuit concentrations for the full (unprojected) directions
FULL = {'r.11.1.1': 4.372, 'r.11.1.2': 7.279, 'r.11.3.1': 4.103, 'r.23.2.1': 3.099, 'r.23.2.3': 3.649}

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()
masks = {}
for t in A8:
    lf = C.leaf(t)
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    masks[t] = (mm, sl)


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
dirs = {}
for t in A8:
    mm, sl = masks[t]
    d = acts[mm].mean(0) - acts[~sl].mean(0)
    dirs[t] = d / d.norm()
Dm = torch.stack([dirs[t] for t in A8])
U, S, Vh = torch.linalg.svd(Dm, full_matrices=False)
shared = Vh[0] / Vh[0].norm()
share_frac = float((S[0] ** 2) / (S ** 2).sum())
print(f'shared direction explains {share_frac:.4f} of the five directions\' variance', flush=True)

res = {}
for t in A8:
    r = dirs[t] - torch.dot(dirs[t], shared) * shared
    res[t] = r / r.norm()
cos = {a: {b: round(float(torch.dot(res[a], res[b])), 4) for b in A8} for a in A8}
off = [abs(cos[a][b]) for i, a in enumerate(A8) for b in A8[i + 1:]]
mean_cos = sum(off) / len(off)
print(f'residual directions: mean pairwise |cos| = {mean_cos:.4f}  (bar for separate < 0.5)', flush=True)

conc = {}
for t in A8:
    d = project_out_dce(KEY, res[t])
    row = {}
    for u in A8:
        mm, sl = masks[u]
        am = float(d[mm].abs().mean()); ag = float(d[~sl].abs().mean())
        row[u] = round(am / ag, 3) if ag > 0 else None
    conc[t] = row
    others = [row[u] for u in A8 if u != t]
    print(f'  residual along {t}: own {row[t]}  mean(others) {sum(others)/len(others):.3f}  '
          f'selective={row[t] > sum(others)/len(others)}  (full was {FULL[t]})  ({time.time()-t0:.0f}s)',
          flush=True)

sel = sum(1 for t in A8 if conc[t][t] > sum(conc[t][u] for u in A8 if u != t) / 4)
shrunk = sum(1 for t in A8 if conc[t][t] < 0.5 * FULL[t])
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude, circuit task',
       'method': 'a8 member-minus-offslice directions; shared component removed as the top right-singular '
                 'vector of the five; residual directions re-tested for orthogonality and for selective '
                 'rank-1 ablation. NOT gradient-descent DAS -- closed-form probe of the same question.',
       'component': KEY, 'circuits': A8,
       'shared_direction_variance_fraction': round(share_frac, 4),
       'residual_mean_pairwise_abs_cos': round(mean_cos, 4),
       'residual_cos': cos, 'residual_rank1_concentration': conc,
       'full_direction_concentration_S2056': FULL,
       'n_selective_of_5': sel, 'n_shrunk_below_half_of_5': shrunk,
       'note': 'read-only; no circuit file modified'}
json.dump(rep, open('circuits/RESIDUAL.json', 'w'), indent=1)
print(f'\npred_a residual mean |cos| {mean_cos:.4f} < 0.5 : {mean_cos < 0.5}')
print(f'pred_b selective for own circuit: {sel}/5  (bar >=3; §2056 had 1/5)')
print(f'pred_c residual concentration below half the full: {shrunk}/5')
print(f'wrote circuits/RESIDUAL.json ({time.time()-t0:.0f}s)', flush=True)
