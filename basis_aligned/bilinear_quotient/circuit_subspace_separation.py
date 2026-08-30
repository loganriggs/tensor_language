# DOES a8 CARRY ITS FIVE CIRCUITS IN ONE SUBSPACE OR FIVE?
#
# §2054 and §2055 localised twelve previously-unlocalised circuits to five components, with attention 8
# taking five of them (r.11.1.1, r.11.1.2, r.11.3.1, r.23.2.1, r.23.2.3) and attention 16 taking three
# (r.3.0, r.3.0.2, r.4.1.1). Both methods ablate a WHOLE component, so neither can distinguish "a8 runs
# five separate circuits" from "a8 runs one thing that five census leaves each see a slice of".
#
# That distinction is what subspace analysis is for, and it is the question DAS answers. This is the
# cheap, closed-form version of it: for each circuit, the direction in a8's output space that separates
# its members from off-slice positions. If the five directions are near-parallel, the five circuits are
# one mechanism cut five ways. If near-orthogonal, a8 genuinely multiplexes.
#
# It then validates the directions causally: project each one out of a8's output (rank-1 ablation) and
# measure whether the damage lands on that circuit specifically or on all five alike.
#
# REGISTERED PREDICTIONS (before running):
#   pred_a  The five a8 directions are NOT mutually parallel: mean pairwise |cos| < 0.5. If FALSE, a8 runs
#           one mechanism and the five census leaves are slices of it, which would mean the census is
#           over-counting circuits at this site.
#   pred_b  Rank-1 projection ablation along a circuit's own direction concentrates on that circuit at
#           >= 2.0x -- a single direction out of 1152 reproduces a useful share of the whole-component
#           effect. If FALSE the circuit is not carried by a single direction and needs true multi-
#           dimensional DAS.
#   pred_c  Each direction concentrates MORE on its own circuit than the mean over the other four a8
#           circuits. This is the selectivity test and the one that would actually establish multiplexing;
#           pred_a alone only shows the directions differ, not that they are functionally separate.
#
# Writes circuits/SUBSPACE.json. Read-only with respect to circuit files.
import json
import time

import torch
import torch.nn.functional as F

import census_lib as C

A8 = ['r.11.1.1', 'r.11.1.2', 'r.11.3.1', 'r.23.2.1', 'r.23.2.3']
A16 = ['r.3.0', 'r.3.0.2', 'r.4.1.1']
GROUPS = {'a8': A8, 'a16': A16}

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()
masks = {}
for t in A8 + A16:
    lf = C.leaf(t)
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    masks[t] = (mm, sl)
print(f'grid {nflat}; a8 carries {len(A8)} circuits, a16 carries {len(A16)}', flush=True)


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
    """dCE when the rank-1 component along unit vector u is removed from `key`'s output."""
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


report = {'schema_version': 1, 'generated': '2026-08-30 by Claude, circuit task',
          'method': 'per-circuit direction in a component output space = unit(mean over members - mean '
                    'off slice); pairwise |cos| between directions; rank-1 projection ablation along each',
          'state': 'census_state_diverse.pt', 'note': 'read-only; no circuit file modified',
          'groups': {}}
t0 = time.time()
for key, tags in GROUPS.items():
    acts = capture(key)
    dirs = {}
    for t in tags:
        mm, sl = masks[t]
        d = acts[mm].mean(0) - acts[~sl].mean(0)
        dirs[t] = d / d.norm()
    cos = {a: {b: round(float(torch.dot(dirs[a], dirs[b])), 4) for b in tags} for a in tags}
    off = [abs(cos[a][b]) for i, a in enumerate(tags) for b in tags[i + 1:]]
    print(f'{key}: mean pairwise |cos| = {sum(off)/len(off):.4f}  (n={len(off)} pairs)', flush=True)

    conc = {}
    for t in tags:
        d = project_out_dce(key, dirs[t])
        row = {}
        for u in tags:
            mm, sl = masks[u]
            am = float(d[mm].abs().mean()); ag = float(d[~sl].abs().mean())
            row[u] = round(am / ag, 3) if ag > 0 else None
        conc[t] = row
        print(f'  rank-1 out of {key} along {t}: own {row[t]}  others '
              f'{ {u: row[u] for u in tags if u != t} }  ({time.time()-t0:.0f}s)', flush=True)
    report['groups'][key] = {'circuits': tags, 'cos': cos,
                             'mean_pairwise_abs_cos': round(sum(off) / len(off), 4),
                             'rank1_projection_concentration': conc}
json.dump(report, open('circuits/SUBSPACE.json', 'w'), indent=1)
print(f'\nwrote circuits/SUBSPACE.json ({time.time()-t0:.0f}s)', flush=True)
