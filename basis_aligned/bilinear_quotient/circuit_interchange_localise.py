# INTERCHANGE (RESAMPLE) ABLATION ON THE FIVE COMPONENTS THE TWELVE CIRCUITS LANDED ON.
#
# §2054 localised twelve previously-unlocalised circuits by MEAN-ablation: replace a component's output
# with its grid mean. That is the weakest counterfactual available -- the mean is off-distribution for
# every position, so some of the damage measures "this component was doing something" rather than "this
# component was doing THIS circuit's thing".
#
# Interchange ablation is the standard stronger test: replace the component's output at each position with
# its output at a DIFFERENT, randomly chosen position. Every injected activation is one the model actually
# produces, so the counterfactual is on-distribution and the surviving damage is specific to the
# position-to-position content rather than to the component's mean level.
#
# The five components §2054 named: a8 (five circuits), a16 (three), a7 (two), a3, m13. Running the same
# concentration measurement on the same twelve circuits makes the two methods directly comparable.
#
# REGISTERED PREDICTIONS (before running):
#   pred_a  For each of the twelve, the component that won under mean-ablation also wins under
#           interchange. If FALSE the two counterfactuals disagree about location and neither localisation
#           should be written into a circuit file.
#   pred_b  Interchange concentration EXCEEDS mean-ablation concentration for the winning component, on at
#           least 8 of 12. On-distribution donors should isolate circuit-specific content better than the
#           mean does. If FALSE the mean was already measuring the specific effect and the extra machinery
#           buys nothing.
#   pred_c  The three circuits §2054 found effectively tied (r.1.3.1 +0.3%, r.13.2.1 +1.1%, r.7.1.1 +6.9%)
#           remain tied under interchange -- top-two margin still under 20%. Registered because a genuinely
#           distributed circuit should stay distributed under a better counterfactual, and if the tie
#           breaks, §2054 under-powered rather than the circuit being diffuse.
#
# Writes circuits/INTERCHANGE.json. Read-only with respect to circuit files: Codex is in the same folder.
import json
import time

import torch

import census_lib as C

TAGS = ['r.11.1.1', 'r.11.1.2', 'r.11.3.1', 'r.13.2.1', 'r.18.2.0', 'r.1.3.1',
        'r.23.2.1', 'r.23.2.3', 'r.3.0', 'r.3.0.2', 'r.4.1.1', 'r.7.1.1']
KEYS = [f'{k}{L}' for k in ('a', 'm') for L in range(18)]
SEED = 20260830

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()
masks = {}
for t in TAGS:
    lf = C.leaf(t)
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    masks[t] = (mm, sl)
print(f'grid {nflat}, {len(TAGS)} circuits, {len(KEYS)} components', flush=True)


@torch.no_grad()
def interchange_dce(key):
    """dCE when `key`'s output at every position is replaced by its output at a random other position.

    Two passes. The first captures the component's real outputs over the whole grid; the second injects a
    fixed permutation of them. The permutation is drawn once per component from a fixed seed so the run
    reproduces, and it is a derangement in expectation only -- a position may keep its own activation with
    probability 1/N, which at N=256,000 is negligible.
    """
    R = C.rows()
    cap = []
    h = C.MODS[key].register_forward_hook(
        lambda mo, i_, o_: cap.append(((o_[0] if isinstance(o_, tuple) else o_)
                                       .detach().float().reshape(-1, C.D).cpu())))
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    acts = torch.cat(cap)                                   # (nflat, D) on cpu
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(acts.shape[0], generator=g)
    donor = acts[perm]

    ces, off = [], 0
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        n = bb.shape[0] * 256
        rep = donor[off:off + n].to(C.DEV)
        off += n

        def fh(mo, i_, o_, rep=rep):
            if isinstance(o_, tuple):
                y, v1 = o_
                return (rep.view_as(y).to(y.dtype), v1)
            return rep.view_as(o_).to(o_.dtype)

        hh = C.MODS[key].register_forward_hook(fh)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = torch.nn.functional.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
        for blkm in C.m.transformer.h:
            x, v1 = blkm(x, v1, x0)
        lg = (30 * torch.tanh(C.m.lm_head(torch.nn.functional.rms_norm(x, (C.D,))) / 30)).float()
        ces.append(torch.nn.functional.cross_entropy(
            lg.view(-1, lg.size(-1)), tg, reduction='none').cpu())
        hh.remove()
    return torch.cat(ces).float() - base


out = {}
t0 = time.time()
for i, key in enumerate(KEYS):
    d = interchange_dce(key)
    for t in TAGS:
        mm, sl = masks[t]
        am = float(d[mm].abs().mean()); ag = float(d[~sl].abs().mean())
        out.setdefault(t, {})[key] = {'abs_dce_members': round(am, 4),
                                      'abs_dce_offslice': round(ag, 4),
                                      'concentration': round(am / ag, 3) if ag > 0 else None}
    print(f'  [{i+1:2d}/36] {key}  ({time.time()-t0:.0f}s)', flush=True)

rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude, circuit task',
       'method': 'interchange (resample) ablation: each component output replaced by its output at a '
                 'random other grid position, fixed seed 20260830; concentration is mean|dCE| on members '
                 'over mean|dCE| off slice',
       'state': 'census_state_diverse.pt', 'seed': SEED,
       'note': 'read-only artifact; no circuit file was modified',
       'by_tag': {}}
for t, per in out.items():
    rank = sorted(((v['concentration'], k) for k, v in per.items() if v['concentration'] is not None),
                  reverse=True)
    rep['by_tag'][t] = {'top': [{'component': k, **per[k]} for c, k in rank[:6]], 'all': per}
json.dump(rep, open('circuits/INTERCHANGE.json', 'w'), indent=1)
print(f'\nwrote circuits/INTERCHANGE.json ({time.time()-t0:.0f}s)', flush=True)
for t, r in rep['by_tag'].items():
    print(f"  {t:10s} best {r['top'][0]['component']:4s} conc {r['top'][0]['concentration']}")
