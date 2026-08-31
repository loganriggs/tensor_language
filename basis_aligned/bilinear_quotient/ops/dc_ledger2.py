# THE DC LEDGER, CENSUS FRAME (rung 97b; rebuild after the assembly-frame void).
#
# CONVENTION (S2135): per-position dCE = CE(intervened) - CE(base) on the census grid, REAL model (the
# battery's frame); positive = damage. d(arm) = mean dCE over the grid. DC share = 1 - d(mean)/d(zero).
# S2187: the assemblers' DC terms carried ~98% of joint knockout damage (assembly frame). Does DC dominance
# generalize to a8 (most circuit-dense), a5 (cliff), m13-m16 (band) on the real model?
#
# REGISTERED PREDICTIONS:
#   (a) DC DOMINATES: median DC share across the six components >= 0.5.
#   (b) a8 TOO: a8 DC share >= 0.5.
#   (c) ANTI-INERTNESS: every zero-arm d >= +0.02 (each component genuinely hurts when zeroed on the real
#       model; the rung-97 void's failure mode is structurally impossible here but checked anyway).
# NULL: DC share small outside the assemblers. PRICE: none (attribution). Self-reviewed.
import json
import os
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'merge_circuit_grain_results.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: census state present; 13 census-frame arms (~16s each)')
    raise SystemExit(0)

import time                                                               # noqa: E402
import statistics as stt                                                  # noqa: E402

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import census_lib as C                                                    # noqa: E402

T0 = time.time()
C.use_state('census_state_diverse.pt')
base = C.base_ce()
KEYS = ['a8', 'a5', 'm13', 'm14', 'm15', 'm16']


@torch.no_grad()
def run_pass(hooks):
    R = C.rows(); ces = []
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
        for blkm in C.m.transformer.h:
            x, v1 = blkm(x, v1, x0)
        lg = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
        ces.append(F.cross_entropy(lg.view(-1, lg.size(-1)), tg, reduction='none').cpu())
    for h in hooks:
        h.remove()
    return torch.cat(ces).float()


# capture means for all six in one pass
caps = {k: [] for k in KEYS}
hs = []
for k in KEYS:
    def mk(k=k):
        def h(mo, i_, o_):
            y = o_[0] if isinstance(o_, tuple) else o_
            caps[k].append(y.detach().float().reshape(-1, C.D).mean(0).cpu())
        return h
    hs.append(C.MODS[k].register_forward_hook(mk()))
print('capture pass', flush=True)
_ = run_pass(hs)
MUS = {k: torch.stack(v).mean(0).to(C.DEV) for k, v in caps.items()}


def abl_hook(key, mu):
    def h(mo, i_, o_):
        y = o_[0] if isinstance(o_, tuple) else o_
        r = (mu.expand_as(y) if mu is not None else torch.zeros_like(y)).to(y.dtype)
        return (r, o_[1]) if isinstance(o_, tuple) else r
    return C.MODS[key].register_forward_hook(h)


res = {}
for k in KEYS:
    row = {}
    for mode, mu in (('zero', None), ('mean', MUS[k])):
        print(f'arm {k}/{mode}', flush=True)
        ce = run_pass([abl_hook(k, mu)])
        row[mode] = round(float((ce - base).mean()), 4)
    row['dc_share'] = round(1 - row['mean'] / max(row['zero'], 1e-9), 3)
    res[k] = row
    print(f'{k}: zero {row["zero"]:+.4f} mean {row["mean"]:+.4f} dc_share {row["dc_share"]}', flush=True)

shares = [res[k]['dc_share'] for k in KEYS]
pa = stt.median(shares) >= 0.5
pb = res['a8']['dc_share'] >= 0.5
pc = all(res[k]['zero'] >= 0.02 for k in KEYS)
out = {'components': res, 'median_dc_share': round(stt.median(shares), 3),
       'convention': 'mean dCE over the census grid vs the real model; dc_share = 1 - d(mean)/d(zero)',
       'pred_a_dc_dominates': bool(pa), 'pred_b_a8_too': bool(pb), 'pred_c_anti_inert': bool(pc),
       'self_reviewed': True, 'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open('dc_ledger2_results.json', 'w'), indent=1)
print(f"(a) median DC share {stt.median(shares):.3f} >= 0.5: {'HELD' if pa else 'FAILED'}")
print(f"(b) a8 {res['a8']['dc_share']:.3f} >= 0.5: {'HELD' if pb else 'FAILED'}")
print(f"(c) all zero-arms >= +0.02: {'HELD' if pc else 'FAILED'}")
print(f'wrote dc_ledger2_results.json ({time.time()-T0:.0f}s)')
