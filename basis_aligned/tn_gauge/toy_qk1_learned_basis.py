"""Is M's selection high-dimensionality INTRINSIC or basis-dependent? (Logan 2026-07-20,
the remaining avenue after F13-F15). F14/F15 found the bilinear output M is ~rank-64 for
layer-1 selection in both the variance and low-rank bases. Test whether a LEARNED
(directly optimized) basis for the M-space makes the layer-1 query/key READS of M sparse
-- the interaction-sparse decomposition variance/rank can't see. The M-input basis rotates
BEFORE rotary, so it is an unconstrained full O(D) rotation.

Stack attn2's QK read maps R = [W_q1; W_k1; W_q2; W_k2] (4D x D). Optimize V in O(D) to
maximize ||R V||_4^4 (validated L4 rotation-to-sparsity, gated by a planted control).
Large L1 drop -> M's selection-relevance IS sparse in a learned basis (basis-dependent).
Small drop (like regime-1's ~7%) -> intrinsic high-dimensionality. Weight-only, tier-1.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, '/workspace/tensor_language')
from deep_model import DeepModel
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN = '/workspace/tensor_language/runs_lm/block2-seed0'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
import json as _j
cfg = _j.load(open(f'{RUN}/config.json'))
D, NH, SPEC = cfg['d_model'], cfg['n_head'], cfg['spec']
VOCAB, N_CTX = cfg['vocab'], cfg['n_ctx']
m = DeepModel(VOCAB, D, NH, SPEC, N_CTX, norm=(cfg['norm'] == 'rms'),
              residual=cfg['residual'], attention=cfg['attention']).to(DEV)
sd = torch.load(f'{RUN}/model.pt', map_location=DEV)
m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) and 'model' in sd else sd)
A2 = m.layers[2]


def l4_rotate(Rm, iters=1500, lr=0.03):
    """maximize ||Rm V||_4^4 over V in O(D). Validated L4 ascent (regime-1)."""
    d = Rm.shape[1]
    V = torch.eye(d, device=DEV)
    for _ in range(iters):
        RV = Rm @ V
        G = Rm.T @ (RV ** 3)
        S = G @ V.T; S = 0.5 * (S - S.T)
        I = torch.eye(d, device=DEV)
        V = torch.linalg.solve(I - lr * S, I + lr * S) @ V
    return V


def hoyer(X):
    n = X.numel(); return float((np.sqrt(n) - (X.abs().sum() / X.norm()).item()) / (np.sqrt(n) - 1))


# ---- planted control: a read matrix sparse in a rotated basis must be recovered ----
S0 = torch.zeros(512, D, device=DEV)
S0[torch.arange(512, device=DEV), torch.randint(0, D, (512,), device=DEV)] = torch.randn(512, device=DEV)
V0, _ = torch.linalg.qr(torch.randn(D, D, device=DEV))
Ac = S0 @ V0.T
Vc = l4_rotate(Ac)
drop_c = 100 * (1 - (Ac @ Vc).abs().sum().item() / Ac.abs().sum().item())
opt_c = 100 * (1 - S0.abs().sum().item() / Ac.abs().sum().item())
print(f'CONTROL planted: L1 drop {drop_c:.1f}% (planted-optimum drop {opt_c:.1f}%) '
      f'{"PASS" if drop_c > 0.8 * opt_c else "FAIL"}', flush=True)
assert drop_c > 0.8 * opt_c, 'L4 optimizer control failed'
# negative control
An = torch.randn(512, D, device=DEV)
Vn = l4_rotate(An)
print(f'CONTROL random: L1 drop {100*(1-(An@Vn).abs().sum().item()/An.abs().sum().item()):.1f}% (small)', flush=True)

# ---- the real object: attn2 QK read maps ----
R = torch.cat([A2.q1.weight.data, A2.k1.weight.data, A2.q2.weight.data, A2.k2.weight.data], 0).float()
V = l4_rotate(R)
l1_0 = R.abs().sum().item(); l1_1 = (R @ V).abs().sum().item()
res = {'l1_before': round(l1_0, 2), 'l1_after': round(l1_1, 2),
       'l1_drop_pct': round(100 * (1 - l1_1 / l1_0), 2),
       'hoyer_before': round(hoyer(R), 3), 'hoyer_after': round(hoyer(R @ V), 3),
       'orth_err': (V @ V.T - torch.eye(D, device=DEV)).abs().max().item()}
res['verdict'] = ('basis-dependent: learned basis sparsifies the QK reads of M'
                  if res['l1_drop_pct'] > 20 else
                  'INTRINSIC: M selection-relevance not sparsifiable by a learned basis (like regime-1)')
print(f"\nattn2 QK reads: L1 {l1_0:.1f} -> {l1_1:.1f} ({res['l1_drop_pct']:.1f}% drop), "
      f"Hoyer {res['hoyer_before']:.3f} -> {res['hoyer_after']:.3f}", flush=True)

# ---- BINDING check: prune reads by magnitude, ΔCE. Does the learned basis let you
# prune MORE at fixed ΔCE than the original basis? (weight-sparsity -> behaviour) ----
import torch.nn.functional as F
val = np.memmap('/workspace/tensor_language/data_text/val.bin', dtype=np.uint16, mode='r')
buf = np.stack([val[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(48)]).astype(np.int64)
Bt = torch.from_numpy(buf).to(DEV); IDX, TGT = Bt[:, :-1], Bt[:, 1:]
for p in m.parameters():
    p.requires_grad_(False)
NAMES = ['q1', 'k1', 'q2', 'k2']
W0 = {n: getattr(A2, n).weight.data.clone() for n in NAMES}


@torch.no_grad()
def ce():
    return F.cross_entropy(m(IDX).float().reshape(-1, VOCAB), TGT.reshape(-1)).item()


CE0 = ce()


def prune_keep(W, frac, basis=None):
    """keep top `frac` of entries by |.|; if basis V given, prune in the V-basis of the input."""
    M_ = W @ basis if basis is not None else W
    k = max(1, int(frac * M_.numel()))
    thr = M_.abs().reshape(-1).topk(k).values.min()
    Mp = torch.where(M_.abs() >= thr, M_, torch.zeros_like(M_))
    return Mp @ basis.T if basis is not None else Mp


res['prune_dce'] = {'original_basis': {}, 'learned_basis': {}}
print('\nBINDING: prune QK reads to keep-fraction, ΔCE (original basis vs learned basis):', flush=True)
Vf = V.float()
for frac in [0.5, 0.25, 0.125, 0.0625]:
    for tag, basis in [('original_basis', None), ('learned_basis', Vf)]:
        for n in NAMES:
            getattr(A2, n).weight.data.copy_(prune_keep(W0[n].float(), frac, basis).to(W0[n].dtype))
        d = ce() - CE0
        res['prune_dce'][tag][frac] = round(d, 4)
        for n in NAMES:
            getattr(A2, n).weight.data.copy_(W0[n])
    print(f"  keep {frac:.4f}: original ΔCE {res['prune_dce']['original_basis'][frac]:+.4f} | "
          f"learned ΔCE {res['prune_dce']['learned_basis'][frac]:+.4f}", flush=True)
res['learned_helps_pruning'] = bool(all(
    res['prune_dce']['learned_basis'][f] <= res['prune_dce']['original_basis'][f] for f in [0.25, 0.125]))
_j.dump(res, open(f'{OUT}/toy_qk1_learned_basis.json', 'w'), indent=2)
print(f"\nweight-L1 verdict: {res['verdict']}", flush=True)
print(f"binding verdict: learned basis prunes better at fixed ΔCE = {res['learned_helps_pruning']}", flush=True)
print('qk1 learned basis done -> toy_qk1_learned_basis.json', flush=True)
