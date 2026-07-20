"""Regime 1 on the FLAGSHIP: OV rotation floor (Logan 2026-07-20).
CORRECTION (caught by the ΔCE gauge check, max|Δlogit|=16.8 on the naive per-layer
version): bilin18 mixes every layer's value with BLOCK-0's value (tier2_model L87-89:
v = (1-lamb)*v + lamb*v1, v1 = block-0 value, same head index). So the value head
subspace is SHARED across all 18 layers -> a per-layer rotation breaks the mixing and
is NOT a gauge. The exact gauge is ONE rotation Q_h per head index, applied to c_v and
c_proj of ALL layers simultaneously. This is the flagship analogue of the residual
bond being shared: the value bus is shared across depth too.

Optimize Q_h in O(d_head) to sparsify the stacked-over-layers value/output maps
(L4/kurtosis rotation-to-sparsity, validated), apply, verify ΔCE=0, report per-head
floor and the per-layer L1-drop profile under the shared gauge. Weight-only, tier-1.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = len(m.transformer.h)
AUDIT = build_eval_tokens(n_chunks=8, seq_len=513)[:6]


def hoyer(X):
    n = X.numel(); l1 = X.abs().sum(); l2 = X.norm()
    return float((np.sqrt(n) - (l1 / l2).item()) / (np.sqrt(n) - 1))


def varimax_Q(A, Bm, iters=700, lr=0.05):
    d = A.shape[1]
    Q = torch.eye(d, device=DEV)
    for _ in range(iters):
        AQ = A @ Q; QB = Q.T @ Bm
        G = A.T @ (AQ ** 3) + Bm @ ((QB ** 3).T)
        S = G @ Q.T; S = 0.5 * (S - S.T)
        I = torch.eye(d, device=DEV)
        Q = torch.linalg.solve(I - lr * S, I + lr * S) @ Q
    return Q


with torch.no_grad():
    LG0 = reference_forward(m, AUDIT[:2, :-1].to(DEV), 'bf16').float()

CV = [m.transformer.h[li].attn.c_v.weight.data.float() for li in range(NL)]
CP = [m.transformer.h[li].attn.c_proj.weight.data.float() for li in range(NL)]
res = {'d_head': HD, 'n_layer': NL, 'per_head_floor': [], 'per_layer_drop_under_shared': []}
print(f'bilin18 regime-1 SHARED-per-head OV gauge ({NH} heads x {NL} layers):', flush=True)

Qs = []
for h in range(NH):
    sl = slice(h * HD, (h + 1) * HD)
    A = torch.cat([CP[li][:, sl] for li in range(NL)], 0)        # (NL*D, HD) readers
    Bm = torch.cat([CV[li][sl, :] for li in range(NL)], 1)       # (HD, NL*D) writers
    Q = varimax_Q(A, Bm); Qs.append(Q)
    l1_0 = (A.abs().sum() + Bm.abs().sum()).item()
    l1_1 = ((A @ Q).abs().sum() + (Q.T @ Bm).abs().sum()).item()
    hb = hoyer(torch.cat([A.reshape(-1), Bm.reshape(-1)]))
    ha = hoyer(torch.cat([(A @ Q).reshape(-1), (Q.T @ Bm).reshape(-1)]))
    res['per_head_floor'].append({'head': h, 'l1_drop_pct': round(100 * (1 - l1_1 / l1_0), 2),
                                  'hoyer_before': round(hb, 3), 'hoyer_after': round(ha, 3)})
    print(f'  head {h}: L1 drop {100*(1-l1_1/l1_0):5.2f}%  Hoyer {hb:.3f}->{ha:.3f}', flush=True)

# per-layer drop profile under the shared gauge
print('  per-layer L1-drop under the shared rotation:', flush=True)
for li in range(NL):
    d0 = d1 = 0.0
    for h in range(NH):
        sl = slice(h * HD, (h + 1) * HD)
        d0 += CP[li][:, sl].abs().sum().item() + CV[li][sl, :].abs().sum().item()
        d1 += (CP[li][:, sl] @ Qs[h]).abs().sum().item() + (Qs[h].T @ CV[li][sl, :]).abs().sum().item()
    res['per_layer_drop_under_shared'].append(round(100 * (1 - d1 / d0), 2))
print('   ' + ' '.join(f'{v:.1f}' for v in res['per_layer_drop_under_shared']), flush=True)

# apply shared gauge, verify ΔCE=0
for li in range(NL):
    cv = m.transformer.h[li].attn.c_v.weight.data
    cp = m.transformer.h[li].attn.c_proj.weight.data
    cv_new = cv.clone(); cp_new = cp.clone()
    for h in range(NH):
        sl = slice(h * HD, (h + 1) * HD)
        Q = Qs[h].to(cv.dtype)
        cv_new[sl, :] = Q.T @ cv[sl, :]
        cp_new[:, sl] = cp[:, sl] @ Q
    cv.copy_(cv_new); cp.copy_(cp_new)
with torch.no_grad():
    LG1 = reference_forward(m, AUDIT[:2, :-1].to(DEV), 'bf16').float()
dmax = (LG1 - LG0).abs().max().item()
res['gauge_check_max_logit_diff'] = dmax
json.dump(res, open(f'{OUT}/bilin18_regime1.json', 'w'), indent=2)
print(f'\nexact-gauge check (SHARED per-head): max|Δlogit| = {dmax:.2e} (must be ~0)', flush=True)
mdrop = np.mean([r['l1_drop_pct'] for r in res['per_head_floor']])
print(f'mean per-head OV L1 drop {mdrop:.2f}%  (vs toy 7%). bilin18 regime1 done.', flush=True)
