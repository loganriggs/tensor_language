"""Flagship query/key RoPE-torus floor (Logan 2026-07-20) — completes regime 1 on
bilin18. Per head/branch, the RoPE-commuting gauge is a HALF=64-angle torus (d_head=128,
rotate-half). q,k are per-layer (no cross-layer mixing, unlike value), so this is a
per-layer-per-head gauge. Fit torus (L4 ascent), report floor, verify ΔCE=0 via
reference_forward. Gated by the same planted-torus control as the toy."""
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
HALF = HD // 2
AUDIT = build_eval_tokens(n_chunks=8, seq_len=513)[:6]


def apply_torus(X, theta):
    a, b = X[:HALF], X[HALF:]
    c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    return torch.cat([c * a - s * b, s * a + c * b], 0)


def hoyer(X):
    n = X.numel(); return float((np.sqrt(n) - (X.abs().sum() / X.norm()).item()) / (np.sqrt(n) - 1))


def fit_torus(Cq, Ck, iters=1200, lr=0.05):
    theta = torch.zeros(HALF, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(iters):
        obj = -(apply_torus(Cq, theta) ** 4).sum() - (apply_torus(Ck, theta) ** 4).sum()
        opt.zero_grad(); obj.backward(); opt.step()
    return theta.detach()


# planted control gate
th0 = (torch.rand(HALF, device=DEV) * 2 - 1) * 1.2
Sq = torch.zeros(HD, D, device=DEV)
Sq[torch.randint(0, HD, (D,), device=DEV), torch.arange(D, device=DEV)] = torch.randn(D, device=DEV)
Aq = apply_torus(Sq, -th0)
th = fit_torus(Aq, Aq)
opt = Sq.abs().sum().item(); got = apply_torus(Aq, th).abs().sum().item()
print(f'CONTROL: L1 {Aq.abs().sum().item():.1f} -> {got:.1f} (opt {opt:.1f}) '
      f'{"PASS" if got <= 1.03 * opt else "FAIL"}', flush=True)
if got > 1.03 * opt:
    print('control failed'); sys.exit(1)

with torch.no_grad():
    LG0 = reference_forward(m, AUDIT[:2, :-1].to(DEV), 'bf16').float()

res = {'torus_dim': HALF, 'per_layer_mean_drop': [], 'floors': {}}
store = {}
BR = [('b1', 'c_q', 'c_k'), ('b2', 'c_q2', 'c_k2')]
print(f'bilin18 QK torus floor ({HALF} angles/head/branch), {NL} layers x {NH} heads:', flush=True)
for li in range(NL):
    drops = []
    for h in range(NH):
        sl = slice(h * HD, (h + 1) * HD)
        for bn, qn, kn in BR:
            Cq = getattr(m.transformer.h[li].attn, qn).weight.data[sl, :].float()
            Ck = getattr(m.transformer.h[li].attn, kn).weight.data[sl, :].float()
            th = fit_torus(Cq, Ck); store[(li, h, bn)] = th
            l0 = (Cq.abs().sum() + Ck.abs().sum()).item()
            l1 = (apply_torus(Cq, th).abs().sum() + apply_torus(Ck, th).abs().sum()).item()
            drops.append(100 * (1 - l1 / l0))
            res['floors'][f'L{li}.H{h}.{bn}'] = round(100 * (1 - l1 / l0), 2)
    res['per_layer_mean_drop'].append(round(float(np.mean(drops)), 2))
    if li % 3 == 0 or li == NL - 1:
        print(f'  L{li:2d}: mean drop {np.mean(drops):.2f}%', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_qk_torus.json', 'w'), indent=2)

# gauge check
for li in range(NL):
    for bn, qn, kn in BR:
        for nm in (qn, kn):
            W = getattr(m.transformer.h[li].attn, nm).weight.data
            Wn = W.clone()
            for h in range(NH):
                sl = slice(h * HD, (h + 1) * HD)
                Wn[sl, :] = apply_torus(W[sl, :].float(), store[(li, h, bn)]).to(W.dtype)
            W.copy_(Wn)
with torch.no_grad():
    LG1 = reference_forward(m, AUDIT[:2, :-1].to(DEV), 'bf16').float()
dmax = (LG1 - LG0).abs().max().item()
res['gauge_max_logit_diff'] = dmax
res['mean_drop_pct'] = round(float(np.mean(list(res['floors'].values()))), 2)
json.dump(res, open(f'{OUT}/bilin18_qk_torus.json', 'w'), indent=2)
print(f'\nexact-gauge check: max|Δlogit| {dmax:.2e} (must be ~0)', flush=True)
print(f'mean flagship QK torus L1 drop {res["mean_drop_pct"]:.2f}%. done.', flush=True)
