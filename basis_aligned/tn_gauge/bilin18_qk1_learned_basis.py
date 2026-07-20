"""Flagship confirmation of F16 (Logan 2026-07-20): are the layer-1 query/key READS sparse
in a learned INPUT-basis rotation on bilin18? F16 (toy) found rotating the QK input basis
(full O(d), residual side, unconstrained -- unlike regime-1's RoPE-constrained head-dim
rotation) lets you prune 75% of attn2's read weights for +0.14 ΔCE. Test the same on
bilin18's second attention (h[1]): stack its QK reads [c_q;c_k;c_q2;c_k2], L4-optimize an
O(D) input rotation to sparsify them, prune by magnitude in that basis, and measure real
end-to-end ΔCE via reference_forward -- vs pruning in the original basis. Gated by a planted
control (optimizer) and by keep=1.0 (ΔCE~0). Weight-only rotation; ΔCE binding.
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
D = cfg['n_embd']
AUD = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = AUD[:, :-1].to(DEV); TGT = AUD[:, 1:].to(DEV)
V = cfg['vocab_size']
A1 = m.transformer.h[1].attn
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
W0 = {n: getattr(A1, n).weight.data.clone() for n in NAMES}


@torch.no_grad()
def ce():
    lg = reference_forward(m, IDX, 'bf16').float()
    return F.cross_entropy(lg.reshape(-1, V), TGT.reshape(-1)).item()


def l4_rotate(Rm, iters=1000, lr=0.02):
    d = Rm.shape[1]; Vv = torch.eye(d, device=DEV)
    for _ in range(iters):
        RV = Rm @ Vv
        G = Rm.T @ (RV ** 3)
        S = G @ Vv.T; S = 0.5 * (S - S.T)
        I = torch.eye(d, device=DEV)
        Vv = torch.linalg.solve(I - lr * S, I + lr * S) @ Vv
    return Vv


# planted control
S0 = torch.zeros(4 * D, D, device=DEV)
S0[torch.arange(4 * D, device=DEV), torch.randint(0, D, (4 * D,), device=DEV)] = torch.randn(4 * D, device=DEV)
V0, _ = torch.linalg.qr(torch.randn(D, D, device=DEV))
Ac = S0 @ V0.T; Vc = l4_rotate(Ac)
dc = 100 * (1 - (Ac @ Vc).abs().sum().item() / Ac.abs().sum().item())
oc = 100 * (1 - S0.abs().sum().item() / Ac.abs().sum().item())
print(f'CONTROL planted L1 drop {dc:.1f}% (opt {oc:.1f}%) {"PASS" if dc > 0.8 * oc else "FAIL"}', flush=True)
assert dc > 0.8 * oc

R = torch.cat([W0[n] for n in NAMES], 0).float()
Vv = l4_rotate(R)
l1d = 100 * (1 - (R @ Vv).abs().sum().item() / R.abs().sum().item())
print(f'flagship h[1] QK reads: L1 drop {l1d:.1f}% under learned input rotation', flush=True)

CE0 = ce()
res = {'baseline_ce': round(CE0, 4), 'l1_drop_pct': round(l1d, 2), 'prune_dce': {'original': {}, 'learned': {}}}
print(f'baseline CE {CE0:.4f}; prune reads -> ΔCE (original vs learned basis):', flush=True)


def prune_keep(W, frac, basis=None):
    Mm = W @ basis if basis is not None else W
    k = max(1, int(frac * Mm.numel()))
    thr = Mm.abs().reshape(-1).topk(k).values.min()
    Mp = torch.where(Mm.abs() >= thr, Mm, torch.zeros_like(Mm))
    return Mp @ basis.T if basis is not None else Mp


for frac in [1.0, 0.5, 0.25, 0.125]:
    for tag, basis in [('original', None), ('learned', Vv)]:
        for n in NAMES:
            getattr(A1, n).weight.data.copy_(prune_keep(W0[n].float(), frac, basis).to(W0[n].dtype))
        d = ce() - CE0
        res['prune_dce'][tag][frac] = round(d, 4)
        for n in NAMES:
            getattr(A1, n).weight.data.copy_(W0[n])
    print(f"  keep {frac:.3f}: original {res['prune_dce']['original'][frac]:+.4f} | "
          f"learned {res['prune_dce']['learned'][frac]:+.4f}", flush=True)
    json.dump(res, open(f'{OUT}/bilin18_qk1_learned_basis.json', 'w'), indent=2)
res['learned_helps'] = bool(res['prune_dce']['learned'][0.25] < res['prune_dce']['original'][0.25])
json.dump(res, open(f'{OUT}/bilin18_qk1_learned_basis.json', 'w'), indent=2)
print(f"\nlearned basis prunes better at keep=0.25: {res['learned_helps']} "
      f"(F16 toy generalizes: {res['learned_helps'] and l1d>15})", flush=True)
print('bilin18 qk1 learned basis done', flush=True)
