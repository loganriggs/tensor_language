"""Converged matched-bits ΔCE comparison of the OV dictionary schemes
(Logan follow-up 2026-07-21): re-run the three schemes with well-converged
dictionaries (4000-step full-batch training) and the REAL cross-entropy audit,
so the scheme choice rests on binding ΔCE at convergence, not the undertrained
sweep. Schemes at matched ~sparsity: per-token top-k, batch-top-k (full-batch,
threshold consistent train/eval), routed/block-sparse (adaptive atoms +
batch-top-k within group). Bits reported alongside."""
import json
import math
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/ov_converged_ce.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
VT = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)
STEPS = 4000


def encode(Xg, Dn, We, b, k, mode):
    z = (Xg - b) @ We.T
    if mode == 'token':
        _, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1), k * len(Xg)
    nnz = k * len(Xg)
    thr = z.abs().reshape(-1).topk(nnz).values.min()
    zc = z * (z.abs() >= thr)
    return b + zc @ Dn, int((zc != 0).sum())


def train(Xg, n, k, mode, seed):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    Dm = Xg[torch.randperm(len(Xg), generator=g)[:n]].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone(); b = Xg.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=3e-3)
    for _ in range(STEPS):
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        xh, _ = encode(Xg, Dn, We, b, k, mode)
        loss = ((xh - Xg) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
    return Dn, We.detach(), b.detach()


@torch.no_grad()
def ce(v_tab):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        bt = AUDIT[i:i + 4].to(DEV)
        x = m.transformer.wte(bt[:, :-1]); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        B, T = x.shape[0], x.shape[1]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
            v = v_tab[bt[:, :-1]].to(x.dtype) if (li == 0 and v_tab is not None) else a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        lg = 30 * torch.tanh(m.lm_head(x) / 30)
        tot += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), bt[:, 1:].reshape(-1)).item() * bt[:, 1:].numel()
        n += bt[:, 1:].numel()
    return tot / n


CE0 = ce(None)
res = {'baseline_ce': CE0, 'arms': {}}
print(f'baseline {CE0:.4f}', flush=True)

# shared-dict schemes at k=8 and k=16
for mode, tag in [('token', 'per-token top-k'), ('batch', 'batch-top-k (full-batch)')]:
    for k in (8, 16):
        vt = torch.empty_like(VT); nnz = 0
        for hh in range(NH):
            Dn, We, b = train(VT[:, hh].to(DEV), 512, k, mode, seed=hh)
            xhat, nz = encode(VT[:, hh].to(DEV), Dn, We, b, k, mode)
            vt[:, hh] = xhat; nnz += nz
        mbits = (512 * HD * NH * 32 + nnz * (32 + math.log2(512))) / 1e6
        d = ce(vt) - CE0
        res['arms'][f'{tag}, n=512 k={k}'] = {'dce': round(d, 4), 'Mbits': round(mbits, 1)}
        print(f'{tag} k={k}: dCE {d:+.4f}  {mbits:.0f}Mbits', flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)

# routed: adaptive atoms + batch-top-k within group, k=8
G = 8
gK = torch.Generator(); gK.manual_seed(1)
C0 = E_hat[torch.randperm(V, generator=gK)[:G]].clone().to(DEV)
for _ in range(10):
    a_ = torch.empty(V, dtype=torch.long, device=DEV)
    for i in range(0, V, 8192):
        xx = E_hat[i:i + 8192].to(DEV)
        a_[i:i + 8192] = ((xx * xx).sum(1, True) - 2 * xx @ C0.T + (C0 * C0).sum(1)[None]).argmin(1)
    Cn = torch.zeros_like(C0); c2 = torch.zeros(G, device=DEV)
    Cn.index_add_(0, a_, E_hat.to(DEV)); c2.index_add_(0, a_, torch.ones(V, device=DEV))
    nz = c2 > 0; C0[nz] = Cn[nz] / c2[nz][:, None]
vt = torch.empty_like(VT); nnz = 0; atomf = 0
for hh in range(NH):
    for gg in range(G):
        rows = (a_ == gg).nonzero().squeeze(1)
        n_g = int(max(64, min(256, round(len(rows) / 40))))
        Dn, We, b = train(VT[rows, hh].to(DEV), n_g, 8, 'batch', seed=hh * G + gg)
        xhat, nz = encode(VT[rows, hh].to(DEV), Dn, We, b, 8, 'batch')
        vt[rows, hh] = xhat; nnz += nz; atomf += n_g * HD
mbits = (atomf * 32 + nnz * (32 + math.log2(256)) + V * math.log2(G)) / 1e6
d = ce(vt) - CE0
res['arms']['routed adaptive + batch-top-k, k=8'] = {'dce': round(d, 4), 'Mbits': round(mbits, 1)}
print(f'routed adaptive+batch k=8: dCE {d:+.4f}  {mbits:.0f}Mbits', flush=True)
json.dump(res, open(OUT, 'w'), indent=2)
print('ov converged ce done', flush=True)
