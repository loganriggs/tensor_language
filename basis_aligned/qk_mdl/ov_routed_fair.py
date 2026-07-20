"""Fair routed test (Logan follow-up 2026-07-21): does block-sparse routing
genuinely beat a single dictionary at matched bits, once BOTH use the strong
per-token top-k encoder (OVD-4 showed batch-top-k is the weaker encoder, so the
earlier routed arm was handicapped)? Converged (4000-step full-batch) dictionaries,
real cross-entropy audit.

Arms (all per-token top-k, k=8):
  - single shared dict, n=512
  - routed G=8, uniform n_g=128
  - routed G=8, adaptive n_g by group size
Plus a matched-bits single-dict reference (n chosen so bits ~= routed) to make the
comparison bits-fair, not just sparsity-fair.
"""
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
OUT = f'{QK}/ov_routed_fair.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
VT = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)
STEPS, K = 4000, 8


def train_encode(Xg, n, k, seed):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    Dm = Xg[torch.randperm(len(Xg), generator=g)[:n]].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone(); b = Xg.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=3e-3)
    for _ in range(STEPS):
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (Xg - b) @ We.T
        _, idx = z.abs().topk(k, 1); coeff = torch.gather(z, 1, idx)
        loss = ((b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1) - Xg) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        z = (Xg - b) @ We.T
        _, idx = z.abs().topk(k, 1); coeff = torch.gather(z, 1, idx)
        xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    return xhat, n * HD


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


# groups
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

CE0 = ce(None)
res = {'baseline_ce': CE0, 'arms': {}}
print(f'baseline {CE0:.4f}', flush=True)


def run_single(n, tag):
    vt = torch.empty_like(VT); atomf = 0
    for hh in range(NH):
        xhat, af = train_encode(VT[:, hh].to(DEV), n, K, seed=hh)
        vt[:, hh] = xhat; atomf += af
    mbits = (atomf * 32 + K * V * NH * (32 + math.log2(n))) / 1e6
    d = ce(vt) - CE0
    res['arms'][tag] = {'dce': round(d, 4), 'Mbits': round(mbits, 1)}
    print(f'{tag}: dCE {d:+.4f}  {mbits:.0f}Mbits', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)


def run_routed(n_g_fn, tag):
    vt = torch.empty_like(VT); atomf = 0; nnz = 0
    for hh in range(NH):
        for gg in range(G):
            rows = (a_ == gg).nonzero().squeeze(1)
            n_g = n_g_fn(len(rows))
            xhat, af = train_encode(VT[rows, hh].to(DEV), n_g, K, seed=hh * G + gg)
            vt[rows, hh] = xhat; atomf += af; nnz += K * len(rows)
    mbits = (atomf * 32 + nnz * (32 + math.log2(256)) + V * math.log2(G)) / 1e6
    d = ce(vt) - CE0
    res['arms'][tag] = {'dce': round(d, 4), 'Mbits': round(mbits, 1)}
    print(f'{tag}: dCE {d:+.4f}  {mbits:.0f}Mbits', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)


run_single(512, 'single dict n=512, per-token k=8')
run_routed(lambda s: 128, 'routed G=8 uniform n_g=128, per-token k=8')
run_routed(lambda s: int(max(64, min(256, round(s / 40)))), 'routed G=8 adaptive n_g, per-token k=8')
# bits-matched single-dict reference (routed uses ~8*128=1024 atoms total vs single 512;
# give single dict n=1024 so its atom bits ~ match routed)
run_single(1024, 'single dict n=1024, per-token k=8 (bits-matched to routed)')
print('ov routed fair done', flush=True)
