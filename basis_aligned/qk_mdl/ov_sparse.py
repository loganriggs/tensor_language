"""OV top-k sparse coding (the basis_aligned e7 move applied to content):
hard vq failed on OV v-tables (+1.38 at k=256 own classes; CE-training recovered
only ~38%). Prediction (tick 10): sparse CODING — each token = signed top-k
combination of shared atoms — rescues content the way it rescued the embedding
(e7: hard-vq +0.87 vs sparse +0.26 CE-trained).

Per layer-0 head: dictionary D_h (n_atoms × 128), token codes = magnitude top-k
(signed). Grid {n_atoms} × {k}, L2-fit ΔCE audit; then CE-train the best small
config (supports frozen, atoms + coefficients trainable through the frozen
model, bf16 + clipping).
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
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/ov_sparse.json'

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 128, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:].to(DEV)
E = m.transformer.wte.weight.detach().float()
VT = (F.rms_norm(E, (D,)) @ m.transformer.h[0].attn.c_v.weight.detach().float().T
      ).view(V, NH, HD)


def train_topk_dict(X, n, k, steps=3000, batch=8192, lr=3e-3, seed=0):
    """Signed magnitude top-k dictionary on rows of X (V, d). Returns
    (Xhat, D_norm, supports, coeffs)."""
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:n]].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    for step in range(steps):
        x = X[torch.randint(0, len(X), (batch,), device=DEV)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        vals, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        z = (X - b) @ We.T
        _, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        Xhat = b.detach() + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    return Xhat, Dn, idx, coeff.detach(), b.detach()


def forward(tokens, v_tab=None, live_parts=None):
    x = m.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = tokens.shape
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        cos, sin = rope_tables(T, HD, tokens.device, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]

        def qk(lin):
            z = lin(h).view(B, T, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cosr, sinr)

        if li == 0 and v_tab is not None:
            v = v_tab[tokens].to(x.dtype)
        elif li == 0 and live_parts is not None:
            Dn, sup, cf, bb = live_parts
            vt = torch.stack(
                [bb[hh] + (cf[hh][tokens].unsqueeze(-1)
                           * Dn[hh][sup[hh][tokens]]).sum(-2)
                 for hh in range(NH)], 2)
            v = vt.to(x.dtype)
        else:
            v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def ce(**kw):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = forward(b[:, :-1], **kw).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = ce()
print(f'baseline {CE0:.4f}')
results = {'baseline_ce': CE0, 'arms': {}}
FULL = 32 * V * HD * NH

best = None
for n, k in [(512, 4), (512, 16), (2048, 4), (2048, 16)]:
    vt = torch.empty_like(VT)
    dl = 0
    parts = None
    for hh in range(NH):
        Xhat, Dn, idx, coeff, b = train_topk_dict(VT[:, hh].contiguous(), n, k)
        vt[:, hh] = Xhat
        dl += 32 * (n * HD + V * k) + V * k * math.log2(n)
    d = ce(v_tab=vt) - CE0
    results['arms'][f'topk n={n} k={k}'] = {'dce': d, 'ratio': dl / FULL}
    print(f'topk n={n} k={k}: dCE {d:+.4f}  DL ratio {dl / FULL:.3f}', flush=True)
    if best is None or d < best[0]:
        best = (d, n, k)
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)

# CE-train the best config (supports frozen; atoms+coeffs+bias trainable)
_, n, k = best
print(f'=== CE-training topk n={n} k={k} (best L2-fit arm)')
m.to(torch.bfloat16)
for p in m.parameters():
    p.requires_grad_(False)
Dn_l, sup_l, cf_l, b_l = [], [], [], []
for hh in range(NH):
    _, Dn, idx, coeff, b = train_topk_dict(VT[:, hh].contiguous(), n, k)
    Dn_l.append(Dn.clone().requires_grad_(True))
    sup_l.append(idx)
    cf_l.append(coeff.clone().requires_grad_(True))
    b_l.append(b.clone().requires_grad_(True))
params = Dn_l + cf_l + b_l
live = (Dn_l, sup_l, cf_l, b_l)
d_before = ce(live_parts=live) - CE0
print(f'before CE-training: dCE {d_before:+.4f} (bf16)')
opt = torch.optim.Adam(params, lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1200)
g = torch.Generator(); g.manual_seed(0)
for step in range(1200):
    bb = TRAIN[torch.randint(0, len(TRAIN), (4,), generator=g)]
    logits = forward(bb[:, :-1], live_parts=live).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           bb[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 300 == 0:
        print(f'  step {step} CE {loss.item():.4f}', flush=True)
d_after = ce(live_parts=live) - CE0
dl = NH * (32 * (n * HD + V * k) + V * k * math.log2(n))
results['ce_trained'] = {'n': n, 'k': k, 'dce_before': d_before,
                         'dce_after': d_after, 'ratio': dl / FULL}
print(f'CE-trained topk n={n} k={k}: dCE {d_before:+.4f} -> {d_after:+.4f}')
with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('ov sparse done')
