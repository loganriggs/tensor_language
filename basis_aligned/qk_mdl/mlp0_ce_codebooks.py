"""CE-train the MLP-0 block codebooks (queue item 1, tick 16): self + cross
blocks with classed inputs (assignments frozen from weights-only k-means,
class-centroid tables trainable through the frozen model; pair block exact).

Per SR-1 (circuit-specific partitions), each block-side gets its OWN table:
  self:  e-direction classes (256, 1152)
  cross: current-side e-classes (256, 1152) + source-side v-classes (NH, 256, 128)
Arms: combined L2-fit baseline (new number), then CE-trained. bf16 + clipping;
train chunks disjoint from audit.
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
K = 256
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/mlp0_ce_codebooks.json'

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 128, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:].to(DEV)
E = m.transformer.wte.weight.detach().float()
EH = F.rms_norm(E, (D,))
VT = (EH @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)


@torch.no_grad()
def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 4096):
            xx = X[i:i + 4096]
            assign[i:i + 4096] = ((xx ** 2).sum(1, keepdim=True) - 2 * xx @ C.T
                                  + (C ** 2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


# frozen assignments (weights-only), separate trainable tables per block-side
C_e, A_e = kmeans(EH, K)
TAB = {
    'self_e': C_e.clone(),
    'cross_e': C_e.clone(),
    'cross_v': torch.stack([kmeans(VT[:, hh].contiguous(), K)[0]
                            for hh in range(NH)]),
}
A_v = torch.stack([kmeans(VT[:, hh].contiguous(), K)[1] for hh in range(NH)], 1)  # (V, NH)


def forward(tokens, tab):
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

        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v_mix = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        att = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v_mix).reshape(B, T, -1))
        x_resid = x
        x = x + att
        if li == 0:
            rms = x.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            a_n = (att.float() / rms).to(x.dtype)
            scale = (x_resid.float().norm(dim=-1, keepdim=True)
                     / EH[tokens].norm(dim=-1, keepdim=True).clamp_min(1e-8))
            e_self = ((tab['self_e'][A_e[tokens]] * scale).float() / rms).to(x.dtype)
            e_cross = ((tab['cross_e'][A_e[tokens]] * scale).float() / rms).to(x.dtype)
            v_c = torch.stack([tab['cross_v'][hh][A_v[:, hh][tokens]]
                               for hh in range(NH)], 2).to(x.dtype)
            att_c = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v_c
                                          ).reshape(B, T, -1))
            a_c = (att_c.float() / rms).to(x.dtype)
            L, R = blk.mlp.Left, blk.mlp.Right
            hidden = (L(e_self) * R(e_self)
                      + L(e_cross) * R(a_c) + L(a_c) * R(e_cross)
                      + L(a_n) * R(a_n))
            x = x + blk.mlp.Down(hidden) + blk.mlp.Down_bias
        else:
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def ce(tab, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        logits = forward(b[:, :-1], tab).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


from tier2_model import reference_forward


@torch.no_grad()
def ce_plain(batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        logits = reference_forward(m, b[:, :-1]).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = ce_plain()
d_before = ce(TAB) - CE0
print(f'baseline {CE0:.4f}; combined L2-fit (self@{K} + cross@{K}x{K}): '
      f'dCE {d_before:+.4f}')

m.to(torch.bfloat16)
for p in m.parameters():
    p.requires_grad_(False)
for kk in TAB:
    TAB[kk].requires_grad_(True)
params = list(TAB.values())
opt = torch.optim.Adam(params, lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1200)
g = torch.Generator(); g.manual_seed(0)
for step in range(1200):
    b = TRAIN[torch.randint(0, len(TRAIN), (4,), generator=g)]
    logits = forward(b[:, :-1], TAB).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 300 == 0:
        print(f'  step {step} CE {loss.item():.4f}', flush=True)
CE0_bf = ce_plain()
d_after = ce(TAB) - CE0_bf
print(f'CE-trained: dCE {d_before:+.4f} -> {d_after:+.4f} (bf16 baseline {CE0_bf:.4f})')
json.dump({'baseline_ce': CE0, 'k': K, 'dce_before': d_before,
           'dce_after': d_after},
          open(OUT, 'w'), indent=2)
torch.save({k: v.detach().cpu() for k, v in TAB.items()},
           '/workspace/tensor_language/basis_aligned/qk_mdl/mlp0_tables.pt')
print('mlp0 ce done')
