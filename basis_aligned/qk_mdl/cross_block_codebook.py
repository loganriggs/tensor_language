"""The V×V cross-block codebook (queued since tick 10; justified by the +0.84
block importance): is block-0's bilinear CROSS term — current token × attention-
out — a (k_t × k_s)-class interaction?

Folded object: cross(i) = Le(i) ⊙ R a(i) + L a(i) ⊙ Re(i), and a(i) is linear
in transported source tokens, so the family is X[t, s] ∈ R^4608 over (current
token t, source token s). Codebook: class the two INPUT sides — current-token
side by classes of the normed embedding direction (k_t), source side by classes
of the per-head OV v-tables (k_s) — ONLY inside the cross term; the self and
pair blocks stay exact. Split-gate: (k_t=∞, k_s=∞) must reproduce baseline.

Arms probe asymmetry: which side of the interaction needs precision?
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/cross_block_codebook.json'

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:]
E = m.transformer.wte.weight.detach().float()
EH = F.rms_norm(E, (D,))                       # per-token direction of the residual
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


def class_table(X, k):
    if k is None:                                  # exact
        return X
    C, assign = kmeans(X, k)
    return C[assign]


@torch.no_grad()
def forward(tokens, kt=None, ks=None):
    """kt: classes for the current-token side of the CROSS term; ks: classes for
    the source-token content (classed v-tables feeding a SECOND attention-out
    used only in the cross term). None = exact."""
    eh_c = class_table(EH, kt)                     # (V, 1152) classed directions
    if ks is None:
        vt_c = None
    else:
        vt_c = torch.empty_like(VT)
        for hh in range(NH):
            vt_c[:, hh] = class_table(VT[:, hh].contiguous(), ks)
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
            # exact block split with frozen empirical rms (gate 2.4e-7, tick 10)
            rms = x.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            e_n = (x_resid.float() / rms).to(x.dtype)
            a_n = (att.float() / rms).to(x.dtype)
            # classed versions used ONLY in the cross term
            scale = x_resid.float().norm(dim=-1, keepdim=True) / \
                EH[tokens].norm(dim=-1, keepdim=True).clamp_min(1e-8)
            e_c = ((eh_c[tokens] * scale).float() / rms).to(x.dtype)
            if vt_c is not None:
                v_c = vt_c[tokens].to(x.dtype)
                v_cm = (1 - a.lamb) * v_c + a.lamb * v_c   # v1 == v at layer 0
                att_c = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v_cm
                                              ).reshape(B, T, -1))
                a_c = (att_c.float() / rms).to(x.dtype)
            else:
                a_c = a_n
            L, R = blk.mlp.Left, blk.mlp.Right
            hidden = (L(e_n) * R(e_n)                       # self: exact
                      + L(e_c) * R(a_c) + L(a_c) * R(e_c)   # cross: classed
                      + L(a_n) * R(a_n))                    # pair: exact
            x = x + blk.mlp.Down(hidden) + blk.mlp.Down_bias
        else:
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


results = {'arms': {}}
# gate: exact-exact through the split path must reproduce the plain model
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
gate = ce(kt=None, ks=None)
print(f'baseline {CE0:.4f}; split-path exact-exact {gate:.4f} '
      f'(gate diff {abs(gate - CE0):.2e})')
results['baseline_ce'] = CE0
results['gate_diff'] = abs(gate - CE0)

for kt, ks in [(256, 256), (1024, 1024), (256, None), (None, 256),
               (4096, 4096), (64, 64)]:
    d = ce(kt=kt, ks=ks) - CE0
    name = f'kt={kt or "exact"}, ks={ks or "exact"}'
    results['arms'][name] = d
    print(f'cross-block {name}: dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)
print('cross block done')
