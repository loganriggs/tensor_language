"""Shared token-class registry test (Logan's question: 'do you just also reduce
the Embedding in the same class structure?').

Arms (all layer-0 of bilin18, dCE-audited, T=512):
  qk_perhead256      QK scores from per-(head,branch) classes  [reference, have ~+0.008]
  ov_own256          v-tables clustered per head with their OWN k-means (k=256)
  ov_qkclasses256    v-tables averaged within the QK branch-1 classes (class MISMATCH test)
  qk_global256       QK factors averaged within ONE global embedding clustering (k=256)
  ov_global256       v-tables averaged within the same global clustering
  both_global256     QK + OV from the single global registry = layer 0 sees 256
                     effective tokens everywhere
  both_global4096    same with k=4096
Global registry = k-means on the rms-normed embedding rows (the shared object both
circuits read).
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/shared_registry.json'

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:]
FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}
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


def class_mean(X, assign, k):
    """Replace each row by its class mean (X: (V, d))."""
    S = torch.zeros(k, X.shape[1], device=X.device)
    cnt = torch.zeros(k, device=X.device)
    S.index_add_(0, assign, X)
    cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
    return (S / cnt.clamp(min=1)[:, None])[assign]


@torch.no_grad()
def forward(tokens, qk_fact=None, v_tab=None):
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
        else:
            v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        if li == 0 and qk_fact is not None:
            s1 = scores_from_factors(*qk_fact[1], tokens, HD).to(x.dtype)
            s2 = scores_from_factors(*qk_fact[2], tokens, HD).to(x.dtype)
        else:
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


def qk_fact_from(assign_fn, k):
    """Build per-branch class-mean factor tensors given assign_fn(head, br)."""
    out = {}
    for br in (1, 2):
        qh, kh = FACT[br]
        qc = torch.stack([class_mean(qh[:, hh], assign_fn(hh, br), k)
                          for hh in range(NH)], 1)
        kc = torch.stack([class_mean(kh[:, hh], assign_fn(hh, br), k)
                          for hh in range(NH)], 1)
        out[br] = (qc, kc)
    return out


# per-head QK assignments (as in the audits)
QK_ASSIGN = {}
for br in (1, 2):
    qh, kh = FACT[br]
    for hh in range(NH):
        _, a_ = kmeans(torch.cat([qh[:, hh], kh[:, hh]], 1), 256)
        QK_ASSIGN[(hh, br)] = a_

# global registry from the embedding itself
GLOB = {}
for k in (256, 4096):
    _, GLOB[k] = kmeans(EH, k)

def build(name):
    if name == 'qk_perhead256':
        return dict(qk_fact=qk_fact_from(lambda hh, br: QK_ASSIGN[(hh, br)], 256))
    if name == 'ov_own256':
        vt = torch.empty_like(VT)
        for hh in range(NH):
            C, a_ = kmeans(VT[:, hh].contiguous(), 256)
            vt[:, hh] = C[a_]
        return dict(v_tab=vt)
    if name == 'ov_qkclasses256':
        vt = torch.empty_like(VT)
        for hh in range(NH):
            vt[:, hh] = class_mean(VT[:, hh], QK_ASSIGN[(hh, 1)], 256)
        return dict(v_tab=vt)
    if name == 'qk_global256':
        return dict(qk_fact=qk_fact_from(lambda hh, br: GLOB[256], 256))
    if name == 'ov_global256':
        vt = torch.empty_like(VT)
        for hh in range(NH):
            vt[:, hh] = class_mean(VT[:, hh], GLOB[256], 256)
        return dict(v_tab=vt)
    if name == 'both_global256':
        kw = build('ov_global256')
        kw['qk_fact'] = qk_fact_from(lambda hh, br: GLOB[256], 256)
        return kw
    if name == 'both_global4096':
        vt = torch.empty_like(VT)
        for hh in range(NH):
            vt[:, hh] = class_mean(VT[:, hh], GLOB[4096], 4096)
        return dict(qk_fact=qk_fact_from(lambda hh, br: GLOB[4096], 4096), v_tab=vt)
    raise KeyError(name)


for name in ['qk_perhead256', 'ov_own256', 'ov_qkclasses256', 'qk_global256',
             'ov_global256', 'both_global256', 'both_global4096']:
    kw = build(name)
    d = ce(**kw) - CE0
    results['arms'][name] = d
    print(f'{name:20s} dCE {d:+.4f}', flush=True)
    del kw
    torch.cuda.empty_cache()

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('shared registry done')
