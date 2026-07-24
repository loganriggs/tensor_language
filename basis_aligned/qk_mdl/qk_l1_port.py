"""TICK 193: LAYER-1 OPENING — does the layer-0 machinery port?

Layer-1 attention reads the residual after block 0 (attention + Bilinear MLP), so the
exact embedding-fold no longer applies. Test the token-identity approximation: build
TOKEN-CONDITIONAL MEAN-RESIDUAL tables — the average block-1 input vector per
vocabulary token, estimated on the DISJOINT co-occurrence corpus (never audited) — push
them through block 1's own q/k projections to get layer-1 factor tables, patch layer 1's
pattern with scores from those tables (rotary handled by scores_from_factors, same
convention as layer 0), and measure held-out damage on the standard 307k audit.

Calibration: (a) whole-layer-1 pattern zeroed (how much layer 1's attention matters at
all); (b) per-head layer-1 zeroing (importance ranking to pick the focus heads).
Verdict: if token-table dCE << whole-layer dCE, layer 1's pattern is mostly
token-identity-driven and the entire dictionary/core/archetype pipeline transfers with
mean-residual tables in place of embeddings. Coverage of the mean estimate is reported
(tokens never seen in the estimation corpus fall back to their raw embedding row).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_EST = 1024                                     # estimation sequences from cooc corpus

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))

# ---------- capture block-1 attention inputs on the disjoint corpus ----------
# reference_forward is a manual (module-bypassing) implementation, so hooks never fire;
# replicate its block-0 body exactly and stop at block 1's lambda-mixed input.
from tier2_model import rope_tables, apply_rot


@torch.no_grad()
def block1_input(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    a = blk.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    nh, hd = a.n_head, a.head_dim
    cos, sin = rope_tables(T, hd, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, nh, hd)
        return apply_rot(F.rms_norm(z, (hd,)), cos, sin)

    v = a.c_v(hcur).view(B, T, nh, hd)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        x = block1_input(idx).float().reshape(-1, D)
        ids = idx.reshape(-1)
        sum_x.index_add_(0, ids, x)
        cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
seen = cnt > 0
mean_x = torch.where(seen[:, None], sum_x / cnt[:, None].clamp_min(1), 0.0)
wte = m.transformer.wte.weight.detach().float().to(DEV)
mean_x[~seen] = wte[~seen]
cov_tok = float(seen.float().mean())
FQP = (torch.bincount(FINEWEB.flatten(), minlength=V).float().to(DEV) + 0.5)
cov_mass = float((FQP * seen.float()).sum() / FQP.sum())
print(f'mean-residual estimate: {int(seen.sum())}/{V} tokens seen '
      f'({cov_tok*100:.1f}% types, {cov_mass*100:.2f}% of audit token mass)', flush=True)
del sum_x, cnt
torch.cuda.empty_cache()

# ---------- layer-1 factor tables from mean residuals ----------
a1 = m.transformer.h[1].attn
with torch.no_grad():
    xn = F.rms_norm(mean_x, (D,))
    # match the model's qk(): project, then PER-HEAD rms_norm (rotary is applied later
    # inside scores_from_factors via the C/S difference expansion)
    L1 = {}
    for name, lin in (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2)):
        z = lin(xn).view(V, NH, HD).float()
        L1[name] = F.rms_norm(z, (HD,)).contiguous()
del mean_x, xn
torch.cuda.empty_cache()


@torch.no_grad()
def audit(patch_fn, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]
        logits = reference_forward(m, idx, 'bf16', score_patch=patch_fn(idx)).float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


BASE = 3.07630                                    # tick-192 full-audit baseline


def patch_tables(idx):
    def p(li, s1, s2):
        if li != 1:
            return s1, s2
        n1 = scores_from_factors(L1['q1'], L1['k1'], idx, HD)
        n2 = scores_from_factors(L1['q2'], L1['k2'], idx, HD)
        return n1.to(s1.dtype), n2.to(s2.dtype)
    return p


def patch_zero_layer(idx):
    def p(li, s1, s2):
        if li != 1:
            return s1, s2
        return torch.zeros_like(s1), s2
    return p


def patch_zero_head(h):
    def mk(idx):
        def p(li, s1, s2):
            if li != 1:
                return s1, s2
            s1 = s1.clone()
            s1[:, h] = 0
            return s1, s2
        return p
    return mk


out = {'coverage_types': round(cov_tok, 4), 'coverage_mass': round(cov_mass, 5),
       'base_ce': BASE}
ce_tab = audit(patch_tables)
out['l1_token_tables_dce'] = round(ce_tab - BASE, 5)
print(f'layer-1 token-table pattern: dCE {ce_tab - BASE:+.5f}', flush=True)
json.dump(out, open(f'{QK}/qk_l1_port.json', 'w'), indent=2)
ce_zero = audit(patch_zero_layer)
out['l1_zero_dce'] = round(ce_zero - BASE, 5)
print(f'layer-1 pattern zeroed: dCE {ce_zero - BASE:+.5f}', flush=True)
json.dump(out, open(f'{QK}/qk_l1_port.json', 'w'), indent=2)
for h in range(NH):
    ce_h = audit(patch_zero_head(h))
    out[f'l1_h{h}_zero_dce'] = round(ce_h - BASE, 5)
    print(f'layer-1 head {h} zeroed: dCE {ce_h - BASE:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_l1_port.json', 'w'), indent=2)
print('L1 PORT DONE', flush=True)
