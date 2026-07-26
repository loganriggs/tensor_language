"""TICK 251: layer-3 port (depth-decay point 4); derived from tick 247.

Same construction as the layer-1 port (tick 193): token-conditional mean residual
at block-2's lambda-mixed input, estimated on the disjoint co-occurrence corpus,
pushed through block 2's own q/k projections into factor tables; patch layer-3's
pattern with table scores; audit on the standard 307k set. Calibration: layer-3
pattern zeroed, and per-head zeroing. The table-vs-zero ratio is the layer's
token-identity share — the first point on the depth-decay curve the compositional
scaling plan predicts, and the denominator for the symbol-pair fold (next tick).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_EST = 1024

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))


@torch.no_grad()
def block3_input(idx):
    """Run blocks 0 and 1 exactly as reference_forward does; return block 2's
    lambda-mixed attention input."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li in (0, 1, 2):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk3 = m.transformer.h[3]
    return blk3.lambdas[0] * x + blk3.lambdas[1] * x0


sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        x = block3_input(idx).float().reshape(-1, D)
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

a2 = m.transformer.h[3].attn
with torch.no_grad():
    xn = F.rms_norm(mean_x, (D,))
    L2 = {}
    for name, lin in (('q1', a2.c_q), ('k1', a2.c_k), ('q2', a2.c_q2), ('k2', a2.c_k2)):
        z = lin(xn).view(V, NH, HD).float()
        L2[name] = F.rms_norm(z, (HD,)).contiguous()
torch.save({'tables': {k: v.cpu() for k, v in L2.items()}, 'seen': seen.cpu()},
           f'{QK}/qk_l3_tables.pt')
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
        if li != 3:
            return s1, s2
        n1 = scores_from_factors(L2['q1'], L2['k1'], idx, HD)
        n2 = scores_from_factors(L2['q2'], L2['k2'], idx, HD)
        return n1.to(s1.dtype), n2.to(s2.dtype)
    return p


def patch_zero_layer(idx):
    def p(li, s1, s2):
        if li != 3:
            return s1, s2
        return torch.zeros_like(s1), s2
    return p


def patch_zero_head(h):
    def mk(idx):
        def p(li, s1, s2):
            if li != 3:
                return s1, s2
            s1 = s1.clone()
            s1[:, h] = 0
            return s1, s2
        return p
    return mk


out = {'coverage_types': round(cov_tok, 4), 'coverage_mass': round(cov_mass, 5),
       'base_ce': BASE}
ce_tab = audit(patch_tables)
out['l3_token_tables_dce'] = round(ce_tab - BASE, 5)
print(f'layer-3 token-table pattern: dCE {ce_tab - BASE:+.5f}', flush=True)
json.dump(out, open(f'{QK}/qk_l3_port.json', 'w'), indent=2)
ce_zero = audit(patch_zero_layer)
out['l3_zero_dce'] = round(ce_zero - BASE, 5)
print(f'layer-3 pattern zeroed: dCE {ce_zero - BASE:+.5f}', flush=True)
json.dump(out, open(f'{QK}/qk_l3_port.json', 'w'), indent=2)
for h in range(NH):
    ce_h = audit(patch_zero_head(h))
    out[f'l3_h{h}_zero_dce'] = round(ce_h - BASE, 5)
    print(f'layer-3 head {h} zeroed: dCE {ce_h - BASE:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_l3_port.json', 'w'), indent=2)
print('L3 PORT DONE', flush=True)
