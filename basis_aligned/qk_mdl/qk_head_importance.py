"""TICK 192 (Logan un-gated layer 1): (A) whole-head ablations re-run on the FULL
standard audit (307k predictions, ~10x tick-191's 64-document subset) — do the quiet
heads (1, 2, 5) stay quiet with more text? (B) cheap weight-space importance candidates
per head, correlated (Spearman) against the full-audit causal numbers — Logan wants a
weight-only correlate of causal importance. (C) layer-1 architecture reconnaissance
printout (modules, dims, whether an MLP exists) to plan the layer-1 program.

Correlate candidates (all weight-only or unigram-weighted, no forwards):
  k1_norm, k2_norm     : Frobenius norms of the head's key tables
  ov_norm              : sum_t p_t ||W_o^h v_t||  (expected output-vector magnitude)
  core_scale           : Frobenius norm of the exact head-space third-moment core
                         M = sum_t p_t k1_t (x) k2_t (x) v_t  (128^3, unnormalized)
  pattern_sq           : E_{i,t~p}[ (q1_i.k1_t)^2 (q2_i.k2_t)^2 ] / d^4  (sampled 4096^2
                         token pairs — expected squared pattern weight under unigram)
  pattern_ov           : E_{i,t~p}[ |s1 s2| ] * ov_norm-ish composite:
                         E[|s1 s2|] x mean ||W_o v||  (pattern-times-write magnitude)
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

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TAB = {}
for br, (qn, kn) in ((1, ('q1', 'k1')), (2, ('q2', 'k2'))):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
with torch.no_grad():
    a0 = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a0.c_v(E).view(V, NH, HD)
    Wo = a0.c_proj.weight.detach().float().view(D, NH, HD)

# ---------- C: layer-1 architecture recon ----------
print('=== layer-1 block structure ===', flush=True)
blk = m.transformer.h[1]
for name, mod in blk.named_children():
    print(f'  {name}: {mod.__class__.__name__}', flush=True)
    for n2, m2 in mod.named_children():
        shp = getattr(getattr(m2, 'weight', None), 'shape', '')
        print(f'    {n2}: {m2.__class__.__name__} {shp}', flush=True)
print(f'  n_layers total: {len(m.transformer.h)}', flush=True)

# ---------- A: whole-head ablations, full audit ----------


@torch.no_grad()
def mean_ce(tabs, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16',
                                   score_patch=None if tabs is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = mean_ce(None)
print(f'baseline CE (full audit) {base:.5f}', flush=True)
out = {'base_ce': round(base, 5), 'n_predictions': int(len(FINEWEB) * (FINEWEB.shape[1] - 1))}
whole = {}
for h in range(NH):
    tabs = {k: v.clone() for k, v in TAB.items()}
    tabs['k1'][:, h] = 0
    tabs['k2'][:, h] = 0
    d = mean_ce(tabs) - base
    whole[h] = d
    print(f'h{h}: whole-head dCE (full audit) {d:+.5f}', flush=True)
    out[f'h{h}_whole_dce'] = round(d, 5)
    json.dump(out, open(f'{QK}/qk_head_importance.json', 'w'), indent=2)
    del tabs
    torch.cuda.empty_cache()

# ---------- B: weight-space correlates ----------
g = torch.Generator().manual_seed(0)
si = torch.multinomial(QP.cpu(), 4096, replacement=True, generator=g).to(DEV)
ti = torch.multinomial(QP.cpu(), 4096, replacement=True, generator=g).to(DEV)
cands = {}
for h in range(NH):
    k1h, k2h = TAB['k1'][:, h], TAB['k2'][:, h]
    q1h, q2h = TAB['q1'][:, h], TAB['q2'][:, h]
    ovn = float((QP * (Vv[:, h] @ Wo[:, h].T).norm(dim=1)).sum())
    Vpi = Vv[:, h] * QP[:, None]
    Mc = torch.stack([k1h.T @ (k2h * Vpi[:, kk:kk + 1]) for kk in range(HD)], 2)
    core_scale = float(Mc.norm())
    s1 = (q1h[si] * k1h[ti]).sum(1) / HD
    s2 = (q2h[si] * k2h[ti]).sum(1) / HD
    patt_sq = float((s1 ** 2 * s2 ** 2).mean())
    patt_abs = float((s1 * s2).abs().mean())
    cands.setdefault('k1_norm', []).append(float(k1h.norm()))
    cands.setdefault('k2_norm', []).append(float(k2h.norm()))
    cands.setdefault('ov_norm', []).append(ovn)
    cands.setdefault('core_scale', []).append(core_scale)
    cands.setdefault('pattern_sq', []).append(patt_sq)
    cands.setdefault('pattern_ov', []).append(patt_abs * ovn)
    del Mc
    torch.cuda.empty_cache()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


wvec = [whole[h] for h in range(NH)]
out['correlates'] = {}
for name, vals in cands.items():
    rho = spearman(vals, wvec)
    out['correlates'][name] = {'values': [round(v, 6) for v in vals],
                               'spearman_vs_whole_dce': round(rho, 3)}
    print(f'correlate {name}: spearman {rho:+.3f} | values '
          + ' '.join(f'{v:.3g}' for v in vals), flush=True)
json.dump(out, open(f'{QK}/qk_head_importance.json', 'w'), indent=2)
print('HEAD IMPORTANCE DONE', flush=True)
