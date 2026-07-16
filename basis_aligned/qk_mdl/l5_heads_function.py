"""Within-window arc, probe 1: WHAT do bilin18's two contextual heads
(L5.H5, L5.H7) compute? Signatures measured on live patterns at L5:
  - positional profile: mean |pattern| by relative offset Δ
  - induction signature: E[pat(i,j)] when t_{j} == t_{i} (same token) and when
    t_{j-?}... classic: j such that t_{j-1} == t_i ('attend after my previous
    occurrence') vs random j
  - copy signature: E[pat(i,j) | t_j == t_i] (attend to my own occurrences)
All heads reported; H5/H7 vs the seven tabled-for-free heads is the contrast."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
L = 5
OUT = f'{QK}/l5_heads_function.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
TOK = build_eval_tokens(n_chunks=32, seq_len=513)[:, :-1]

pos_prof = torch.zeros(NH, 64, device=DEV)      # |pat| by Δ (1..64)
pos_cnt = torch.zeros(64, device=DEV)
sig = {k: torch.zeros(NH, device=DEV) for k in ('copy', 'induction', 'random')}
sig_n = {k: 0 for k in ('copy', 'induction', 'random')}


@torch.no_grad()
def run(idx):
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if li == L:
            ap = pat.abs()
            for dlt in range(1, 65):
                vals = torch.diagonal(ap, offset=-dlt, dim1=2, dim2=3)  # (B,NH,T-dlt)
                pos_prof[:, dlt - 1] += vals.sum((0, 2))
                pos_cnt[dlt - 1] += vals.shape[0] * vals.shape[2]
            same = (idx[:, :, None] == idx[:, None, :])                 # t_j == t_i
            prev_match = torch.zeros_like(same)
            prev_match[:, :, 1:] = (idx[:, :, None] == idx[:, None, :-1])  # t_{j-1} == t_i
            tri = mask[None].expand(B, -1, -1) & (~torch.eye(T, dtype=torch.bool, device=DEV))[None]
            for key, cond in (('copy', same & tri),
                              ('induction', prev_match & tri),
                              ('random', tri)):
                c = cond[:, None].expand(-1, NH, -1, -1)
                n = cond.sum().item()
                if n:
                    sig[key] += (pat * c).sum((0, 2, 3)) / n
                    sig_n[key] += 1
            return
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))


for i in range(0, len(TOK), 4):
    run(TOK[i:i + 4].to(DEV))
prof = (pos_prof / pos_cnt.clamp_min(1)).cpu()
res = {'note': 'L5 heads; contextual = H5, H7',
       'signatures_mean_pattern': {
           k: [round(v, 5) for v in (sig[k] / max(sig_n[k], 1)).cpu().tolist()]
           for k in sig},
       'pos_profile_delta1_4_16_64': {
           f'H{h}': [round(prof[h, d].item(), 5) for d in (0, 3, 15, 63)]
           for h in range(NH)}}
for k in ('copy', 'induction'):
    ratio = (sig[k] / sig['random'].clamp_min(1e-9)).cpu()
    res[f'{k}_over_random'] = {f'H{h}': round(ratio[h].item(), 2) for h in range(NH)}
    print(k, res[f'{k}_over_random'], flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
print('l5 heads function done', flush=True)
