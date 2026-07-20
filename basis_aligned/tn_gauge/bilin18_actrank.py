"""Does the flagship have the LOW-RANK residual stream the code-propagation regime
needs? (Logan 2026-07-20). Gate 2b on the toy (d=128) found the coding residual is
isotropic because the d=128 activations are near-full-rank (eff-rank ~115/128) -- too
small to test the regime's premise (activations low-rank relative to width). Here:
measure the effective rank of the residual stream vs depth on bilin18 (d=1152), the
cheap premise test (no dictionary needed). If eff-rank << 1152 and stays modest with
depth, the sparse-code regime is worth pursuing on the flagship; if it climbs toward
full rank, the regime is in the same trouble as the toy."""
import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TOK = build_eval_tokens(n_chunks=8, seq_len=513)[:8]


def eff_rank(X):
    s = torch.linalg.svdvals(X.double())
    p = s / s.sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


def rank_at(X, frac):
    s = torch.linalg.svdvals(X.double()) ** 2
    c = torch.cumsum(s, 0) / s.sum()
    return int((c < frac).sum().item()) + 1


# inline forward capturing the residual stream at each block input (rms-normed)
res = {'d_model': D, 'n_layer': len(m.transformer.h), 'per_bond': []}
Hcap = []
with torch.no_grad():
    x = m.transformer.wte(TOK[:, :-1].to(DEV))
    x0 = x
    B, T = x.shape[0], x.shape[1]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        Hcap.append(F.rms_norm(x, (D,)).reshape(-1, D).float().cpu())   # bond input
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))

print(f'bilin18 d={D}, {len(Hcap)} bonds, {Hcap[0].shape[0]} tokens', flush=True)
print('  bond | eff-rank | rank@90% | rank@99% | (of d={})'.format(D), flush=True)
for li, Hc in enumerate(Hcap):
    Xc = Hc - Hc.mean(0)
    er = eff_rank(Xc); r90 = rank_at(Xc, 0.90); r99 = rank_at(Xc, 0.99)
    res['per_bond'].append({'bond': li, 'eff_rank': round(er, 1), 'rank90': r90, 'rank99': r99})
    if li % 2 == 0 or li == len(Hcap) - 1:
        print(f'  {li:2d}   |  {er:6.1f}  |   {r90:4d}   |   {r99:4d}   |', flush=True)
json.dump(res, open(f'{OUT}/bilin18_actrank.json', 'w'), indent=2)
print('bilin18 actrank done -> bilin18_actrank.json', flush=True)
