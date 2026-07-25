"""TICK 203: data-weighted rank of the MLP -> layer-1-QK channels. Weight-space ranks
(tick 202: median 68/128) count directions the data never visits; the program's
recurring lesson (folding, metric-decides-superposition) says measure on the manifold.

For 64 held-out documents: capture the MLP's actual per-position OUTPUT (Down applied
to gated activations), push through each of the 36 layer-1 reader maps (W_head, per
q1/k1/q2/k2), and measure the effective rank and r90 of the resulting 128-dim output
COVARIANCE (p-realized, mean-removed). Also, per channel, the token-identity split of
the channel output (variance explained by token-conditional means, split-half) — the
context share of what each reader actually receives from the MLP. If data ranks
collapse (e.g. r90 ~ 10-30 vs weight 92), the circuit-relative decomposition should be
fit on the manifold: tables + low-rank context channel per reader.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_CAP = 64

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
blk0 = m.transformer.h[0]
mlp = blk0.mlp
a1 = m.transformer.h[1].attn


@torch.no_grad()
def mlp_out_batch(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    x = blk0.lambdas[0] * x + blk0.lambdas[1] * x0
    a = blk0.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cos, sin)

    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    return mlp(F.rms_norm(x, (x.size(-1),)))          # (B, T, D) MLP output


print('capturing MLP outputs...', flush=True)
OUTS, IDS = [], []
with torch.no_grad():
    for i in range(0, N_CAP, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        OUTS.append(mlp_out_batch(idx).float().reshape(-1, D).cpu())
        IDS.append(idx.reshape(-1).cpu())
MO = torch.cat(OUTS)
IDS = torch.cat(IDS)
N = MO.shape[0]
print(f'{N} positions', flush=True)

out = {}
ranks = {}
tokr2 = {}
half = N // 2
cnt_a = torch.bincount(IDS[:half], minlength=V).float()
ok = cnt_a >= 4
for mapname, lin in (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2)):
    W = lin.weight.detach().float().cpu()
    for h in range(NH):
        Y = MO @ W[h * HD:(h + 1) * HD].T                # (N, 128)
        Yg = (Y - Y.mean(0)).to(DEV)
        C = Yg.T @ Yg / N
        ev = torch.linalg.eigvalsh(C).clamp_min(0)
        eff = float(ev.sum() ** 2 / (ev ** 2).sum())
        cs = ev.flip(0).cumsum(0) / ev.sum()
        r90 = int((cs < 0.90).sum()) + 1
        ranks[f'{mapname}_h{h}'] = {'eff': round(eff, 1), 'r90': r90}
        # token-identity split (split-half)
        sum_a = torch.zeros(V, HD)
        sum_a.index_add_(0, IDS[:half], Y[:half])
        mean_a = sum_a / cnt_a[:, None].clamp_min(1)
        selb = ok[IDS[half:]]
        pred = mean_a[IDS[half:][selb]]
        resid = Y[half:][selb] - pred
        r2 = 1 - float((resid ** 2).sum()) / float(
            ((Y[half:][selb] - Y[half:][selb].mean(0)) ** 2).sum())
        tokr2[f'{mapname}_h{h}'] = round(r2, 3)
        del Yg, C
        torch.cuda.empty_cache()
effs = [v['eff'] for v in ranks.values()]
r90s = [v['r90'] for v in ranks.values()]
r2s = list(tokr2.values())
out['data_rank'] = {'min_eff': min(effs), 'median_eff': float(np.median(effs)),
                    'max_eff': max(effs), 'median_r90': float(np.median(r90s)),
                    'detail': ranks}
out['token_r2'] = {'median': float(np.median(r2s)), 'min': min(r2s), 'max': max(r2s),
                   'detail': tokr2}
print(f'DATA channel rank (of 128): min {min(effs):.0f} median {np.median(effs):.0f} '
      f'max {max(effs):.0f}; median r90 {np.median(r90s):.0f} (weight-space median was '
      f'68 / r90 92)', flush=True)
print(f'channel-output token-identity R2: median {np.median(r2s):.3f} '
      f'range [{min(r2s):.3f}, {max(r2s):.3f}]', flush=True)
json.dump(out, open(f'{QK}/qk_mlp_datarank.json', 'w'), indent=2)
print('MLP DATARANK DONE', flush=True)
