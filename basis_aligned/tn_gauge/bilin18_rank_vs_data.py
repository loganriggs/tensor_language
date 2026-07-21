"""Rank vs amount of data (Logan 2026-07-21): does the effective rank of the QK-input activation
covariance keep growing as we double the tokens (i.e. is the ~128 effective rank real, or just
undersampling)? Accumulate C = sum x x^T token-by-token; snapshot eff-rank and rank@90% at doubling
token counts. Datasets: FineWeb (the model's TRAINING distribution) and the Pile. Layers 1/5/9.
Writes fig_rank_vs_data.png + json.
"""
import sys, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
rms = lambda x: F.rms_norm(x, (D,))
LAYERS = [1, 5, 9]
MAXL = max(LAYERS)


@torch.no_grad()
def qk_inputs_batch(idx):
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None; out = {}
    for li in range(MAXL + 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        if li in LAYERS:
            out[li] = h.reshape(-1, D)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(B, T, D))
        x = x + blk.mlp(rms(x))
    return out


def effrank(Cn):
    s = torch.linalg.svdvals(Cn.double()); p = s / s.sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


def rank_at(Cn, frac):
    s = torch.linalg.svdvals(Cn.double()); c = torch.cumsum(s, 0) / s.sum()
    return int((c < frac).sum()) + 1


# token-count thresholds (doubling); batch=1 early for fine granularity
THRESH = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


def run_dataset(seqs):
    """seqs: (N, T). Returns {layer: [(ntok, effrank, r90, r99), ...]} snapshotted at THRESH."""
    N = seqs.shape[0]
    Csum = {li: torch.zeros(D, D, device=DEV, dtype=torch.float64) for li in LAYERS}
    ntok = 0; ti = 0; res = {li: [] for li in LAYERS}
    i = 0
    while i < N:
        bs = 1 if ntok < 20000 else 20                    # fine granularity early, fast later
        idx = seqs[i:i + bs, :-1].to(DEV)
        acts = qk_inputs_batch(idx)
        for li in LAYERS:
            Csum[li] += (acts[li].double().T @ acts[li].double())
        ntok += idx.numel(); i += bs
        while ti < len(THRESH) and ntok >= THRESH[ti]:
            for li in LAYERS:
                Cn = Csum[li] / ntok
                res[li].append((ntok, effrank(Cn), rank_at(Cn, 0.90), rank_at(Cn, 0.99)))
            ti += 1
    for li in LAYERS:
        Cn = Csum[li] / ntok
        res[li].append((ntok, effrank(Cn), rank_at(Cn, 0.90), rank_at(Cn, 0.99)))
    return res


# FineWeb (training distribution)
fw = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
print(f'FineWeb {fw.shape}; running...', flush=True)
RES = {'fineweb': run_dataset(fw)}
# Pile
pile = build_eval_tokens(n_chunks=600, seq_len=513)[:600]
print(f'Pile {pile.shape}; running...', flush=True)
RES['pile'] = run_dataset(pile)

out = {ds: {li: [[int(t), round(er, 1), r90, r99] for (t, er, r90, r99) in rows] for li, rows in d.items()}
       for ds, d in RES.items()}
json.dump(out, open(f'{OUT}/bilin18_rank_vs_data.json', 'w'), indent=2)
print('\nlayer-1 eff-rank & rank@90% vs tokens:', flush=True)
for ds in ['fineweb', 'pile']:
    print(f'  {ds}:', flush=True)
    for (t, er, r90, r99) in RES[ds][1]:
        print(f'    {t:7d} tok: eff-rank {er:5.1f}  rank@90% {r90:4d}  rank@99% {r99:4d}', flush=True)

fig, ax = plt.subplots(1, len(LAYERS), figsize=(4.2 * len(LAYERS), 4.2), sharey=True)
for j, li in enumerate(LAYERS):
    for ds, col in [('fineweb', '#1a9850'), ('pile', '#3b7dd8')]:
        rows = RES[ds][li]
        ts = [r[0] for r in rows]
        ax[j].plot(ts, [r[1] for r in rows], 'o-', color=col, label=f'{ds} eff-rank')
        ax[j].plot(ts, [r[2] for r in rows], 's--', color=col, alpha=0.6, label=f'{ds} rank@90%')
    ax[j].set_xscale('log', base=2); ax[j].set_xlabel('tokens'); ax[j].set_title(f'layer {li} QK input')
    ax[j].grid(alpha=0.3, which='both')
    if j == 0:
        ax[j].set_ylabel(f'covariance rank (of {D})'); ax[j].legend(fontsize=7)
fig.suptitle('QK-input covariance rank vs amount of data (FineWeb=train dist, Pile); does rank keep growing?')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rank_vs_data.png', dpi=120)
print('\nsaved fig_rank_vs_data.png', flush=True)
