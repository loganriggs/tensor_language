"""TICK 206 (Logan): (A) data-sufficiency ladder for the manifold-rank claims;
(B) weight-space block decomposition of the MLP -> layer-1-QK channels over the
upstream writers (embedding vs each layer-0 attention head).

(A) The tick-203 channel ranks used 32,704 positions. Recompute effective rank and
r90 for all 36 channels at 32k / 131k / 523k positions (streamed covariances). If
"median effective rank 10" grows materially with data, the manifold claim shrinks.

(B) STAY IN WEIGHT SPACE: the MLP input is h = rms(x) with x = lambda-mixed embedding
+ attention-0 output; because the layer is bilinear, its output splits EXACTLY into
blocks: emb x emb (pure token identity — a weight-exact table), emb x attn0-head (9
cross blocks), attn0 x attn0. The per-position rms scalar multiplies everything
equally (a positive gauge), so BLOCK SHARES are gauge-clean. We evaluate the p-weighted
second moment of each block's contribution to each reader channel by Monte Carlo drawn
entirely from weights + the frozen unigram convention (no forward passes, no contexts):
  emb side:   a ~ hat-e_t, t ~ p (rms-normed embedding rows; exact writer)
  attn side:  b_h ~ g_h * (W_o^h v_t'), t' ~ p, with g_h^2 = E[P^2] pattern mass from
              the tick-192 weight/unigram sampler (per-head scale)
Per reader channel: share(emb^2), share(emb x attn_h) for each h, share(attn x attn),
computed from 8192 MC samples of each writer. Deliverable: the weight-space analog of
the data-side token-identity split (compare to the measured 0.56 median), and the
ranking of which layer-0 heads' outputs are actually READ by each layer-1 channel
through the MLP — Logan's "fold embedding and attn0 into the next QK" in weight space.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
NMC = 8192

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
blk0 = m.transformer.h[0]
mlp = blk0.mlp
Lw = mlp.Left.weight.detach().float()
Rw = mlp.Right.weight.detach().float()
Dw = mlp.Down.weight.detach().float()
a1 = m.transformer.h[1].attn
MAPS = (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2))
out = {}

# ================= (A) data-sufficiency ladder =================


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
    return mlp(F.rms_norm(x, (x.size(-1),)))


READERS = []
for mapname, lin in MAPS:
    W = lin.weight.detach().float()
    for h in range(NH):
        READERS.append((f'{mapname}_h{h}', W[h * HD:(h + 1) * HD].to(DEV)))

CHECKPOINTS = (64, 256, 1024)
sums = {name: torch.zeros(HD, device=DEV) for name, _ in READERS}
grams = {name: torch.zeros(HD, HD, device=DEV) for name, _ in READERS}
npos = 0
ladder = {}
with torch.no_grad():
    for i in range(0, 1024, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        mo = mlp_out_batch(idx).float().reshape(-1, D)
        for name, Wr in READERS:
            Y = mo @ Wr.T
            sums[name] += Y.sum(0)
            grams[name] += Y.T @ Y
        npos += mo.shape[0]
        if (i + 4) in CHECKPOINTS:
            effs, r90s = [], []
            for name, _ in READERS:
                mu = sums[name] / npos
                C = grams[name] / npos - torch.outer(mu, mu)
                ev = torch.linalg.eigvalsh(C).clamp_min(0)
                effs.append(float(ev.sum() ** 2 / (ev ** 2).sum()))
                cs = ev.flip(0).cumsum(0) / ev.sum()
                r90s.append(int((cs < 0.90).sum()) + 1)
            ladder[str(npos)] = {'median_eff': float(np.median(effs)),
                                 'max_eff': float(np.max(effs)),
                                 'median_r90': float(np.median(r90s))}
            print(f'{npos} positions: median eff-rank {np.median(effs):.1f} '
                  f'(max {np.max(effs):.1f}), median r90 {np.median(r90s):.0f}',
                  flush=True)
out['rank_ladder'] = ladder
json.dump(out, open(f'{QK}/qk_mlp_blocks.json', 'w'), indent=2)
del sums, grams
torch.cuda.empty_cache()

# ================= (B) weight-space block decomposition =================
wte = m.transformer.wte.weight.detach().float().to(DEV)
EMB = F.rms_norm(wte, (D,))
q1f, k1f = branch_factors(m, 1)
q2f, k2f = branch_factors(m, 2)
with torch.no_grad():
    a0 = blk0.attn
    Vv0 = a0.c_v(EMB).view(V, NH, HD)
    Wo0 = a0.c_proj.weight.detach().float().view(D, NH, HD)
# per-head pattern-mass scale g_h^2 = E_{i,t~p x p}[P(i,t)^2] (weights+unigram sampler)
g = torch.Generator().manual_seed(0)
gh2 = []
K1t, K2t = k1f.float().to(DEV), k2f.float().to(DEV)
Q1t, Q2t = q1f.float().to(DEV), q2f.float().to(DEV)
for h in range(NH):
    si = torch.multinomial(QP.cpu(), 4096, replacement=True, generator=g).to(DEV)
    ti = torch.multinomial(QP.cpu(), 4096, replacement=True, generator=g).to(DEV)
    s1 = (Q1t[si, h] * K1t[ti, h]).sum(1) / HD
    s2 = (Q2t[si, h] * K2t[ti, h]).sum(1) / HD
    gh2.append(float((s1 ** 2 * s2 ** 2).mean()))
gh = torch.tensor(gh2, device=DEV).sqrt()

lam0 = blk0.lambdas.detach().float()
scale_emb = float(lam0[0] + lam0[1])
gmc = torch.Generator().manual_seed(1)
ti_a = torch.multinomial(QP.cpu(), NMC, replacement=True, generator=gmc).to(DEV)
A_emb = scale_emb * EMB[ti_a]                                   # (NMC, D) emb writer
B_att = {}
for h in range(NH):
    tv = torch.multinomial(QP.cpu(), NMC, replacement=True, generator=gmc).to(DEV)
    B_att[h] = gh[h] * (Vv0[tv, h] @ Wo0[:, h].T)               # (NMC, D)
B_sum = sum(B_att.values())

Lg, Rg, Dg = Lw.to(DEV), Rw.to(DEV), Dw.to(DEV)


def channel_second_moment(Wr, xa, xb):
    """E || Wr Down( (L xa) . (R xb) sym ) ||^2 over MC pairs (xa_i, xb_i)."""
    acts = 0.5 * ((xa @ Lg.T) * (xb @ Rg.T) + (xb @ Lg.T) * (xa @ Rg.T))
    Y = acts @ Dg.T @ Wr.T
    return float((Y ** 2).sum(1).mean())


blocks = {}
for name, Wr in READERS[:9] + READERS[9:18]:                    # q1 + k1 channels
    row = {}
    ee = channel_second_moment(Wr, A_emb, A_emb)
    row['emb2'] = ee
    for h in range(NH):
        row[f'embxattn_h{h}'] = 2.0 * channel_second_moment(Wr, A_emb, B_att[h])
    row['attn2'] = channel_second_moment(Wr, B_sum, B_sum)
    tot = sum(row.values())
    blocks[name] = {k: round(v / tot, 4) for k, v in row.items()}
    cross = {h: blocks[name][f'embxattn_h{h}'] for h in range(NH)}
    toph = max(cross, key=cross.get)
    print(f'{name}: emb2 {blocks[name]["emb2"]:.3f} attn2 {blocks[name]["attn2"]:.3f} '
          f'cross-total {sum(cross.values()):.3f} (top l0-head {toph} '
          f'{cross[toph]:.3f})', flush=True)
out['blocks'] = blocks
emb2s = [b['emb2'] for b in blocks.values()]
out['summary'] = {'emb2_median': float(np.median(emb2s)),
                  'cross_median': float(np.median([sum(v for k, v in b.items()
                                                       if k.startswith('embx'))
                                                   for b in blocks.values()])),
                  'attn2_median': float(np.median([b['attn2'] for b in blocks.values()]))}
print('summary:', out['summary'], flush=True)
json.dump(out, open(f'{QK}/qk_mlp_blocks.json', 'w'), indent=2)
print('MLP BLOCKS DONE', flush=True)
