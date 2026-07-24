"""TICK 200: structure of layer-1's CONTEXT-DEPENDENT pattern remainder — the part the
token tables miss (+0.027 of +2.70) and the part that self-repaired in tick 197.

For 64 held-out co-occurrence documents, capture at every position: the actual layer-1
query/key factors (pre-rotary, per-head rms-normed, exactly as the model computes them),
the per-l0-head attention outputs y_h (head space, pre-projection), and the block-0 MLP
output. Deviation = actual factor minus the (shrunk) token-table factor for that token.

Questions, per layer-1 head:
  (1) SIZE: p-weighted relative norm of the deviation vs the factor itself.
  (2) RANK: singular spectrum of the deviation matrix (65k positions x 128) — variance
      explained by top 1/4/16/64 directions. Low-rank => a small context summary
      suffices; the program's first two-layer object would be tables + r context dims.
  (3) SOURCE: ridge regression R^2 of the deviation onto (a) the nine layer-0 attention
      head outputs at that position (1152 features), (b) the block-0 MLP output (1152),
      (c) both — does layer 1's context-dependence read layer-0 attention, the MLP, or
      information neither carries linearly?
Both query (q1) and key (k1) deviations analyzed (branch 1; branch 2 spot-checked on
one head for symmetry of conclusions).
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
N_EST, TAU, N_CAP = 1024, 8.0, 64

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()


@torch.no_grad()
def block0_forward(idx, capture=False):
    """Returns block-1 input; if capture, also per-l0-head outputs y (B,T,NH,HD) and
    block-0 mlp output (B,T,D)."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    a = blk.attn
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
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v)                 # (B,T,NH,HD)
    x = x + a.c_proj(y.reshape(B, T, -1))
    mlp_out = blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = x + mlp_out
    blk1 = m.transformer.h[1]
    xin1 = blk1.lambdas[0] * x + blk1.lambdas[1] * x0
    if capture:
        return xin1, y, mlp_out
    return xin1


# ---- shrunk token tables (pre-rotary factors) ----
print('estimating shrunk means...', flush=True)
sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        x = block0_forward(idx).float().reshape(-1, D)
        ids = idx.reshape(-1)
        sum_x.index_add_(0, ids, x)
        cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
wte = m.transformer.wte.weight.detach().float().to(DEV)
mean_x = torch.where((cnt > 0)[:, None], sum_x / cnt[:, None].clamp_min(1), wte)
shr = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
del sum_x, mean_x
torch.cuda.empty_cache()
a1 = m.transformer.h[1].attn
with torch.no_grad():
    xn = F.rms_norm(shr, (D,))
    Tq1 = F.rms_norm(a1.c_q(xn).view(V, NH, HD).float(), (HD,))
    Tk1 = F.rms_norm(a1.c_k(xn).view(V, NH, HD).float(), (HD,))
del shr, xn
torch.cuda.empty_cache()

# ---- capture actual factors + sources ----
print('capturing actual factors and sources...', flush=True)
DQ, DK, YS, MLP, IDS = [], [], [], [], []
with torch.no_grad():
    for i in range(0, N_CAP, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        xin1, y, mlp_out = block0_forward(idx, capture=True)
        B, T = idx.shape
        h1n = F.rms_norm(xin1, (D,))
        q1a = F.rms_norm(a1.c_q(h1n).view(B, T, NH, HD).float(), (HD,))
        k1a = F.rms_norm(a1.c_k(h1n).view(B, T, NH, HD).float(), (HD,))
        ids = idx.reshape(-1)
        DQ.append((q1a.reshape(-1, NH, HD) - Tq1[ids]).cpu())
        DK.append((k1a.reshape(-1, NH, HD) - Tk1[ids]).cpu())
        YS.append(y.reshape(-1, NH * HD).float().cpu())
        MLP.append(mlp_out.reshape(-1, D).float().cpu())
        IDS.append(ids.cpu())
DQ = torch.cat(DQ)
DK = torch.cat(DK)
YS = torch.cat(YS)
MLP = torch.cat(MLP)
IDS = torch.cat(IDS)
N = DQ.shape[0]
print(f'{N} positions captured', flush=True)


def ridge_r2(X, Y, lam=1e-3):
    X = X.to(DEV)
    Y = Y.to(DEV)
    Xm = X - X.mean(0)
    Ym = Y - Y.mean(0)
    G = Xm.T @ Xm + lam * N * torch.eye(X.shape[1], device=DEV)
    W = torch.linalg.solve(G, Xm.T @ Ym)
    resid = Ym - Xm @ W
    r2 = 1 - float((resid ** 2).sum()) / float((Ym ** 2).sum())
    del X, Y, Xm, Ym, G, W, resid
    torch.cuda.empty_cache()
    return r2


out = {}
for h in range(NH):
    row = {}
    for name, Dv, Tf in (('q1', DQ[:, h], Tq1), ('k1', DK[:, h], Tk1)):
        rel = float(Dv.norm() / Tf[IDS.to(DEV)][:, h].cpu().norm().clamp_min(1e-9))
        Dg = Dv.to(DEV)
        U_, S_, _ = torch.linalg.svd(Dg - Dg.mean(0), full_matrices=False)
        tot = float((S_ ** 2).sum())
        spec = {r: round(float((S_[:r] ** 2).sum()) / tot, 3) for r in (1, 4, 16, 64)}
        r2_att = ridge_r2(YS, Dv)
        r2_mlp = ridge_r2(MLP, Dv)
        r2_both = ridge_r2(torch.cat([YS, MLP], 1), Dv)
        row[name] = {'rel_norm': round(rel, 3), 'var_top_r': spec,
                     'R2_l0attn': round(r2_att, 3), 'R2_mlp': round(r2_mlp, 3),
                     'R2_both': round(r2_both, 3)}
        del Dg
        torch.cuda.empty_cache()
    out[f'h{h}'] = row
    print(f'l1 h{h} q1: rel {row["q1"]["rel_norm"]} top16 {row["q1"]["var_top_r"][16]} '
          f'R2 attn/mlp/both {row["q1"]["R2_l0attn"]}/{row["q1"]["R2_mlp"]}/{row["q1"]["R2_both"]} | '
          f'k1: rel {row["k1"]["rel_norm"]} top16 {row["k1"]["var_top_r"][16]} '
          f'R2 {row["k1"]["R2_l0attn"]}/{row["k1"]["R2_mlp"]}/{row["k1"]["R2_both"]}', flush=True)
    json.dump(out, open(f'{QK}/qk_l1_context.json', 'w'), indent=2)
print('L1 CONTEXT DONE', flush=True)
