"""LINEAR + k BILINEAR FEATURES (Logan): keep the linear part, then add back a LOW-RANK bilinear
component fit to the remainder.

Each block already is a sum of rank-1 bilinear features:
    mlp(x) = Down(Lx o Rx) = sum_p Down[:,p] * (L_p . x)(R_p . x)      (4608 features)
So the model class "linear plus k bilinear" is exactly:
    mlp_k(x) = W x + b  +  sum_{p in S_k} u_p * h_p(x),   h_p(x) = (L_p . x)(R_p . x)
with S_k the k highest-contribution features and u_p REFIT by least squares against the residual.
(This differs from the earlier probe, which projected the residual's OUTPUT onto k directions while
still computing the full quadratic; here the quadratic itself is rank-k.)

All 18 blocks, sequential fits (each block fit under the already-approximated upstream).
Held FW[448:600,:128] = 152 sequences x 127 scored tokens; paired standard errors over sequences.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0); DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18'); NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 6
KS = [0, 8, 32, 128, 512]; KTOP = 512
LIN, SEL, UMAT = {}, {}, {}


@torch.no_grad()
def fwd(idx, mode=None, k=0, upto=None, fit_layer=None, acc=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        hcur = F.rms_norm(x, (D,))
        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,))
        mlp = blk.mlp
        hp = mlp.Left(xhat) * mlp.Right(xhat)              # (B,T,4608) the rank-1 bilinear features
        mo_true = mlp.Down(hp) + mlp.Down_bias
        if fit_layer is not None and li == fit_layer:
            acc.append((xhat.detach(), mo_true.detach(), hp.detach()))
        if mode == 'approx' and li in LIN and (upto is None or li < upto):
            W, b = LIN[li]; mo = xhat @ W + b
            if k > 0:
                mo = mo + hp[..., SEL[li][:k]] @ UMAT[li][:k]
        else:
            mo = mo_true
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T - 1)
    return ce


def held(**kw):
    return torch.cat([fwd(HELD[i:i + B0], **kw) for i in range(0, HELD.shape[0], B0)])


base = held(); BASE = float(base.mean())
n_tok = base.shape[0] * base.shape[1]
print(f'GATE base CE {BASE:.4f}  |  held tokens scored: {base.shape[0]} seqs x {base.shape[1]} = {n_tok}', flush=True)
res = {'meta': {'base_ce': round(BASE, 4), 'held_sequences': int(base.shape[0]),
                'held_scored_tokens': int(n_tok), 'ks': KS,
                'model_class': 'mlp_k(x) = W x + b + sum_{p in S_k} u_p (L_p.x)(R_p.x), u refit by least squares'}}

print('fitting 18 blocks sequentially (linear map + top-512 feature refit) ...', flush=True)
for li in range(NL):
    # pass 1: linear map + feature selection statistics
    A = torch.zeros(D + 1, D + 1, device=DEV, dtype=torch.float64)
    Bm = torch.zeros(D + 1, D, device=DEV, dtype=torch.float64)
    hsq = torch.zeros(4608, device=DEV, dtype=torch.float64); hs = torch.zeros(4608, device=DEV, dtype=torch.float64); n = 0
    for i in range(0, TRAIN.shape[0], B0):
        acc = []; fwd(TRAIN[i:i + B0], mode='approx', upto=li, fit_layer=li, acc=acc)
        xh, mo, hp = acc[0]
        xa = torch.cat([xh, torch.ones_like(xh[..., :1])], -1).reshape(-1, D + 1).double()
        A += xa.T @ xa; Bm += xa.T @ mo.reshape(-1, D).double()
        h2 = hp.reshape(-1, 4608).double(); hsq += (h2 ** 2).sum(0); hs += h2.sum(0); n += h2.shape[0]
    sol = torch.linalg.solve(A + 1.0 * torch.eye(D + 1, device=DEV, dtype=torch.float64), Bm).float()
    LIN[li] = (sol[:D], sol[D])
    dn = m.transformer.h[li].mlp.Down.weight.detach()          # (1152, 4608)
    score = (hsq / n - (hs / n) ** 2).clamp_min(0).sqrt().float() * dn.norm(dim=0)
    SEL[li] = score.topk(KTOP).indices
    # pass 2: refit output vectors of the selected features against the residual
    C = torch.zeros(KTOP, KTOP, device=DEV, dtype=torch.float64)
    Bh = torch.zeros(KTOP, D, device=DEV, dtype=torch.float64)
    for i in range(0, TRAIN.shape[0], B0):
        acc = []; fwd(TRAIN[i:i + B0], mode='approx', upto=li, fit_layer=li, acc=acc)
        xh, mo, hp = acc[0]
        r = (mo - (xh @ LIN[li][0] + LIN[li][1])).reshape(-1, D).double()
        hsel = hp[..., SEL[li]].reshape(-1, KTOP).double()
        C += hsel.T @ hsel; Bh += hsel.T @ r
    UMAT[li] = torch.linalg.solve(C + 1e-2 * torch.eye(KTOP, device=DEV, dtype=torch.float64), Bh).float()
    print(f'  layer {li} fitted', flush=True)

print('=== all 18 blocks: linear + k bilinear features ===', flush=True)
cells = {}
for k in KS:
    ce = held(mode='approx', k=k); d = ce - base
    v = (round(float(d.mean()), 4), round(float(d.mean(1).std() / np.sqrt(d.shape[0])), 4))
    frac = round(float(k) / 4608, 4)
    print(f'  k={k:4d} ({frac:6.2%} of the 4608 features)   dCE {v[0]:+.4f} +- {v[1]:.4f}', flush=True)
    cells[f'k{k}'] = v
res['cells'] = cells
json.dump(res, open(f'{QK}/qk_lin_plus_bilin.json', 'w'), indent=1)
print('SAVED qk_lin_plus_bilin.json', flush=True)
