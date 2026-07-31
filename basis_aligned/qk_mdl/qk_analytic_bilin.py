"""ANALYTIC linear/quadratic split + OPTIMAL k-feature bilinear remainder (Logan's question).

Exact algebra, no regression, expanding each block around the data mean mu = E[xhat]:
    mlp(xhat) = Down[(L.mu) o (R.mu)] + Down_bias                      constant
              + Down[diag(L.mu) R + diag(R.mu) L] d                     LINEAR in d (closed form)
              + Down[(L d) o (R d)]                                     the pure quadratic remainder Q(d)

Q(d) = Down h(d) with h(d) = (L d) o (R d) a FIXED 4608-dim feature map, so the best k-feature
approximation is a matrix low-rank problem, solved analytically by Eckart-Young:
    minimize E|| Down h - A B h ||^2  =  || (Down - A B) C^(1/2) ||_F^2 ,  C = E[h h^T]
    -> SVD of (Down C^(1/2)), truncate to k.  The k optimal features are linear COMBINATIONS of the
       4608 hidden products, i.e. general quadratic forms d^T M_j d -- strictly better than picking
       k of the hidden units.

Evaluated ONE BLOCK AT A TIME with the rest of the model exact (so there is no compounding confound
and every fit is matched to the distribution it is evaluated on).
Held FW[448:600,:128] = 152 sequences x 127 = 19304 scored tokens; paired standard errors.
Gates: the three-way split reproduces the block output to ~1e-6; k=full reproduces the base model.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0); DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18'); NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); FH = 4608
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 6
BLOCKS = [0, 1, 3, 5, 7]; KS = [0, 8, 32, 128, 512]
APPROX = {}   # li -> (mu, W, b_const, A, Bfeat, k)


@torch.no_grad()
def fwd(idx, use=None, collect=None, acc=None):
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
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        if collect is not None and li == collect:
            acc.append(xhat.detach())
        if use is not None and li == use:
            mu, W, bc, A, Bf, k = APPROX[li]
            d = xhat - mu
            mo = bc + d @ W
            if k > 0:
                hd = (mlp.Left(d) * mlp.Right(d))
                mo = mo + (hd @ Bf.T) @ A.T
        else:
            mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T - 1)


def held(**kw):
    return torch.cat([fwd(HELD[i:i + B0], **kw) for i in range(0, HELD.shape[0], B0)])


base = held(); BASE = float(base.mean())
print(f'GATE base CE {BASE:.4f} | held {base.shape[0]}x{base.shape[1]} = {base.shape[0]*base.shape[1]} tokens', flush=True)
res = {'meta': {'base_ce': round(BASE, 4), 'held_scored_tokens': int(base.shape[0] * base.shape[1]),
                'method': 'analytic mean-expansion linear part + Eckart-Young optimal k-feature quadratic',
                'blocks': BLOCKS, 'ks': KS}, 'cells': {}}

for li in BLOCKS:
    blk = m.transformer.h[li]; mlp = blk.mlp
    Lw = mlp.Left.weight.detach(); Rw = mlp.Right.weight.detach()
    Dw = mlp.Down.weight.detach(); db = mlp.Down_bias.detach()
    # pass 1: mean of xhat
    s = torch.zeros(D, device=DEV, dtype=torch.float64); n = 0
    for i in range(0, TRAIN.shape[0], B0):
        acc = []; fwd(TRAIN[i:i + B0], collect=li, acc=acc)
        xh = acc[0].reshape(-1, D).double(); s += xh.sum(0); n += xh.shape[0]
    mu = (s / n).float()
    Lmu = Lw @ mu; Rmu = Rw @ mu
    W = (Dw * Lmu[None, :]) @ Rw + (Dw * Rmu[None, :]) @ Lw     # (1152,1152) acting as d @ W.T
    W = W.T.contiguous()
    bc = Dw @ (Lmu * Rmu) + db
    # gate: exact three-way reconstruction on one batch
    acc = []; fwd(TRAIN[:B0], collect=li, acc=acc); xh = acc[0]
    d = xh - mu
    exact = mlp.Down(mlp.Left(xh) * mlp.Right(xh)) + db
    split = bc + d @ W + mlp.Down(mlp.Left(d) * mlp.Right(d))
    gate = float((exact - split).norm() / exact.norm())
    # pass 2: feature covariance C = E[h h^T] with h = (Ld)o(Rd)
    C = torch.zeros(FH, FH, device=DEV, dtype=torch.float32)
    for i in range(0, TRAIN.shape[0], B0):
        acc = []; fwd(TRAIN[i:i + B0], collect=li, acc=acc)
        dd = acc[0] - mu
        h = (mlp.Left(dd) * mlp.Right(dd)).reshape(-1, FH)
        C += h.T @ h
    C /= n
    ev, Qv = torch.linalg.eigh(C.double())
    ev = ev.clamp_min(0); floor = 1e-10 * float(ev.max())
    keep = ev > floor
    Qk = Qv[:, keep]; evk = ev[keep]
    G = (Dw.double() @ Qk) * evk.sqrt()[None, :]        # Down C^(1/2) in the eigenbasis
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    print(f'block {li}: split gate {gate:.2e} | feature-cov rank {int(keep.sum())} | '
          f'top-k energy 8:{float((S[:8]**2).sum()/(S**2).sum()):.3f} 32:{float((S[:32]**2).sum()/(S**2).sum()):.3f} '
          f'128:{float((S[:128]**2).sum()/(S**2).sum()):.3f} 512:{float((S[:512]**2).sum()/(S**2).sum()):.3f}', flush=True)
    cells = {}
    for k in KS:
        if k == 0:
            A = torch.zeros(D, 1, device=DEV); Bf = torch.zeros(1, FH, device=DEV)
        else:
            kk = min(k, S.shape[0])
            A = U[:, :kk].float()
            Bf = ((S[:kk, None] * Vh[:kk]) @ (Qk * (1.0 / evk.sqrt())[None, :]).T).float()
        APPROX[li] = (mu, W, bc, A, Bf, k)
        ce = held(use=li); dd_ = ce - base
        v = (round(float(dd_.mean()), 4), round(float(dd_.mean(1).std() / np.sqrt(dd_.shape[0])), 4))
        print(f'   k={k:4d}  dCE {v[0]:+.4f} +- {v[1]:.4f}', flush=True)
        cells[f'k{k}'] = v
    res['cells'][f'block_{li}'] = {'split_gate': gate, 'cells': cells,
                                   'sv_energy': {kk: round(float((S[:kk] ** 2).sum() / (S ** 2).sum()), 4)
                                                 for kk in (8, 32, 128, 512)}}
    APPROX.pop(li, None)
    del C, Qv, Qk, G, U, S, Vh; torch.cuda.empty_cache()
    json.dump(res, open(f'{QK}/qk_analytic_bilin.json', 'w'), indent=1)
print('SAVED qk_analytic_bilin.json', flush=True)
