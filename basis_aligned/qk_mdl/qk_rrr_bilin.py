"""THE FRONTIER: least-squares linear part + OPTIMAL k bilinear features on ITS residual.

The previous run split analytically (expansion about the mean), which is exact but whose linear term is
NOT the best linear map -- least squares also absorbs the quadratic's mean and its third-moment
correlation with d. Here:

    step 1  W,b = argmin E|| mlp(xhat) - xhat W - b ||^2                (ordinary least squares)
    step 2  r   = mlp(xhat) - xhat W - b       (by construction E[r]=0 and E[d^T r]=0)
    step 3  approximate r by k bilinear features h(d) = (L d) o (R d), d = xhat - mu:
              min_{A,B} E|| r - h B A ||^2   with rank k
            -> REDUCED-RANK REGRESSION, analytic:
               Bo = C_hh^{-1} C_hr  (ordinary least squares on the features)
               M  = Bo^T C_hh Bo    (covariance of the fitted values)
               U_k = top-k eigenvectors of M   ->   approximation = h Bo U_k U_k^T

This is the optimal member of the class "linear map plus k bilinear features" for each k.
One block at a time, rest of the model exact (matched; no compounding).
Held FW[448:600,:128] = 152 x 127 = 19304 scored tokens; paired standard errors.
GATE: k=0 must reproduce the least-squares linear caps from qk_degree_ablation.json.
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
AP = {}


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
            W, b, mu, hmean, Bk, Ak = AP[li]
            mo = xhat @ W + b
            if Bk is not None:
                d = xhat - mu
                h = (mlp.Left(d) * mlp.Right(d)) - hmean
                mo = mo + (h @ Bk) @ Ak
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
prev = json.load(open(f'{QK}/qk_degree_ablation.json'))['block_cells']
res = {'meta': {'base_ce': round(BASE, 4), 'held_scored_tokens': int(base.shape[0] * base.shape[1]),
                'method': 'least-squares linear + reduced-rank-regression optimal k bilinear features',
                'blocks': BLOCKS, 'ks': KS}, 'cells': {}}

for li in BLOCKS:
    mlp = m.transformer.h[li].mlp
    # ---- pass 1: mean of xhat ----
    s = torch.zeros(D, device=DEV, dtype=torch.float64); n = 0
    for i in range(0, TRAIN.shape[0], B0):
        acc = []; fwd(TRAIN[i:i + B0], collect=li, acc=acc)
        xh = acc[0].reshape(-1, D).double(); s += xh.sum(0); n += xh.shape[0]
    mu = (s / n).float()
    # ---- pass 2: all cross-moments ----
    Sxx = torch.zeros(D + 1, D + 1, device=DEV, dtype=torch.float64)
    Sxm = torch.zeros(D + 1, D, device=DEV, dtype=torch.float64)
    Shh = torch.zeros(FH, FH, device=DEV, dtype=torch.float64)
    Shm = torch.zeros(FH, D, device=DEV, dtype=torch.float64)
    Shx = torch.zeros(FH, D + 1, device=DEV, dtype=torch.float64)
    hsum = torch.zeros(FH, device=DEV, dtype=torch.float64)
    for i in range(0, TRAIN.shape[0], B0):
        acc = []; fwd(TRAIN[i:i + B0], collect=li, acc=acc)
        xh = acc[0]
        mo = (mlp.Down(mlp.Left(xh) * mlp.Right(xh)) + mlp.Down_bias).reshape(-1, D).double()
        d = xh - mu
        h = (mlp.Left(d) * mlp.Right(d)).reshape(-1, FH).double()
        xa = torch.cat([xh.reshape(-1, D), torch.ones(xh.reshape(-1, D).shape[0], 1, device=DEV)], 1).double()
        Sxx += xa.T @ xa; Sxm += xa.T @ mo
        Shh += h.T @ h; Shm += h.T @ mo; Shx += h.T @ xa; hsum += h.sum(0)
    sol = torch.linalg.solve(Sxx + 1.0 * torch.eye(D + 1, device=DEV, dtype=torch.float64), Sxm)
    W = sol[:D].float(); b = sol[D].float()
    hmean = (hsum / n)
    # centered feature covariance and cross-covariance with the residual r = mo - xa @ sol
    Chh = Shh / n - torch.outer(hmean, hmean)
    Chr = (Shm - Shx @ sol) / n            # E[r]=0 by the normal equations, so no mean correction needed
    Bo = torch.linalg.solve(Chh + 1e-3 * torch.eye(FH, device=DEV, dtype=torch.float64) * float(Chh.diagonal().mean()), Chr)
    M = Bo.T @ Chh @ Bo
    evals, U = torch.linalg.eigh(M); U = U.flip(1); evals = evals.flip(0).clamp_min(0)
    energy = {kk: round(float(evals[:kk].sum() / evals.sum()), 4) for kk in (8, 32, 128, 512)}
    gate_ref = prev.get(f'L{li}_linear', [None])[0]
    print(f'block {li}: RRR eigen-energy {energy} | k=0 should reproduce {gate_ref}', flush=True)
    cells = {}
    for k in KS:
        if k == 0:
            AP[li] = (W, b, mu, hmean.float(), None, None)
        else:
            Uk = U[:, :k]
            Bk = (Bo @ Uk).float()          # FH x k
            Ak = Uk.T.float()               # k x D
            AP[li] = (W, b, mu, hmean.float(), Bk, Ak)
        ce = held(use=li); dd = ce - base
        v = (round(float(dd.mean()), 4), round(float(dd.mean(1).std() / np.sqrt(dd.shape[0])), 4))
        print(f'   k={k:4d}  dCE {v[0]:+.4f} +- {v[1]:.4f}', flush=True)
        cells[f'k{k}'] = v
    res['cells'][f'block_{li}'] = {'rrr_eigen_energy': energy, 'gate_k0_ref': gate_ref, 'cells': cells}
    AP.pop(li, None)
    del Shh, Shm, Shx, Chh, Chr, Bo, M, U
    torch.cuda.empty_cache()
    json.dump(res, open(f'{QK}/qk_rrr_bilin.json', 'w'), indent=1)
print('SAVED qk_rrr_bilin.json', flush=True)
