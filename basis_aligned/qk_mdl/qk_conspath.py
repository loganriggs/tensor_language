"""CONSUMPTION-PATH SPLIT for block 0 (Logan): is block 0's (linear) write there for the readout, or for
downstream nonlinear consumers (norms + products)?

A contribution c added at block 0's MLP output reaches the final pre-readout residual by TWO routes:
  (i)  DIRECT skip: multiplied only by the residual-mixing factors, Lambda = prod_{l=1..17} lambda0[l].
       (attn/mlp additions are separate contributions; the skip carries c linearly.)
  (ii) INDIRECT: read by every intermediate block's rms_norm and bilinear projections -> nonlinear responses.

Test: mean-ablate block 0 (so intermediate blocks see the per-position mean, i.e. NONE of the token-specific
deviation d0), then add Lambda * d0 to x just before the readout rms_norm. That reproduces route (i) exactly
and removes route (ii). Compare to injecting d0 normally at block 0 (both routes).

    dCE(normal) - dCE(bypass)  =  the causal effect that flows ONLY through downstream nonlinearities.

Done for d0 = the LINEAR-cap deviation and for d0 = the FULL block-0 deviation.
Held FW[448:600,:128] = 152 x 127 = 19304 tokens; paired SE. Gates: full model ~0; mean floor +1.2341;
linear normal +0.0796 (reproduce qk_degree_ablation).
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

# residual-mixing product Lambda = prod_{l=1..17} lambda0[l]
with torch.no_grad():
    lam = torch.ones(D, device=DEV)
    for l in range(1, NL):
        l0 = m.transformer.h[l].lambdas[0]
        lam = lam * (l0.to(DEV) if torch.is_tensor(l0) else float(l0))
LAM = lam
print(f'Lambda (skip-propagation factor) mean {float(LAM.mean()):.4f} min {float(LAM.min()):.4f} max {float(LAM.max()):.4f}', flush=True)

LINW = None; LINB = None; MEAN0 = None


@torch.no_grad()
def fwd(idx, mode=None, collect0=None, acc=None):
    """mode: None(full) | 'mean' | 'lin' | 'lin_bypass' | 'full_bypass'."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    stash = None
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
        xhat = F.rms_norm(x, (D,))
        mo_true = blk.mlp(xhat)
        if collect0 is not None and li == 0:
            acc.append((xhat.detach(), mo_true.detach()))
        if li == 0 and mode in ('mean', 'lin', 'lin_bypass', 'full_bypass'):
            lincap = xhat @ LINW + LINB
            mean0 = MEAN0.unsqueeze(0)
            if mode == 'mean':
                mo = mean0.expand(B, -1, -1)
            elif mode == 'lin':
                mo = lincap
            elif mode == 'lin_bypass':
                stash = LAM * (lincap - mean0)          # deviation, direct-route only
                mo = mean0.expand(B, -1, -1)
            elif mode == 'full_bypass':
                stash = LAM * (mo_true - mean0)
                mo = mean0.expand(B, -1, -1)
        else:
            mo = mo_true
        x = x + mo
    if stash is not None:
        x = x + stash
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T - 1)


def held(**kw):
    return torch.cat([fwd(HELD[i:i + B0], **kw) for i in range(0, HELD.shape[0], B0)])


# fit linear cap + per-position mean for block 0
print('fitting block-0 linear cap and per-position mean ...', flush=True)
A = torch.zeros(D + 1, D + 1, device=DEV, dtype=torch.float64)
Bm = torch.zeros(D + 1, D, device=DEV, dtype=torch.float64)
for i in range(0, TRAIN.shape[0], B0):
    acc = []; fwd(TRAIN[i:i + B0], collect0=0, acc=acc)
    xh, mo = acc[0]
    xa = torch.cat([xh, torch.ones_like(xh[..., :1])], -1).reshape(-1, D + 1).double()
    A += xa.T @ xa; Bm += xa.T @ mo.reshape(-1, D).double()
sol = torch.linalg.solve(A + 1.0 * torch.eye(D + 1, device=DEV, dtype=torch.float64), Bm).float()
LINW, LINB = sol[:D], sol[D]
S, T = HELD.shape
msum = torch.zeros(T, D, device=DEV)
for i in range(0, S, B0):
    acc = []; fwd(HELD[i:i + B0], collect0=0, acc=acc); msum += acc[0][1].sum(0)
MEAN0 = msum / S

base = held(); BASE = float(base.mean())
print(f'GATE base CE {BASE:.4f} | held {base.shape[0]}x{base.shape[1]} = {base.shape[0]*base.shape[1]} tokens', flush=True)


def rep(name, mode):
    ce = held(mode=mode); d = ce - base
    v = (round(float(d.mean()), 4), round(float(d.mean(1).std() / np.sqrt(d.shape[0])), 4))
    print(f'  {name:32s} dCE {v[0]:+.4f} +- {v[1]:.4f}', flush=True)
    return v


res = {'meta': {'base_ce': round(BASE, 4), 'lambda_mean': round(float(LAM.mean()), 4),
                'held_scored_tokens': int(base.shape[0] * base.shape[1])}}
res['floor_mean'] = rep('mean floor', 'mean')
res['linear_normal'] = rep('linear cap (both routes)', 'lin')
res['linear_bypass'] = rep('linear cap, DIRECT route only', 'lin_bypass')
res['full_bypass'] = rep('full block0, DIRECT route only', 'full_bypass')

fl = res['floor_mean'][0]
lin_n = fl - res['linear_normal'][0]           # recovered by linear cap, both routes
lin_b = fl - res['linear_bypass'][0]           # recovered by linear cap, direct route only
full_b = fl - res['full_bypass'][0]
res['analysis'] = {
    'floor_nats': fl,
    'linear_recovers_bothroutes': round(lin_n, 4),
    'linear_recovers_directonly': round(lin_b, 4),
    'linear_indirect_share': round(1 - lin_b / lin_n, 3),
    'full_recovers_directonly': round(full_b, 4),
    'full_indirect_share': round(1 - full_b / fl, 3),
}
print('=== analysis ===', flush=True)
print(f"  linear cap recovers {lin_n:.4f} nats via both routes, {lin_b:.4f} via direct-to-readout only", flush=True)
print(f"  -> {res['analysis']['linear_indirect_share']:.1%} of block 0's linear effect flows through DOWNSTREAM NONLINEARITIES", flush=True)
print(f"  full block 0: {full_b:.4f} of {fl:.4f} recovered by direct route -> {res['analysis']['full_indirect_share']:.1%} is intermediate-nonlinear", flush=True)
json.dump(res, open(f'{QK}/qk_conspath.json', 'w'), indent=1)
print('SAVED qk_conspath.json', flush=True)
