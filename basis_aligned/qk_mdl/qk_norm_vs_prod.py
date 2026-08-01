"""Does block 0's (linear) effect flow through the NORM's nonlinearity, the downstream PRODUCTS, or a
linear response? Two clean tests that do NOT rebuild the model (Logan).

Setup: block 0's deviation d0 = (linear cap) - (per-position mean) enters the post-block-0 stream x1.
The downstream map F: x1 -> logits (blocks 1..17 + readout), with x0=embedding and v1=block-0 value held
fixed (neither depends on block 0's MLP output).

TEST 1 -- FREEZE DENOMINATOR (isolates the norm): record every downstream rms denominator on the
    block0=mean pass, then evaluate block0=lincap REUSING those denominators (each norm becomes division
    by a constant = linear). If block 0's recovery survives, the norm's nonlinear response is not the channel.

TEST 2 -- FIRST-ORDER RESPONSE (linear vs nonlinear consumption overall): compare the true effect
    F(x1+d0)-F(x1) to the finite-difference linear response [F(x1+eps*d0)-F(x1)]/eps. Fraction captured
    first-order = how much of block 0 is consumed by an (operating-point-dependent) LINEAR map.

Held FW[448:600,:128]. Gates: manual-rms floor reproduces +1.2341, lincap +0.0796.
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
EPS = torch.finfo(torch.float32).eps
LINW = LINB = MEAN0 = None


def rms(x, rec=None, use=None, key=None):
    if use is not None:
        return x / use[key]
    den = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    if rec is not None:
        rec[key] = den
    return x / den


@torch.no_grad()
def to_block0(idx):
    """run through block 0, return (x1_lincap, d0, x0, v1) with block-0 output = linear cap."""
    B, T = idx.shape
    x = rms(m.transformer.wte(idx)); x0 = x
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    blk = m.transformer.h[0]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
    hcur = rms(x)
    def qk(l):
        z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v1 = a.c_v(hcur).view(B, T, NH, HD)
    v = v1
    q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
    xhat = rms(x)
    lincap = xhat @ LINW + LINB
    d0 = lincap - MEAN0.unsqueeze(0)
    x1 = x + lincap
    return x1, d0, x0, v1


@torch.no_grad()
def downstream(x1, x0, v1, rec=None, use=None):
    B, T, _ = x1.shape
    cos, sin = rope_tables(T, HD, DEV, x1.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    x = x1
    for li in range(1, NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        hcur = rms(x, rec, use, f'h{li}')
        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = rms(x, rec, use, f'm{li}')
        x = x + blk.mlp(xhat)
    xf = rms(x, rec, use, 'final')
    return 30 * torch.tanh(m.lm_head(xf) / 30)


def ce_from_logits(logits, idx):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(idx.shape[0], idx.shape[1] - 1)


# fit linear cap + per-position mean (block 0), reusing a plain forward
@torch.no_grad()
def block0_io(idx):
    B, T = idx.shape
    x = rms(m.transformer.wte(idx)); x0 = x
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    blk = m.transformer.h[0]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
    hcur = rms(x)
    def qk(l):
        z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD)
    q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
    xhat = rms(x)
    return xhat.detach(), blk.mlp(xhat).detach()


print('fitting block-0 linear cap + mean ...', flush=True)
A = torch.zeros(D + 1, D + 1, device=DEV, dtype=torch.float64); Bm = torch.zeros(D + 1, D, device=DEV, dtype=torch.float64)
for i in range(0, TRAIN.shape[0], B0):
    xh, mo = block0_io(TRAIN[i:i + B0])
    xa = torch.cat([xh, torch.ones_like(xh[..., :1])], -1).reshape(-1, D + 1).double()
    A += xa.T @ xa; Bm += xa.T @ mo.reshape(-1, D).double()
sol = torch.linalg.solve(A + torch.eye(D + 1, device=DEV, dtype=torch.float64), Bm).float()
LINW, LINB = sol[:D], sol[D]
S, T = HELD.shape; msum = torch.zeros(T, D, device=DEV)
for i in range(0, S, B0):
    _, mo = block0_io(HELD[i:i + B0]); msum += mo.sum(0)
MEAN0 = msum / S

# main loop
floor_ce, full_ce, frozen_ce, lin_ce = [], [], [], []
EPSJVP = 0.05
for i in range(0, HELD.shape[0], B0):
    idx = HELD[i:i + B0]
    x1, d0, x0, v1 = to_block0(idx)
    x1_floor = x1 - d0
    rec = {}
    lf = downstream(x1_floor, x0, v1, rec=rec)      # floor + record denominators
    lfull = downstream(x1_floor + d0, x0, v1)       # true lincap
    lfroz = downstream(x1_floor + d0, x0, v1, use=rec)   # lincap with frozen (floor) denominators
    leps = downstream(x1_floor + EPSJVP * d0, x0, v1)
    llin = lf + (leps - lf) / EPSJVP                # first-order response
    floor_ce.append(ce_from_logits(lf, idx)); full_ce.append(ce_from_logits(lfull, idx))
    frozen_ce.append(ce_from_logits(lfroz, idx)); lin_ce.append(ce_from_logits(llin, idx))
floor = torch.cat(floor_ce); full = torch.cat(full_ce); frozen = torch.cat(frozen_ce); lin = torch.cat(lin_ce)

# base CE (unmodified model, manual rms) for gate
@torch.no_grad()
def base_ce(idx):
    x1, d0, x0, v1 = to_block0(idx)
    return ce_from_logits(downstream(x1 - d0 + (block0_io(idx)[1]), x0, v1), idx)
# simpler gate: floor should be ~ base+1.2341; report floor/full means
mf, mfl = float(floor.mean()), float(full.mean())
print(f'GATE floor CE {mf:.4f}  lincap CE {mfl:.4f}  (block-0 effect = {mf-mfl:.4f} nats)', flush=True)
recov_full = mf - mfl
recov_frozen = mf - float(frozen.mean())
recov_lin = mf - float(lin.mean())
res = {'floor_ce': round(mf, 4), 'lincap_ce': round(mfl, 4),
       'recovery_full': round(recov_full, 4),
       'recovery_frozen_denominator': round(recov_frozen, 4),
       'recovery_first_order': round(recov_lin, 4),
       'norm_channel_share': round(1 - recov_frozen / recov_full, 3),
       'nonlinear_channel_share': round(1 - recov_lin / recov_full, 3),
       'eps_jvp': EPSJVP}
print('=== TEST 1: freeze denominator (norm) ===', flush=True)
print(f"  recovery with real norms      {recov_full:+.4f}", flush=True)
print(f"  recovery with FROZEN norms    {recov_frozen:+.4f}  -> norm channel carries {res['norm_channel_share']:.1%}", flush=True)
print('=== TEST 2: first-order response ===', flush=True)
print(f"  recovery first-order (linear) {recov_lin:+.4f}  -> nonlinear channel carries {res['nonlinear_channel_share']:.1%}", flush=True)
json.dump(res, open(f'{QK}/qk_norm_vs_prod.json', 'w'), indent=1)
print('SAVED', flush=True)
