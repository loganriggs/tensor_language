"""Is RMSNorm the reason block 0's linear write is non-absorbable? (Logan)

Mechanistic claim: block 0's effect flows through downstream RMSNorms nonlinearly -- it sets which
direction survives normalization before the downstream squares read it. Test: replace each downstream
norm with its FIRST-ORDER TAYLOR expansion around the block-0-ablated (floor) operating point, so the
norm becomes a genuine LINEAR operator (stable, unlike freezing the denominator). Keep the products real.

    rms_lin(x; xf) = rms(xf) + J_rms(xf) . (x - xf)
    J_rms(xf).v = v/df - xf * ( <xf,v> / (D * df^3) ) ,  df = sqrt(mean(xf^2)+eps)

If block 0's recovery COLLAPSES under linear norms -> the norm's nonlinearity is the channel / obstruction
(claim confirmed). If it SURVIVES -> block 0 matters through the products instead, not the norm.

Gate: with block0=mean in both passes, rms_lin == rms exactly (x=xf). Floor +1.2341, full lincap +0.0796.
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
LINSET = set(range(1, NL))  # which downstream blocks' norms to linearize


def rms(x):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def rms_norm_site(x, rec, use, key):
    """real rms, optionally recording xf; or linearized around recorded xf."""
    blk_idx = int(key[1:]) if key[0] in 'hm' else 999
    if use is not None and blk_idx in LINSET:
        xf = use[key]
        df = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        v = x - xf
        jv = v / df - xf * ((xf * v).sum(-1, keepdim=True) / (D * df.pow(3)))
        return xf / df + jv
    if rec is not None:
        rec[key] = x.detach()
    return rms(x)


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
    return x, xhat, v.view(B, T, NH, HD), x0


@torch.no_grad()
def run(idx, block0='full', rec=None, use=None):
    """block0: 'full' | 'mean' | 'lincap'. rec/use control downstream norm linearization."""
    B, T = idx.shape
    x_pre, xhat0, v1, x0 = block0_io(idx)
    blk0 = m.transformer.h[0]
    if block0 == 'mean':
        mo0 = MEAN0.unsqueeze(0).expand(B, -1, -1)
    elif block0 == 'lincap':
        mo0 = xhat0 @ LINW + LINB
    else:
        mo0 = blk0.mlp(xhat0)
    x = x_pre + mo0
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(1, NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        hcur = rms_norm_site(x, rec, use, f'h{li}')
        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = rms_norm_site(x, rec, use, f'm{li}')
        x = x + blk.mlp(xhat)
    xf = rms_norm_site(x, rec, use, 'final')
    logits = 30 * torch.tanh(m.lm_head(xf) / 30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T - 1)


# fit block-0 linear cap + mean
print('fitting ...', flush=True)
A = torch.zeros(D + 1, D + 1, device=DEV, dtype=torch.float64); Bm = torch.zeros(D + 1, D, device=DEV, dtype=torch.float64)
for i in range(0, TRAIN.shape[0], B0):
    _, xhat0, _, _ = block0_io(TRAIN[i:i + B0]); mo = m.transformer.h[0].mlp(xhat0)
    xa = torch.cat([xhat0, torch.ones_like(xhat0[..., :1])], -1).reshape(-1, D + 1).double()
    A += xa.T @ xa; Bm += xa.T @ mo.reshape(-1, D).double()
sol = torch.linalg.solve(A + torch.eye(D + 1, device=DEV, dtype=torch.float64), Bm).float()
LINW, LINB = sol[:D], sol[D]
S, T = HELD.shape; msum = torch.zeros(T, D, device=DEV)
for i in range(0, S, B0):
    _, xhat0, _, _ = block0_io(HELD[i:i + B0]); msum += m.transformer.h[0].mlp(xhat0).sum(0)
MEAN0 = msum / S

# real-norm reference
def held_ce(block0, use_linnorm=False, linset=None):
    out = []
    for i in range(0, HELD.shape[0], B0):
        idx = HELD[i:i + B0]
        if use_linnorm:
            global LINSET
            if linset is not None: LINSET = linset
            rec = {}; run(idx, block0='mean', rec=rec)          # record floor operating points
            out.append(run(idx, block0=block0, use=rec))
        else:
            out.append(run(idx, block0=block0))
    return torch.cat(out)


base = held_ce('full')
floor = held_ce('mean'); full = held_ce('lincap')
BASE = float(base.mean())
print(f'GATE base {BASE:.4f} | floor {float(floor.mean())-BASE:+.4f} | lincap {float(full.mean())-BASE:+.4f}', flush=True)

# linearized-norm, restricted to the consumption window (deeper norms stay real to re-stabilize)
res = {'base_ce': round(BASE, 4), 'floor': round(float(floor.mean()) - BASE, 4),
       'lincap_real': round(float(full.mean()) - BASE, 4),
       'recovery_real_norm': round(float(floor.mean()) - float(full.mean()), 4), 'windows': {}}
recov_real = float(floor.mean()) - float(full.mean())
print(f'block 0 recovery, REAL norms  {recov_real:+.4f}', flush=True)
for name, ls in [('block1', {1}), ('blocks1-2', {1, 2}), ('blocks1-3', {1, 2, 3})]:
    hh = {f'h{i}' for i in ls} | {f'm{i}' for i in ls}
    fl = held_ce('mean', use_linnorm=True, linset=hh)
    lc = held_ce('lincap', use_linnorm=True, linset=hh)
    rec = float(fl.mean()) - float(lc.mean())
    ok_gate = float(fl.mean()) - BASE
    share = round(1 - rec / recov_real, 3)
    res['windows'][name] = {'floor_gate': round(ok_gate, 4), 'recovery_linnorm': round(rec, 4), 'norm_share': share}
    print(f'  linearize {name:10s} norms: floor-gate {ok_gate:+.4f} | recovery {rec:+.4f} | norm carries {share:.1%}', flush=True)
json.dump(res, open(f'{QK}/qk_linnorm.json', 'w'), indent=1)
print('SAVED', flush=True)
