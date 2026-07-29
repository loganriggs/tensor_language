"""Extend the fully-analytic truncated chain through layers 3-5 at K=576 (64 dims/head, the width
that solved compounding). For each target layer LT in {3,4,5}: interface test (MLP_LT's output
computed by the chain m0..mLT from truncated attention streams; residual otherwise real) vs that
interface's floor. Plus the six-MLP JOINT arm (all m0..m5 replaced by chained versions in the
residual, attention streams oracle-real -- caveat noted) vs the measured six-MLP joint floor.
Coefficient bookkeeping via the lambda recurrence: coeffs(xb_L) = lamL0*coeffs(xb_{L-1}) + lamL1*e
+ a_L + m_L.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
LED = json.load(open(f'{QK}/qk_completeness_ledger.json'))
SUBBASE = LED['subset_base']; MF = LED['mlp_floor']
LMAX = 5; PER = 64
BLKS = [m.transformer.h[i] for i in range(LMAX+1)]
WT = {}
for li, blk in enumerate(BLKS):
    WT[li] = (blk.mlp.Left.weight.detach().float(), blk.mlp.Right.weight.detach().float(),
              blk.mlp.Down.weight.detach().float(), blk.mlp.Down_bias.detach().float())
def T_ev(li, u, v):
    Lw, Rw, Dw, _ = WT[li]
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)

# coefficient bookkeeping: for each layer l, coefficients of x_pre_l over {e, a_0..a_l, m_0..m_{l-1}}
lam = [(blk.lambdas[0].item(), blk.lambdas[1].item()) for blk in BLKS]
CO = []   # CO[l] = dict: 'e': float, ('a',j): float, ('m',j): float  for x_pre_l (includes a_l coeff 1)
cur = {'e': lam[0][0] + lam[0][1]}   # xb_{-1} := x0; block0: x_into = (l00+l01) e
for l in range(LMAX+1):
    xpre = dict(cur); xpre[('a', l)] = 1.0
    CO.append(xpre)
    nxt = dict(xpre); nxt[('m', l)] = 1.0                      # xb_l = x_pre_l + m_l
    if l < LMAX:
        cur = {k: lam[l+1][0]*v for k, v in nxt.items()}
        cur['e'] = cur.get('e', 0.0) + lam[l+1][1]

# bases: PCA-64/head for attention layers 0..5
accs = [torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for _ in range(LMAX+1)]
@torch.no_grad()
def run_all(idx, want_heads=False):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    outs = {'e': x0.reshape(-1, D)}; x = None; v1 = None
    for li in range(LMAX+1):
        blk = BLKS[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if want_heads: outs[f'ah{li}'] = yh.reshape(-1, NH, HD)
        aout = a.c_proj(yh.reshape(B, T, -1)); outs[f'a{li}'] = aout.reshape(-1, D)
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,))); outs[f'm{li}'] = mo.reshape(-1, D); x = x + mo
    return outs
for i in range(0, 64, 8):
    o = run_all(COOC[i:i+8].to(DEV)[:, :128], want_heads=True)
    for li in range(LMAX+1):
        accs[li] += torch.einsum('nhd,nhe->hde', o[f'ah{li}'].double(), o[f'ah{li}'].double())
QB = []
for li in range(LMAX+1):
    cw = BLKS[li].attn.c_proj.weight.detach().float(); cs = []
    for hh in range(NH):
        ev, evec = torch.linalg.eigh(accs[li][hh])
        cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:PER]].float())
    Qx, _ = torch.linalg.qr(torch.cat(cs, 1)); QB.append(Qx)
print("bases ready (K=576 x 6 layers)", flush=True)


def chain(x0f, a_list, upto):
    """chained m-hats 0..upto from truncated attention streams; returns list of m-hats."""
    at = [(a_list[l] @ QB[l]) @ QB[l].T for l in range(upto+1)]
    mh = []
    for l in range(upto+1):
        co = CO[l]
        xp = co['e']*x0f
        for j in range(l+1):
            if ('a', j) in co: xp = xp + co[('a', j)]*at[j]
            if ('m', j) in co: xp = xp + co[('m', j)]*mh[j]
        r = xp.pow(2).sum(1)/D
        mh.append(T_ev(l, xp, xp)/r.unsqueeze(1) + WT[l][3])
    return mh


@torch.no_grad()
def audit(mode, LT=None):
    """mode 'iface' (substitute only MLP_LT via chain) | 'joint6' | 'floor6'"""
    if mode == 'floor6':
        # mean-input floors for all six MLP interfaces simultaneously
        s = [torch.zeros(D, device=DEV, dtype=torch.float64) for _ in range(LMAX+1)]; nn = 0
        for i in range(0, 64, 8):
            o = run_all(COOC[i:i+8].to(DEV)[:, :128])
            # reconstruct hin per layer: hin_l = rms(x_pre_l) -- recompute from streams
            x0f = o['e']
            for l in range(LMAX+1):
                co = CO[l]; xp = co['e']*x0f
                for j in range(l+1):
                    if ('a', j) in co: xp = xp + co[('a', j)]*o[f'a{j}']
                    if ('m', j) in co: xp = xp + co[('m', j)]*o[f'm{j}']
                s[l] += F.rms_norm(xp, (D,)).double().sum(0)
            nn += x0f.shape[0]
        MU = [F.rms_norm((si/nn).float(), (D,)) for si in s]
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
        cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        cache_a = {}; cache_m = {}
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T2, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
            aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
            x = x + aout; hin = F.rms_norm(x, (D,))
            if li <= LMAX: cache_a[li] = aout.reshape(-1, D)
            if mode == 'floor6' and li <= LMAX:
                x = x + blk.mlp(MU[li].expand(B, T2, D)); continue
            if mode == 'iface' and li == LT:
                mh = chain(x0.reshape(-1, D), [cache_a[j] for j in range(LT+1)], LT)
                x = x + mh[LT].view(B, T2, D).to(x.dtype); continue
            if mode == 'joint6' and li == LMAX:
                mh = chain(x0.reshape(-1, D), [cache_a[j] for j in range(LMAX+1)], LMAX)
                delta = sum((mh[l] - cache_m[l]) for l in range(LMAX)).view(B, T2, D)
                x = x + delta.to(x.dtype) + mh[LMAX].view(B, T2, D).to(x.dtype); continue
            mo = blk.mlp(hin)
            if li <= LMAX: cache_m[li] = mo.reshape(-1, D)
            x = x + mo
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

res = {}
for LT in (3, 4, 5):
    d = audit('iface', LT) - SUBBASE
    fl = MF[str(LT)]
    res[f'iface_mlp{LT}'] = {'dCE': round(d, 5), 'frac': round(1-d/fl, 4)}
    print(f"chain interface MLP{LT} (depth {LT+1}): +{d:.5f} ({1-d/fl:.1%} of floor {fl:.3f})", flush=True)
d6 = audit('joint6') - SUBBASE
f6 = audit('floor6') - SUBBASE
res['joint6_dCE'] = round(d6, 5); res['floor6_dCE'] = round(f6, 5)
res['joint6_frac'] = round(1 - d6/f6, 4)
print(f"JOINT six-MLP chain: +{d6:.5f} vs six-MLP floor +{f6:.5f} -> {1-d6/f6:.1%} (oracle attention streams; caveat)", flush=True)
json.dump(res, open(f'{QK}/qk_chain_deep.json', 'w'), indent=2)
print("QK CHAIN DEEP DONE", flush=True)
