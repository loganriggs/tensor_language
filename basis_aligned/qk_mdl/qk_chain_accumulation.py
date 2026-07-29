"""Deep-chain error accumulation: does compounding through chained quadratics (depth-2 95.1% ->
depth-3 49.5%) flatten with RICHER STREAM BASES, or does it require RE-ANCHORING?
All arms measured arm-B-style at MLP2's interface (fully/partially truncated chain computes MLP2's
output; residual otherwise real; floor 0.1818):
  basis sweep: per-head PCA basis of K/9 dims per attention layer, K in {144, 288, 576}
  anchor sweep at K=144: anchor-none (full 3-chain), anchor-m0 (m0 real, m1 chained), anchor-both
  (m0,m1 real = one-hop reference 93.9%).
The K-curve vs the anchor-curve says which lever flattens compounding per unit of description.
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
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']
FLOOR2 = 0.18181
B0, B1, B2 = m.transformer.h[0], m.transformer.h[1], m.transformer.h[2]
WT = {}
for tag, blk in [('0', B0), ('1', B1), ('2', B2)]:
    WT[tag] = (blk.mlp.Left.weight.detach().float(), blk.mlp.Right.weight.detach().float(),
               blk.mlp.Down.weight.detach().float(), blk.mlp.Down_bias.detach().float())
def T_ev(tag, u, v):
    Lw, Rw, Dw, _ = WT[tag]
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)

lam00 = (B0.lambdas[0]+B0.lambdas[1]).item()
l10, l11 = B1.lambdas[0].item(), B1.lambdas[1].item()
l20, l21 = B2.lambdas[0].item(), B2.lambdas[1].item()
CE_ = l20*(l10*lam00 + l11) + l21; C01 = l20*l10; C1 = l20


@torch.no_grad()
def run_three(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    outs = {'e': x0.reshape(-1, D)}
    x = lam00*x0; v1c = None
    for li, blk in enumerate([B0, B1, B2]):
        a = blk.attn
        if li: x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1c is None: v1c = v
        v = (1-a.lamb)*v + a.lamb*v1c.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        outs[f'a{li}h'] = yh.reshape(-1, NH, HD)
        aout = a.c_proj(yh.reshape(B, T, -1)); outs[f'a{li}'] = aout.reshape(-1, D)
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,))); outs[f'm{li}'] = mo.reshape(-1, D); x = x + mo
    return outs

# per-head PCA covariances of all three attention layers
accs = [torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for _ in range(3)]
for i in range(0, 64, 8):
    o = run_three(COOC[i:i+8].to(DEV)[:, :128])
    for li in range(3):
        accs[li] += torch.einsum('nhd,nhe->hde', o[f'a{li}h'].double(), o[f'a{li}h'].double())
BLKS = [B0, B1, B2]
def basis(K):
    per = K // NH; Q = []
    for li in range(3):
        cw = BLKS[li].attn.c_proj.weight.detach().float(); cs = []
        for hh in range(NH):
            ev, evec = torch.linalg.eigh(accs[li][hh])
            cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:per]].float())
        Qx, _ = torch.linalg.qr(torch.cat(cs, 1)); Q.append(Qx)
    return Q
BAS = {K: basis(K) for K in (144, 288, 576)}
print("bases ready", flush=True)


def chain(x0f, a0f, a1f, a2f, K, anchor, m0real=None, m1real=None):
    Q0, Q1, Q2 = BAS[K]
    a0t = (a0f @ Q0) @ Q0.T; a1t = (a1f @ Q1) @ Q1.T; a2t = (a2f @ Q2) @ Q2.T
    if anchor in ('m0', 'both'):
        m0h = m0real
    else:
        xp0 = lam00*x0f + a0t; r0 = xp0.pow(2).sum(1)/D
        m0h = T_ev('0', xp0, xp0)/r0.unsqueeze(1) + WT['0'][3]
    if anchor == 'both':
        m1h = m1real
    else:
        xp1 = (l10*lam00 + l11)*x0f + l10*a0t + l10*m0h + a1t; r1 = xp1.pow(2).sum(1)/D
        m1h = T_ev('1', xp1, xp1)/r1.unsqueeze(1) + WT['1'][3]
    xp2 = CE_*x0f + C01*a0t + C01*m0h + C1*a1t + C1*m1h + a2t; r2 = xp2.pow(2).sum(1)/D
    return T_ev('2', xp2, xp2)/r2.unsqueeze(1) + WT['2'][3]


@torch.no_grad()
def audit(K, anchor):
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = lam00*x0; v1 = None
        cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        cache = {}
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            if li: x = blk.lambdas[0]*x + blk.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T2, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
            aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
            x = x + aout; hin = F.rms_norm(x, (D,))
            if li <= 2: cache[f'a{li}'] = aout.reshape(-1, D)
            if li == 2:
                mo = chain(x0.reshape(-1, D), cache['a0'], cache['a1'], cache['a2'], K, anchor,
                           cache.get('m0'), cache.get('m1')).view(B, T2, D)
                x = x + mo.to(x.dtype); continue
            mo = blk.mlp(hin)
            if li <= 1: cache[f'm{li}'] = mo.reshape(-1, D)
            x = x + mo
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

res = {}
for K in (144, 288, 576):
    d = audit(K, 'none') - SUBBASE
    res[f'fullchain_K{K}'] = {'dCE': round(d, 5), 'frac': round(1-d/FLOOR2, 4)}
    print(f"full chain K={K}: +{d:.5f} ({1-d/FLOOR2:.1%})", flush=True)
for anchor in ('m0', 'both'):
    d = audit(144, anchor) - SUBBASE
    res[f'anchor_{anchor}_K144'] = {'dCE': round(d, 5), 'frac': round(1-d/FLOOR2, 4)}
    print(f"anchor={anchor} K=144: +{d:.5f} ({1-d/FLOOR2:.1%})", flush=True)
json.dump(res, open(f'{QK}/qk_chain_accumulation.json', 'w'), indent=2)
print("QK CHAIN ACCUMULATION DONE", flush=True)
