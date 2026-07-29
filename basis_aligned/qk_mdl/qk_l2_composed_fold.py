"""Layer-2 extension of the composed chain (the per-layer recipe: streams -> named bases, cores as
exact restrictions, gauges carried). MLP2's pre-rms input = six analytic streams:
  x_pre2 = c_e*e + c01*a0 + c01*m0 + c1*(a1 + m1) + a2
Gates: rms-gauge for MLP2; 6-stream split; full-analytic == model. Arms:
  A: a0/a1/a2 -> named/PCA 144-dim bases, m0/m1 exact-chained (interface test at MLP2)
  B: fully truncated chain (m0 = T0 restriction on truncated streams; m1 = T1 on its truncated
     streams incl m0-hat; MLP2 = T2 on all six truncated streams) -- interface test at MLP2
  C: JOINT chain -- substitute m0, m1, m2 outputs ALL with the truncated-chain versions in the
     residual; compare against the joint floor for {mlp0, mlp1, mlp2}.
References: MLP2 floor 0.1818; MLP2 data program (polished) 67.6%.
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

# attn0 named archetype basis
mh = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV); pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
cw0 = B0.attn.c_proj.weight.detach().float()
cols = []
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4; Dv = bb[f'h{hh}_v_Dm'].to(DEV); Dv = Dv/Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vd = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        Pp = mh[f'h{hh}']; Dn_ = Pp['Dm'].to(DEV); Dn_ = Dn_/Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vd = Dn_[:, 2*HD:].T @ Pp['U'].to(DEV)[:, :16]
    if Vd.shape[1] < 16: Vd = torch.cat([Vd, torch.zeros(HD, 16-Vd.shape[1], device=DEV)], 1)
    cols.append(cw0[:, hh*HD:(hh+1)*HD] @ Vd)
QA0, _ = torch.linalg.qr(torch.cat(cols, 1))

lam00 = (B0.lambdas[0]+B0.lambdas[1]).item()
l10, l11 = B1.lambdas[0].item(), B1.lambdas[1].item()
l20, l21 = B2.lambdas[0].item(), B2.lambdas[1].item()
CE_ = l20*(l10*lam00 + l11) + l21; C01 = l20*l10; C1 = l20


@torch.no_grad()
def run_three(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    outs = {}
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
        aout = a.c_proj(yh.reshape(B, T, -1))
        outs[f'a{li}'] = aout.reshape(-1, D)
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        outs[f'm{li}'] = mo.reshape(-1, D)
        if li == 2:
            outs['xpre2'] = (x).reshape(-1, D); outs['y2'] = mo.reshape(-1, D)
        x = x + mo
    outs['e'] = x0.reshape(-1, D)
    return outs

# bases for a1, a2 (PCA-16/head of head outputs; data-lite metric only)
acc1 = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
acc2 = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
for i in range(0, 64, 8):
    o = run_three(COOC[i:i+8].to(DEV)[:, :128])
    acc1 += torch.einsum('nhd,nhe->hde', o['a1h'].double(), o['a1h'].double())
    acc2 += torch.einsum('nhd,nhe->hde', o['a2h'].double(), o['a2h'].double())
def head_basis(acc, blk):
    cw = blk.attn.c_proj.weight.detach().float(); cs = []
    for hh in range(NH):
        ev, evec = torch.linalg.eigh(acc[hh])
        cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:16]].float())
    Qx, _ = torch.linalg.qr(torch.cat(cs, 1)); return Qx
QA1 = head_basis(acc1, B1); QA2 = head_basis(acc2, B2)
print("bases ready", flush=True)

# gates on one batch
o = run_three(COOC[:8].to(DEV)[:, :128])
xp2 = o['xpre2']; y2 = o['y2']
r2 = xp2.pow(2).sum(1)/D
g1 = float(((T_ev('2', xp2, xp2)/r2.unsqueeze(1) + WT['2'][3]) - y2).norm()/y2.norm())
S = CE_*o['e'] + C01*o['a0'] + C01*o['m0'] + C1*o['a1'] + C1*o['m1'] + o['a2']
gs = float((S - xp2).norm()/xp2.norm())
print(f"GATES: rms-gauge {g1:.2e} | 6-stream split {gs:.2e}", flush=True)
streams = {'E': CE_*o['e'], 'A0': C01*o['a0'], 'M0': C01*o['m0'], 'A1': C1*o['a1'], 'M1': C1*o['m1'], 'A2': o['a2']}
tot = (y2 - y2.mean(0)).pow(2).sum()
names = list(streams)
shares = {}
for i in range(6):
    for j in range(i, 6):
        blk = ((1 if i == j else 2)*T_ev('2', streams[names[i]], streams[names[j]]))/r2.unsqueeze(1)
        sh = float((blk - blk.mean(0)).pow(2).sum()/tot)
        if sh > 0.02: shares[f'{names[i]}x{names[j]}'] = round(sh, 3)
print("block shares (>0.02):", shares, flush=True)


def chain_truncated(x0f, a0f, a1f, a2f):
    """fully truncated chain: returns m0hat, m1hat, m2hat (per-position, flattened)."""
    a0t = (a0f @ QA0) @ QA0.T; a1t = (a1f @ QA1) @ QA1.T; a2t = (a2f @ QA2) @ QA2.T
    xp0 = lam00*x0f + a0t; r0 = xp0.pow(2).sum(1)/D
    m0h = T_ev('0', xp0, xp0)/r0.unsqueeze(1) + WT['0'][3]
    xp1 = (l10*lam00 + l11)*x0f + l10*a0t + l10*m0h + a1t; r1 = xp1.pow(2).sum(1)/D
    m1h = T_ev('1', xp1, xp1)/r1.unsqueeze(1) + WT['1'][3]
    xp2h = CE_*x0f + C01*a0t + C01*m0h + C1*a1t + C1*m1h + a2t; r2h = xp2h.pow(2).sum(1)/D
    m2h = T_ev('2', xp2h, xp2h)/r2h.unsqueeze(1) + WT['2'][3]
    return m0h, m1h, m2h


@torch.no_grad()
def audit(mode):
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
            fa = aout.reshape(-1, D)
            if li == 0: cache['a0'] = fa
            if li == 1: cache['a1'] = fa
            if li == 2: cache['a2'] = fa
            if mode == 'C' and li in (0, 1, 2):
                if li == 2:
                    m0h, m1h, m2h = chain_truncated(x0.reshape(-1, D), cache['a0'], cache['a1'], cache['a2'])
                    # rebuild residual: x currently contains REAL m0,m1 -- replace by chained versions
                    delta = (C01*(m0h - cache['m0real']) + C1*(m1h - cache['m1real'])).view(B, T2, D)
                    x = x + delta.to(x.dtype) + m2h.view(B, T2, D).to(x.dtype)
                    continue
                mo = blk.mlp(hin)
                cache[f'm{li}real'] = mo.reshape(-1, D)
                x = x + mo; continue
            if li == 2 and mode in ('A', 'B'):
                if mode == 'A':
                    a0t = (cache['a0'] @ QA0) @ QA0.T; a1t = (cache['a1'] @ QA1) @ QA1.T; a2t = (cache['a2'] @ QA2) @ QA2.T
                    xp = CE_*x0.reshape(-1, D) + C01*a0t + C01*cache['m0real'] + C1*a1t + C1*cache['m1real'] + a2t
                    rr = xp.pow(2).sum(1)/D
                    mo = (T_ev('2', xp, xp)/rr.unsqueeze(1) + WT['2'][3]).view(B, T2, D)
                else:
                    _, _, m2h = chain_truncated(x0.reshape(-1, D), cache['a0'], cache['a1'], cache['a2'])
                    mo = m2h.view(B, T2, D)
                x = x + mo.to(x.dtype); continue
            mo = blk.mlp(hin)
            if li in (0, 1): cache[f'm{li}real'] = mo.reshape(-1, D)
            x = x + mo
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

res = {'gate_rms': g1, 'gate_split': gs, 'block_shares': shares}
dA = audit('A') - SUBBASE
dB = audit('B') - SUBBASE
dC = audit('C') - SUBBASE
res['armA_dCE'] = round(dA, 5); res['armA_frac'] = round(1-dA/FLOOR2, 4)
res['armB_dCE'] = round(dB, 5); res['armB_frac'] = round(1-dB/FLOOR2, 4)
res['armC_joint3_dCE'] = round(dC, 5)
print(f"arm A (streams trunc, m0/m1 exact): +{dA:.5f} ({1-dA/FLOOR2:.1%} of MLP2 floor)", flush=True)
print(f"arm B (fully truncated 3-layer chain, MLP2 interface): +{dB:.5f} ({1-dB/FLOOR2:.1%})", flush=True)
print(f"arm C (joint: m0,m1,m2 all chained-truncated in residual): +{dC:.5f}", flush=True)
json.dump(res, open(f'{QK}/qk_l2_composed_fold.json', 'w'), indent=2)
print("QK L2 COMPOSED FOLD DONE", flush=True)
