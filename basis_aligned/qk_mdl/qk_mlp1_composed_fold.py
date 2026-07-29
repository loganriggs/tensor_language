"""MLP1 composed fold (extends qk_mlp0_composed_fold one layer up). MLP1's pre-rms input splits
into FOUR analytic streams: x_pre1 = c_e*e_t + lam*a0 + lam*m0 + a1  (e = rms embedding; a0 = attn0
out, exactly folded; m0 = MLP0 out, itself analytic by the verified composed chain; a1 = attn1 out,
whose pattern is now also exact by the pattern-gauge result). Bilinearity gives 10 interaction
blocks. GATES: rms-as-gauge for MLP1; 4-stream split exactness; full-analytic substitution == model.
TRUNCATED ARMS: (A) a0 -> named 144-dim archetype write-basis, a1 -> per-head 16-dim PCA basis
(data-lite, noted), m0 exact-chained; (B) additionally m0 -> MLP0's archetype-composed output (fully
truncated interpretable chain). Substitution dCE vs MLP1 floor 2.151; reference: data program 96.1%.
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
FLOOR1 = 2.15118
b0, b1 = m.transformer.h[0], m.transformer.h[1]
L1 = b1.mlp.Left.weight.detach().float(); R1 = b1.mlp.Right.weight.detach().float()
D1 = b1.mlp.Down.weight.detach().float(); bias1 = b1.mlp.Down_bias.detach().float()
L0m = b0.mlp.Left.weight.detach().float(); R0m = b0.mlp.Right.weight.detach().float()
D0m = b0.mlp.Down.weight.detach().float(); bias0 = b0.mlp.Down_bias.detach().float()

def T1(u, v):
    return 0.5*(((u @ L1.T) * (v @ R1.T)) @ D1.T + ((v @ L1.T) * (u @ R1.T)) @ D1.T)
def T0(u, v):
    return 0.5*(((u @ L0m.T) * (v @ R0m.T)) @ D0m.T + ((v @ L0m.T) * (u @ R0m.T)) @ D0m.T)

# archetype write-basis for attn0 (as in qk_mlp0_composed_fold)
mh = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV); pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
cw0 = b0.attn.c_proj.weight.detach().float()
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


@torch.no_grad()
def streams(idx):
    """e-part, a0, m0, a1, x_pre1, true mlp1_out per position."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    # block 0
    x = b0.lambdas[0]*x0 + b0.lambdas[1]*x0; a = b0.attn; hcur = F.rms_norm(x, (D,))
    def qk0(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD); v1c = v
    q, k, q2, k2 = qk0(a.c_q), qk0(a.c_k), qk0(a.c_q2), qk0(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat = (s1*s2).masked_fill(~mask, 0.0)
    a0 = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
    xpre0 = x + a0; hin0 = F.rms_norm(xpre0, (D,)); m0 = b0.mlp(hin0)
    xb0 = xpre0 + m0
    # block 1 attention
    x = b1.lambdas[0]*xb0 + b1.lambdas[1]*x0; a = b1.attn; hcur = F.rms_norm(x, (D,))
    def qk1(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    vv = a.c_v(hcur).view(B, T, NH, HD); vv = (1-a.lamb)*vv + a.lamb*v1c.view_as(vv)
    q, k, q2, k2 = qk1(a.c_q), qk1(a.c_k), qk1(a.c_q2), qk1(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat1 = (s1*s2).masked_fill(~mask, 0.0)
    a1 = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat1, vv).reshape(B, T, -1))
    xpre1 = x + a1; hin1 = F.rms_norm(xpre1, (D,)); y1 = b1.mlp(hin1)
    lam0, lam1 = b1.lambdas[0].item(), b1.lambdas[1].item()
    e_part = (lam0*(b0.lambdas[0]+b0.lambdas[1]) + lam1) * x0     # embedding stream inside x_pre1
    return (e_part.reshape(-1, D), (lam0*a0).reshape(-1, D), (lam0*m0).reshape(-1, D),
            a1.reshape(-1, D), xpre1.reshape(-1, D), y1.reshape(-1, D), hin0.reshape(-1, D))

E, A0, M0, A1, XP, Y1, HIN0 = streams(COOC[:8].to(DEV)[:, :128])
rho2 = XP.pow(2).sum(1)/D
g1 = float(((T1(XP, XP)/rho2.unsqueeze(1) + bias1) - Y1).norm() / Y1.norm())
S = E + A0 + M0 + A1
g_split = float((S - XP).norm() / XP.norm())
print(f"GATE rms-gauge MLP1: {g1:.2e} | 4-stream sum == x_pre1: {g_split:.2e}", flush=True)
# 10-block variance shares
names = ['E', 'A0', 'M0', 'A1']; ST = [E, A0, M0, A1]
tot = (Y1 - Y1.mean(0)).pow(2).sum()
shares = {}
for i in range(4):
    for j in range(i, 4):
        blk = ((1 if i == j else 2) * T1(ST[i], ST[j])) / rho2.unsqueeze(1)
        shares[f'{names[i]}x{names[j]}'] = round(float((blk - blk.mean(0)).pow(2).sum()/tot), 3)
print("block shares:", shares, flush=True)

# a1 PCA basis per head (data-lite: 16 dims/head of attn1 head outputs)
@torch.no_grad()
def a1_heads(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    x = b0.lambdas[0]*x0 + b0.lambdas[1]*x0; a = b0.attn; hcur = F.rms_norm(x, (D,))
    def qk0(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD); v1c = v
    q, k, q2, k2 = qk0(a.c_q), qk0(a.c_k), qk0(a.c_q2), qk0(a.c_k2)
    pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
    a0_ = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
    xpre0 = x + a0_; xb0 = xpre0 + b0.mlp(F.rms_norm(xpre0, (D,)))
    x = b1.lambdas[0]*xb0 + b1.lambdas[1]*x0; a = b1.attn; hcur = F.rms_norm(x, (D,))
    def qk1(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    vv = a.c_v(hcur).view(B, T, NH, HD); vv = (1-a.lamb)*vv + a.lamb*v1c.view_as(vv)
    q, k, q2, k2 = qk1(a.c_q), qk1(a.c_k), qk1(a.c_q2), qk1(a.c_k2)
    pat1 = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
    return torch.einsum('bhqk,bkhd->bqhd', pat1, vv).reshape(-1, NH, HD)
acc = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
for i in range(0, 64, 8):
    f = a1_heads(COOC[i:i+8].to(DEV)[:, :128]).double()
    acc += torch.einsum('nhd,nhe->hde', f, f)
cw1 = b1.attn.c_proj.weight.detach().float()
cols1 = []
for hh in range(NH):
    ev, evec = torch.linalg.eigh(acc[hh])
    cols1.append(cw1[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:16]].float())
QA1, _ = torch.linalg.qr(torch.cat(cols1, 1))
print("a1 basis ready", flush=True)


@torch.no_grad()
def audit(mode):
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = x0; v1 = None
        cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        a0c = m0c = None
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            x = blk.lambdas[0]*x + blk.lambdas[1]*x0 if li else blk.lambdas[0]*x0 + blk.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T2, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            aout = a.c_proj(yh4.reshape(B, T2, -1)); x = x + aout; hin = F.rms_norm(x, (D,))
            if li == 0:
                a0c = aout.reshape(-1, D); mo = blk.mlp(hin); m0c = mo.reshape(-1, D); x = x + mo; continue
            if li == 1:
                lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
                e_part = (lam0*(b0.lambdas[0]+b0.lambdas[1]) + lam1) * x0.reshape(-1, D)
                a0s = lam0 * a0c; a1s = aout.reshape(-1, D)
                if mode in ('A', 'B'):
                    a0s = (a0s @ QA0) @ QA0.T; a1s = (a1s @ QA1) @ QA1.T
                if mode == 'B':
                    # m0 via MLP0's archetype-composed output (fully truncated chain)
                    e0 = (b0.lambdas[0]+b0.lambdas[1]) * x0.reshape(-1, D)
                    a0t = (a0c @ QA0) @ QA0.T
                    xp0 = e0 + a0t; r0 = xp0.pow(2).sum(1)/D
                    m0s = lam0 * (T0(xp0, xp0)/r0.unsqueeze(1) + bias0)
                else:
                    m0s = lam0 * m0c
                xps = e_part + a0s + m0s + a1s
                r2 = xps.pow(2).sum(1)/D
                mo = (T1(xps, xps)/r2.unsqueeze(1) + bias1).view(B, T2, D).to(x.dtype)
                x = x + mo; continue
            x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

d_exact = audit('exact') - SUBBASE
d_A = audit('A') - SUBBASE
d_B = audit('B') - SUBBASE
print(f"full-analytic (gate): +{d_exact:.5f} | arm A (a0/a1 truncated, m0 exact-chain): +{d_A:.5f} "
      f"({1-d_A/FLOOR1:.1%}) | arm B (fully truncated chain): +{d_B:.5f} ({1-d_B/FLOOR1:.1%})", flush=True)
res = {'gate_rms_gauge': g1, 'gate_stream_split': g_split, 'block_shares': shares,
       'full_analytic_dCE': round(d_exact, 5), 'armA_dCE': round(d_A, 5), 'armA_frac': round(1-d_A/FLOOR1, 4),
       'armB_dCE': round(d_B, 5), 'armB_frac': round(1-d_B/FLOOR1, 4)}
json.dump(res, open(f'{QK}/qk_mlp1_composed_fold.json', 'w'), indent=2)
print("QK MLP1 COMPOSED FOLD DONE", flush=True)
