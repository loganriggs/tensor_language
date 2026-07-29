"""The 'everything compressed, everything analytic' cell: the two-layer composed chain with
(1) attention streams truncated to their 144-dim bases, (2) MLP0's output replaced by the truncated
chain, AND (3) both MLP tensors T0, T1 truncated to rank-R interaction bases chosen WEIGHT-NATIVELY
under the composed-stream metric (directions = generalized eigendirections of the output-weighted
input Gram, whitened by the composed streams' second moment; output maps by closed-form tensor
projection). Data enters ONLY as (a) the 144-dim a1 basis and (b) the stream second moments used
as whitening metrics. References: arm B full-tensor chain +0.105; DATA-fit joint MLP0+MLP1 programs
+0.193 (qk_ledger_v2 'both'). Joint floor for {mlp0,mlp1}: measured here (both interfaces
mean-input simultaneously).
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
b0, b1 = m.transformer.h[0], m.transformer.h[1]
WT = {}
for tag, blk in [('0', b0), ('1', b1)]:
    WT[tag] = (blk.mlp.Left.weight.detach().float(), blk.mlp.Right.weight.detach().float(),
               blk.mlp.Down.weight.detach().float(), blk.mlp.Down_bias.detach().float())

def T_ev(tag, u, v):
    Lw, Rw, Dw, _ = WT[tag]
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)

# archetype bases (attn0 named; attn1 PCA-16/head as in qk_mlp1_composed_fold)
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
def run_two(idx):
    """streams for both layers + a1 head outputs."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    x = (b0.lambdas[0]+b0.lambdas[1])*x0; a = b0.attn; hcur = F.rms_norm(x, (D,))
    def qk0(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD); v1c = v
    q, k, q2, k2 = qk0(a.c_q), qk0(a.c_k), qk0(a.c_q2), qk0(a.c_k2)
    pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
    a0 = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
    xpre0 = x + a0; m0 = b0.mlp(F.rms_norm(xpre0, (D,))); xb0 = xpre0 + m0
    x = b1.lambdas[0]*xb0 + b1.lambdas[1]*x0; a = b1.attn; hcur = F.rms_norm(x, (D,))
    def qk1(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    vv = a.c_v(hcur).view(B, T, NH, HD); vv = (1-a.lamb)*vv + a.lamb*v1c.view_as(vv)
    q, k, q2, k2 = qk1(a.c_q), qk1(a.c_k), qk1(a.c_q2), qk1(a.c_k2)
    pat1 = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
    a1h = torch.einsum('bhqk,bkhd->bqhd', pat1, vv)
    a1 = a.c_proj(a1h.reshape(B, T, -1))
    return x0.reshape(-1, D), a0.reshape(-1, D), a1.reshape(-1, D), a1h.reshape(-1, NH, HD)

# a1 basis + composed-stream second moments (metric data: cooc 64 seqs)
accH = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
M0acc = torch.zeros(D, D, device=DEV, dtype=torch.float64)
M1acc = torch.zeros(D, D, device=DEV, dtype=torch.float64)
nacc = 0
lam00 = (b0.lambdas[0]+b0.lambdas[1]).item(); lam10, lam11 = b1.lambdas[0].item(), b1.lambdas[1].item()
_, _, _, a1h_probe = run_two(COOC[:8].to(DEV)[:, :128])
for i in range(0, 64, 8):
    e, a0, a1, a1h = run_two(COOC[i:i+8].to(DEV)[:, :128])
    accH += torch.einsum('nhd,nhe->hde', a1h.double(), a1h.double())
    a0t = (a0 @ QA0) @ QA0.T
    xp0 = lam00*e + a0t
    M0acc += (xp0.double().T @ xp0.double()); nacc += xp0.shape[0]
cw1 = b1.attn.c_proj.weight.detach().float()
cols1 = []
for hh in range(NH):
    ev, evec = torch.linalg.eigh(accH[hh])
    cols1.append(cw1[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:16]].float())
QA1, _ = torch.linalg.qr(torch.cat(cols1, 1))
M0 = (M0acc/nacc).float()
# second pass for M1 (needs truncated m0 which needs T0 features... use FULL-tensor truncated chain
# for the metric -- metric choice only)
for i in range(0, 64, 8):
    e, a0, a1, _ = run_two(COOC[i:i+8].to(DEV)[:, :128])
    a0t = (a0 @ QA0) @ QA0.T; a1t = (a1 @ QA1) @ QA1.T
    xp0 = lam00*e + a0t; r0 = xp0.pow(2).sum(1)/D
    m0t = T_ev('0', xp0, xp0)/r0.unsqueeze(1) + WT['0'][3]
    xp1 = (lam10*lam00 + lam11)*e + lam10*a0t + lam10*m0t + a1t
    M1acc += (xp1.double().T @ xp1.double())
M1 = (M1acc/nacc).float()
print("bases + metrics ready", flush=True)

def trunc_tensor(tag, Mmet, R):
    """weight-native rank-R interaction basis for T_tag; coefficients by GAUSSIAN projection under
    the metric (Isserlis 4th moments) with an intercept -- fixes the isotropic-Frobenius mismatch:
    minimize E_{x~N(0,M)} |T(x,x) - c - sum_r u_r (a_r.x)^2|^2, all closed form."""
    Lw, Rw, Dw, _ = WT[tag]
    wp = Dw.norm(dim=0)
    G = torch.einsum('p,pi,pj->ij', wp, Lw, Lw) + torch.einsum('p,pi,pj->ij', wp, Rw, Rw)
    W = torch.linalg.cholesky(Mmet + 1e-4*torch.eye(D, device=DEV))
    Gw = W.T @ G @ W
    ev, evec = torch.linalg.eigh(Gw)
    A = (W @ evec[:, ev.argsort(descending=True)[:R]]).T
    A = A / A.norm(dim=1, keepdim=True)
    MA = A @ Mmet                                                  # (R, D) rows = M a_r
    q = (A * MA).sum(1)                                            # a_r^T M a_r
    # E[T(x,x)] = T : M  (contract with metric), per output dim
    TM = Dw @ ((Lw @ Mmet) * Rw).sum(1)                            # sum_ij T_oij M_ij
    # cross moments: E[T(x,x)(a_r.x)^2] = (T:M) q_r + 2 T(Ma_r, Ma_r)
    TMa = Dw @ ((Lw @ MA.T) * (Rw @ MA.T))                         # (D, R): T(Ma_r, Ma_r)
    b_r = TM.unsqueeze(1) * q.unsqueeze(0) + 2*TMa                 # (D, R)
    # feature Gram: E[(a_r.x)^2 (a_s.x)^2] = q_r q_s + 2 (a_r^T M a_s)^2 ; E[(a_r.x)^2] = q_r
    C = A @ MA.T                                                   # a_r^T M a_s
    Gram = q.unsqueeze(1)*q.unsqueeze(0) + 2*C**2
    # solve with intercept: [[1, q^T],[q, Gram]] [c; U] = [TM; b]
    n1 = torch.cat([torch.ones(1, device=DEV), q])
    Big = torch.cat([n1.unsqueeze(0), torch.cat([q.unsqueeze(1), Gram], 1)], 0)
    rhs = torch.cat([TM.unsqueeze(0), b_r.T], 0)                   # (R+1, D)
    sol = torch.linalg.solve(Big + 1e-6*torch.eye(R+1, device=DEV), rhs)
    c0, U = sol[0], sol[1:]
    return A.contiguous(), U.contiguous(), c0.contiguous()

PROG = {}
for R in (256, 512):
    PROG[('0', R)] = trunc_tensor('0', M0, R)
    PROG[('1', R)] = trunc_tensor('1', M1, R)
print("truncated tensors built", flush=True)


@torch.no_grad()
def audit(mode, R=256):
    """mode: 'alltrunc' | 'jointfloor' | 'armB_ref'"""
    tot = 0.0; n = 0
    if mode == 'jointfloor':
        mu0 = None; mu1 = None
        # mean inputs for both interfaces from cooc
        s0 = torch.zeros(D, device=DEV, dtype=torch.float64); s1_ = torch.zeros(D, device=DEV, dtype=torch.float64); nn = 0
        for i in range(0, 64, 8):
            idx = COOC[i:i+8].to(DEV)[:, :128]
            B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
            e, a0, a1, _ = run_two(idx)
            xp0 = lam00*e + a0
            m0 = b0.mlp(F.rms_norm(xp0, (D,)).view(B, T, D)).reshape(-1, D)
            xp1 = (lam10*lam00+lam11)*e + lam10*a0 + lam10*m0 + a1
            s0 += F.rms_norm(xp0, (D,)).double().sum(0); s1_ += F.rms_norm(xp1, (D,)).double().sum(0); nn += xp0.shape[0]
        mu0 = F.rms_norm((s0/nn).float(), (D,)); mu1 = F.rms_norm((s1_/nn).float(), (D,))
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = x0; v1 = None
        cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        a0c = None
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            x = blk.lambdas[0]*x + blk.lambdas[1]*x0 if li else (b0.lambdas[0]+b0.lambdas[1])*x0
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
                if mode == 'jointfloor':
                    x = x + b0.mlp(mu0.expand(B, T2, D)); a0c = aout.reshape(-1, D); continue
                a0t = ((aout.reshape(-1, D)) @ QA0) @ QA0.T; a0c = a0t
                xp0 = lam00*x0.reshape(-1, D) + a0t; r0 = xp0.pow(2).sum(1)/D
                A_, U_, c0_ = PROG[('0', R)]
                m0t = (c0_ + ((xp0 @ A_.T)**2) @ U_)/r0.unsqueeze(1) + WT['0'][3]
                x = x + m0t.view(B, T2, D).to(x.dtype); PROGm0 = m0t; continue
            if li == 1:
                if mode == 'jointfloor':
                    x = x + b1.mlp(mu1.expand(B, T2, D)); continue
                a1t = ((aout.reshape(-1, D)) @ QA1) @ QA1.T
                xp1 = (lam10*lam00+lam11)*x0.reshape(-1, D) + lam10*a0c + lam10*PROGm0 + a1t
                r1 = xp1.pow(2).sum(1)/D
                A_, U_, c1_ = PROG[('1', R)]
                m1t = (c1_ + ((xp1 @ A_.T)**2) @ U_)/r1.unsqueeze(1) + WT['1'][3]
                x = x + m1t.view(B, T2, D).to(x.dtype); continue
            x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

floor_joint = audit('jointfloor') - SUBBASE
res = {'joint_floor_mlp01': round(floor_joint, 5), 'ref_armB_fulltensor': 0.10473, 'ref_data_joint': 0.19308}
for R in (256, 512):
    d = audit('alltrunc', R) - SUBBASE
    res[f'alltrunc_R{R}'] = {'dCE': round(d, 5), 'joint_frac': round(1 - d/floor_joint, 4)}
    print(f"ALL-TRUNC R={R}: dCE +{d:.5f} -> {1-d/floor_joint:.1%} of joint floor "
          f"(refs: armB full-tensor +0.105, data-joint +0.193)", flush=True)
json.dump(res, open(f'{QK}/qk_composed_alltrunc.json', 'w'), indent=2)
print("QK COMPOSED ALLTRUNC DONE", flush=True)
