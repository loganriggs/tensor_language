"""TICK 187 (Logan): warm-started joint polish for heads 0/4 in the ASYMMETRIC
mode-separated form — the tick-186 recipe (true deflation warm start, gamma-ramped
sketched moment matching) applied to the hard heads.

Per head (h0: m=2048/mode k=4; h4: m=1024/mode k=4): stagewise = three per-mode SAEs
(seed 0, identical to ticks 183/185/186a) + asymmetric CP R=64. Joint phase trains all
three SAEs AND the three factor matrices together; factors warm-started as
  FA_r = (V * scale0 * lam_r)^(1/3) a_r  (same for FB from b_r, FC from c_r)
so the CP model equals the stagewise code core at step 0. Loss: sum of per-mode
reconstruction MSE + gamma(step) * sketched third-moment match between codes and CP
model (8 probes, EMA-normalized), gamma ramped 0 -> 0.05 after step 500, 4000 steps.
Metrics: asym moment gate before/after, CP refit rel-err R=64 before/after, factor
drift, transfer of stagewise factors onto the joint core. Polished SAEs + factors
saved to qk_h04_polish.pt for the artifact refresh.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR = 12000, 2048, 3e-3
R_CP, GAMMA_MAX, JSTEPS, N_PROBE = 64, 0.05, 4000, 8

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
q1, k1 = branch_factors(m, 1)
q2, k2 = branch_factors(m, 2)
K1, K2 = k1.float().to(DEV), k2.float().to(DEV)
with torch.no_grad():
    a0 = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a0.c_v(E).view(V, NH, HD)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()
QP_CPU = QP.cpu()


def train_sae(Y, m_atoms, k_code, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:m_atoms].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (Y * QP[:, None]).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(m_atoms, device=DEV)
    for step in range(STEPS):
        kk = max(k_code, int(round(2 * k_code - k_code * min(1.0, 2 * step / STEPS))))
        bi = torch.multinomial(QP_CPU, BATCH, replacement=True, generator=g).to(DEV)
        y = Y[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((y - b) @ We.T)
        vals, idx = z.topk(kk, dim=1)
        yhat = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), (vals > 1e-8).float().reshape(-1))
        loss = ((yhat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    z_ = torch.relu((Y - b) @ We.T)
                    v_, i_ = z_.topk(k_code, dim=1)
                    rec = b + (v_.unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - Y) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = Y[worst] / Y[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
                    del z_, rec
            fired.zero_()
    return Dm.detach(), b.detach(), We.detach()


def encode(Y, Dm, b, We, kc):
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((Y - b) @ We.T)
        vals, idx = z.topk(kc, dim=1)
        rec = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
    return idx, vals, rec


@torch.no_grad()
def asym_moment_residual(Ys, recs, n_probe=256, seed=3):
    g = torch.Generator(device='cpu').manual_seed(seed)
    num = den = 0.0
    for _ in range(n_probe):
        pr = [torch.randn(Y.shape[1], generator=g).to(DEV) for Y in Ys]
        t = (QP * (Ys[0] @ pr[0]) * (Ys[1] @ pr[1]) * (Ys[2] @ pr[2])).sum()
        th = (QP * (recs[0] @ pr[0]) * (recs[1] @ pr[1]) * (recs[2] @ pr[2])).sum()
        num += float((t - th) ** 2)
        den += float(t ** 2)
    return num / max(den, 1e-30)


@torch.no_grad()
def build_core(idxs, coeffs, mm):
    ka, kb, kc_ = (ii.shape[1] for ii in idxs)
    keys_all, vals_all = [], []
    for s in range(0, V, 4096):
        i1, i2, i3 = (ii[s:s + 4096].long() for ii in idxs)
        c1, c2, c3 = (cc[s:s + 4096] for cc in coeffs)
        w = QP[s:s + 4096, None] * c1
        a = i1[:, :, None, None].expand(-1, ka, kb, kc_)
        b_ = i2[:, None, :, None].expand(-1, ka, kb, kc_)
        c_ = i3[:, None, None, :].expand(-1, ka, kb, kc_)
        v = w[:, :, None, None] * c2[:, None, :, None] * c3[:, None, None, :]
        keys_all.append(((a * mm + b_) * mm + c_).reshape(-1))
        vals_all.append(v.reshape(-1))
    uk, inv = torch.unique(torch.cat(keys_all), return_inverse=True)
    cv = torch.zeros(len(uk), device=DEV)
    cv.scatter_add_(0, inv, torch.cat(vals_all))
    return (torch.div(uk, mm * mm, rounding_mode='floor'),
            torch.div(uk, mm, rounding_mode='floor') % mm, uk % mm, cv)


class AsymCore:
    def __init__(self, ai, bi, ci, vals, mm):
        self.mm = mm
        self.scale = float(vals.norm().clamp_min(1e-30))
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / self.scale

    def mode_mat(self, mode, x, y):
        out = torch.zeros(self.mm, device=DEV)
        if mode == 0:
            out.scatter_add_(0, self.ai, self.vals * x[self.bi] * y[self.ci])
        elif mode == 1:
            out.scatter_add_(0, self.bi, self.vals * x[self.ai] * y[self.ci])
        else:
            out.scatter_add_(0, self.ci, self.vals * x[self.ai] * y[self.bi])
        return out

    def triple(self, a, b, c):
        return float((self.vals * a[self.ai] * b[self.bi] * c[self.ci]).sum())


def cp_fit(sp, R, seed, iters=40, n_starts=6):
    mm = sp.mm
    gg = torch.Generator().manual_seed(seed)
    A = torch.zeros(mm, 0, device=DEV)
    B = torch.zeros(mm, 0, device=DEV)
    C = torch.zeros(mm, 0, device=DEV)
    lv = torch.zeros(0, device=DEV)

    def res_mode(mode, x, y):
        out = sp.mode_mat(mode, x, y)
        if lv.numel():
            if mode == 0:
                out -= A @ (lv * (B.T @ x) * (C.T @ y))
            elif mode == 1:
                out -= B @ (lv * (A.T @ x) * (C.T @ y))
            else:
                out -= C @ (lv * (A.T @ x) * (B.T @ y))
        return out

    lams = []
    for r in range(R):
        best = None
        for s in range(n_starts):
            a, b, c = (torch.rand(mm, generator=gg).to(DEV) for _ in range(3))
            a, b, c = a / a.norm(), b / b.norm(), c / c.norm()
            for _ in range(iters):
                a = res_mode(0, b, c).clamp_min(0)
                a = a / a.norm().clamp_min(1e-20)
                b = res_mode(1, a, c).clamp_min(0)
                b = b / b.norm().clamp_min(1e-20)
                c = res_mode(2, a, b).clamp_min(0)
                nc = float(c.norm())
                if nc < 1e-20:
                    break
                c = c / nc
            lam = sp.triple(a, b, c) - (float((lv * (A.T @ a) * (B.T @ b) * (C.T @ c)).sum())
                                        if lv.numel() else 0.0)
            if best is None or lam > best[0]:
                best = (lam, a, b, c)
        if best[0] <= 0:
            break
        lams.append(best[0])
        A = torch.cat([A, best[1][:, None]], 1)
        B = torch.cat([B, best[2][:, None]], 1)
        C = torch.cat([C, best[3][:, None]], 1)
        lv = torch.tensor(lams, device=DEV)
    return A, B, C, lv


def eval_on_core(sp, A, B, C):
    R = A.shape[1]
    h = torch.tensor([sp.triple(A[:, r], B[:, r], C[:, r]) for r in range(R)], device=DEV)
    G = (A.T @ A) * (B.T @ B) * (C.T @ C)
    lam = torch.clamp(torch.linalg.solve(G + 1e-8 * torch.eye(R, device=DEV), h), min=0)
    L = float(torch.linalg.eigvalsh(G)[-1].clamp_min(1e-12))
    for _ in range(300):
        lam = torch.clamp(lam - (G @ lam - h) / L, min=0)
    res2 = 1.0 - 2.0 * float(lam @ h) + float(lam @ G @ lam)
    return max(res2, 0.0) ** 0.5


out = {}
save = {}
for h, (mm, kc) in ((0, (2048, 4)), (4, (1024, 4))):
    Ys = [K1[:, h], K2[:, h], Vv[:, h]]
    saes = [train_sae(Y, mm, kc, seed=0) for Y in Ys]
    encs = [encode(Y, *s, kc) for Y, s in zip(Ys, saes)]
    idxs = [e[0] for e in encs]
    coeffs = [e[1] for e in encs]
    mres0 = asym_moment_residual(Ys, [e[2] for e in encs])
    sp0 = AsymCore(*build_core(idxs, coeffs, mm), mm)
    A0, B0, C0, lam0 = cp_fit(sp0, R_CP, 0)
    rel0 = eval_on_core(sp0, A0, B0, C0)
    scale0 = sp0.scale
    del sp0
    torch.cuda.empty_cache()
    print(f'h{h} stagewise: mres {mres0:.4f} cpR64 {rel0:.4f}', flush=True)

    g = torch.Generator(device='cpu').manual_seed(11 + h)
    params = []
    joint = []
    for (Dm0, b0, We0) in saes:
        Dm = Dm0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        We = We0.clone().requires_grad_(True)
        joint.append((Dm, b, We))
        params += [Dm, b, We]
    scal = (V * scale0 * lam0.clamp_min(1e-12)) ** (1.0 / 3.0)
    FA = (A0 * scal).clone().requires_grad_(True)
    FB = (B0 * scal).clone().requires_grad_(True)
    FC = (C0 * scal).clone().requires_grad_(True)
    params += [FA, FB, FC]
    opt = torch.optim.Adam(params, lr=1e-3)
    ema_m3 = 1.0
    for step in range(JSTEPS):
        gamma = GAMMA_MAX * max(0.0, (step - 500) / (JSTEPS - 500))
        bi = torch.multinomial(QP_CPU, 4096, replacement=True, generator=g).to(DEV)
        loss = 0.0
        ss = []
        for (Dm, b, We), Y in zip(joint, Ys):
            y = Y[bi]
            Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            z = torch.relu((y - b) @ We.T)
            with torch.no_grad():
                tidx = z.topk(kc, dim=1).indices
            s = torch.zeros_like(z).scatter_(1, tidx, torch.gather(z, 1, tidx))
            ss.append(s)
            loss = loss + ((b + s @ Dn - y) ** 2).mean()
        if gamma > 0:
            mm3 = 0.0
            for _ in range(N_PROBE):
                u, v_, wv = (torch.randn(mm, generator=g).to(DEV) for _ in range(3))
                m3 = ((ss[0] @ u) * (ss[1] @ v_) * (ss[2] @ wv)).mean()
                cp3 = ((FA.clamp_min(0).T @ u) * (FB.clamp_min(0).T @ v_)
                       * (FC.clamp_min(0).T @ wv)).sum() / V
                mm3 = mm3 + (m3 - cp3) ** 2
            mm3 = mm3 / N_PROBE
            ema_m3 = 0.99 * ema_m3 + 0.01 * float(mm3)
            loss = loss + gamma * mm3 / max(ema_m3, 1e-20)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        encsJ = [encode(Y, Dm.detach(), b.detach(), We.detach(), kc)
                 for (Dm, b, We), Y in zip(joint, Ys)]
    idxsJ = [e[0] for e in encsJ]
    coeffsJ = [e[1] for e in encsJ]
    mresJ = asym_moment_residual(Ys, [e[2] for e in encsJ])
    spJ = AsymCore(*build_core(idxsJ, coeffsJ, mm), mm)
    AJ, BJ, CJ, lamJ = cp_fit(spJ, R_CP, 0)
    relJ = eval_on_core(spJ, AJ, BJ, CJ)
    drift = float(np.mean([float((X.T @ X0).abs().max(1).values.mean())
                           for X, X0 in ((AJ, A0), (BJ, B0), (CJ, C0))]))
    transfer0 = eval_on_core(spJ, A0, B0, C0)
    with torch.no_grad():
        Fn = [X.clamp_min(0) / X.clamp_min(0).norm(dim=0, keepdim=True).clamp_min(1e-12)
              for X in (FA, FB, FC)]
    transferF = eval_on_core(spJ, *[f.detach() for f in Fn])
    del spJ
    torch.cuda.empty_cache()
    out[f'h{h}'] = {'stagewise': {'mres': round(mres0, 4), 'cp_relerr_R64': round(rel0, 4)},
                    'joint_polish': {'mres': round(mresJ, 4), 'cp_relerr_R64': round(relJ, 4),
                                     'factor_drift_cos': round(drift, 4),
                                     'stage_factors_on_jointcore': round(transfer0, 4),
                                     'trained_factors_on_jointcore': round(transferF, 4)}}
    print(f'h{h} polish: mres {mresJ:.4f} cpR64 {relJ:.4f} (stage {rel0:.4f}) | '
          f'drift {drift:.3f} stage-transfer {transfer0:.4f} trained-transfer {transferF:.4f}',
          flush=True)
    for t, nm in enumerate(('k1', 'k2', 'v')):
        save[f'h{h}_{nm}_Dm'] = joint[t][0].detach().cpu()
        save[f'h{h}_{nm}_b'] = joint[t][1].detach().cpu()
        save[f'h{h}_{nm}_We'] = joint[t][2].detach().cpu()
    save[f'h{h}_AJ'], save[f'h{h}_BJ'], save[f'h{h}_CJ'] = AJ.cpu(), BJ.cpu(), CJ.cpu()
    save[f'h{h}_lamJ'] = lamJ.cpu()
    save[f'h{h}_idxsJ'] = [i.cpu() for i in idxsJ]
    save[f'h{h}_coeffsJ'] = [c.cpu() for c in coeffsJ]
    json.dump(out, open(f'{QK}/qk_h04_polish.json', 'w'), indent=2)
    torch.save(save, f'{QK}/qk_h04_polish.pt')
print('H04 POLISH DONE', flush=True)
