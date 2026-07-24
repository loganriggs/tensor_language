"""TICK 183: repair the permutation-null statistic (tick-182 h5 control exposed it).

Flaw: spec check 3 compared CP fit quality on TWO DIFFERENT tensors (real core vs
permuted core). A mode-permuted core approaches the product of independent marginals,
which is intrinsically near-low-rank, so it can fit BETTER than a genuinely structured
real core — h5, whose symmetric m=512 structure is solidly validated, ties its
asymmetric null (0.132 vs 0.136), proving the statistic is broken at these capacities.

Corrected statistic — everything evaluated on the SAME real core:
  real-fit    : rel-err of CP factors fit on the real core (as before);
  null-factors: fit factors on the PERMUTED core, transplant them to the real core,
                refit only the nonneg weights lambda by Gram solve + projected descent,
                report rel-err on the real core;
  marg-rank1  : rank-1 product-of-marginals baseline (u_mode = p-weighted mean code),
                lambda optimal, rel-err on the real core.
Structure exists iff real-fit << null-factors (the null's directions do not describe
the real tensor), with marg-rank1 as the floor a trivial explanation achieves.

Also fixes the asymmetry meter: cos in TOKEN space, cos(S1 a_r, S2 b_r) — comparing
loadings over different mode dictionaries (tick 182) was meaningless.

Applied to: asymmetric h0 (m=2048,k=4), h4 (m=1024,k=4), h5 (m=128,k=2) — SAEs
retrained with the same seeds as tick 182 (deterministic, identical codes) — and
symmetric h0/h4 at m=4096 (codes loaded from qk_h04_sparse_core.pt), re-adjudicating
the tick-180 "fails the null" verdict.
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
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((Y - b) @ We.T)
        vals, idx = z.topk(k_code, dim=1)
    return idx, vals.detach()


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
        nrm = vals.norm().clamp_min(1e-30)
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / nrm

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


def cp_fit(sp, R, seed, iters=40, n_starts=6, symmetric=False):
    mm = sp.mm
    gg = torch.Generator().manual_seed(seed)
    A = torch.zeros(mm, 0, device=DEV)
    B = torch.zeros(mm, 0, device=DEV)
    C = torch.zeros(mm, 0, device=DEV)
    lv = torch.zeros(0, device=DEV)

    def res_mode(mode, x, y):
        out = sp.mode_mat(mode, x, y)
        if lv.numel():
            F1, F2, F3 = (A, B, C)
            if mode == 0:
                out -= F1 @ (lv * (F2.T @ x) * (F3.T @ y))
            elif mode == 1:
                out -= F2 @ (lv * (F1.T @ x) * (F3.T @ y))
            else:
                out -= F3 @ (lv * (F1.T @ x) * (F2.T @ y))
        return out

    lams = []
    for r in range(R):
        best = None
        for s in range(n_starts):
            if symmetric:
                u = torch.rand(mm, generator=gg).to(DEV)
                u = u / u.norm()
                for _ in range(60):
                    u = res_mode(0, u, u).clamp_min(0)
                    n = float(u.norm())
                    if n < 1e-20:
                        break
                    u = u / n
                a = b = c = u
            else:
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
    return A, B, C


def eval_on_core(sp, A, B, C, ridge=1e-8, polish=300):
    """Nonneg lambda refit on THIS core for fixed factors; returns rel-err."""
    R = A.shape[1]
    h = torch.tensor([sp.triple(A[:, r], B[:, r], C[:, r]) for r in range(R)], device=DEV)
    G = (A.T @ A) * (B.T @ B) * (C.T @ C)
    lam = torch.clamp(torch.linalg.solve(G + ridge * torch.eye(R, device=DEV), h), min=0)
    L = float(torch.linalg.eigvalsh(G)[-1].clamp_min(1e-12))
    for _ in range(polish):
        lam = torch.clamp(lam - (G @ lam - h) / L, min=0)
    res2 = 1.0 - 2.0 * float(lam @ h) + float(lam @ G @ lam)
    return max(res2, 0.0) ** 0.5


def marg_rank1(sp, S_list):
    us = []
    for t, S in enumerate(S_list):
        u = (QP[:, None] * S).sum(0)
        us.append(u / u.norm().clamp_min(1e-20))
    return eval_on_core(sp, us[0][:, None], us[1][:, None], us[2][:, None])


results = {}
R = 32

# ---------------- asymmetric heads ----------------
CFG = {0: (2048, 4), 4: (1024, 4), 5: (128, 2)}
for h, (mm, kc) in CFG.items():
    Ys = [K1[:, h], K2[:, h], Vv[:, h]]
    parts = [train_sae(Y, mm, kc, seed=0) for Y in Ys]
    idxs = [p[0] for p in parts]
    coeffs = [p[1] for p in parts]
    S = []
    for t in range(3):
        Sd = torch.zeros(V, mm, device=DEV)
        Sd.scatter_(1, idxs[t], coeffs[t])
        S.append(Sd)
    sp_real = AsymCore(*build_core(idxs, coeffs, mm), mm)
    gp = torch.Generator().manual_seed(7)
    S1p = S[0].clone()
    for f in range(mm):
        S1p[:, f] = S1p[torch.randperm(V, generator=gp).to(DEV), f]
    vn, in_ = S1p.topk(min(2 * kc, mm), dim=1)
    del S1p
    sp_null = AsymCore(*build_core([in_, idxs[1], idxs[2]], [vn, coeffs[1], coeffs[2]], mm), mm)

    A, B, C = cp_fit(sp_real, R, 0)
    An, Bn, Cn = cp_fit(sp_null, R, 0)
    real_fit = eval_on_core(sp_real, A, B, C)
    null_on_real = eval_on_core(sp_real, An, Bn, Cn)
    m1 = marg_rank1(sp_real, S)
    tok_asym = [float(F.cosine_similarity(S[0] @ A[:, r], S[1] @ B[:, r], dim=0))
                for r in range(A.shape[1])]
    row = {'form': 'asym', 'm_per_mode': mm, 'k': kc,
           'real_fit_R32': round(real_fit, 4),
           'null_factors_on_real_R32': round(null_on_real, 4),
           'marg_rank1': round(m1, 4),
           'token_space_cos_b1_b2_mean': round(sum(tok_asym) / len(tok_asym), 3)}
    results[f'h{h}_asym'] = row
    print(f'h{h} asym: real-fit {real_fit:.4f} | null-factors-on-real {null_on_real:.4f} '
          f'| marg-rank1 {m1:.4f} | token-space cos(b1,b2) {row["token_space_cos_b1_b2_mean"]}',
          flush=True)
    json.dump(results, open(f'{QK}/qk_null_repair.json', 'w'), indent=2)
    del sp_real, sp_null, S
    torch.cuda.empty_cache()

# ---------------- symmetric h0/h4 at m=4096 (tick-180 codes) ----------------
blob = torch.load(f'{QK}/qk_h04_sparse_core.pt', map_location=DEV)
for h in (0, 4):
    mm = 4096
    idx = blob[f'h{h}_idx'].long().to(DEV)
    coeff = blob[f'h{h}_coeff'].to(DEV)
    sp_real = AsymCore(*build_core([idx] * 3, [coeff] * 3, mm), mm)
    Sd = torch.zeros(V, mm, device=DEV)
    Sd.scatter_(1, idx, coeff)
    gp = torch.Generator().manual_seed(7)
    for f in range(mm):
        Sd[:, f] = Sd[torch.randperm(V, generator=gp).to(DEV), f]
    vn, in_ = Sd.topk(12, dim=1)
    sp_null = AsymCore(*build_core([in_] * 3, [vn] * 3, mm), mm)
    U, _, _ = cp_fit(sp_real, R, 0, symmetric=True)
    Un, _, _ = cp_fit(sp_null, R, 0, symmetric=True)
    real_fit = eval_on_core(sp_real, U, U, U)
    null_on_real = eval_on_core(sp_real, Un, Un, Un)
    Sd0 = torch.zeros(V, mm, device=DEV)
    Sd0.scatter_(1, idx, coeff)
    m1 = marg_rank1(sp_real, [Sd0] * 3)
    row = {'form': 'sym', 'm': mm,
           'real_fit_R32': round(real_fit, 4),
           'null_factors_on_real_R32': round(null_on_real, 4),
           'marg_rank1': round(m1, 4)}
    results[f'h{h}_sym'] = row
    print(f'h{h} sym: real-fit {real_fit:.4f} | null-factors-on-real {null_on_real:.4f} '
          f'| marg-rank1 {m1:.4f}', flush=True)
    json.dump(results, open(f'{QK}/qk_null_repair.json', 'w'), indent=2)
    del sp_real, sp_null, Sd, Sd0
    torch.cuda.empty_cache()
print('NULL REPAIR DONE', flush=True)
