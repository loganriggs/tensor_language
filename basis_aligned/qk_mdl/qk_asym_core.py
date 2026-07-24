"""TICK 182: mode-separated ASYMMETRIC core for the CP-refractory heads (0, 4), with
head 5 as a positive control.

Hypothesis: h0/h4's symmetric cores fail the permutation null (tick 180) because their
mechanism is asymmetric — key-class X on branch 1 with DIFFERENT key-class Y on branch 2
writing value-class Z. A symmetric archetype u^(x3) cannot express X!=Y triples; the
mode-separated core can:
  T_abc = sum_t p_t s1_ta s2_tb sv_tc,  from three separate per-mode SAEs on the
  (V,128) rows k1, k2, v — mirroring mu_i = M(q1_i, q2_i, .) mode-for-mode.
Stage 3 becomes asymmetric nonneg CP  T ~= sum_r lam_r a_r (x) b_r (x) c_r via
alternating power iteration (HOPM) + deflation. Planted known-answer test (asymmetric
archetypes, 3 modes) gates the fitter BEFORE real data. Permutation null: permute token
assignment per feature column of mode 1 only (kills cross-mode co-occurrence, preserves
all marginals). Components read as (branch-1 key class) x (branch-2 key class) ->
(value class): top tokens dumped per mode.

Per-mode SAE ladder: h0/h4 m in {1024, 2048, 4096} at k=4; h5 m in {128, 256, 512} at
k=2 (its tick-181 optimum scale). Gate: asymmetric sketched moment residual < 0.05
(probes u, v, w applied to k1, k2, v separately).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
STEPS, BATCH, LR, GATE = 12000, 2048, 3e-3, 0.05
tok = AutoTokenizer.from_pretrained('gpt2')

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
        rec = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
    return Dn.detach(), b.detach(), idx, vals.detach(), rec.detach()


@torch.no_grad()
def asym_moment_residual(Ys, recs, n_probe=256, seed=3):
    """Probes applied per mode: T(u,v,w) = sum_t p_t (k1.u)(k2.v)(vv.w)."""
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
def build_asym_core(idxs, coeffs, mm):
    """COO core T_abc = sum_t p_t s1_ta s2_tb sv_tc from three k-sparse codes."""
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
    keys = torch.cat(keys_all)
    vals = torch.cat(vals_all)
    uk, inv = torch.unique(keys, return_inverse=True)
    cv = torch.zeros(len(uk), device=DEV)
    cv.scatter_add_(0, inv, vals)
    ai = torch.div(uk, mm * mm, rounding_mode='floor')
    bi = torch.div(uk, mm, rounding_mode='floor') % mm
    ci = uk % mm
    return ai, bi, ci, cv


class AsymCore:
    def __init__(self, ai, bi, ci, vals, mm):
        self.mm = mm
        nrm = vals.norm().clamp_min(1e-30)
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / nrm

    def mode_mat(self, mode, x, y):
        """Contract the two OTHER modes: mode=0 -> out[a] = sum vals*x[b]*y[c], etc."""
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


def cp_fit_asym(sp, R, seed, iters=40, n_starts=6):
    """Asymmetric nonneg CP: alternating power (HOPM) + deflation on the sparse core.
    Deflation handled implicitly: residual contractions subtract the low-rank terms."""
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
    inner = torch.tensor([sp.triple(A[:, r], B[:, r], C[:, r]) for r in range(len(lams))],
                         device=DEV)
    G = (A.T @ A) * (B.T @ B) * (C.T @ C)
    prefix = {}
    for P in (16, 32, 64):
        p = min(P, len(lams))
        lp = lv[:p]
        res2 = 1.0 - 2.0 * float(lp @ inner[:p]) + float(lp @ G[:p, :p] @ lp)
        prefix[P] = max(res2, 0.0) ** 0.5
    return (A, B, C), lv, prefix


# ---------------- planted known-answer test (asymmetric) ----------------
print('=== planted test: asymmetric sparse CP at m=2048 ===', flush=True)
mm0, R0 = 2048, 24
g = torch.Generator().manual_seed(5)
MUs = []
for _ in range(3):
    MU = torch.zeros(mm0, R0, device=DEV)
    for r in range(R0):
        sup = torch.randperm(mm0, generator=g)[:6]
        MU[sup.to(DEV), r] = (0.3 + torch.rand(6, generator=g)).to(DEV)
    MUs.append(MU / MU.norm(dim=0, keepdim=True))
LAM0 = (10 ** (2 * torch.rand(R0, generator=g))).to(DEV)
pk, pv = [], []
for r in range(R0):
    acts = [MU[:, r].nonzero().squeeze(1) for MU in MUs]
    ws = [MUs[i][acts[i], r] for i in range(3)]
    a = acts[0][:, None, None].expand(6, 6, 6)
    b_ = acts[1][None, :, None].expand(6, 6, 6)
    c_ = acts[2][None, None, :].expand(6, 6, 6)
    v = LAM0[r] * ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    pk.append(((a * mm0 + b_) * mm0 + c_).reshape(-1))
    pv.append(v.reshape(-1))
uk, inv = torch.unique(torch.cat(pk), return_inverse=True)
cv = torch.zeros(len(uk), device=DEV)
cv.scatter_add_(0, inv, torch.cat(pv))
sp_pl = AsymCore(torch.div(uk, mm0 * mm0, rounding_mode='floor'),
                 torch.div(uk, mm0, rounding_mode='floor') % mm0, uk % mm0, cv, mm0)
(Ap, Bp, Cp), _, prel = cp_fit_asym(sp_pl, R0, 0)
mcos = float(np.mean([float((X.T @ MU).abs().max(0).values.mean())
                      for X, MU in ((Ap, MUs[0]), (Bp, MUs[1]), (Cp, MUs[2]))]))
print(f'planted: matched-cos {mcos:.4f} rel-err@24 {prel[32]:.4f} '
      f'-> {"PASS" if mcos >= 0.95 else "FAIL"}', flush=True)
assert mcos >= 0.95, 'asymmetric CP fitter failed the planted gate'

# ---------------- real heads ----------------
HEADS = {0: ((1024, 2048, 4096), 4), 4: ((1024, 2048, 4096), 4), 5: ((128, 256, 512), 2)}
results = {'planted_matched_cos': round(mcos, 4)}
blob = {}
for h, (ladder, kc) in HEADS.items():
    Ys = [K1[:, h], K2[:, h], Vv[:, h]]
    fit = None
    for mm in ladder:
        parts = [train_sae(Y, mm, kc, seed=0) for Y in Ys]
        mres = asym_moment_residual(Ys, [p[4] for p in parts])
        print(f'h{h} per-mode m={mm} k={kc}: asym moment residual {mres:.4f}'
              + (' PASS' if mres < GATE else ''), flush=True)
        if mres < GATE:
            fit = (mm, parts, mres)
            break
        fit = (mm, parts, mres)
    mm, parts, mres = fit
    row = {'m_per_mode': mm, 'k': kc, 'moment_rel_err': round(mres, 4),
           'gate_pass': bool(mres < GATE)}
    idxs = [p[2] for p in parts]
    coeffs = [p[3] for p in parts]
    sp = AsymCore(*build_asym_core(idxs, coeffs, mm), mm)
    row['core_nnz'] = int(len(sp.vals))
    Us3, prels = [], []
    for seed in range(3):
        F3, lv, pr = cp_fit_asym(sp, 64, seed)
        Us3.append(F3)
        prels.append(pr)
        print(f'  h{h} seed {seed}: rel-err 16/32/64 = '
              f'{pr[16]:.4f}/{pr[32]:.4f}/{pr[64]:.4f}', flush=True)
    for P in (16, 32, 64):
        row[f'R{P}_relerr'] = round(min(p[P] for p in prels), 4)
    stab = []
    for i in range(3):
        for j in range(i + 1, 3):
            cs = [(Us3[i][t][:, :32].T @ Us3[j][t][:, :32]).abs().max(1).values.mean()
                  for t in range(3)]
            stab.append(float(sum(cs) / 3))
    row['R32_stability'] = round(sum(stab) / len(stab), 3)
    best = int(np.argmin([p[64] for p in prels]))
    (A, B, C), lv, _ = cp_fit_asym(sp, 64, best)
    for t, nm in ((0, 'A'), (1, 'B'), (2, 'C')):
        blob[f'h{h}_{nm}'] = (A, B, C)[t].cpu()
    blob[f'h{h}_lam'] = lv.cpu()
    # archetype dump: top tokens per mode for top-5 components
    S = []
    for t in range(3):
        Sd = torch.zeros(V, mm, device=DEV)
        Sd.scatter_(1, idxs[t], coeffs[t])
        S.append(Sd)
    arch = []
    for r in lv[:32].argsort(descending=True)[:5].tolist():
        entry = {}
        for t, nm, U in ((0, 'branch1_keys', A), (1, 'branch2_keys', B), (2, 'values', C)):
            top = (S[t] @ U[:, r]).argsort(descending=True)[:6]
            entry[nm] = [tok.decode([x]).replace('\n', '\\n') for x in top.tolist()]
        arch.append(entry)
    row['top_archetypes'] = arch
    # asymmetry meter: how different are the three factors of each component?
    d_ab = [float(F.cosine_similarity(A[:, r], B[:, r], dim=0)) for r in range(min(32, A.shape[1]))]
    row['mean_cos_A_B_top32'] = round(sum(d_ab) / len(d_ab), 3)
    # permutation null: shuffle mode-1 token assignment per feature column
    gp = torch.Generator().manual_seed(7)
    S1 = S[0]
    for f in range(mm):
        S1[:, f] = S1[torch.randperm(V, generator=gp).to(DEV), f]
    vn, in_ = S1.topk(min(2 * kc, mm), dim=1)
    del S
    torch.cuda.empty_cache()
    sp_n = AsymCore(*build_asym_core([in_, idxs[1], idxs[2]], [vn, coeffs[1], coeffs[2]], mm), mm)
    _, _, pr_n = cp_fit_asym(sp_n, 32, 0)
    row['R32_relerr_null'] = round(pr_n[32], 4)
    print(f'h{h}: null R32 {pr_n[32]:.4f} (real {row["R32_relerr"]}) | '
          f'mean cos(A,B) {row["mean_cos_A_B_top32"]}', flush=True)
    del sp, sp_n
    torch.cuda.empty_cache()
    results[f'h{h}'] = row
    json.dump(results, open(f'{QK}/qk_asym_core.json', 'w'), indent=2)
    torch.save(blob, f'{QK}/qk_asym_core.pt')
print('ASYM CORE DONE', flush=True)
