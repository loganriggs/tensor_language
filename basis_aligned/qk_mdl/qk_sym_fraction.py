"""TICK 189 (Logan): how much of heads 0/4 is ACTUALLY symmetric? Fit-based test.

Cosine branch-agreement under-credits components whose top tokens match but whose tails
differ (and cross-mode dictionaries add gauge noise). Proper test: (1) train ONE shared
key dictionary on the stacked branch-1 + branch-2 key rows (2V datapoints), so both key
modes live in the same feature space; (2) fit CP where at every deflation step BOTH a
tied rank-1 (a = b: one key detector used by both branches — the simpler, symmetric
form) and a free rank-1 (a != b) are fit, and the tied one is ACCEPTED whenever its
lambda is within 5% of the free one's. Count tied components and their interaction
mass; estimate the description-length saving (a tied component stores one key factor
instead of two). Gate first: the shared-key SAE must still pass the asymmetric moment
gate (<0.05) — if sharing the key dictionary already breaks fidelity, symmetry claims
are moot. Configs as ticks 187-188: h0 m=2048 k=4, h4 m=1024 k=4; value SAE separate.
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
STEPS, BATCH, LR, R_CP, TIE_EPS = 12000, 2048, 3e-3, 64, 0.05
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


def train_sae(Y, pw_cpu, m_atoms, k_code, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:m_atoms].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    pw = (pw_cpu / pw_cpu.sum()).to(DEV)
    b = (Y * pw[:, None]).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(m_atoms, device=DEV)
    for step in range(STEPS):
        kk = max(k_code, int(round(2 * k_code - k_code * min(1.0, 2 * step / STEPS))))
        bi = torch.multinomial(pw_cpu, BATCH, replacement=True, generator=g).to(DEV)
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
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / vals.norm().clamp_min(1e-30)

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


def hybrid_cp(sp, R, seed, iters=40, n_starts=6):
    """Deflation where each component is tied (a=b) if lambda_tied >= (1-eps)*lambda_free."""
    mm = sp.mm
    gg = torch.Generator().manual_seed(seed)
    A = torch.zeros(mm, 0, device=DEV)
    B = torch.zeros(mm, 0, device=DEV)
    C = torch.zeros(mm, 0, device=DEV)
    lv = torch.zeros(0, device=DEV)
    tied_flags, lams = [], []

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

    def lam_of(a, b, c):
        base = sp.triple(a, b, c)
        if lv.numel():
            base -= float((lv * (A.T @ a) * (B.T @ b) * (C.T @ c)).sum())
        return base

    for r in range(R):
        best_free = None
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
            lam = lam_of(a, b, c)
            if best_free is None or lam > best_free[0]:
                best_free = (lam, a, b, c)
        best_tied = None
        for s in range(n_starts):
            u = torch.rand(mm, generator=gg).to(DEV)
            u = u / u.norm()
            c = torch.rand(mm, generator=gg).to(DEV)
            c = c / c.norm()
            for _ in range(iters):
                u = (0.5 * (res_mode(0, u, c) + res_mode(1, u, c))).clamp_min(0)
                u = u / u.norm().clamp_min(1e-20)
                c = res_mode(2, u, u).clamp_min(0)
                nc = float(c.norm())
                if nc < 1e-20:
                    break
                c = c / nc
            lam = lam_of(u, u, c)
            if best_tied is None or lam > best_tied[0]:
                best_tied = (lam, u, c)
        use_tied = best_tied[0] >= (1 - TIE_EPS) * best_free[0] and best_tied[0] > 0
        if use_tied:
            lam, a, b, c = best_tied[0], best_tied[1], best_tied[1], best_tied[2]
        else:
            lam, a, b, c = best_free
        if lam <= 0:
            break
        tied_flags.append(bool(use_tied))
        lams.append(lam)
        A = torch.cat([A, a[:, None]], 1)
        B = torch.cat([B, b[:, None]], 1)
        C = torch.cat([C, c[:, None]], 1)
        lv = torch.tensor(lams, device=DEV)
    return A, B, C, lv, tied_flags


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
for h, (mm, kc) in ((0, (2048, 4)), (4, (1024, 4))):
    Yk1, Yk2, Yv = K1[:, h], K2[:, h], Vv[:, h]
    Ystack = torch.cat([Yk1, Yk2], 0)
    pw2 = torch.cat([QP, QP]).cpu() / 2
    Dk, bk, Wk = train_sae(Ystack, pw2, mm, kc, seed=0)
    Dv, bv, Wv = train_sae(Yv, QP.cpu().clone(), mm, kc, seed=0)
    i1, c1, r1 = encode(Yk1, Dk, bk, Wk, kc)
    i2, c2, r2 = encode(Yk2, Dk, bk, Wk, kc)
    iv, cv_, rv = encode(Yv, Dv, bv, Wv, kc)
    mres = asym_moment_residual([Yk1, Yk2, Yv], [r1, r2, rv])
    print(f'h{h} shared-key SAE: moment residual {mres:.4f} '
          f'{"PASS" if mres < 0.05 else "FAIL"}', flush=True)
    sp = AsymCore(*build_core([i1, i2, iv], [c1, c2, cv_], mm), mm)
    A, B, C, lv, tied = hybrid_cp(sp, R_CP, 0)
    rel_hybrid = eval_on_core(sp, A, B, C)
    mass = (lv.abs() / lv.abs().sum()).cpu().numpy()
    n_tied = int(sum(tied))
    mass_tied = float(mass[[i for i, t in enumerate(tied) if t]].sum()) if n_tied else 0.0
    bits_saved = n_tied * mm * 32
    out[f'h{h}'] = {'shared_key_mres': round(mres, 4), 'gate_pass': bool(mres < 0.05),
                    'hybrid_relerr_R64': round(rel_hybrid, 4),
                    'n_tied': n_tied, 'n_total': len(tied),
                    'tied_mass_frac': round(mass_tied, 3),
                    'tied_flags_first16': tied[:16],
                    'key_factor_bits_saved_Mbit': round(bits_saved / 1e6, 2)}
    print(f'h{h}: hybrid rel-err {rel_hybrid:.4f} | tied {n_tied}/{len(tied)} components '
          f'({mass_tied*100:.0f}% of mass) | key-factor bits saved {bits_saved/1e6:.2f} Mbit',
          flush=True)
    json.dump(out, open(f'{QK}/qk_sym_fraction.json', 'w'), indent=2)
    del sp
    torch.cuda.empty_cache()
print('SYM FRACTION DONE', flush=True)
