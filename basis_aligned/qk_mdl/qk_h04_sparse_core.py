"""TICK 180 (Logan un-gated the mechanism path for h0/h4): full sparse-core pipeline for
heads 0 and 4 at m=4096, k=8 — the capacity where tick-176 extrapolation (residual halves
per doubling: 0.173 -> 0.097 -> 0.055) predicts the moment gate opens.

Dense Stage 2 is infeasible at m=4096 (m^3 fp32 = 275 GB), so the core is held in
coalesced COO form: V tokens x k^3=512 ordered slot-triples = 25.8M raw entries.
Stage 3 CP (power iteration + deflation, the planted-validated fitter) is re-implemented
against the sparse core; deflation terms stay low-rank ((u_r . u)^2 corrections), and
prefix rel-errors at R in {16,32,64} come from the exact Gram identity
  |res|^2 = |M|^2 - 2 sum_r lam_r <M, u_r^x3> + sum_rs lam_r lam_s (u_r . u_s)^3.
Greedy deflation makes the rank-16/32 fits exact prefixes of the rank-64 run (generator
state is consumed per component), so one run per seed covers all ranks.

ORDER OF OPERATIONS (positive-control rule): the sparse fitter first re-passes the
planted known-answer test (24 sparse archetypes at m=4096, gate matched-cos >= 0.95)
before any real-data fitting. Then per head: Stage-1 triple SAE (m=4096, k=8, unigram +
nonneg, annealed 2k->k), sketched-moment gate (<0.05), sparse core + diagonal-mass split,
CP with 3 seeds (stability), column-permutation null at R=32, archetype top-token dumps.
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
M_ATOMS, K_CODE, STEPS, BATCH, LR = 4096, 8, 12000, 2048, 3e-3
GATE = 0.05
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


def train_triple(Y, pw, seed=0, nonneg=True):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Y[torch.randperm(len(Y), generator=g)[:M_ATOMS].to(DEV)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (Y * pw[:, None]).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(M_ATOMS, device=DEV)
    pw_cpu = pw.cpu()
    for step in range(STEPS):
        kk = max(K_CODE, int(round(2 * K_CODE - K_CODE * min(1.0, 2 * step / STEPS))))
        bi = torch.multinomial(pw_cpu, BATCH, replacement=True, generator=g).to(DEV)
        y = Y[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (y - b) @ We.T
        if nonneg:
            z = torch.relu(z)
        vals, idx = z.abs().topk(kk, dim=1)
        coeff = torch.gather(z, 1, idx)
        yhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), (coeff.abs() > 1e-8).float().reshape(-1))
        loss = ((yhat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    z_ = (Y - b) @ We.T
                    if nonneg:
                        z_ = torch.relu(z_)
                    v_, i_ = z_.abs().topk(K_CODE, dim=1)
                    rec = b + (torch.gather(z_, 1, i_).unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - Y) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = Y[worst] / Y[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
                    del z_, rec
            fired.zero_()
        if (step + 1) % 2000 == 0:
            print(f'  step {step + 1} loss {float(loss):.5f} kk {kk}', flush=True)
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (Y - b) @ We.T
        if nonneg:
            z = torch.relu(z)
        vals, idx = z.abs().topk(K_CODE, dim=1)
        coeff = torch.gather(z, 1, idx)
        rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    return Dn.detach(), b.detach(), We.detach(), idx, coeff.detach(), rec.detach()


@torch.no_grad()
def moment_residual(Y, rec, pw, n_probe=256, seed=3):
    g = torch.Generator(device='cpu').manual_seed(seed)
    num = den = 0.0
    for _ in range(n_probe):
        u, v_, wv = (torch.randn(Y.shape[1], generator=g).to(DEV) for _ in range(3))
        t = (pw * (Y @ u) * (Y @ v_) * (Y @ wv)).sum()
        th = (pw * (rec @ u) * (rec @ v_) * (rec @ wv)).sum()
        num += float((t - th) ** 2)
        den += float(t ** 2)
    return num / max(den, 1e-30)


# ---------------- sparse third-moment core ----------------

@torch.no_grad()
def build_sparse_core(idx, coeff, mm, pw, chunk=4096):
    """Coalesced COO core M_abc = sum_t p_t s_ta s_tb s_tc from k-sparse codes."""
    k = idx.shape[1]
    keys_all, vals_all = [], []
    for s in range(0, idx.shape[0], chunk):
        ii = idx[s:s + chunk].long()
        cc = coeff[s:s + chunk]
        w = pw[s:s + chunk, None] * cc
        a = ii[:, :, None, None].expand(-1, k, k, k)
        b_ = ii[:, None, :, None].expand(-1, k, k, k)
        c_ = ii[:, None, None, :].expand(-1, k, k, k)
        v = w[:, :, None, None] * cc[:, None, :, None] * cc[:, None, None, :]
        keys_all.append(((a * mm + b_) * mm + c_).reshape(-1))
        vals_all.append(v.reshape(-1))
    keys = torch.cat(keys_all)
    vals = torch.cat(vals_all)
    del keys_all, vals_all
    uk, inv = torch.unique(keys, return_inverse=True)
    cv = torch.zeros(len(uk), device=DEV)
    cv.scatter_add_(0, inv, vals)
    ai = torch.div(uk, mm * mm, rounding_mode='floor')
    bi = torch.div(uk, mm, rounding_mode='floor') % mm
    ci = uk % mm
    return ai, bi, ci, cv


class SparseCore:
    def __init__(self, ai, bi, ci, vals, mm):
        self.mm = mm
        nrm = vals.norm().clamp_min(1e-30)
        self.ai, self.bi, self.ci = ai, bi, ci
        self.vals = vals / nrm
        self.diag_mass_frac = float((self.vals[(ai == bi) & (bi == ci)] ** 2).sum()
                                    / (self.vals ** 2).sum())

    def matvec(self, u, Um=None, lv=None):
        """(M - sum_r lam_r u_r^x3) contracted on modes 2,3 with u."""
        out = torch.zeros(self.mm, device=DEV)
        out.scatter_add_(0, self.ai, self.vals * u[self.bi] * u[self.ci])
        if Um is not None and Um.shape[1]:
            d = Um.T @ u
            out -= Um @ (lv * d ** 2)
        return out

    def triple(self, u):
        return float((self.vals * u[self.ai] * u[self.bi] * u[self.ci]).sum())


def cp_fit_sparse(sp, R, seed, n_starts=8, iters=60):
    """Symmetric nonneg CP by power iteration + deflation on the sparse core.
    Returns U (m, R'), lam, and prefix rel-errors via the Gram identity (|M|=1)."""
    mm = sp.mm
    gg = torch.Generator().manual_seed(seed)
    Us, lams = [], []
    Um = torch.zeros(mm, 0, device=DEV)
    lv = torch.zeros(0, device=DEV)
    for r in range(R):
        best_u, best_lam = None, -1.0
        for s in range(n_starts):
            u = torch.rand(mm, generator=gg).to(DEV)
            u = u / u.norm()
            for _ in range(iters):
                u = sp.matvec(u, Um, lv).clamp_min(0)
                n = float(u.norm())
                if n < 1e-20:
                    break
                u = u / n
            lam = sp.triple(u) - float(((Um.T @ u) ** 3 * lv).sum())
            if lam > best_lam:
                best_lam, best_u = lam, u
        if best_lam <= 0:
            break
        Us.append(best_u)
        lams.append(best_lam)
        Um = torch.stack(Us, 1)
        lv = torch.tensor(lams, device=DEV)
    inner = torch.tensor([sp.triple(u) for u in Us], device=DEV)   # <M, u_r^x3>
    G = (Um.T @ Um) ** 3
    prefix_rel = {}
    for P in (16, 32, 64):
        p = min(P, len(lams))
        lp = lv[:p]
        res2 = 1.0 - 2.0 * float(lp @ inner[:p]) + float(lp @ G[:p, :p] @ lp)
        prefix_rel[P] = max(res2, 0.0) ** 0.5
    return Um, lv, prefix_rel


def stability(Us, P):
    vals = []
    for i in range(len(Us)):
        for j in range(i + 1, len(Us)):
            C = (Us[i][:, :P].T @ Us[j][:, :P]).abs()
            vals.append(float(C.max(1).values.mean()))
    return sum(vals) / len(vals)


# ---------------- planted known-answer test for the SPARSE fitter ----------------
print('=== planted test: sparse CP fitter at m=4096 ===', flush=True)
g = torch.Generator().manual_seed(5)
R0 = 24
MU = torch.zeros(M_ATOMS, R0, device=DEV)
for r in range(R0):
    sup = torch.randperm(M_ATOMS, generator=g)[:6]
    MU[sup.to(DEV), r] = (0.3 + torch.rand(6, generator=g)).to(DEV)
MU = MU / MU.norm(dim=0, keepdim=True)
LAM0 = (10 ** (2 * torch.rand(R0, generator=g))).to(DEV)
pk_keys, pk_vals = [], []
for r in range(R0):
    act = MU[:, r].nonzero().squeeze(1)
    w = MU[act, r]
    a = act[:, None, None].expand(6, 6, 6)
    b_ = act[None, :, None].expand(6, 6, 6)
    c_ = act[None, None, :].expand(6, 6, 6)
    v = LAM0[r] * w[:, None, None] * w[None, :, None] * w[None, None, :]
    pk_keys.append(((a * M_ATOMS + b_) * M_ATOMS + c_).reshape(-1))
    pk_vals.append(v.reshape(-1))
uk, inv = torch.unique(torch.cat(pk_keys), return_inverse=True)
cv = torch.zeros(len(uk), device=DEV)
cv.scatter_add_(0, inv, torch.cat(pk_vals))
sp_pl = SparseCore(torch.div(uk, M_ATOMS * M_ATOMS, rounding_mode='floor'),
                   torch.div(uk, M_ATOMS, rounding_mode='floor') % M_ATOMS,
                   uk % M_ATOMS, cv, M_ATOMS)
Upl, _, prel = cp_fit_sparse(sp_pl, R0, 0)
mcos = float((Upl.T @ MU).abs().max(0).values.mean())
print(f'planted: matched-cos {mcos:.4f} rel-err@24 {prel[16]:.4f}/{prel[32]:.4f} '
      f'-> {"PASS" if mcos >= 0.95 else "FAIL"}', flush=True)
assert mcos >= 0.95, 'sparse CP fitter failed the planted gate; not touching real data'

# ---------------- real heads 0 and 4 ----------------
results = {'planted_matched_cos': round(mcos, 4)}
blob = {}
for h in (0, 4):
    print(f'=== head {h}: Stage 1 (m={M_ATOMS}, k={K_CODE}) ===', flush=True)
    Y = torch.cat([K1[:, h], K2[:, h], Vv[:, h]], 1)
    Dn, b, We, idx, coeff, rec = train_triple(Y, QP, seed=0, nonneg=True)
    mres = moment_residual(Y, rec, QP)
    row = {'moment_rel_err': round(mres, 4), 'gate_pass': bool(mres < GATE)}
    print(f'h{h} m=4096 k=8: moment-rel-err {mres:.4f} '
          f'{"PASS" if mres < GATE else "FAIL"}', flush=True)
    blob[f'h{h}_Dn'] = Dn.cpu()
    blob[f'h{h}_b'] = b.cpu()
    blob[f'h{h}_idx'] = idx.to(torch.int16).cpu()
    blob[f'h{h}_coeff'] = coeff.cpu()
    del We, rec
    torch.cuda.empty_cache()

    print(f'=== head {h}: Stage 2 sparse core ===', flush=True)
    sp = SparseCore(*build_sparse_core(idx, coeff, M_ATOMS, QP), M_ATOMS)
    row['core_nnz'] = int(len(sp.vals))
    row['diag_mass_frac'] = round(sp.diag_mass_frac, 4)
    print(f'h{h}: core nnz {row["core_nnz"]} diag-mass {sp.diag_mass_frac:.4f}', flush=True)

    print(f'=== head {h}: Stage 3 CP (3 seeds, prefix ranks 16/32/64) ===', flush=True)
    Us, prels = [], []
    for seed in range(3):
        Um, lv, prel_ = cp_fit_sparse(sp, 64, seed)
        Us.append(Um)
        prels.append(prel_)
        print(f'  seed {seed}: rel-err 16/32/64 = '
              f'{prel_[16]:.4f}/{prel_[32]:.4f}/{prel_[64]:.4f}', flush=True)
    for P in (16, 32, 64):
        row[f'R{P}_relerr'] = round(min(p[P] for p in prels), 4)
        row[f'R{P}_stability'] = round(stability(Us, P), 3)
    best = int(np.argmin([p[64] for p in prels]))
    Um, lv, _ = cp_fit_sparse(sp, 64, best)
    blob[f'h{h}_U'] = Um.cpu()
    blob[f'h{h}_lam'] = lv.cpu()

    S_dense = torch.zeros(V, M_ATOMS, device=DEV)
    S_dense.scatter_(1, idx, coeff)
    arch = []
    for r in lv[:32].argsort(descending=True)[:5].tolist():
        load = S_dense @ Um[:, r]
        top = load.argsort(descending=True)[:8]
        arch.append([tok.decode([t]).replace('\n', '\\n') for t in top.tolist()])
    row['top_archetype_tokens'] = arch

    print(f'=== head {h}: permutation null (R=32) ===', flush=True)
    gp = torch.Generator().manual_seed(7)
    for f in range(M_ATOMS):
        S_dense[:, f] = S_dense[torch.randperm(V, generator=gp).to(DEV), f]
    vals_n, idx_n = S_dense.topk(12, dim=1)
    del S_dense
    torch.cuda.empty_cache()
    sp_n = SparseCore(*build_sparse_core(idx_n, vals_n, M_ATOMS, QP, chunk=2048), M_ATOMS)
    _, _, prel_n = cp_fit_sparse(sp_n, 32, 0)
    row['R32_relerr_null'] = round(prel_n[32], 4)
    print(f'h{h}: null R32 rel-err {prel_n[32]:.4f} (real {row["R32_relerr"]})', flush=True)
    del sp, sp_n
    torch.cuda.empty_cache()

    results[f'h{h}'] = row
    json.dump(results, open(f'{QK}/qk_h04_sparse_core.json', 'w'), indent=2)
    torch.save(blob, f'{QK}/qk_h04_sparse_core.pt')
print('H04 SPARSE CORE DONE', flush=True)
