"""TICK 208 (measure comparison)

Same fit as tick 207 but with the token modes weighted by the FROZEN unigram
convention: rows of Lt/Rt scaled by p_t^(1/2) (the only data ingredient, 307k token
counts). If sparsity and a null gap appear here where the unweighted fit was dense,
the conclusion is: the composed circuit IS sparsely structured, but only under the
measure the model actually operates in.

Original tick-207 header follows.
"""
"""TICK 207 (Logan): a NEW basis, sparse relative to everything — CP decomposition of
the COMPOSED weight tensor embedding -> bilinear MLP -> layer-1 QK read, in token
space, weight-only (no data at all, not even unigram, in the fit).

Object per reader channel H (a layer-1 head's q1 or k1 map):
  G_H[o, s, t] = sum_j A_H[o, j] * (L_j . e_s)(R_j . e_t),   A_H = W_read @ Down
— a (128 x V x V) tensor in FACTORED form (Lt = E L^T, Rt = E R^T, both V x 4608).
No basis is assumed: components are fit jointly as (u_r in R^V left token class,
v_r in R^V right token class, w_r in R^128 output direction), nonneg token modes,
signed output mode, HOPM + deflation entirely in the factored form (all contractions
via V x 4608 matmuls; norm via the triple-Gram identity). This is exactly Logan's
"some embeddings might sparsely interact through the second QK layer through the
bilinear composition": if true, the fitted token classes are SPARSE and the rel-err
at modest rank is low; the basis is learned, not inherited from embedding/attention.

Validation: (i) dense-subset consistency check (512 random tokens: factored-path CP
evaluated against the densely materialized subtensor); (ii) corrected null — refit on
a token-misaligned null (rows of Rt permuted), transplant those factors onto the real
G with lambda refit. Interpretation hooks: top tokens per (u_r, v_r); cosine of w_r
against the layer-1 head's OWN archetype key detectors (does the composed circuit
write onto the archetype axes layer 1 already uses?). Readers: q1_h5, k1_h2 (largest
attention/cross blocks in Section 7f), k1_h1 (subword giant), q1_h7 (determiner reader).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
R_CP, ITERS, STARTS = 32, 40, 5
tok = AutoTokenizer.from_pretrained('gpt2')

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
blk0 = m.transformer.h[0]
mlp = blk0.mlp
Lw = mlp.Left.weight.detach().float().to(DEV)
Rw = mlp.Right.weight.detach().float().to(DEV)
Dw = mlp.Down.weight.detach().float().to(DEV)
NJ = Lw.shape[0]
a1 = m.transformer.h[1].attn
wte = m.transformer.wte.weight.detach().float().to(DEV)
EMB = F.rms_norm(wte, (D,))
FINEWEB_ = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
QPw = (torch.bincount(FINEWEB_.flatten(), minlength=V).float() + 0.5).to(DEV)
QPw = (QPw / QPw.sum()).sqrt()[:, None]
Lt = QPw * (EMB @ Lw.T)                                      # p^(1/2)-weighted rows
Rt = QPw * (EMB @ Rw.T)
GLL = Lt.T @ Lt                                              # (4608, 4608) Grams
GRR = Rt.T @ Rt
l1_23 = torch.load(f'{QK}/qk_l1_stage23.pt', map_location='cpu')
l1_s1 = torch.load(f'{QK}/qk_l1_stage1.pt', map_location='cpu')

READERS = {'q1_h5': (a1.c_q, 5), 'k1_h2': (a1.c_k, 2), 'k1_h1': (a1.c_k, 1),
           'q1_h7': (a1.c_q, 7)}
out = {}


def fit_channel(A, seed=0, R=R_CP):
    """HOPM + deflation on factored G. Returns U (V,R), Vf (V,R), W (128,R), lam."""
    gg = torch.Generator().manual_seed(seed)
    GAA = A.T @ A                                            # (4608, 4608)
    norm2 = float((GAA * GLL * GRR).sum())
    Us, Vs, Ws, lams = [], [], [], []

    def contract_lam(u, v, w):
        base = float(w @ (A @ ((Lt.T @ u) * (Rt.T @ v))))
        for uu, vv, ww, ll in zip(Us, Vs, Ws, lams):
            base -= ll * float((uu @ u) * (vv @ v) * (ww @ w))
        return base

    for r in range(R):
        best = None
        for s in range(STARTS):
            u = torch.rand(V, generator=gg).to(DEV)
            u = u / u.norm()
            v = torch.rand(V, generator=gg).to(DEV)
            v = v / v.norm()
            w = torch.randn(128, generator=gg).to(DEV)
            w = w / w.norm()
            for _ in range(ITERS):
                aw = A.T @ w
                u_new = Lt @ (aw * (Rt.T @ v))
                for uu, vv, ww, ll in zip(Us, Vs, Ws, lams):
                    u_new -= ll * uu * float((vv @ v) * (ww @ w))
                u = u_new.clamp_min(0)
                n = float(u.norm())
                if n < 1e-20:
                    break
                u = u / n
                v_new = Rt @ (aw * (Lt.T @ u))
                for uu, vv, ww, ll in zip(Us, Vs, Ws, lams):
                    v_new -= ll * vv * float((uu @ u) * (ww @ w))
                v = v_new.clamp_min(0)
                n = float(v.norm())
                if n < 1e-20:
                    break
                v = v / n
                w_new = A @ ((Lt.T @ u) * (Rt.T @ v))
                for uu, vv, ww, ll in zip(Us, Vs, Ws, lams):
                    w_new -= ll * ww * float((uu @ u) * (vv @ v))
                n = float(w_new.norm())
                if n < 1e-20:
                    break
                w = w_new / n
            lam = contract_lam(u, v, w)
            if best is None or abs(lam) > abs(best[0]):
                best = (lam, u, v, w)
        if best is None or abs(best[0]) < 1e-12:
            break
        lams.append(best[0])
        Us.append(best[1])
        Vs.append(best[2])
        Ws.append(best[3])
    U = torch.stack(Us, 1)
    Vf = torch.stack(Vs, 1)
    W = torch.stack(Ws, 1)
    lv = torch.tensor(lams, device=DEV)
    # rel-err via Gram identity
    inner = torch.tensor([float(W[:, r] @ (A @ ((Lt.T @ U[:, r]) * (Rt.T @ Vf[:, r]))))
                          for r in range(U.shape[1])], device=DEV)
    Gm = (U.T @ U) * (Vf.T @ Vf) * (W.T @ W)
    res2 = norm2 - 2 * float(lv @ inner) + float(lv @ Gm @ lv)
    rel = (max(res2, 0.0) / max(norm2, 1e-30)) ** 0.5
    return U, Vf, W, lv, rel, norm2


def lam_refit_rel(A, U, Vf, W, norm2):
    R = U.shape[1]
    inner = torch.tensor([float(W[:, r] @ (A @ ((Lt.T @ U[:, r]) * (Rt.T @ Vf[:, r]))))
                          for r in range(R)], device=DEV)
    Gm = (U.T @ U) * (Vf.T @ Vf) * (W.T @ W)
    lam = torch.linalg.solve(Gm + 1e-8 * torch.eye(R, device=DEV), inner)
    res2 = norm2 - 2 * float(lam @ inner) + float(lam @ Gm @ lam)
    return (max(res2, 0.0) / max(norm2, 1e-30)) ** 0.5


def sparsity(u, k=16):
    e = u ** 2
    return float(e.sort(descending=True).values[:k].sum() / e.sum())


for rname, (lin, h) in READERS.items():
    Wr = lin.weight.detach().float()[h * HD:(h + 1) * HD].to(DEV)
    A = Wr @ Dw
    U, Vf, W, lv, rel, norm2 = fit_channel(A, seed=0)
    # dense-subset consistency check
    gsub = torch.Generator().manual_seed(3)
    sub = torch.randperm(V, generator=gsub)[:512].to(DEV)
    Gd = torch.einsum('oj,sj,tj->ost', A, Lt[sub], Rt[sub])
    Gfit = torch.einsum('or,sr,tr,r->ost', W, U[sub], Vf[sub], lv)
    sub_rel = float((Gd - Gfit).norm() / Gd.norm())
    # corrected null: permute Rt token alignment, fit, transplant
    gp = torch.Generator().manual_seed(7)
    perm = torch.randperm(V, generator=gp).to(DEV)
    Rt_save = Rt.clone()
    globals()['Rt'] = Rt[perm]
    globals()['GRR'] = Rt.T @ Rt
    Un, Vn, Wn, _, _, _ = fit_channel(A, seed=0)
    globals()['Rt'] = Rt_save
    globals()['GRR'] = Rt.T @ Rt
    null_rel = lam_refit_rel(A, Un, Vn[perm.argsort()], Wn, norm2)
    # interpretation
    comps = []
    Dn1 = l1_s1[f'h{h}_Dn'].float()
    U1 = l1_23[f'h{h}_U'].float()
    arch_keys = (Dn1[:, :HD].T @ U1).to(DEV)                 # (128, R1) l1 archetype g1
    arch_keys = arch_keys / arch_keys.norm(dim=0, keepdim=True).clamp_min(1e-9)
    for r in lv.abs().argsort(descending=True)[:5].tolist():
        cu = (W[:, r] / W[:, r].norm().clamp_min(1e-9)) @ arch_keys
        comps.append({'lam': round(float(lv[r]), 4),
                      'left': [tok.decode([t]).replace('\n', '\\n')
                               for t in U[:, r].argsort(descending=True)[:6].tolist()],
                      'right': [tok.decode([t]).replace('\n', '\\n')
                                for t in Vf[:, r].argsort(descending=True)[:6].tolist()],
                      'left_top16_mass': round(sparsity(U[:, r]), 3),
                      'right_top16_mass': round(sparsity(Vf[:, r]), 3),
                      'max_cos_l1_archetype': round(float(cu.abs().max()), 3)})
    out[rname] = {'rel_R32': round(rel, 4), 'dense_subset_check': round(sub_rel, 4),
                  'null_on_real': round(null_rel, 4), 'top_components': comps}
    print(f'{rname}: rel {rel:.4f} | subset-check {sub_rel:.4f} | null-on-real '
          f'{null_rel:.4f} | c0 left {comps[0]["left"][:3]} right '
          f'{comps[0]["right"][:3]} (top16 {comps[0]["left_top16_mass"]}/'
          f'{comps[0]["right_top16_mass"]}, arch-cos {comps[0]["max_cos_l1_archetype"]})',
          flush=True)
    json.dump(out, open(f'{QK}/qk_composed_cp_uw.json', 'w'), indent=2)
    del A, U, Vf, W, Un, Vn, Wn, Gd, Gfit
    torch.cuda.empty_cache()
print('COMPOSED CP DONE', flush=True)
