"""Lesson 1 (the exact fold) demonstration data. A bilinear layer's separate weight matrices
ARE one interaction tensor — folded exactly, no information lost, no gauge added.
(1) Tiny 2D unit: out=(l.x)(r.x) = x^T (l r^T) x. Show the 2x2 matrix and verify.
(2) The modular-addition toy: its weight matrices (Wl,Wr,Wout) fold to ONE interaction matrix
    per output class, M_c = sum_k Wout[c,k] wl_k wr_k^T, and logit_c = e_a^T M_c e_b EXACTLY.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# (1) tiny 2D unit
l = np.array([1.0, -0.5]); r = np.array([0.4, 1.0]); M2 = np.outer(l, r)
xs = np.random.default_rng(0).standard_normal((6, 2))
direct = (xs @ l) * (xs @ r); folded = np.einsum('ni,ij,nj->n', xs, M2, xs)
tiny = {'l': l.round(2).tolist(), 'r': r.round(2).tolist(), 'M': M2.round(2).tolist(),
        'max_abs_err': round(float(np.abs(direct - folded).max()), 8)}

# (2) modular-addition toy (retrain quickly, then fold)
P, d, m = 23, 24, 48
E = nn.Embedding(P, d).to(DEV); Wl = nn.Linear(d, m, bias=False).to(DEV)
Wr = nn.Linear(d, m, bias=False).to(DEV); Wout = nn.Linear(m, P).to(DEV)
opt = torch.optim.Adam(list(E.parameters())+list(Wl.parameters())+list(Wr.parameters())+list(Wout.parameters()), lr=3e-3, weight_decay=1e-3)
aa, bb = torch.meshgrid(torch.arange(P), torch.arange(P), indexing='ij')
A, B = aa.reshape(-1).to(DEV), bb.reshape(-1).to(DEV); Y = (A+B) % P
def fwd(a, b): return Wout((E(a) @ Wl.weight.T) * (E(b) @ Wr.weight.T))
for step in range(6000):
    loss = F.cross_entropy(fwd(A, B), Y); opt.zero_grad(); loss.backward(); opt.step()
acc = float((fwd(A, B).argmax(1) == Y).float().mean())
with torch.no_grad():
    Em = E.weight.detach(); WlM = Wl.weight.detach(); WrM = Wr.weight.detach(); Wo = Wout.weight.detach(); bo = Wout.bias.detach()
    # fold: M_c = sum_k Wo[c,k] wl_k wr_k^T   (d x d) per class
    Mfold = torch.einsum('ck,ki,kj->cij', Wo, WlM, WrM)             # (P,d,d)
    # verify logit_c = e_a^T M_c e_b + bias  == model logit
    logit_folded = torch.einsum('ni,cij,nj->nc', Em[A], Mfold, Em[B]) + bo
    logit_model = fwd(A, B)
    fold_err = float((logit_folded - logit_model).norm() / logit_model.norm())
    # render one class's folded interaction matrix (in the Fourier-sorted embedding basis for readability)
    c_show = 0
    Mc = Mfold[c_show].cpu().numpy()
    # also the effective rank of the folded tensor stacked (how low-rank the whole map is)
    Mflat = Mfold.reshape(P, -1).cpu().numpy()
    sv = np.linalg.svd(Mflat, compute_uv=False)
    pr = sv/sv.sum(); efr = float(np.exp(-(pr*np.log(pr+1e-12)).sum()))
    # fp64 fold error (algebraic-identity receipt): recompute in double precision
    Em64 = Em.double(); Wo64 = Wo.double(); WlM64 = WlM.double(); WrM64 = WrM.double(); bo64 = bo.double()
    Mfold64 = torch.einsum('ck,ki,kj->cij', Wo64, WlM64, WrM64)
    lf64 = torch.einsum('ni,cij,nj->nc', Em64[A], Mfold64, Em64[B]) + bo64
    lm64 = ((Em64[A] @ WlM64.T) * (Em64[B] @ WrM64.T)) @ Wo64.T + bo64
    fold_err64 = float((lf64 - lm64).norm() / lm64.norm())
    # symmetrization: only the (i,j)-symmetric part is observable (both legs get an embedding)
    Msym = 0.5 * (Mfold + Mfold.transpose(1, 2))
    lsym = torch.einsum('ni,cij,nj->nc', Em[A], Msym, Em[B]) + bo
    sym_maxdiff = float((lsym - logit_model).abs().max())
    antisym_frac_mod = float((Mfold - Msym).norm() / Mfold.norm())
    # all-inputs verification scatter (folded vs layer), a few hundred points
    idx = torch.randperm(len(A))[:400]
    scatter = [[round(float(logit_model[i, Y[i]]), 3), round(float(logit_folded[i, Y[i]]), 3)] for i in idx.tolist()]
mod = {'task': 'a+b mod 23', 'accuracy': round(acc, 3), 'weight_matrices': ['Wl (24x48)', 'Wr (24x48)', 'Wout (48x23)'],
       'folded_to': '23 interaction matrices M_c (24x24), logit_c = e_a^T M_c e_b',
       'fold_relative_error_fp32': fold_err, 'fold_relative_error_fp64': fold_err64,
       'symmetrized_output_maxdiff': sym_maxdiff, 'antisym_fraction': round(antisym_frac_mod, 3),
       'Mc_class0': [[round(float(v), 2) for v in row] for row in Mc.tolist()],
       'Mc_slices': [[[round(float(v), 2) for v in row] for row in Mfold[c].cpu().numpy().tolist()] for c in [0, 1, 5]],
       'verify_scatter': scatter, 'folded_effective_rank': round(efr, 2), 'n_classes': P}
json.dump({'tiny': tiny, 'modular': mod}, open('tn_fold_demo.json', 'w'), indent=1)
print(f"tiny 2D fold err {tiny['max_abs_err']:.2e}", flush=True)
print(f"modular toy acc {acc:.3f}, fold rel-err {fold_err:.2e} (EXACT), folded eff-rank {efr:.2f} of {P}", flush=True)
print('TN FOLD DEMO DONE', flush=True)
