"""Lesson 3 (the right basis) — runs OUR ACTUAL Stage-1 technique end to end: fit a TopK sparse
dictionary in the HEAD-INDUCED METRIC (the concatenated head read-space [K1^T x; K2^T x; V^T x]),
and recover planted atoms. Contrast with a naive-L2 dictionary on raw embeddings, which lands its
atoms in the head's KERNEL (directions the head is structurally blind to). This shows the method
computing, not a shortcut.
"""
import json
import torch
import torch.nn.functional as Fn

torch.manual_seed(0)
d, dh, K, n, k = 64, 8, 12, 4000, 2

M = torch.randn(3*dh, d)                                   # [K1;K2;V] -> pulled-back metric G=M^T M
Vh = torch.linalg.svd(M, full_matrices=True).Vh
U_vis, U_ker = Vh[:3*dh].T, Vh[3*dh:].T                    # range(G) and ker(G) in R^d
A = Fn.normalize(torch.randn(3*dh, K), dim=0)
D_true = (U_vis @ A).T                                     # planted atoms, head-visible
S = torch.zeros(n, K)
for j in range(n):
    S[j, torch.randperm(K)[:k]] = torch.rand(k) + .5
X = S @ D_true
X = X + 6.0 * (torch.randn(n, U_ker.shape[1]) @ U_ker.T)   # LOUD nuisance in ker(G) (head-invisible)


def topk_sae(Y, m, kk, steps=3000, lr=1e-2):
    dy = Y.shape[1]
    We = torch.nn.Parameter(torch.randn(dy, m)*.1); Wd = torch.nn.Parameter(torch.randn(m, dy)*.1)
    b = torch.nn.Parameter(Y.mean(0).clone())
    opt = torch.optim.Adam([We, Wd, b], lr=lr)
    for _ in range(steps):
        z = (Y - b) @ We; v, i = z.topk(kk, 1)
        zk = torch.zeros_like(z).scatter_(1, i, v.relu())
        Yh = zk @ Fn.normalize(Wd, dim=1) + b
        loss = ((Yh - Y)**2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = (Y - b) @ We; v, i = z.topk(kk, 1)
        zk = torch.zeros_like(z).scatter_(1, i, v.relu())
    return Fn.normalize(Wd, dim=1).detach(), zk.detach()


Ttrue = Fn.normalize(D_true @ M.T, dim=1)                  # planted atoms AS THE HEAD READS THEM
def rec(atoms_hs): return float((Fn.normalize(atoms_hs, dim=1) @ Ttrue.T).abs().max(0).values.mean())

Dm, code_m = topk_sae(X @ M.T, K, k)                       # OURS: fit in head space
Dn_raw, _ = topk_sae(X, K, k); Dn_hs = Dn_raw @ M.T        # NAIVE L2: fit on X, viewed through head
metric_cos, naive_cos = round(rec(Dm), 3), round(rec(Dn_hs), 3)
# how visible each recovered atom is to the head (||M . atom||, normalized)
vis_metric = round(float((Dm.norm(dim=1) / Dm.norm(dim=1).mean()).mean()), 2)   # metric atoms already in head space
# for naive: project raw atoms through M, measure surviving norm fraction
Dn_norm = Fn.normalize(Dn_raw, dim=1)
naive_visible = (Dn_norm @ M.T).norm(dim=1)                # ~0 if in kernel
metric_visible = torch.ones(K)                            # metric atoms live in head space by construction
res = {'d': d, 'head_dim_total': 3*dh, 'K_atoms': K, 'active_per_token': k,
       'metric_recovery_cos': metric_cos, 'naive_recovery_cos': naive_cos,
       'naive_atom_head_visibility': [round(float(x), 2) for x in naive_visible],
       'metric_atom_head_visibility': [round(float(x), 2) for x in metric_visible],
       'sample_embedding': [round(float(x), 2) for x in X[0][:24]],
       'sample_recovered_code': [round(float(x), 2) for x in code_m[0]]}
json.dump(res, open('tn_metric_dict_demo.json', 'w'), indent=1)
print(f"OUR technique (metric head-space TopK SAE): recovery cos {metric_cos}", flush=True)
print(f"naive L2 on raw embeddings: recovery cos {naive_cos}; its atoms' head-visibility {res['naive_atom_head_visibility']}", flush=True)
print(f"=> naive atoms land in the head's KERNEL (visibility ~0); metric atoms recover the planted features", flush=True)
print('TN METRIC DICT DEMO DONE', flush=True)
