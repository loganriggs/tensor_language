"""e8b: weights-only gradient fits of tensor networks to E (Logan's question:
"did you optimize a hierarchical Tucker to match the matrix directly?").

No data, no model — pure tensor approximation of E under relative Frobenius
error (= 1 - cos^2 with optimal scale: the tensor generalization of cosine
similarity, accounting for scale). Also extends the TT-SVD rank sweep upward
("how do you make FVU smaller: increase rmax").

Arms (all on the padded, centered, permuted tensor; FVU on real rows):
  tt_svd     rmax in {384, 512, 768, 1024}, semantic + random orderings
  tt_grad    chain TT, cores initialized from TT-SVD(256), Adam on FVU
  ht_grad    BALANCED-tree HT: leaves A1..A4 (16 x r), pair cores B12,B34
             (r,r,r2), root C (r2,r2,1024). r=16 (exact leaves), r2=48
             -> 2.39M params, matched to TT rmax=256 (2.42M).
  svd ref    matched-param SVD for context (from e6: r=50, 2.57M, fvu 0.836).

CORRECTION found while preparing this: FINDING 10 said the rmax=256 TT is
"2.4% budget" — it is 2.43M/51.5M = 4.7%. At matched params plain SVD (0.836)
slightly beats semantic TT-SVD (0.848). Fixed in RESULTS.md.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6
from e8_tensor_train import (E_C, VPAD, DIGITS, B, make_orderings, tt_svd,
                             fvu_real)

DEV = 'cuda'
BASE = '/workspace/tensor_language/basis_aligned'
V, D_MODEL = e6.E.shape
torch.manual_seed(0)

orderings = make_orderings()
torch.save({k: v.cpu() for k, v in orderings.items()}, f'{BASE}/e8_orderings.pt')
results = {'rows': []}


def report(tag, ordering, params, Ehat_perm, perm, extra=None):
    fvu, _ = fvu_real(Ehat_perm, perm)
    row = {'method': tag, 'ordering': ordering, 'params': params, 'fvu': fvu}
    row.update(extra or {})
    results['rows'].append(row)
    print(f"{tag:10s} {ordering:9s} params {params / 1e6:6.2f}M "
          f"({params / (V * D_MODEL):5.1%} of E)  fvu {fvu:.4f}", flush=True)
    return fvu


# ---- 1. TT-SVD rank sweep upward
for name in ['semantic', 'random']:
    perm = orderings[name]
    for rmax in [384, 512, 768, 1024]:
        cores, params, R = tt_svd(E_C[perm], rmax)
        report('tt_svd', name, params, R, perm, {'rmax': rmax})
        del cores, R
        torch.cuda.empty_cache()


# ---- 2. gradient chain-TT from TT-SVD(256) init
def tt_recon(cores):
    R = cores[0].reshape(-1, cores[0].shape[-1])
    for c in cores[1:DIGITS]:
        R = torch.einsum('ab,bcd->acd', R, c).reshape(R.shape[0] * c.shape[1], -1)
    return R @ cores[DIGITS]


for name in ['semantic', 'random']:
    perm = orderings[name]
    T = E_C[perm]
    denom = (T ** 2).sum()
    cores, params, R = tt_svd(T, 256)
    fvu0 = float(((R - T) ** 2).sum() / denom)
    del R
    cores = [c.clone().requires_grad_(True) for c in cores]
    opt = torch.optim.Adam(cores, lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
    for step in range(3000):
        loss = ((tt_recon(cores) - T) ** 2).sum() / denom
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            print(f'  tt_grad {name} step {step:5d} padded-fvu {loss.item():.4f}',
                  flush=True)
    with torch.no_grad():
        report('tt_grad', name, params, tt_recon(cores), perm,
               {'rmax': 256, 'padded_fvu_init': fvu0})
    del cores
    torch.cuda.empty_cache()


# ---- 3. gradient balanced-tree HT (r=16 exact leaves, r2=48 ~ matched params)
def ht_recon(A, B12, B34, C):
    X12 = torch.einsum('ar,bs,rst->abt', A[0], A[1], B12).reshape(B * B, -1)
    X34 = torch.einsum('ar,bs,rst->abt', A[2], A[3], B34).reshape(B * B, -1)
    return torch.einsum('at,bu,tuD->abD', X12, X34, C).reshape(VPAD, D_MODEL)


R_LEAF, R2 = 16, 48
for name in ['semantic', 'random']:
    perm = orderings[name]
    T = E_C[perm]
    denom = (T ** 2).sum()
    g = torch.Generator(device='cpu'); g.manual_seed(0)

    def rnd(*s, scale):
        return (torch.randn(*s, generator=g) * scale).to(DEV).requires_grad_(True)

    A = [rnd(B, R_LEAF, scale=0.3) for _ in range(4)]
    B12 = rnd(R_LEAF, R_LEAF, R2, scale=0.1)
    B34 = rnd(R_LEAF, R_LEAF, R2, scale=0.1)
    C = rnd(R2, R2, D_MODEL, scale=0.1)
    params = sum(t.numel() for t in A + [B12, B34, C])
    opt = torch.optim.Adam(A + [B12, B34, C], lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000)
    for step in range(4000):
        loss = ((ht_recon(A, B12, B34, C) - T) ** 2).sum() / denom
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            print(f'  ht_grad {name} step {step:5d} padded-fvu {loss.item():.4f}',
                  flush=True)
    with torch.no_grad():
        report('ht_grad', name, params, ht_recon(A, B12, B34, C), perm,
               {'r_leaf': R_LEAF, 'r2': R2})
    torch.cuda.empty_cache()

with open(f'{BASE}/e8b_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print('e8b done')
