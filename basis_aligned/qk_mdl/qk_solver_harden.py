"""SOLVER HARDENING (tick 171): make the dictionary pass the planted gate (spec 8A check 1).
Tick 170: current signed-topk trainer recovers 0.884 mean|cos| on the CORRELATED planted
problem (0.964 uncorrelated control) vs a 0.99 gate — correlated features absorb into
mixed atoms. Variants tested here (each on both DGPs, 5 seeds each, selection by
reconstruction only — never by ground truth):
  base       current recipe (reference)
  nonneg     ReLU codes before top-k (spec recommendation; planted coefficients positive)
  anneal     k annealed 6 -> 3 over the first half of training
  nn+anneal  both
Report per variant: mean/max MCC over seeds, MCC of the best-by-reconstruction seed,
fraction of atoms above 0.9. Gate re-check on the winner.
"""
import sys
import json
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')

torch.manual_seed(0)
DEV = 'cuda'
n, d, m0, gsize = 8192, 64, 48, 16
STEPS, BATCH, LR = 12000, 2048, 3e-3


def make_dgp(correlated, seed=1):
    g = torch.Generator().manual_seed(seed)
    D_true = torch.linalg.qr(torch.randn(d, d, generator=g))[0][:m0].to(DEV)
    Z = torch.zeros(n, m0, device=DEV)
    color = torch.randint(0, gsize, (n,), generator=g)
    if correlated:
        block = color // 4
        shape = torch.where(torch.rand(n, generator=g) < 0.7,
                            block * 4 + torch.randint(0, 4, (n,), generator=g),
                            torch.randint(0, gsize, (n,), generator=g))
    else:
        shape = torch.randint(0, gsize, (n,), generator=g)
    texture = torch.randint(0, gsize, (n,), generator=g)
    for gi, pick in enumerate((color, shape, texture)):
        Z[torch.arange(n), gi * gsize + pick.to(DEV)] = (0.5 + torch.rand(n, generator=g)).to(DEV)
    return Z @ D_true, D_true


def train(X, k, seed, nonneg=False, anneal=False):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:m0].to(X.device)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=LR)
    fired = torch.zeros(m0, device=X.device)
    for step in range(STEPS):
        kk = k
        if anneal:
            kk = max(k, int(round(6 - 3 * min(1.0, 2 * step / STEPS))))
        x = X[torch.randint(0, len(X), (BATCH,), device=X.device)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        if nonneg:
            z = torch.relu(z)
        vals, idx = z.abs().topk(kk, dim=1)
        coeff = torch.gather(z, 1, idx)
        xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), (coeff.abs() > 1e-8).float().reshape(-1))
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    z_ = (X - b) @ We.T
                    if nonneg:
                        z_ = torch.relu(z_)
                    v_, i_ = z_.abs().topk(k, dim=1)
                    rec = b + (torch.gather(z_, 1, i_).unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - X) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = X[worst] / X[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
            fired.zero_()
    with torch.no_grad():
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (X - b) @ We.T
        if nonneg:
            z = torch.relu(z)
        v_, i_ = z.abs().topk(k, dim=1)
        rec = b + (torch.gather(z, 1, i_).unsqueeze(-1) * Dn[i_]).sum(1)
        mse = float(((rec - X) ** 2).sum() / (X - X.mean(0)).pow(2).sum())
    return Dn.detach(), mse


VARIANTS = {'base': {}, 'nonneg': {'nonneg': True}, 'anneal': {'anneal': True},
            'nn+anneal': {'nonneg': True, 'anneal': True}}
out = {}
for corr in (True, False):
    E, D_true = make_dgp(corr)
    tag = 'correlated' if corr else 'independent'
    out[tag] = {}
    for vname, kw in VARIANTS.items():
        mccs, mses = [], []
        for seed in range(5):
            Dn, mse = train(E, 3, seed, **kw)
            mccs.append(float((Dn @ D_true.T).abs().max(1).values.mean()))
            mses.append(mse)
        best = int(torch.tensor(mses).argmin())
        row = {'mcc_mean': round(sum(mccs) / 5, 4), 'mcc_max': round(max(mccs), 4),
               'mcc_best_by_recon': round(mccs[best], 4),
               'mcc_all': [round(x, 4) for x in mccs]}
        out[tag][vname] = row
        print(f'{tag:11s} {vname:10s} mcc mean {row["mcc_mean"]} max {row["mcc_max"]} '
              f'best-by-recon {row["mcc_best_by_recon"]}', flush=True)
json.dump(out, open('/workspace/tensor_language/basis_aligned/qk_mdl/qk_solver_harden.json', 'w'),
          indent=2)
print('HARDEN DONE', flush=True)
