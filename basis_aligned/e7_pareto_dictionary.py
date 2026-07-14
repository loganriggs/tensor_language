"""e7 stage A: Pareto frontier over (n_atoms, L0) for representing pythia-410m's
embedding as a sparse dictionary — learned, not k-means.

Each token row x is coded as a SIGNED top-k combination of dictionary atoms:
  z = (x - b_pre) @ W_e^T ;  keep top-k by |z| ;  xhat = b + sum_j z_j * D_j
(sparse coding flavor, magnitude top-k with signed coefficients; decoder rows
unit-norm; dead atoms reinitialized to worst-reconstructed rows periodically).

Grid: n_atoms x k, trained on the 50304 embedding rows under MSE, audited with
FVU and swapped-in delta-CE (e6 machinery). Corners for context: SVD (dense
codes), k-means (k=1 unlearned), the identity (n=V, k=1, dCE=0).

Stage B (e7b) CE-finetunes the per-n frontier configs through the model.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6

DEV = 'cuda'
BASE = '/workspace/tensor_language/basis_aligned'
E = e6.E  # (V, d) float32 on GPU
V, D_MODEL = E.shape
torch.manual_seed(0)


def train_topk_dict(n, k, steps=4000, batch=8192, lr=3e-3, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    idx0 = torch.randperm(V, generator=g)[:n]
    Dm = E[idx0].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b_pre = E.mean(0).clone()
    b = E.mean(0).clone()
    for t in [Dm, We, b_pre, b]:
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b_pre, b], lr=lr)
    usage = torch.zeros(n, device=DEV)

    def encode_decode(x, Dn):
        z = (x - b_pre) @ We.T
        vals, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        return xhat, idx, coeff

    for step in range(steps):
        x = E[torch.randint(0, V, (batch,), device=DEV)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        xhat, idx, _ = encode_decode(x, Dn)
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            usage.index_add_(0, idx.flatten(),
                             torch.ones(idx.numel(), device=DEV))
        if step % 500 == 499:  # revive dead atoms on worst-reconstructed rows
            with torch.no_grad():
                dead = (usage == 0).nonzero().flatten()
                if len(dead):
                    Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sample = E[torch.randint(0, V, (4096,), device=DEV)]
                    xhat, _, _ = encode_decode(sample, Dn)
                    err = ((xhat - sample) ** 2).sum(1)
                    worst = sample[err.topk(min(len(dead), 4096)).indices]
                    w = worst[:len(dead)] - b
                    w = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    Dm.data[dead[:len(w)]] = w
                    We.data[dead[:len(w)]] = w
                usage.zero_()

    with torch.no_grad():
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        # full-vocab codes
        supports = torch.empty(V, k, dtype=torch.long, device=DEV)
        coeffs = torch.empty(V, k, device=DEV)
        Ehat = torch.empty_like(E)
        for i in range(0, V, 8192):
            x = E[i:i + 8192]
            z = (x - b_pre) @ We.T
            _, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            supports[i:i + 8192] = idx
            coeffs[i:i + 8192] = coeff
            Ehat[i:i + 8192] = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        n_alive = int((torch.bincount(supports.flatten(), minlength=n) > 0).sum())
    return Ehat, {'D': Dn.detach().cpu(), 'supports': supports.cpu(),
                  'coeffs': coeffs.detach().cpu(), 'b': b.detach().cpu()}, n_alive


if __name__ == '__main__':
    res = {'baseline_ce': None, 'rows': []}
    print('baseline CE...')
    res['baseline_ce'] = e6.eval_ce()
    print(f"baseline {res['baseline_ce']:.4f}")

    for n in [1024, 4096, 16384, 32768]:
        for k in [1, 4, 16, 64]:
            Ehat, state, n_alive = train_topk_dict(n, k)
            row = {'method': 'topk_dict', 'n_atoms': n, 'k': k, 'n_alive': n_alive,
                   'fvu': e6.fvu(Ehat), 'ce': e6.eval_ce(Ehat)}
            row['dce'] = row['ce'] - res['baseline_ce']
            res['rows'].append(row)
            print(f"n={n:6d} k={k:3d} alive={n_alive:6d}  fvu {row['fvu']:.4f}  "
                  f"dCE {row['dce']:+.4f}", flush=True)
            torch.save(state, f'{BASE}/e7_dict_n{n}_k{k}.pt')
            del Ehat
            torch.cuda.empty_cache()
            with open(f'{BASE}/e7_results.json', 'w') as fh:
                json.dump(res, fh, indent=2)
    print('stage A done; saved e7_results.json')
