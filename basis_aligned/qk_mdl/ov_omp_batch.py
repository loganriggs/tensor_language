"""Least-squares / marginal-error batch allocation (Logan 2026-07-21).
The linear-encoder top-k compares raw coefficient magnitudes across words,
which is invalid for an overcomplete dictionary. The principled fix: select
atoms by TRUE marginal error (greedy orthogonal matching pursuit with a
least-squares refit at every step), then allocate a GLOBAL budget by pooling
the per-step residual reductions. If Logan is right, batch (variable per-word
sparsity by marginal error) then beats per-token (fixed k) even in the
overcomplete case, restoring the orthonormal-regime guarantee.

Head 0, dictionary of 512 atoms, average k=8. FVU = fraction of head variance
unexplained. All coefficients are least-squares-optimal on the chosen support."""
import json
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
X = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)[:, 0].to(DEV)
b = X.mean(0)
Y = X - b                                         # (V, 128) center
TOTVAR = (Y ** 2).sum().item()
K, KMAX = 8, 28

# dictionary: reuse a per-token-trained one's atoms (unit norm). Quick train.
g = torch.Generator(device='cpu'); g.manual_seed(0)
Dm = X[torch.randperm(V, generator=g)[:512]].clone()
Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
We = Dm.clone(); bb = b.clone()
for t in (Dm, We, bb):
    t.requires_grad_(True)
opt = torch.optim.Adam([Dm, We, bb], lr=3e-3)
for _ in range(2500):
    Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    z = (X - bb) @ We.T
    _, idx = z.abs().topk(K, 1); coeff = torch.gather(z, 1, idx)
    loss = ((bb + (coeff.unsqueeze(-1) * Dn[idx]).sum(1) - X) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()   # (512, 128)
print('dictionary trained', flush=True)


@torch.no_grad()
def omp(nsteps):
    """Vectorized greedy OMP for all V words to nsteps atoms, LS-refit each step.
    Returns per-step residual-SSE (V, nsteps+1) and supports (V, nsteps)."""
    r = Y.clone()                                 # residual
    res_sse = [ (r ** 2).sum(1).cpu() ]           # step 0 = bias only
    sup = torch.full((V, nsteps), -1, device=DEV, dtype=torch.long)
    chosen = torch.zeros(V, 512, dtype=torch.bool, device=DEV)
    for s in range(nsteps):
        corr = (r @ Dn.T).abs()                   # (V, 512)
        corr[chosen] = -1
        a = corr.argmax(1)                         # (V,)
        sup[:, s] = a
        chosen[torch.arange(V, device=DEV), a] = True
        # LS refit on current support of size s+1
        Ds = Dn[sup[:, :s + 1]]                    # (V, s+1, 128)
        G = torch.bmm(Ds, Ds.transpose(1, 2))      # (V, s+1, s+1)
        rhs = torch.bmm(Ds, Y.unsqueeze(-1)).squeeze(-1)   # (V, s+1)
        c = torch.linalg.solve(G + 1e-6 * torch.eye(s + 1, device=DEV), rhs)
        recon = torch.bmm(c.unsqueeze(1), Ds).squeeze(1)   # (V, 128)
        r = Y - recon
        res_sse.append((r ** 2).sum(1).cpu())
    return torch.stack(res_sse, 1), sup            # (V, nsteps+1)


# per-token OMP to k=8
sse_pt, _ = omp(K)
fvu_pt = sse_pt[:, K].sum().item() / TOTVAR
print(f'per-token OMP (LS, k=8): FVU {fvu_pt:.4f}', flush=True)

# batch: run to KMAX, marginal gains = sse[:,s]-sse[:,s+1]; pool, take top K*V
sse_full, _ = omp(KMAX)
gains = (sse_full[:, :-1] - sse_full[:, 1:])       # (V, KMAX) marginal reductions, decreasing/word
flat = gains.reshape(-1)
thr = flat.topk(K * V).values.min()
take = gains >= thr                                # (V, KMAX)
# because gains decrease along a row (greedy OMP), take is a per-word prefix; p_t = its sum
p = take.sum(1)                                    # (V,) atoms per word
# residual after p_t steps:
sse_batch = sse_full.gather(1, p.unsqueeze(1)).squeeze(1)
fvu_batch = sse_batch.sum().item() / TOTVAR
print(f'batch OMP (LS, marginal-error alloc, avg k={p.float().mean():.2f}): FVU {fvu_batch:.4f}', flush=True)

res = {'per_token_OMP_LS_k8': round(fvu_pt, 4),
       'batch_OMP_LS_avgk8': round(fvu_batch, 4),
       'batch_avg_k': round(float(p.float().mean()), 3),
       'batch_min_k': int(p.min()), 'batch_max_k': int(p.max()),
       'batch_beats_pertoken': bool(fvu_batch <= fvu_pt),
       'note': 'if True, Logan validated: proper marginal-error batch >= per-token even overcomplete'}
print(f'VERDICT: batch {"<=" if res["batch_beats_pertoken"] else ">"} per-token '
      f'-> Logan {"VALIDATED" if res["batch_beats_pertoken"] else "still not (see gap)"}', flush=True)
json.dump(res, open(f'{QK}/ov_omp_batch.json', 'w'), indent=2)
torch.save(p.cpu(), f'{QK}/ov_omp_counts.pt')
print('ov omp batch done', flush=True)
