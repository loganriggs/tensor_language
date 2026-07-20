"""Decisive test of 'batch-top-k >= per-token top-k' (Logan's premise).
Take ONE trained dictionary and encode the SAME vectors two ways at the SAME
total budget (k*V nonzeros): per-token top-k (each row its own best k) vs
batch-top-k (global top k*V). If batch >= token always, batch's FVU <= token's.
Prediction (from theory): per-token gives each row its LOCALLY OPTIMAL k-term
code, so batch (global budget) can only help when per-row needs are
heterogeneous, and can HURT rows it starves — so it does NOT dominate."""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs

torch.manual_seed(0)
DEV = 'cuda'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
X = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)[:, 0].to(DEV)
TOTVAR = ((X - X.mean(0)) ** 2).sum().item()

# train ONE dictionary with per-token top-k, full-batch, well-converged
n, k = 512, 16
g = torch.Generator(device='cpu'); g.manual_seed(0)
Dm = X[torch.randperm(V, generator=g)[:n]].clone()
Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
We = Dm.clone(); b = X.mean(0).clone()
for t in (Dm, We, b):
    t.requires_grad_(True)
opt = torch.optim.Adam([Dm, We, b], lr=3e-3)
for step in range(4000):
    Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    z = (X - b) @ We.T
    _, idx = z.abs().topk(k, 1); coeff = torch.gather(z, 1, idx)
    xh = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    loss = ((xh - X) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
    z = (X - b) @ We.T
    # per-token top-k
    _, idx = z.abs().topk(k, 1); coeff = torch.gather(z, 1, idx)
    xh_tok = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    fvu_tok = ((xh_tok - X) ** 2).sum().item() / TOTVAR
    # batch-top-k on the SAME dict, SAME budget k*V
    nnz = k * V
    thr = z.abs().reshape(-1).topk(nnz).values.min()
    zc = z * (z.abs() >= thr)
    xh_bat = b + zc @ Dn
    fvu_bat = ((xh_bat - X) ** 2).sum().item() / TOTVAR
    per_row = (zc != 0).sum(1)
print(f'SAME dictionary, SAME budget (k*V nonzeros):', flush=True)
print(f'  per-token top-k  FVU {fvu_tok:.4f}', flush=True)
print(f'  batch-top-k      FVU {fvu_bat:.4f}', flush=True)
print(f'  batch per-row atom count: min {per_row.min().item()} '
      f'median {per_row.median().item()} max {per_row.max().item()} '
      f'(words with 0 atoms: {(per_row == 0).sum().item()})', flush=True)
print(f'  verdict: batch {"<=" if fvu_bat <= fvu_tok else ">"} token '
      f'-> premise {"holds" if fvu_bat <= fvu_tok else "FALSE (per-token dominates)"}', flush=True)
print('same test done', flush=True)
