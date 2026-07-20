"""Training loss curves + convergence/batch-size diagnosis for the layer-0
content dictionaries (Logan 2026-07-21).

Logan's concern: batch-top-k should be able to REPLICATE per-token top-k
(allocate exactly k to everyone), so its optimum must be >= as good; coming
out worse points to a convergence or train/eval mismatch. Suspected cause:
the batch-top-k THRESHOLD is computed within a minibatch during training but
over the full vocabulary at eval — a mismatch that also makes training noisy.

This script trains, recording fraction-of-variance-unexplained (FVU) every few
steps, on one representative head:
  Panel A (shared dict of 512, convergence diagnosis, FULL-BATCH gradient so
  there is no sampling noise and the batch-top-k train threshold == eval
  threshold exactly):
    - per-token top-k, k=16
    - batch-top-k avg=16, trained on MINIBATCHES of 8192 (the original)
    - batch-top-k avg=16, trained FULL-BATCH (the fix)
  Panel B (routed / block-sparse, Logan's requests): adaptive group dict sizes
  AND batch-top-k within each group; per-group curves + aggregate.
FVU is always measured against the whole head's variance, so every curve is
comparable."""
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
VT = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)
X = VT[:, 0].to(DEV)                          # head 0, (V, 128)
STEPS, REC = 2500, 25
gmean = X.mean(0)
TOTVAR = ((X - gmean) ** 2).sum().item()


def encode(Xg, Dn, We, b, k, mode):
    z = (Xg - b) @ We.T
    if mode == 'token':
        _, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    nnz = k * len(Xg)
    thr = z.abs().reshape(-1).topk(nnz).values.min()
    return b + (z * (z.abs() >= thr)) @ Dn


def train_curve(rows, n, k, mode, seed, minibatch=None):
    """minibatch=None -> full-batch (all rows each step)."""
    Xg = X[rows]
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    Dm = Xg[torch.randperm(len(Xg), generator=g)[:n]].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone(); b = Xg.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=3e-3)
    curve = []
    for step in range(STEPS + 1):
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        if step % REC == 0:
            with torch.no_grad():                     # eval always full-set
                xhat = encode(Xg, Dn.detach(), We.detach(), b.detach(), k, mode)
                curve.append((step, ((xhat - Xg) ** 2).sum().item() / TOTVAR))
        if step == STEPS:
            break
        xb = Xg if minibatch is None else Xg[torch.randint(0, len(Xg), (minibatch,), device=DEV)]
        xh = encode(xb, Dn, We, b, k, mode)
        loss = ((xh - xb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return curve


ALL = torch.arange(V, device=DEV)
out = {'shared': {}, 'routed': {}}
print('shared-dict convergence diagnosis (full-batch)...', flush=True)
out['shared']['per-token k=16 (full-batch)'] = train_curve(ALL, 512, 16, 'token', 0)
out['shared']['batch-top-k avg=16 (minibatch 8192)'] = train_curve(ALL, 512, 16, 'batch', 2, minibatch=8192)
out['shared']['batch-top-k avg=16 (full-batch)'] = train_curve(ALL, 512, 16, 'batch', 2)
for nm, c in out['shared'].items():
    print(f'  {nm:38s} final FVU {c[-1][1]:.4f}', flush=True)

# routed: adaptive group sizes + batch-top-k within groups
G = 8
gK = torch.Generator(); gK.manual_seed(1)
C0 = E_hat[torch.randperm(V, generator=gK)[:G]].clone().to(DEV)
for _ in range(10):
    a_ = torch.empty(V, dtype=torch.long, device=DEV)
    for i in range(0, V, 8192):
        xx = E_hat[i:i + 8192].to(DEV)
        a_[i:i + 8192] = ((xx * xx).sum(1, True) - 2 * xx @ C0.T + (C0 * C0).sum(1)[None]).argmin(1)
    Cn = torch.zeros_like(C0); c2 = torch.zeros(G, device=DEV)
    Cn.index_add_(0, a_, E_hat.to(DEV)); c2.index_add_(0, a_, torch.ones(V, device=DEV))
    nz = c2 > 0; C0[nz] = Cn[nz] / c2[nz][:, None]
sizes = torch.bincount(a_, minlength=G)
print('routed groups (adaptive n_g, batch-top-k within group, full-batch)...', flush=True)
for gg in range(G):
    rows = (a_ == gg).nonzero().squeeze(1)
    # adaptive: atoms proportional to group size (bounded), same average sparsity
    n_g = int(max(64, min(256, round(len(rows) / 40))))
    out['routed'][f'group{gg} (words={len(rows)}, atoms={n_g})'] = train_curve(
        rows, n_g, 8, 'batch', 200 + gg)
    print(f'  group {gg}: {len(rows)} words, {n_g} atoms, final FVU '
          f'{out["routed"][f"group{gg} (words={len(rows)}, atoms={n_g})"][-1][1]:.4f}', flush=True)

json.dump(out, open(f'{QK}/ov_train_curves.json', 'w'))
print('ov train curves done', flush=True)
