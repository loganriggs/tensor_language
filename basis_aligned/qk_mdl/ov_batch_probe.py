"""Why is batch-top-k worse, and can it be fixed? (Logan 2026-07-21)
Thesis: for an ORTHONORMAL basis batch-top-k is provably <= per-token top-k
(error = sum of dropped coeff^2, so keeping the globally-largest coeffs is
optimal and per-token is a constrained version). For an OVERCOMPLETE dictionary
the linear-encoder coefficients are not optimal and their magnitudes are not
comparable across words, which breaks the guarantee.

Tests (head 0, V=50257 inputs, FVU = fraction of head variance unexplained):
  1. ORTHONORMAL (128-dim SVD basis): per-token vs batch-top-k. Prediction:
     batch <= per-token (validates the intuition).
  2. OVERCOMPLETE (512-atom trained dict): per-token vs batch. batch > per-token.
  3. Fixes for the overcomplete case:
     (a) min-1-atom floor (guarantee every word >=1 atom, rest by global budget)
     (b) per-word-normalized selection (compare relative-within-word importance)
     (c) warm-start: per-token dict, then finetune with batch-top-k
  4. Histogram of per-word atom counts under plain batch-top-k, and per-word
     reconstruction error vs atom count / content norm (are 0-atom words hurt,
     or just small-norm and well-served by the bias?).
"""
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
TOTVAR = ((X - X.mean(0)) ** 2).sum().item()
K = 8
res = {'n_inputs_per_head': V, 'n_heads': NH}


def fvu(xhat):
    return ((xhat - X) ** 2).sum().item() / TOTVAR


# ---- 1. ORTHONORMAL basis (SVD, coefficients ARE optimal) ----
mu = X.mean(0)
U, Sg, Vh = torch.linalg.svd(X - mu, full_matrices=False)   # 128-dim
Cf = U * Sg                                                  # (V,128) coeffs in ortho basis
# per-token top-k
vals, idx = Cf.abs().topk(K, 1)
Cp = torch.zeros_like(Cf).scatter_(1, idx, torch.gather(Cf, 1, idx))
res['ortho per-token'] = round(fvu(mu + Cp @ Vh), 4)
# batch-top-k
thr = Cf.abs().reshape(-1).topk(K * V).values.min()
Cb = Cf * (Cf.abs() >= thr)
res['ortho batch-top-k'] = round(fvu(mu + Cb @ Vh), 4)
print(f'ORTHONORMAL: per-token {res["ortho per-token"]}  batch {res["ortho batch-top-k"]}  '
      f'(batch<=token? {res["ortho batch-top-k"] <= res["ortho per-token"]})', flush=True)


# ---- overcomplete dictionary trainer ----
def train(n, mode, steps=2500, warm=None):
    if warm is not None:
        Dm, We, b = (t.clone() for t in warm)
    else:
        g = torch.Generator(device='cpu'); g.manual_seed(0)
        Dm = X[torch.randperm(V, generator=g)[:n]].clone()
        Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        We = Dm.clone(); b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=3e-3)
    for _ in range(steps):
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (X - b) @ We.T
        if mode == 'token':
            _, idx = z.abs().topk(K, 1); coeff = torch.gather(z, 1, idx)
            xh = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        else:
            thr = z.abs().reshape(-1).topk(K * V).values.min()
            xh = b + (z * (z.abs() >= thr)) @ Dn
        loss = ((xh - X) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(), We.detach(), b.detach()


def encode(Dn, We, b, mode, floor0=False, pernorm=False):
    z = (X - b) @ We.T
    if mode == 'token':
        _, idx = z.abs().topk(K, 1); coeff = torch.gather(z, 1, idx)
        counts = torch.full((V,), K, device=DEV)
        return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1), counts
    zsel = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8) if pernorm else z
    thr = zsel.abs().reshape(-1).topk(K * V).values.min()
    keep = zsel.abs() >= thr
    if floor0:  # guarantee >=1 atom per word: force each row's own top-1
        top1 = z.abs().argmax(1)
        keep[torch.arange(V, device=DEV), top1] = True
    zc = z * keep
    return b + zc @ Dn, keep.sum(1)


# ---- 2. OVERCOMPLETE per-token vs batch ----
Dt = train(512, 'token'); xh, _ = encode(*Dt, 'token'); res['overc per-token'] = round(fvu(xh), 4)
Db = train(512, 'batch'); xh_b, cnt_b = encode(*Db, 'batch'); res['overc batch'] = round(fvu(xh_b), 4)
print(f'OVERCOMPLETE: per-token {res["overc per-token"]}  batch {res["overc batch"]}', flush=True)

# ---- 3. fixes (all on the batch-trained dict unless noted) ----
xh, _ = encode(*Db, 'batch', floor0=True); res['overc batch + min-1-atom'] = round(fvu(xh), 4)
xh, _ = encode(*Db, 'batch', pernorm=True); res['overc batch + per-word-normalized select'] = round(fvu(xh), 4)
Dw = train(512, 'batch', steps=1000, warm=Dt); xh, cnt_w = encode(*Dw, 'batch')
res['overc batch warm-started from per-token'] = round(fvu(xh), 4)
for kname in ['overc batch + min-1-atom', 'overc batch + per-word-normalized select',
              'overc batch warm-started from per-token']:
    print(f'  {kname}: {res[kname]}', flush=True)

# ---- 4. histogram + error-vs-count for plain batch ----
counts = cnt_b.cpu()
res['atom_count_hist'] = {str(i): int((counts == i).sum()) for i in range(0, 25)}
res['atom_count_hist']['25+'] = int((counts >= 25).sum())
per_err = ((xh_b - X) ** 2).sum(1).cpu()
cnorm = ((X - X.mean(0)) ** 2).sum(1).cpu()
z0 = counts == 0
res['zero_atom_words'] = int(z0.sum())
res['mean_err_zero_atom'] = round(float(per_err[z0].mean()), 4)
res['mean_err_nonzero'] = round(float(per_err[~z0].mean()), 4)
res['mean_contentnorm_zero_atom'] = round(float(cnorm[z0].mean()), 4)
res['mean_contentnorm_nonzero'] = round(float(cnorm[~z0].mean()), 4)
print(f'zero-atom words: {res["zero_atom_words"]}; their mean err {res["mean_err_zero_atom"]} '
      f'vs nonzero {res["mean_err_nonzero"]}; their content-norm {res["mean_contentnorm_zero_atom"]} '
      f'vs {res["mean_contentnorm_nonzero"]}', flush=True)

json.dump(res, open(f'{QK}/ov_batch_probe.json', 'w'), indent=2)
torch.save(counts, f'{QK}/ov_batch_counts.pt')
print('ov batch probe done', flush=True)
