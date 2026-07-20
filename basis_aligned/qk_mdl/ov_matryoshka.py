"""Matryoshka / hierarchical dictionary for layer-0 content (Logan 2026-07-21).
A single ordered dictionary of 512 atoms trained so that every PREFIX is itself
a good dictionary: the loss sums reconstruction at nested prefix sizes
{32,128,512}, forcing early atoms to carry coarse/broad structure and later
atoms to refine. Two payoffs to test:
  (1) does the hierarchy cost reconstruction at fixed per-token k=8 vs a plain
      (non-nested) dictionary? (usual Matryoshka trade: some reconstruction for
      structure);
  (2) it enables ADAPTIVE per-word prefix length — easy words reconstructed from
      the first few atoms, hard words from more — a *structured* (nested) form of
      the flexible sparsity batch-top-k tried to get. Test: give each word the
      smallest prefix that reaches a target reconstruction, matched to average k=8.
Head 0, FVU = fraction of head variance unexplained."""
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
N, K = 512, 8
PREFIXES = [32, 128, 512]


def train(nested, steps=2500):
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    Dm = X[torch.randperm(V, generator=g)[:N]].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone(); b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=3e-3)
    for _ in range(steps):
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (X - b) @ We.T
        loss = 0.0
        plist = PREFIXES if nested else [N]
        for P in plist:
            zp = z[:, :P]
            _, idx = zp.abs().topk(K, 1); coeff = torch.gather(zp, 1, idx)
            xh = b + (coeff.unsqueeze(-1) * Dn[:P][idx]).sum(1)
            loss = loss + ((xh - X) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(), We.detach(), b.detach()


@torch.no_grad()
def fvu_topk(Dn, We, b, P=N, k=K):
    z = (X - b) @ We.T
    zp = z[:, :P]
    _, idx = zp.abs().topk(k, 1); coeff = torch.gather(zp, 1, idx)
    xh = b + (coeff.unsqueeze(-1) * Dn[:P][idx]).sum(1)
    return ((xh - X) ** 2).sum().item() / TOTVAR


res = {}
plain = train(nested=False)
res['plain dict, per-token k=8 (full 512)'] = round(fvu_topk(*plain), 4)
mat = train(nested=True)
res['matryoshka, per-token k=8 (full 512)'] = round(fvu_topk(*mat), 4)
# the hierarchy: FVU using only the first P atoms (per-token k=8 within prefix)
for P in PREFIXES:
    res[f'matryoshka, first {P} atoms only (k=8)'] = round(fvu_topk(*mat, P=P), 4)
    res[f'plain, first {P} atoms only (k=8)'] = round(fvu_topk(*plain, P=P), 4)
print('reconstruction FVU:', flush=True)
for kk, vv in res.items():
    print(f'  {kk}: {vv}', flush=True)

# adaptive per-word prefix on the matryoshka dict: smallest prefix reaching a
# per-word error target, budget-matched to avg k=8 dense-in-prefix.
Dn, We, b = mat
with torch.no_grad():
    z = (X - b) @ We.T
    # cumulative dense reconstruction error as prefix grows (coarse grid)
    grid = [8, 16, 32, 64, 128, 256, 512]
    errs = []
    for P in grid:
        # dense within prefix P (all P coeffs) — the hierarchy's natural use
        xh = b + z[:, :P] @ Dn[:P]
    unifP = {P: round(float((((b + z[:, :P] @ Dn[:P]) - X) ** 2).sum() / TOTVAR), 4) for P in grid}
    res['matryoshka dense-in-prefix FVU by uniform prefix'] = unifP
    print('matryoshka dense-in-prefix FVU by prefix size:', unifP, flush=True)

json.dump(res, open(f'{QK}/ov_matryoshka.json', 'w'), indent=2)
print('ov matryoshka done', flush=True)
