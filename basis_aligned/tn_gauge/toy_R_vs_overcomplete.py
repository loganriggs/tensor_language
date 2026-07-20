"""The MISSING anchor (Logan 2026-07-20): the exact R-rotation baseline, and the
overcomplete Φ measured AGAINST it (not against zero).

R baseline: a square orthonormal rotation of the residual bond, optimized to
concentrate the stream (min mean|Rx|, Cayley/Stiefel). Applied as an EXACT gauge it
leaves the model identical (ΔCE=0 at full rank) and stays a tensor network. A top-k
code in the rotated basis is the 'perfect but square' sparse description; at k=d it is
lossless. This is the reference the overcomplete arm must beat.

Overcomplete Φ: m>d dictionary (already built in toy_code_propagation). Sparser at
matched k, but the code is a lossy description -> a real ΔCE cost, now read against R.

Compare end-to-end ΔCE vs k (every bond coded) for:
  R (m=d=128, orthonormal, EXACT at k=128) | Φ m=512 | Φ m=2048
Toy block2, real TinyStories. No SAE-as-forward-mechanism framing: R is a gauge.
"""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language')
from deep_model import DeepModel
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN = '/workspace/tensor_language/runs_lm/block2-seed0'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
cfg = json.load(open(f'{RUN}/config.json'))
D, NH, SPEC = cfg['d_model'], cfg['n_head'], cfg['spec']
VOCAB, N_CTX = cfg['vocab'], cfg['n_ctx']
m = DeepModel(VOCAB, D, NH, SPEC, N_CTX, norm=(cfg['norm'] == 'rms'),
              residual=cfg['residual'], attention=cfg['attention']).to(DEV)
sd = torch.load(f'{RUN}/model.pt', map_location=DEV)
m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) and 'model' in sd else sd)
m.eval()
for p in m.parameters():
    p.requires_grad_(False)
val = np.memmap('/workspace/tensor_language/data_text/val.bin', dtype=np.uint16, mode='r')
buf = np.stack([val[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(64)]).astype(np.int64)
B = torch.from_numpy(buf).to(DEV); IDX, TGT = B[:, :-1], B[:, 1:]
rms = lambda x: F.rms_norm(x, (x.size(-1),))


@torch.no_grad()
def ce_of(model):
    return F.cross_entropy(model(IDX).float().reshape(-1, VOCAB), TGT.reshape(-1)).item()


CE0 = ce_of(m)
# pooled normalized bond activations
Hs = []
with torch.no_grad():
    x = m.embed(IDX)
    for layer in m.layers:
        Hs.append(rms(x).reshape(-1, D)); x = layer(x)
POOL = torch.cat(Hs, 0)
bmean = POOL.mean(0)
res = {'baseline_ce': round(CE0, 4), 'arms': {}}
print(f'baseline CE {CE0:.4f}', flush=True)

# ---- optimize an orthonormal R to concentrate the stream (min mean|R (x-b)|) ----
Xc = POOL - bmean
R = torch.eye(D, device=DEV)
lr = 0.2
for step in range(400):
    Z = Xc @ R.T                                   # (N, D) rotated codes
    G = torch.sign(Z).T @ Xc / Xc.shape[0]         # d mean|Z| / dR
    A = G @ R.T; A = 0.5 * (A - A.T)               # skew (tangent of O(D))
    I = torch.eye(D, device=DEV)
    R = torch.linalg.solve(I + lr * A, I - lr * A) @ R
orth_err = (R @ R.T - torch.eye(D, device=DEV)).abs().max().item()
print(f'R optimized, orthogonality error {orth_err:.1e}', flush=True)


def make_R_encode(R, b):
    @torch.no_grad()
    def enc(h, k):
        z = (h - b) @ R.T                          # coeffs in rotated basis (orthonormal)
        thr = z.abs().topk(k, 1).values[:, -1:]
        zt = torch.where(z.abs() >= thr, z, torch.zeros_like(z))
        return b + zt @ R                          # exact inverse (R orthonormal)
    return enc


# overcomplete dictionaries (train quickly, LS-refit encode) -- reuse the recipe
def train_dict(X, mm, steps=2500):
    g = torch.Generator(device='cpu').manual_seed(0)
    Phi = X[torch.randperm(X.shape[0], generator=g)[:mm]].clone().T
    Phi = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
    We = Phi.clone(); b = X.mean(0).clone()
    Phi.requires_grad_(True); We.requires_grad_(True); b.requires_grad_(True)
    opt = torch.optim.Adam([Phi, We, b], lr=3e-3)
    for _ in range(steps):
        Pn = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
        z = (X - b) @ We
        _, idx = z.abs().topk(16, 1); coeff = torch.gather(z, 1, idx)
        rec = b + torch.einsum('nk,nkd->nd', coeff, Pn.T[idx])
        (((rec - X) ** 2).mean()).backward(); opt.step(); opt.zero_grad()
    return (Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)).detach(), We.detach(), b.detach()


def make_dict_encode(Phi, We, b):
    @torch.no_grad()
    def enc(h, k):
        z = (h - b) @ We
        _, idx = z.abs().topk(k, 1)
        Psup = Phi[:, idx].permute(1, 2, 0)
        G = torch.bmm(Psup, Psup.transpose(1, 2))
        rhs = torch.bmm(Psup, (h - b).unsqueeze(-1))
        c = torch.linalg.solve(G + 1e-4 * torch.eye(k, device=DEV), rhs)
        return b + torch.bmm(c.transpose(1, 2), Psup).squeeze(1)
    return enc


class CodeNorm(nn.Module):
    def __init__(self, enc, k):
        super().__init__(); self.enc = enc; self.k = k
    def forward(self, x):
        h = rms(x); return self.enc(h.reshape(-1, D), self.k).reshape(h.shape)


orig = [l.norm for l in m.layers]


def sweep(tag, enc, ks):
    row = {}
    for k in ks:
        for l in m.layers:
            l.norm = CodeNorm(enc, k)
        dce = ce_of(m) - CE0
        for l, nm in zip(m.layers, orig):
            l.norm = nm
        row[f'k={k}'] = round(dce, 4)
        print(f'  {tag} k={k}: ΔCE {dce:+.4f}', flush=True)
    res['arms'][tag] = row
    json.dump(res, open(f'{OUT}/toy_R_vs_overcomplete.json', 'w'), indent=2)


print('R baseline (square orthonormal, EXACT gauge; k=128 must give ΔCE≈0):', flush=True)
sweep('R square m=128 (exact gauge)', make_R_encode(R, bmean), [16, 32, 64, 128])
for mm in (512, 2048):
    Phi, We, b = train_dict(POOL, mm)
    print(f'overcomplete Φ m={mm}:', flush=True)
    sweep(f'overcomplete m={mm}', make_dict_encode(Phi, We, b), [16, 32, 64])
print('R vs overcomplete done -> toy_R_vs_overcomplete.json', flush=True)
