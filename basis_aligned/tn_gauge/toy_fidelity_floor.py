"""Gate 2: is F2's negative "regime dead" or "dictionary too small"? (Logan 2026-07-20)
Sweep dictionary size m and SHARED (one Φ, pooled bonds) vs PER-BOND (a Φ per bond),
at fixed k, and measure end-to-end ΔCE (every bond coded). If ΔCE -> ~0 with larger m
or per-bond dicts, the code-propagation regime is viable and F2 was just underpowered;
if it plateaus high, the regime itself is lossy. Report bits alongside (structural:
m*d*32 per dict + estimation k*log2(m) per token) so "viable" is bits-honest.
Toy block2, real TinyStories, LS-refit coeffs (fair)."""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math
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
# collect per-bond normalized activations
bacts = [[] for _ in range(len(SPEC))]
with torch.no_grad():
    x = m.embed(IDX)
    for li, layer in enumerate(m.layers):
        bacts[li].append(rms(x).reshape(-1, D)); x = layer(x)
BOND = [torch.cat(a, 0) for a in bacts]


def train_dict(X, mm, steps=2000):
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
        loss = ((rec - X) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)).detach(), We.detach(), b.detach()


def make_encode(Phi, We, b):
    @torch.no_grad()
    def enc(h, k):
        z = (h - b) @ We
        _, idx = z.abs().topk(k, 1)
        Psup = Phi[:, idx].permute(1, 2, 0)
        y = h - b
        G = torch.bmm(Psup, Psup.transpose(1, 2))
        rhs = torch.bmm(Psup, y.unsqueeze(-1))
        c = torch.linalg.solve(G + 1e-4 * torch.eye(k, device=DEV), rhs)
        return b + torch.bmm(c.transpose(1, 2), Psup).squeeze(1)
    return enc


class CodeNorm(nn.Module):
    def __init__(self, enc, k):
        super().__init__(); self.enc = enc; self.k = k
    def forward(self, x):
        h = rms(x); return self.enc(h.reshape(-1, D), self.k).reshape(h.shape)


res = {'baseline_ce': round(CE0, 4), 'arms': {}}
print(f'baseline CE {CE0:.4f}', flush=True)
K = 32
orig = [l.norm for l in m.layers]
for mm in (512, 2048):
    # SHARED: one Φ on pooled bonds
    Phi, We, b = train_dict(torch.cat(BOND, 0), mm)
    enc = make_encode(Phi, We, b)
    for l in m.layers:
        l.norm = CodeNorm(enc, K)
    dce = ce_of(m) - CE0
    for l, nm in zip(m.layers, orig):
        l.norm = nm
    bits = mm * D * 32 + K * math.log2(mm) * BOND[0].shape[0] * len(SPEC)
    res['arms'][f'shared m={mm} k={K}'] = {'dce': round(dce, 4), 'Mbits': round(bits / 1e6, 1)}
    print(f'shared m={mm} k={K}: ΔCE {dce:+.4f}  {bits/1e6:.1f}Mbit', flush=True)
    # PER-BOND: a Φ per bond
    encs = []
    for li in range(len(SPEC)):
        Pi, Wi, bi = train_dict(BOND[li], mm)
        encs.append(make_encode(Pi, Wi, bi))
    for li, l in enumerate(m.layers):
        l.norm = CodeNorm(encs[li], K)
    dce = ce_of(m) - CE0
    for l, nm in zip(m.layers, orig):
        l.norm = nm
    bits = len(SPEC) * mm * D * 32 + K * math.log2(mm) * BOND[0].shape[0] * len(SPEC)
    res['arms'][f'per-bond m={mm} k={K}'] = {'dce': round(dce, 4), 'Mbits': round(bits / 1e6, 1)}
    print(f'per-bond m={mm} k={K}: ΔCE {dce:+.4f}  {bits/1e6:.1f}Mbit', flush=True)
    json.dump(res, open(f'{OUT}/toy_fidelity_floor.json', 'w'), indent=2)
print('toy fidelity floor done', flush=True)
