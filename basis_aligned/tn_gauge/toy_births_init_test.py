"""Regime 2, the useful version (Logan 2026-07-20): F10 showed write-seeded atoms are
the right DIRECTION but fixed seeds are catastrophic (+2.8 ΔCE) — they must be trained.
So the real question: does write-seeded INITIALIZATION + training converge faster / to a
lower ΔCE than random init? If yes, weight-informed births earn their keep as a warm start
(a legal search with a good prior), which is exactly Logan's 'legal search, constrained
output' framing.

Per bond ℓ∈{1,2,3}, train an overcomplete dict (m=512, k=32) two ways: WRITE-init (seed
atoms = sampled upstream write deltas) vs RANDOM-init. Track reconstruction loss vs step;
then measure end-to-end ΔCE with each init's TRAINED dicts (bond 0 exact). Toy block2.
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
def ce():
    return F.cross_entropy(m(IDX).float().reshape(-1, VOCAB), TGT.reshape(-1)).item()


CE0 = ce()
H, DELTA = [], []
with torch.no_grad():
    x = m.embed(IDX)
    for layer in m.layers:
        xb = x; H.append(rms(xb).reshape(-1, D)); x = layer(x); DELTA.append((x - xb).reshape(-1, D))
M, K = 512, 32
CHECK = [50, 200, 800, 2000]


def unit(A):
    return A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)


def train(X, init, seed, steps=2000):
    Phi = init.T.clone().contiguous()          # init: (m, D) atoms -> Phi (D, m)
    We = Phi.clone(); b = X.mean(0).clone()
    Phi.requires_grad_(True); We.requires_grad_(True); b.requires_grad_(True)
    opt = torch.optim.Adam([Phi, We, b], lr=3e-3)
    curve = {}
    for step in range(steps + 1):
        Pn = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
        z = (X - b) @ We
        _, idx = z.abs().topk(K, 1); coeff = torch.gather(z, 1, idx)
        rec = b + torch.einsum('nk,nkd->nd', coeff, Pn.T[idx])
        loss = ((rec - X) ** 2).mean()
        if step in CHECK:
            curve[step] = round(loss.item(), 5)
        if step < steps:
            opt.zero_grad(); loss.backward(); opt.step()
    return (Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)).detach(), We.detach(), b.detach(), curve


class CodeNorm(nn.Module):
    def __init__(self, tp):
        super().__init__(); self.tp = tp
    def forward(self, x):
        h = rms(x)
        if self.tp is None:
            return h
        Phi, We, b = self.tp; shp = h.shape; Y = h.reshape(-1, D)
        z = (Y - b) @ We
        _, idx = z.abs().topk(K, 1)
        Psup = Phi[:, idx].permute(1, 2, 0)
        G = torch.bmm(Psup, Psup.transpose(1, 2))
        rhs = torch.bmm(Psup, (Y - b).unsqueeze(-1))
        c = torch.linalg.solve(G + 1e-4 * torch.eye(K, device=DEV), rhs)
        return (b + torch.bmm(c.transpose(1, 2), Psup).squeeze(1)).reshape(shp)


orig = [l.norm for l in m.layers]
res = {'baseline_ce': round(CE0, 4), 'm': M, 'k': K, 'curves': {}, 'final_dce': {}}
print(f'baseline CE {CE0:.4f}; write-init vs random-init training (m={M}, k={K}):', flush=True)
for initkind in ['write', 'random']:
    tps = {0: None}
    curves = {}
    for li in (1, 2, 3):
        X = H[li]
        if initkind == 'write':
            gg = torch.Generator(device='cpu').manual_seed(li)
            init0 = unit(torch.cat(DELTA[:li], 0)[torch.randperm(torch.cat(DELTA[:li], 0).shape[0], generator=gg)[:M]].clone())
        else:
            init0 = unit(torch.randn(M, D, generator=torch.Generator(device='cpu').manual_seed(100 + li)).to(DEV))
        Phi, We, b, curve = train(X, init0, li)
        tps[li] = (Phi, We, b); curves[f'bond{li}'] = curve
    for li, l in enumerate(m.layers):
        l.norm = CodeNorm(tps[li])
    dce = ce() - CE0
    for l, nm in zip(m.layers, orig):
        l.norm = nm
    res['curves'][initkind] = curves
    res['final_dce'][initkind] = round(dce, 4)
    print(f'  {initkind:6s}: loss@steps ' +
          ' '.join(f'{s}:{curves["bond2"][s]:.4f}' for s in CHECK) +
          f'  -> final ΔCE {dce:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/toy_births_init_test.json', 'w'), indent=2)
dw, dr = res['final_dce']['write'], res['final_dce']['random']
print(f'\nfinal ΔCE: write-init {dw:+.4f} vs random-init {dr:+.4f}  '
      f'(write-init better: {dw < dr})', flush=True)
print('births init test done', flush=True)
