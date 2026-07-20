"""Closes the F11 open question (Logan 2026-07-20): write-seeded init trained WORSE than
random because write atoms are CLUSTERED. Is that fixable by orthogonalization (Logan's
own dedup prescription), or is write-information useless once you train? Test a
DE-CLUSTERED write init: the write subspace's principal directions (SVD of upstream write
deltas -> orthonormal, diverse), used as the first atoms, random-filled to m. Compare
final ΔCE and convergence to pure random init and clustered-write init.

  random         : m=512 random unit atoms (F11 winner, +0.35)
  clustered-write: m=512 sampled write deltas (F11 loser, +0.50)
  ortho-write    : [top-128 write-PCA orthonormal] + [384 random]  (diverse + write-informed)
If ortho-write ~ random  -> write-info useless for a trained dict (training finds the subspace).
If ortho-write < random  -> orthogonalization rescues write-info (clustering was the problem).
Toy block2, per bond 1-3, m=512 k=32.
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


def init_for(kind, li):
    g = torch.Generator(device='cpu').manual_seed(li)
    Wup = torch.cat(DELTA[:li], 0)
    if kind == 'random':
        return unit(torch.randn(M, D, generator=torch.Generator(device='cpu').manual_seed(100 + li)).to(DEV))
    if kind == 'clustered-write':
        return unit(Wup[torch.randperm(Wup.shape[0], generator=g)[:M]].clone())
    # ortho-write: top-128 write-PCA directions (diverse) + 384 random
    sub = Wup[torch.randperm(Wup.shape[0], generator=g)[:20000]]
    U, S, Vh = torch.linalg.svd(sub.double() - sub.double().mean(0), full_matrices=False)
    pca = Vh[:D].float()                                   # (128, 128) orthonormal write dirs
    rnd = unit(torch.randn(M - D, D, generator=torch.Generator(device='cpu').manual_seed(200 + li)).to(DEV))
    return torch.cat([pca, rnd], 0)


def train(X, init0, steps=2000):
    Phi = init0.T.clone().contiguous()
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
print(f'baseline CE {CE0:.4f}; F11-closing ortho-init test (m={M} k={K}):', flush=True)
for kind in ['random', 'clustered-write', 'ortho-write']:
    tps = {0: None}; curves = {}
    for li in (1, 2, 3):
        Phi, We, b, curve = train(H[li], init_for(kind, li))
        tps[li] = (Phi, We, b); curves[f'bond{li}'] = curve
    for li, l in enumerate(m.layers):
        l.norm = CodeNorm(tps[li])
    dce = ce() - CE0
    for l, nm in zip(m.layers, orig):
        l.norm = nm
    res['curves'][kind] = curves; res['final_dce'][kind] = round(dce, 4)
    print(f'  {kind:16s}: loss@ ' + ' '.join(f'{s}:{curves["bond2"][s]:.4f}' for s in CHECK) +
          f'  -> ΔCE {dce:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/toy_births_ortho_init_test.json', 'w'), indent=2)
rr, ow = res['final_dce']['random'], res['final_dce']['ortho-write']
res['ortho_beats_random'] = bool(ow < rr - 0.01)
res['verdict'] = ('orthogonalization rescues write-info' if ow < rr - 0.01
                  else 'write-info useless for a trained dict (training finds the subspace)')
json.dump(res, open(f'{OUT}/toy_births_ortho_init_test.json', 'w'), indent=2)
print(f"\nrandom {rr:+.4f} vs ortho-write {ow:+.4f} -> {res['verdict']}", flush=True)
print('ortho init test done', flush=True)
