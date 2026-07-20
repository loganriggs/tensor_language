"""Gate 2b: can the PROPAGATION-COMPATIBLE (shared Φ) regime reach faithfulness?
(Logan 2026-07-20). Additive code propagation (codes flow without re-solving) needs
ONE shared Φ across all bonds (x_{l+1}=x_l+write => Φ(c+w) requires same Φ). But
gate 2 showed shared Φ is the lossy config. Question: does scaling a SHARED Φ reach
end-to-end ΔCE<0.05, or does it plateau (=> propagation regime is capped and you must
re-encode per bond, abandoning regime (c))? Sweep shared m in {2048,4096,8192}, k in
{32,64}. Bits reported. Toy block2, real TinyStories, LS-refit coeffs."""
import json, sys, math
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
bacts = [[] for _ in range(len(SPEC))]
with torch.no_grad():
    x = m.embed(IDX)
    for li, layer in enumerate(m.layers):
        bacts[li].append(rms(x).reshape(-1, D)); x = layer(x)
POOL = torch.cat([torch.cat(a, 0) for a in bacts], 0)
NTOK = POOL.shape[0] // len(SPEC)


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
        loss = ((rec - X) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)).detach(), We.detach(), b.detach()


def make_encode(Phi, We, b):
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


res = {'baseline_ce': round(CE0, 4), 'arms': {}}
print(f'baseline CE {CE0:.4f}', flush=True)
orig = [l.norm for l in m.layers]
for mm in (2048, 4096, 8192):
    Phi, We, b = train_dict(POOL, mm)
    enc = make_encode(Phi, We, b)
    for k in (32, 64):
        for l in m.layers:
            l.norm = CodeNorm(enc, k)
        dce = ce_of(m) - CE0
        for l, nm in zip(m.layers, orig):
            l.norm = nm
        bits = mm * D * 32 + k * math.log2(mm) * NTOK * len(SPEC)
        res['arms'][f'shared m={mm} k={k}'] = {'dce': round(dce, 4), 'Mbits': round(bits / 1e6, 1)}
        print(f'shared m={mm} k={k}: ΔCE {dce:+.4f}  {bits/1e6:.1f}Mbit', flush=True)
        json.dump(res, open(f'{OUT}/toy_shared_scaling.json', 'w'), indent=2)
print('toy shared scaling done', flush=True)
