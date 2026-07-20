"""Regime 2 at the BINDING metric (Logan 2026-07-20). F9 showed write-seeded atoms
reconstruct the deep stream better than token/random SEEDS (un-confounded). But this
program's recurring lesson is reconstruction != behavior, so the binding test is ΔCE:
apply a FIXED seeded dictionary at every bond (bond 0 exact by lookup per Logan's
calibration b) and measure end-to-end cross-entropy. Does write-seeding win at ΔCE?

Per bond ℓ>=1, fixed dictionary of m atoms, sparse code k (corr top-k + LS refit),
three seedings compared at matched (m,k): WRITE (upstream write deltas), TOKEN
(embedding), RANDOM. Atoms are fixed seeds, never trained -> un-confounded.
Toy block2, real TinyStories.
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
# collect write deltas (for seeding the dictionaries)
DELTA = []
with torch.no_grad():
    x = m.embed(IDX)
    for layer in m.layers:
        xb = x; x = layer(x); DELTA.append((x - xb).reshape(-1, D))
Etok = rms(m.embed.weight.data.float())
M, K = 512, 32


def unit(A):
    return A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)


def sample(A, seed):
    gg = torch.Generator(device='cpu').manual_seed(seed)
    return unit(A[torch.randperm(A.shape[0], generator=gg)[:M]].clone())


def build_dicts(kind, seed):
    dicts = {}
    for li in range(len(SPEC)):
        if li == 0:
            dicts[li] = None                     # bond 0 exact (Logan calibration b)
        elif kind == 'write':
            dicts[li] = sample(torch.cat(DELTA[:li], 0), seed)
        elif kind == 'token':
            dicts[li] = sample(Etok, seed)
        else:
            dicts[li] = unit(torch.randn(M, D, generator=torch.Generator(device='cpu').manual_seed(seed)).to(DEV))
    return dicts


class CodeNorm(nn.Module):
    def __init__(self, Dict):
        super().__init__(); self.Dict = Dict
    def forward(self, x):
        h = rms(x)
        if self.Dict is None:
            return h
        shp = h.shape; Y = h.reshape(-1, D)
        Yc = Y - Y.mean(0)
        z = Yc @ self.Dict.T
        _, idx = z.abs().topk(K, 1)
        Psup = self.Dict[idx].transpose(1, 2)
        G = torch.bmm(Psup.transpose(1, 2), Psup)
        rhs = torch.bmm(Psup.transpose(1, 2), Yc.unsqueeze(-1))
        c = torch.linalg.solve(G + 1e-4 * torch.eye(K, device=DEV), rhs)
        return (Y.mean(0) + torch.bmm(Psup, c).squeeze(-1)).reshape(shp)


orig = [l.norm for l in m.layers]
res = {'baseline_ce': round(CE0, 4), 'm': M, 'k': K, 'arms': {}}
print(f'baseline CE {CE0:.4f}; regime-2 ΔCE (bond 0 exact, bonds 1-3 coded, m={M}, k={K}):', flush=True)
for kind in ['write', 'token', 'random']:
    dces = []
    for seed in range(3):
        dicts = build_dicts(kind, 10 * seed + 1)
        for li, l in enumerate(m.layers):
            l.norm = CodeNorm(dicts[li])
        dces.append(ce() - CE0)
        for l, nm in zip(m.layers, orig):
            l.norm = nm
    res['arms'][kind] = {'dce_mean': round(float(np.mean(dces)), 4), 'dce_std': round(float(np.std(dces)), 4)}
    print(f'  {kind:6s}: ΔCE {np.mean(dces):+.4f} ± {np.std(dces):.4f}', flush=True)
    json.dump(res, open(f'{OUT}/toy_births_dce_test.json', 'w'), indent=2)
w, t, r = (res['arms'][k]['dce_mean'] for k in ('write', 'token', 'random'))
res['write_beats_token'] = bool(w < t); res['write_beats_random'] = bool(w < r)
json.dump(res, open(f'{OUT}/toy_births_dce_test.json', 'w'), indent=2)
print(f'\nwrite beats token at ΔCE: {w < t} ; beats random: {w < r}  '
      f'(does the reconstruction win survive the binding metric?)', flush=True)
print('births ΔCE test done', flush=True)
