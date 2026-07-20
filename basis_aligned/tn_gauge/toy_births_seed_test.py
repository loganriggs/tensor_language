"""Regime 2, first step — the UN-CONFOUNDED atom-birth test (Logan 2026-07-20).
Logan's step 2: birth atoms from weight-side WRITE directions. Logan's step-4 concern:
the earlier write-span diagnostic (F4/flagship) was confounded because Φ was TRAINED on
activations (silently absorbing manufactured features). Fix: don't train Φ — SEED its
atoms and leave them fixed, then compare seedings. If bond activations sparse-code better
in write-seeded atoms than in boundary-derived (token) atoms or random atoms, manufactured
features ARE write directions and weight-informed births work — un-confounded (atoms are
seeds, never trained, so nothing is silently absorbed).

Per bond ℓ, fixed dictionary of m atoms, sparse code k (correlation top-k + LS refit),
report FVU for three seedings:
  WRITE  = sampled upstream write vectors (the deltas each upstream layer adds), unit-norm
  TOKEN  = sampled token embeddings (boundary/rotation-stage dictionary, "no births")
  RANDOM = random unit vectors
Toy block2, real TinyStories. NOTE: atoms are FIXED seeds (not optimized) — this measures
seeding quality, the birth hypothesis, not a trained dictionary's ceiling.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
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
IDX = torch.from_numpy(buf).to(DEV)[:, :-1]
rms = lambda x: F.rms_norm(x, (x.size(-1),))

# collect bond inputs H[li] and per-layer write deltas
H, DELTA = [], []
with torch.no_grad():
    x = m.embed(IDX)
    for layer in m.layers:
        xb = x; H.append(rms(xb).reshape(-1, D)); x = layer(x); DELTA.append((x - xb).reshape(-1, D))
Etok = rms(m.embed.weight.data.float())
M, K = 512, 16
g = torch.Generator(device='cpu').manual_seed(0)


def unit(A):
    return A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)


def sample(A, mm, seed):
    gg = torch.Generator(device='cpu').manual_seed(seed)
    return unit(A[torch.randperm(A.shape[0], generator=gg)[:mm]].clone())


@torch.no_grad()
def fvu_fixed_dict(Hc, Dict, k):
    """sparse-code rows of Hc in FIXED atoms Dict (m,D): corr top-k support + LS refit."""
    mu = Hc.mean(0)
    Y = Hc - mu
    z = Y @ Dict.T                                   # (N, m) correlations
    _, idx = z.abs().topk(k, 1)
    Psup = Dict[idx].transpose(1, 2)                 # (N, D, k)
    G = torch.bmm(Psup.transpose(1, 2), Psup)        # (N, k, k)
    rhs = torch.bmm(Psup.transpose(1, 2), Y.unsqueeze(-1))
    c = torch.linalg.solve(G + 1e-4 * torch.eye(k, device=DEV), rhs)
    rec = torch.bmm(Psup, c).squeeze(-1)
    return ((rec - Y) ** 2).sum().item() / (Y ** 2).sum().item()


res = {'m': M, 'k': K, 'per_bond': {}}
NSEED = 5
print(f'un-confounded births test (m={M} fixed seed atoms, k={K}, {NSEED} subsamples); '
      f'FVU per bond (mean+-std):', flush=True)
print('  bond | WRITE-seed | TOKEN-seed | RANDOM-seed', flush=True)


def avg_fvu(source, ell, base):
    vals = [fvu_fixed_dict(H[ell], sample(source, M, 100 * base + s)
                           if source is not None else unit(torch.randn(M, D,
                           generator=torch.Generator(device='cpu').manual_seed(100 * base + s)).to(DEV)), K)
            for s in range(NSEED)]
    return float(np.mean(vals)), float(np.std(vals))


for ell in range(len(SPEC)):
    tw, ts = avg_fvu(Etok, ell, 1)
    rw, rs = avg_fvu(None, ell, 3)
    if ell == 0:
        ww_, ws_ = float('nan'), 0.0
    else:
        ww_, ws_ = avg_fvu(torch.cat(DELTA[:ell], 0), ell, 2)
    res['per_bond'][f'bond{ell}({SPEC[ell]})'] = {
        'write': None if ell == 0 else [round(ww_, 4), round(ws_, 4)],
        'token': [round(tw, 4), round(ts, 4)], 'random': [round(rw, 4), round(rs, 4)]}
    wstr = '  n/a       ' if ell == 0 else f'{ww_:.3f}±{ws_:.3f}'
    print(f'  {ell}({SPEC[ell][0]})  | {wstr} | {tw:.3f}±{ts:.3f} | {rw:.3f}±{rs:.3f}', flush=True)
json.dump(res, open(f'{OUT}/toy_births_seed_test.json', 'w'), indent=2)
# verdict
deep = [f'bond{e}({SPEC[e]})' for e in range(1, len(SPEC))]
ww = np.mean([res['per_bond'][b]['write'][0] for b in deep])
tt = np.mean([res['per_bond'][b]['token'][0] for b in deep])
rr = np.mean([res['per_bond'][b]['random'][0] for b in deep])
res['verdict'] = {'write_mean': round(float(ww), 4), 'token_mean': round(float(tt), 4),
                  'random_mean': round(float(rr), 4),
                  'write_beats_token': bool(ww < tt), 'write_beats_random': bool(ww < rr)}
json.dump(res, open(f'{OUT}/toy_births_seed_test.json', 'w'), indent=2)
print(f'\ndeep-bond mean FVU: WRITE {ww:.3f} | TOKEN {tt:.3f} | RANDOM {rr:.3f}', flush=True)
print(f'write beats token: {ww<tt} ; write beats random: {ww<rr}  (birth hypothesis)', flush=True)
print('births seed test done', flush=True)
