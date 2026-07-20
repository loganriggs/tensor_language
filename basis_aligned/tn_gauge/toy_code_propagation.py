"""Overcomplete shared-dictionary code propagation on the TOY (Logan 2026-07-20).

Tests the crux of Logan's construction: represent the residual stream by ONE shared
overcomplete dictionary Phi (d x m, unit-norm columns, m>d), encode each bond to a
sparse code c (TopK), and run the model in code coordinates. The gates:

  G1 FAITHFULNESS   a single shared Phi codes EVERY bond's (RMSNorm'd) activation
                    -- FVU per bond vs sparsity k. (Step-4 "one dictionary per
                    stream" claim: does one Phi serve all bonds?)
  G2 END-TO-END CE  replace norm(x) at every layer input with its Phi-sparse
                    reconstruction; measure real ΔCE vs k. This is the propagation
                    regime's ceiling -- if coding every bond destroys the model,
                    the regime is dead. Binding metric.
  G3 AMPLIFICATION  Logan's Step-5 prediction: a degree-2 layer roughly DOUBLES
                    relative error (y = code + 2 T(Phi c, eps) + T(eps,eps)).
                    Measure rel_err_out / rel_err_in per MLP. Falsifiable.

Toy: runs_lm/block2-seed0 = [attn,mlp,attn,mlp], d=128, 4 heads, hidden=512,
vocab 1024, RMSNorm, lerp residual. Real TinyStories val tokens.
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language')
from deep_model import DeepModel

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN = '/workspace/tensor_language/runs_lm/block2-seed0'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
cfg = json.load(open(f'{RUN}/config.json'))
D, NH, SPEC = cfg['d_model'], cfg['n_head'], cfg['spec']
VOCAB, N_CTX = cfg['vocab'], cfg['n_ctx']
M = 512                     # overcomplete dictionary size (4x d)

m = DeepModel(VOCAB, D, NH, SPEC, N_CTX, norm=(cfg['norm'] == 'rms'),
              residual=cfg['residual'], attention=cfg['attention']).to(DEV)
sd = torch.load(f'{RUN}/model.pt', map_location=DEV)
m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) and 'model' in sd else sd)
m.eval()
for p in m.parameters():
    p.requires_grad_(False)

# real TinyStories val tokens
val = np.memmap('/workspace/tensor_language/data_text/val.bin', dtype=np.uint16, mode='r')
NW = 64
buf = np.stack([val[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(NW)]).astype(np.int64)
B = torch.from_numpy(buf).to(DEV)
IDX, TGT = B[:, :-1], B[:, 1:]


@torch.no_grad()
def ce_of(model):
    lg = model(IDX).float()
    return F.cross_entropy(lg.reshape(-1, VOCAB), TGT.reshape(-1)).item()


CE0 = ce_of(m)
print(f'toy {SPEC}  baseline CE {CE0:.4f}  (real TinyStories val)', flush=True)

# ---- collect normalized activations at each layer INPUT bond (what norm() sees) ----
rms = lambda x: F.rms_norm(x, (x.size(-1),))
bond_acts = [[] for _ in range(len(SPEC))]
with torch.no_grad():
    x = m.embed(IDX)
    for li, layer in enumerate(m.layers):
        bond_acts[li].append(rms(x).reshape(-1, D))     # normalized input to this layer
        x = layer(x)
BOND = [torch.cat(a, 0) for a in bond_acts]              # per-bond (Ntok, D)
POOL = torch.cat(BOND, 0)                                # shared-stream pool
print(f'collected {POOL.shape[0]} normalized bond activations across {len(SPEC)} bonds', flush=True)

# ---- learn ONE shared overcomplete dictionary Phi with a TopK linear encoder ----
KTRAIN = 16
g = torch.Generator(device='cpu').manual_seed(0)
sel = torch.randperm(POOL.shape[0], generator=g)[:M]
Phi = POOL[sel].clone().T                               # (D, M) columns = atoms
Phi = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
We = Phi.clone()                                        # (D, M) encoder (read as We^T)
b = POOL.mean(0).clone()
Phi.requires_grad_(True); We.requires_grad_(True); b.requires_grad_(True)
opt = torch.optim.Adam([Phi, We, b], lr=3e-3)
Xtr = POOL
for step in range(2500):
    Pn = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
    z = (Xtr - b) @ We                                  # (N, M)
    val_, idx = z.abs().topk(KTRAIN, 1)
    coeff = torch.gather(z, 1, idx)                     # (N, K)
    rec = b + torch.einsum('nk,nkd->nd', coeff, Pn.T[idx])   # (N, D)
    loss = ((rec - Xtr) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f'  dict step {step}: recon MSE {loss.item():.5f}', flush=True)
Phi = (Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)).detach()
We = We.detach(); b = b.detach()


@torch.no_grad()
def encode(h, k):
    """TopK support from the linear encoder, then LEAST-SQUARES refit of the k
    coefficients on the selected atoms (removes the fixed-k encoder-calibration
    artifact so FVU is monotone in k -- the OMP/LS lesson)."""
    z = (h - b) @ We
    _, idx = z.abs().topk(k, 1)                          # (N, k) support
    Psup = Phi[:, idx].permute(1, 2, 0)                  # (N, k, D)
    y = (h - b)                                          # (N, D)
    G = torch.bmm(Psup, Psup.transpose(1, 2))           # (N, k, k)
    rhs = torch.bmm(Psup, y.unsqueeze(-1))              # (N, k, 1)
    c = torch.linalg.solve(G + 1e-4 * torch.eye(k, device=DEV), rhs)  # (N,k,1)
    rec = b + torch.bmm(c.transpose(1, 2), Psup).squeeze(1)
    return rec


def fvu(h, hhat):
    return (((hhat - h) ** 2).sum() / ((h - h.mean(0)) ** 2).sum()).item()


res = {'baseline_ce': round(CE0, 4), 'm': M, 'spec': SPEC, 'G1_fvu_per_bond': {},
       'G2_end_to_end_dce': {}, 'G3_amplification': {}}

# ---- G1: shared-Phi faithfulness per bond vs k ----
print('\nG1 shared-dictionary faithfulness (FVU) per bond:', flush=True)
for k in (4, 8, 16, 32, 64):
    row = [round(fvu(BOND[li], encode(BOND[li], k)), 4) for li in range(len(SPEC))]
    res['G1_fvu_per_bond'][f'k={k}'] = row
    print(f'  k={k:2d}: ' + '  '.join(f'bond{li}({SPEC[li][0]}) {row[li]:.3f}' for li in range(len(SPEC))), flush=True)


# ---- G2: replace every layer-input norm with Phi-code reconstruction, real ΔCE ----
class CodeNorm(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k = k
    def forward(self, x):
        h = F.rms_norm(x, (x.size(-1),))
        shp = h.shape
        return encode(h.reshape(-1, D), self.k).reshape(shp)


print('\nG2 end-to-end ΔCE with EVERY bond coded (shared Phi):', flush=True)
orig_norms = [layer.norm for layer in m.layers]
for k in (4, 8, 16, 32, 64):
    for layer in m.layers:
        layer.norm = CodeNorm(k)
    dce = ce_of(m) - CE0
    res['G2_end_to_end_dce'][f'k={k}'] = round(dce, 4)
    print(f'  k={k:2d}: ΔCE {dce:+.4f}', flush=True)
for layer, nm in zip(m.layers, orig_norms):
    layer.norm = nm

# ---- G3: per-MLP error amplification (rel_err_out / rel_err_in) ----
print('\nG3 error amplification through each bilinear MLP (Step-5 ~2x prediction):', flush=True)
mlp_layers = [i for i, s in enumerate(SPEC) if s == 'mlp']
for li in mlp_layers:
    layer = m.layers[li]
    h = BOND[li]                                        # true normalized input (Ntok,D)
    core = lambda hh: layer.D(layer.L(hh) * layer.R(hh))
    y_true = core(h)
    for k in (8, 16, 32):
        hhat = encode(h, k)
        y_code = core(hhat)
        rin = (hhat - h).norm() / h.norm()
        rout = (y_code - y_true).norm() / y_true.norm()
        amp = (rout / rin).item()
        res['G3_amplification'].setdefault(f'bond{li}', {})[f'k={k}'] = {
            'rel_err_in': round(rin.item(), 4), 'rel_err_out': round(rout.item(), 4),
            'amplification': round(amp, 2)}
        print(f'  bond{li} k={k:2d}: in {rin.item():.3f} -> out {rout.item():.3f}  '
              f'(x{amp:.2f})', flush=True)

json.dump(res, open(f'{OUT}/toy_code_propagation.json', 'w'), indent=2)
print('\ntoy code propagation done -> toy_code_propagation.json', flush=True)
