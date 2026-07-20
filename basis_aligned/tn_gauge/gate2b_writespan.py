"""Gate 2b: the residual-in-write-span diagnostic (Logan 2026-07-20).
Refines the theory: additivity forces COMPATIBILITY (bond-l atoms exist at l+1),
not identity -> nested growing dictionary Phi_{l+1} >= Phi_l (shared core + per-bond
atom BIRTHS). Births are necessary because layers manufacture features with no
preimage at bond 0 ("prev token was X, through L0 OV"). Step-4 closure assumed NO
births; the depth-degrading FVU is that assumption failing.

Decisive test (distinguishes atom-birth vs capacity vs regime-suspect BEFORE the
capacity sweep can): take each bond's coding residual r_l = h_l - Phi c_l and ask how
much of its variance lies in the span of the UPSTREAM layers' WRITE mechanisms (the
actual residual-stream deltas each upstream layer adds) versus the TOKEN-embedding
span versus a random subspace of equal dimension.
  residual mostly in WRITE span, >> token span  -> ATOM-BIRTH (regime survives; fix =
    birth atoms from write directions, weight-derived, TN-pure)
  residual in TOKEN span                          -> capacity (just need more atoms)
  residual in NEITHER                             -> regime suspect

Also calibration (b): bond 0 should be EXACT by token lookup (not coded). Re-measure
end-to-end ΔCE with bond 0 exact vs all-coded.
Toy block2, real TinyStories, shared Phi m=2048, LS-refit coeffs.
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
# collect per-bond normalized inputs h[li] and per-layer raw write deltas d[li]
H, DELTA = [], []
with torch.no_grad():
    x = m.embed(IDX)
    for li, layer in enumerate(m.layers):
        xb = x
        H.append(rms(xb).reshape(-1, D))
        x = layer(x)
        DELTA.append((x - xb).reshape(-1, D))     # write of layer li into the stream
print(f'baseline CE {CE0:.4f}; collected {H[0].shape[0]} tokens x {len(SPEC)} bonds', flush=True)

M = 2048
POOL = torch.cat(H, 0)


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


Phi, We, b = train_dict(POOL, M)


@torch.no_grad()
def encode(h, k):
    z = (h - b) @ We
    _, idx = z.abs().topk(k, 1)
    Psup = Phi[:, idx].permute(1, 2, 0)
    G = torch.bmm(Psup, Psup.transpose(1, 2))
    rhs = torch.bmm(Psup, (h - b).unsqueeze(-1))
    c = torch.linalg.solve(G + 1e-4 * torch.eye(k, device=DEV), rhs)
    return b + torch.bmm(c.transpose(1, 2), Psup).squeeze(1)


def top_basis(X, K):
    """top-K left-singular directions (uncentered) of rows of X, as (D,K) orthonormal."""
    U, S, Vh = torch.linalg.svd(X.double() - 0, full_matrices=False)
    return Vh[:K].T.float()                          # (D,K): row-space directions


def captured(r, U):
    """fraction of ||r||^2 lying in subspace spanned by orthonormal cols of U."""
    return ((r @ U) ** 2).sum().item() / (r ** 2).sum().item()


# token-embedding subspace (normalized embedding rows)
Etok = rms(m.embed.weight.data.float())
res = {'baseline_ce': round(CE0, 4), 'm': M, 'k': 32, 'writespan': {}, 'per_upstream_layer': {},
       'calibration_bond0_exact': {}}
KPROJ = 32
Utok = top_basis(Etok, KPROJ)
def eff_rank(X):
    s = torch.linalg.svdvals(X.double())
    p = s / s.sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


print('\ngate2b residual-in-span (k=32 codes; K=32 subspaces): '
      'frac of bond residual variance captured', flush=True)
print('  bond | write-span | token-span | random | SELF(ceiling) | eff-rank', flush=True)
for ell in range(len(SPEC)):
    r = H[ell] - encode(H[ell], 32)
    if ell == 0:
        wcap = float('nan')                          # no upstream writes
    else:
        Wup = torch.cat(DELTA[:ell], 0)              # upstream write vectors
        Uwrite = top_basis(Wup, KPROJ)
        wcap = captured(r, Uwrite)
    tcap = captured(r, Utok)
    self_cap = captured(r, top_basis(r, KPROJ))      # best possible K-dim capture
    er = eff_rank(r)
    er_act = eff_rank(H[ell] - H[ell].mean(0))        # intrinsic dim of the activations
    rnd = KPROJ / D
    res['writespan'][f'bond{ell}({SPEC[ell]})'] = {
        'write_span': None if ell == 0 else round(wcap, 4),
        'token_span': round(tcap, 4), 'random': round(rnd, 4),
        'self_ceiling': round(self_cap, 4), 'eff_rank': round(er, 1), 'act_eff_rank': round(er_act, 1),
        'resid_fvu': round((r ** 2).sum().item() / ((H[ell] - H[ell].mean(0)) ** 2).sum().item(), 4)}
    ws = 'n/a  ' if ell == 0 else f'{wcap:.3f}'
    print(f'  {ell}({SPEC[ell][0]})  |   {ws}    |   {tcap:.3f}    |  {rnd:.3f} |     {self_cap:.3f}     |  {er:.1f}  | act-rank {er_act:.1f}', flush=True)

# per-upstream-layer breakdown for the deepest bond
ell = len(SPEC) - 1
r = H[ell] - encode(H[ell], 32)
print(f'\nper-upstream-layer capture of bond{ell} residual (K=32 each):', flush=True)
for j in range(ell):
    Uj = top_basis(DELTA[j], KPROJ)
    cj = captured(r, Uj)
    res['per_upstream_layer'][f'layer{j}({SPEC[j]})'] = round(cj, 4)
    print(f'  layer {j} ({SPEC[j]}) writes capture {cj:.3f} of bond{ell} residual', flush=True)


# calibration (b): bond 0 exact by lookup, code the rest
class CodeNorm(nn.Module):
    def __init__(self, k, code=True):
        super().__init__(); self.k = k; self.code = code
    def forward(self, x):
        h = rms(x)
        if not self.code:
            return h
        return encode(h.reshape(-1, D), self.k).reshape(h.shape)


orig = [l.norm for l in m.layers]
print('\ncalibration (b) end-to-end ΔCE, k=32:', flush=True)
for tag, code0 in [('all bonds coded', True), ('bond0 EXACT (lookup), rest coded', False)]:
    for li, l in enumerate(m.layers):
        l.norm = CodeNorm(32, code=(code0 or li != 0))
    dce = ce_of(m) - CE0
    for l, nm in zip(m.layers, orig):
        l.norm = nm
    res['calibration_bond0_exact'][tag] = round(dce, 4)
    print(f'  {tag}: ΔCE {dce:+.4f}', flush=True)

json.dump(res, open(f'{OUT}/gate2b_writespan.json', 'w'), indent=2)
print('\ngate2b done -> gate2b_writespan.json', flush=True)
