"""Layer-1 QK: per-source atom RANK for selection (Logan 2026-07-20, follows F13).
F13 found attn2's QK selection runs on the source graph over {E=embedding, A=attn0 OV out,
M=mlp1 bilinear out}, dominated by M×M. Now decompose each source into atoms (PCA of its
contribution to the normed residual that QK reads) and ask, at the binding metric: how few
atoms of each source does selection need? Project a source onto its top-r principal atoms,
recompute the exact source-block decomposition (rotary included), rebuild the attn2 pattern,
run to logits, ΔCE vs r. r=D is the exact-gauge check (ΔCE=0). Toy block2, real TinyStories.
"""
import json, sys, itertools
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language')
from deep_model import DeepModel
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN = '/workspace/tensor_language/runs_lm/block2-seed0'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
cfg = json.load(open(f'{RUN}/config.json'))
D, NH, SPEC = cfg['d_model'], cfg['n_head'], cfg['spec']
DH = D // NH
VOCAB, N_CTX = cfg['vocab'], cfg['n_ctx']
m = DeepModel(VOCAB, D, NH, SPEC, N_CTX, norm=(cfg['norm'] == 'rms'),
              residual=cfg['residual'], attention=cfg['attention']).to(DEV)
sd = torch.load(f'{RUN}/model.pt', map_location=DEV)
m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) and 'model' in sd else sd)
m.eval()
for p in m.parameters():
    p.requires_grad_(False)
val = np.memmap('/workspace/tensor_language/data_text/val.bin', dtype=np.uint16, mode='r')
buf = np.stack([val[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(48)]).astype(np.int64)
B = torch.from_numpy(buf).to(DEV); IDX, TGT = B[:, :-1], B[:, 1:]
T = IDX.shape[1]
MASK = torch.tril(torch.ones(T, T, device=DEV))
rms = lambda x: F.rms_norm(x, (D,))
A2 = m.layers[2]; rot = A2.rotary
SRC = ['E', 'A', 'M']


def hd(x):
    return x.view(x.shape[0], x.shape[1], NH, DH)


def qk(lin, hn):
    return rot(hd(lin(hn)))


with torch.no_grad():
    x0 = m.embed(IDX); x1 = m.layers[0](x0); x2 = m.layers[1](x1)
    nx2 = rms(x2); s = nx2.norm(dim=-1, keepdim=True) / x2.norm(dim=-1, keepdim=True)
    PARTS0 = {'E': 0.5 * x0 * s, 'A': (x1 - 0.5 * x0) * s, 'M': (x2 - x1) * s}


def pca_basis(P):
    """top principal directions of the source's normed contribution (D x D, ordered)."""
    X = P.reshape(-1, D).double()
    U, S, Vh = torch.linalg.svd(X - X.mean(0), full_matrices=False)
    return Vh.float(), X.mean(0).float()               # (D,D) rows=components, mean


BASES = {p: pca_basis(PARTS0[p]) for p in SRC}


def project(P, p, r):
    if r >= D:
        return P
    Vh, mu = BASES[p]
    Xc = P.reshape(-1, D) - mu
    Xr = (Xc @ Vh[:r].T) @ Vh[:r] + mu
    return Xr.view_as(P)


@torch.no_grad()
def dce_for(parts):
    Q1 = {p: qk(A2.q1, parts[p]) for p in SRC}; K1 = {p: qk(A2.k1, parts[p]) for p in SRC}
    Q2 = {p: qk(A2.q2, parts[p]) for p in SRC}; K2 = {p: qk(A2.k2, parts[p]) for p in SRC}
    s1 = sum(torch.einsum('bqnh,bknh->bnqk', Q1[a], K1[b]) for a, b in itertools.product(SRC, SRC))
    s2 = sum(torch.einsum('bqnh,bknh->bnqk', Q2[a], K2[b]) for a, b in itertools.product(SRC, SRC))
    pat = (s1 * s2) / DH**2 * MASK
    v2 = hd(A2.v(rms(x2)))
    z = torch.einsum('bnqk,bknh->bqnh', pat, v2).reshape(x2.shape[0], T, D)
    xo = torch.lerp(x2, A2.o(z), A2.scale)
    lg = m.head(m.layers[3](xo)).float()
    return F.cross_entropy(lg.reshape(-1, VOCAB), TGT.reshape(-1)).item()


with torch.no_grad():
    CEreal = F.cross_entropy(m(IDX).float().reshape(-1, VOCAB), TGT.reshape(-1)).item()
CE0 = dce_for(PARTS0)
res = {'ce_real': round(CEreal, 4), 'ce_full': round(CE0, 4), 'gate_dce': round(CE0 - CEreal, 6),
       'rank_sweep': {}}
print(f'CE real {CEreal:.4f}; full-parts {CE0:.4f} (gate {CE0-CEreal:+.2e})', flush=True)
RANKS = [2, 4, 8, 16, 32, 64, 128]
for p in SRC:
    print(f'\nsource {p}: ΔCE vs rank (other sources full):', flush=True)
    row = {}
    for r in RANKS:
        parts = dict(PARTS0); parts[p] = project(PARTS0[p], p, r)
        d = dce_for(parts) - CE0
        row[r] = round(d, 4)
        print(f'  r={r:3d}: ΔCE {d:+.4f}', flush=True)
    res['rank_sweep'][p] = row
    json.dump(res, open(f'{OUT}/toy_qk1_source_rank.json', 'w'), indent=2)

# combined: M and E each at a small rank, A dropped entirely
for rM, rE in [(8, 8), (16, 16), (32, 16), (16, 32)]:
    parts = dict(PARTS0)
    parts['M'] = project(PARTS0['M'], 'M', rM); parts['E'] = project(PARTS0['E'], 'E', rE)
    parts['A'] = project(PARTS0['A'], 'A', 0) if False else PARTS0['A']
    d = dce_for(parts) - CE0
    res['rank_sweep'][f'M{rM}_E{rE}'] = round(d, 4)
    print(f'M rank {rM} + E rank {rE} (A full): ΔCE {d:+.4f}', flush=True)
json.dump(res, open(f'{OUT}/toy_qk1_source_rank.json', 'w'), indent=2)
print('\nqk1 source rank done -> toy_qk1_source_rank.json', flush=True)
