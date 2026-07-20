"""TN-gauge primitives on a TOY bilinear transformer (Logan 2026-07-20).

Before building any DMRG-style interaction-sparsity sweep on bilin18, verify the
GAUGE PRIMITIVES are (a) exactly output-preserving and (b) do not blow up, on a
real trained toy: runs_lm/block2-seed0 = [attn, mlp, attn, mlp], d=128, 4 heads,
d_head=32, d_hidden=512, vocab 1024, RMSNorm, lerp residual. Fully polynomial.

The proposal framed this as "choose a basis per RESIDUAL BOND to sparsify
adjacent interaction cores, sweep DMRG-style between two fixed boundaries
(embedding, unembedding)." This script pressure-tests that framing and finds the
structure is different in a load-bearing way:

  CHECK A  global residual rotation Q is an exact gauge (RMSNorm-equivariant)
  CHECK B  ...but pinning the embedding basis (rank d) forces Q = I: the shared
           residual bond has ZERO middle freedom. The two boundaries do not
           bracket a free interior -- they pin the whole trunk.
  CHECK C  the real, INDEPENDENT freedoms are per-layer PRIVATE bonds:
           C1 attention OV head-subspace = full O(d_head), exact gauge
           C2 attention QK head-subspace = constrained (RoPE breaks a free rot)
           C3 MLP hidden basis: rotation BREAKS output (elementwise * pins it);
              only permutation+scaling is free
  CHECK D  one real L1-minimizing OV gauge (Cayley steps): L1 drops, CE unchanged
           -> the primitive optimizes and does not blow up
  CHECK E  cross-layer interaction DAG (composition norms in the FIXED residual
           basis) is sparse -> "which layers interact" is a weight-only readout,
           decoupled from the within-layer gauges.

Everything is exact arithmetic on a 128-dim model; runs in seconds on CPU/GPU.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language')
from deep_model import DeepModel, SPECS

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN = '/workspace/tensor_language/runs_lm/block2-seed0'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
cfg = json.load(open(f'{RUN}/config.json'))
D, NH = cfg['d_model'], cfg['n_head']
DH = D // NH
SPEC = cfg['spec']
VOCAB, N_CTX = cfg['vocab'], cfg['n_ctx']

m = DeepModel(VOCAB, D, NH, SPEC, N_CTX, norm=(cfg['norm'] == 'rms'),
              residual=cfg['residual'], attention=cfg['attention']).to(DEV)
sd = torch.load(f'{RUN}/model.pt', map_location=DEV)
sd = sd.get('model', sd) if isinstance(sd, dict) and 'model' in sd else sd
m.load_state_dict(sd)
m.eval()
for p in m.parameters():
    p.requires_grad_(False)

# fixed random token windows. Gauge INVARIANCE is exact for any input, so the
# check needs no real corpus; CE here is on random tokens (baseline is high) and
# only serves as a coarse "did anything change" -- the binding metric is max|Δlogit|.
gtok = torch.Generator(device='cpu').manual_seed(0)
B = torch.randint(0, VOCAB, (16, N_CTX + 1), generator=gtok).to(DEV)
IDX, TGT = B[:, :-1], B[:, 1:]


@torch.no_grad()
def logits_ce(model):
    lg = model(IDX).float()
    ce = F.cross_entropy(lg.reshape(-1, VOCAB), TGT.reshape(-1)).item()
    return lg, ce


LG0, CE0 = logits_ce(m)
res = {'run': RUN, 'baseline_ce': round(CE0, 5), 'checks': {}}
print(f'toy loaded: {SPEC}  baseline CE {CE0:.5f}', flush=True)


def maxdiff(model):
    lg, ce = logits_ce(model)
    return (lg - LG0).abs().max().item(), ce


def rand_orth(n, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diagonal(R)))   # unique orthogonal
    return Q.to(DEV)


def clone_model():
    m2 = DeepModel(VOCAB, D, NH, SPEC, N_CTX, norm=(cfg['norm'] == 'rms'),
                   residual=cfg['residual'], attention=cfg['attention']).to(DEV)
    m2.load_state_dict(m.state_dict())
    m2.eval()
    for p in m2.parameters():
        p.requires_grad_(False)
    return m2


def W_(layer, name):
    return getattr(layer, name).weight.data


# ---------- CHECK A: global residual rotation is an exact gauge ----------
Q = rand_orth(D, 1)
mA = clone_model()
mA.embed.weight.data = m.embed.weight.data @ Q.T
mA.head.weight.data = m.head.weight.data @ Q.T
for li, kind in enumerate(SPEC):
    L = mA.layers[li]
    if kind == 'attn':
        for nm in ('q1', 'k1', 'q2', 'k2', 'v'):
            W_(L, nm).copy_(getattr(m.layers[li], nm).weight.data @ Q.T)
        W_(L, 'o').copy_(Q @ m.layers[li].o.weight.data)
    else:
        for nm in ('L', 'R'):
            W_(L, nm).copy_(getattr(m.layers[li], nm).weight.data @ Q.T)
        W_(L, 'D').copy_(Q @ m.layers[li].D.weight.data)
dA, ceA = maxdiff(mA)
res['checks']['A_global_residual_gauge'] = {'max_logit_diff': dA, 'ce': round(ceA, 5),
                                            'exact': bool(dA < 1e-3)}
print(f'A global residual Q: max|Δlogit| {dA:.2e}  CE {ceA:.5f}  -> {"EXACT gauge" if dA<1e-3 else "NOT invariant"}', flush=True)

# ---------- CHECK B: pinning embedding kills the residual freedom ----------
# embed is (V, d) with V>d; its rank = d, so the only Q with embed @ Q.T == embed
# is Q = I. The shared residual bond has no interior gauge once the ends are fixed.
emb_rank = int(torch.linalg.matrix_rank(m.embed.weight.data.float()).item())
head_rank = int(torch.linalg.matrix_rank(m.head.weight.data.float()).item())
# residual gauge group dim that FIXES embed: dim{ Q in O(d): Q E^T = E^T } = 0 when rank=d
res['checks']['B_boundary_pins_residual'] = {
    'embed_rank': emb_rank, 'd_model': D, 'head_rank': head_rank,
    'residual_free_dofs_after_pinning_embed': 0 if emb_rank == D else '(d-rank interior)',
    'note': 'embed rank == d_model => only Q=I keeps embed fixed => no residual middle gauge'}
print(f'B embed rank {emb_rank}/{D}, head rank {head_rank}/{D} -> residual interior gauge DOFs = '
      f'{0 if emb_rank==D else "nonzero"}', flush=True)

# ---------- CHECK C1: attention OV head subspace = full O(d_head), exact ----------
attn_layers = [i for i, k in enumerate(SPEC) if k == 'attn']
li = attn_layers[0]
mC1 = clone_model()
Rh = [rand_orth(DH, 10 + h) for h in range(NH)]
v_new = m.layers[li].v.weight.data.clone()
o_new = m.layers[li].o.weight.data.clone()
for h in range(NH):
    sl = slice(h * DH, (h + 1) * DH)
    v_new[sl, :] = Rh[h] @ m.layers[li].v.weight.data[sl, :]     # rotate value rows
    o_new[:, sl] = m.layers[li].o.weight.data[:, sl] @ Rh[h].T   # unrotate o input cols
mC1.layers[li].v.weight.data.copy_(v_new)
mC1.layers[li].o.weight.data.copy_(o_new)
dC1, ceC1 = maxdiff(mC1)
res['checks']['C1_OV_head_gauge'] = {'max_logit_diff': dC1, 'exact': bool(dC1 < 1e-3)}
print(f'C1 OV per-head O(d_head): max|Δlogit| {dC1:.2e}  -> {"EXACT gauge" if dC1<1e-3 else "NOT invariant"}', flush=True)

# ---------- CHECK C2: attention QK head rotation is broken by RoPE ----------
mC2 = clone_model()
q_new = m.layers[li].q1.weight.data.clone()
k_new = m.layers[li].k1.weight.data.clone()
for h in range(NH):
    sl = slice(h * DH, (h + 1) * DH)
    q_new[sl, :] = Rh[h] @ m.layers[li].q1.weight.data[sl, :]
    k_new[sl, :] = Rh[h] @ m.layers[li].k1.weight.data[sl, :]    # same R on q,k preserves q.k PRE-rope
mC2.layers[li].q1.weight.data.copy_(q_new)
mC2.layers[li].k1.weight.data.copy_(k_new)
dC2, ceC2 = maxdiff(mC2)
# control: WITHOUT rope it would be exact; measure that by rotating in a rope-commuting way (identity here)
res['checks']['C2_QK_head_rotation_vs_rope'] = {
    'max_logit_diff': dC2, 'exact_gauge': bool(dC2 < 1e-3),
    'note': 'same R on q,k preserves the raw dot product but NOT after RoPE -> QK basis is rope-constrained'}
print(f'C2 QK per-head rotation (pre-RoPE dot preserved): max|Δlogit| {dC2:.2e}  -> '
      f'{"still exact (rope-commuting)" if dC2<1e-3 else "BROKEN by RoPE (constrained freedom)"}', flush=True)

# ---------- CHECK C3: MLP hidden basis -- rotation breaks, perm+scale preserves ----------
mlp_layers = [i for i, k in enumerate(SPEC) if k == 'mlp']
lj = mlp_layers[0]
H = m.layers[lj].L.weight.data.shape[0]
# C3a rotation of hidden units
Mrot = rand_orth(H, 99)
mC3a = clone_model()
mC3a.layers[lj].L.weight.data.copy_(Mrot @ m.layers[lj].L.weight.data)
mC3a.layers[lj].R.weight.data.copy_(Mrot @ m.layers[lj].R.weight.data)
mC3a.layers[lj].D.weight.data.copy_(m.layers[lj].D.weight.data @ Mrot.T)
dC3a, _ = maxdiff(mC3a)
# C3b permutation + scaling
g = torch.Generator(device='cpu').manual_seed(7)
perm = torch.randperm(H, generator=g).to(DEV)
a = (0.5 + torch.rand(H, generator=g)).to(DEV)          # L scale
b_ = (0.5 + torch.rand(H, generator=g)).to(DEV)         # R scale
mC3b = clone_model()
mC3b.layers[lj].L.weight.data.copy_((a[:, None] * m.layers[lj].L.weight.data)[perm])
mC3b.layers[lj].R.weight.data.copy_((b_[:, None] * m.layers[lj].R.weight.data)[perm])
Dcols = m.layers[lj].D.weight.data / (a * b_)[None, :]
mC3b.layers[lj].D.weight.data.copy_(Dcols[:, perm])
dC3b, _ = maxdiff(mC3b)
res['checks']['C3_mlp_hidden_pinned_by_elementwise'] = {
    'rotation_max_logit_diff': dC3a, 'rotation_breaks_output': bool(dC3a > 1e-2),
    'perm_scale_max_logit_diff': dC3b, 'perm_scale_exact': bool(dC3b < 1e-3),
    'note': 'elementwise * pins hidden basis: only permutation+scaling is a gauge'}
print(f'C3 MLP hidden: rotation max|Δlogit| {dC3a:.2e} ({"BREAKS (pinned)" if dC3a>1e-2 else "?"}) ; '
      f'perm+scale {dC3b:.2e} ({"EXACT" if dC3b<1e-3 else "?"})', flush=True)

# ---------- CHECK D: a real L1-minimizing OV gauge -- optimizes, no blow-up ----------
# layer-0 head-0 token->value table VT = E_hat @ v_h^T   (V x d_head). Rotate its
# columns by R in O(d_head) to minimize ||VT R^T||_1 (varimax-type). Cayley steps.
E_hat = F.rms_norm(m.embed.weight.data.float(), (D,))
h0 = 0
vh = m.layers[li].v.weight.data.float()[h0 * DH:(h0 + 1) * DH, :]     # (d_head, d)
VT = E_hat @ vh.T                                                     # (V, d_head)
l1_0 = VT.abs().sum().item() / VT.numel()
R = torch.eye(DH, device=DEV)
lr = 0.5
for step in range(300):
    M = (VT @ R.T)
    # subgradient of mean|.| wrt R:  d/dR sum|VT R^T| = sign(M)^T VT
    G = torch.sign(M).T @ VT / VT.numel()                            # (d_head, d_head)
    # project to skew (tangent of O(n)) and take a Cayley retraction step
    A = G @ R.T
    A = 0.5 * (A - A.T)
    Cay = torch.linalg.solve(torch.eye(DH, device=DEV) + lr * A,
                             torch.eye(DH, device=DEV) - lr * A)
    R = Cay @ R
l1_1 = (VT @ R.T).abs().sum().item() / VT.numel()
orth_err = (R @ R.T - torch.eye(DH, device=DEV)).abs().max().item()
# apply this gauge to the model and confirm CE unchanged (it is a gauge)
mD = clone_model()
v_new = m.layers[li].v.weight.data.clone()
o_new = m.layers[li].o.weight.data.clone()
sl = slice(h0 * DH, (h0 + 1) * DH)
v_new[sl, :] = R.to(v_new.dtype) @ m.layers[li].v.weight.data[sl, :]
o_new[:, sl] = m.layers[li].o.weight.data[:, sl] @ R.T.to(o_new.dtype)
mD.layers[li].v.weight.data.copy_(v_new)
mD.layers[li].o.weight.data.copy_(o_new)
dD, ceD = maxdiff(mD)
res['checks']['D_l1_ov_gauge'] = {
    'l1_before': round(l1_0, 5), 'l1_after': round(l1_1, 5),
    'l1_drop_pct': round(100 * (1 - l1_1 / l1_0), 1),
    'orth_error': orth_err, 'ce_after_applying': round(ceD, 5),
    'max_logit_diff': dD, 'ce_unchanged': bool(dD < 1e-3)}
print(f'D L1 OV gauge: mean|VT| {l1_0:.4f} -> {l1_1:.4f} ({100*(1-l1_1/l1_0):.1f}% drop), '
      f'orth_err {orth_err:.1e}, CE {ceD:.5f} (Δlogit {dD:.1e})', flush=True)

# ---------- CHECK E: cross-layer interaction DAG (fixed residual basis) ----------
# writer directions: columns of o (attn) / D (mlp) live in residual space.
# reader directions: rows of q1,k1,v (attn) / L,R (mlp) read residual space.
# composition(writer i -> reader j) = ||Wread_j @ Wwrite_i||_F normalized. Sparse?
writers, readers = {}, {}
for i, kind in enumerate(SPEC):
    Lr = m.layers[i]
    if kind == 'attn':
        writers[i] = Lr.o.weight.data.float()                        # (d, d) cols=write dirs
        readers[i] = torch.cat([Lr.q1.weight.data, Lr.k1.weight.data,
                                Lr.v.weight.data], 0).float()         # (3d, d) rows=read dirs
    else:
        writers[i] = Lr.D.weight.data.float()                        # (d, h)
        readers[i] = torch.cat([Lr.L.weight.data, Lr.R.weight.data], 0).float()  # (2h, d)
comp = np.zeros((len(SPEC), len(SPEC)))
for i in writers:
    Wi = writers[i]
    wn = Wi.norm()
    for j in readers:
        if j <= i:
            continue
        Rj = readers[j]
        c = (Rj @ Wi).norm().item() / (Rj.norm().item() * wn.item() + 1e-9)
        comp[i, j] = c
off = comp[np.triu_indices(len(SPEC), 1)]
off = off[off > 0]
res['checks']['E_cross_layer_dag'] = {
    'spec': SPEC,
    'composition_matrix': [[round(float(x), 4) for x in row] for row in comp],
    'max': round(float(off.max()), 4), 'min': round(float(off.min()), 4),
    'ratio_max_min': round(float(off.max() / off.min()), 1)}
print('E cross-layer composition (writer i -> reader j), normalized:', flush=True)
for i in range(len(SPEC)):
    print('   ' + ' '.join(f'{comp[i,j]:.3f}' for j in range(len(SPEC))), flush=True)
print(f'   spread max/min over ordered pairs = {off.max()/off.min():.1f}x', flush=True)

json.dump(res, open(f'{OUT}/toy_gauge_probe.json', 'w'), indent=2)
print('\ntoy gauge probe done -> toy_gauge_probe.json', flush=True)
