"""Layer-1 QK interaction graph over upstream sources (Logan 2026-07-20).
Focus: the second attention (attn2 in block2 = [attn0,mlp1,attn2,mlp3]). Its query/key
reads the residual x2 = E + A + M, the three upstream WRITE sources:
  E = 0.5*x0        (embedding contribution, through the lerp)
  A = x1 - 0.5*x0   (attn0 OV output)
  M = x2 - x1       (mlp1 bilinear output)
The QK score is bilinear in the (RMS-normed) residual, so it splits EXACTLY into a 3x3
block structure of source interactions: score_b[q,k] = sum_{pq,pk in {E,A,M}}
q_b(pq_q)·k_b(pk_k), for each branch b (rotary included, since rotary+linear+dot are all
linear/bilinear). This is the COARSEST decomposition (each source = one unit) of what
Logan wants — the source-level QK interaction graph. Measure each block's Frobenius mass
and its causal ΔCE (ablate blocks, keep subsets), to see which interactions the layer-1
selection actually uses. Sets up the finer per-source atom decomposition next.
Gate: sum of 9 blocks == real score (fp32). Toy block2, real TinyStories.
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
A2 = m.layers[2]           # the layer-1 attention
rot = A2.rotary
SRC = ['E', 'A', 'M']


def heads(x):
    return x.view(x.shape[0], x.shape[1], NH, DH)


def qk(lin, hn):
    return rot(heads(lin(hn)))                       # (B,T,NH,DH) with rotary


@torch.no_grad()
def blocks_and_parts():
    x0 = m.embed(IDX)
    x1 = m.layers[0](x0)
    x2 = m.layers[1](x1)
    nx2 = rms(x2)
    s = nx2.norm(dim=-1, keepdim=True) / x2.norm(dim=-1, keepdim=True)   # exact per-token RMS scale
    parts = {'E': 0.5 * x0 * s, 'A': (x1 - 0.5 * x0) * s, 'M': (x2 - x1) * s}
    # per-branch, per-source q/k
    Q1 = {p: qk(A2.q1, parts[p]) for p in SRC}; K1 = {p: qk(A2.k1, parts[p]) for p in SRC}
    Q2 = {p: qk(A2.q2, parts[p]) for p in SRC}; K2 = {p: qk(A2.k2, parts[p]) for p in SRC}
    blk1, blk2 = {}, {}
    for pq, pk in itertools.product(SRC, SRC):
        blk1[(pq, pk)] = torch.einsum('bqnh,bknh->bnqk', Q1[pq], K1[pk])
        blk2[(pq, pk)] = torch.einsum('bqnh,bknh->bnqk', Q2[pq], K2[pk])
    return x0, x1, x2, parts, blk1, blk2


x0, x1, x2, parts, blk1, blk2 = blocks_and_parts()
# GATE: sum of blocks == real score
nx2 = rms(x2)
score1_real = torch.einsum('bqnh,bknh->bnqk', qk(A2.q1, nx2), qk(A2.k1, nx2))
score2_real = torch.einsum('bqnh,bknh->bnqk', qk(A2.q2, nx2), qk(A2.k2, nx2))
s1_sum = sum(blk1.values()); s2_sum = sum(blk2.values())
g1 = (s1_sum - score1_real).abs().max().item(); g2 = (s2_sum - score2_real).abs().max().item()
print(f'GATE sum-of-blocks vs real score: branch1 {g1:.2e}, branch2 {g2:.2e} (must be ~0)', flush=True)
assert g1 < 1e-2 and g2 < 1e-2, 'block decomposition gate failed'


@torch.no_grad()
def run_with_pattern(pattern2):
    """forward the toy but override attn2's pattern; return CE."""
    x = x2
    v2 = heads(A2.v(rms(x2)))
    z = torch.einsum('bnqk,bknh->bqnh', pattern2, v2).reshape(x2.shape[0], T, D)
    xo = torch.lerp(x2, A2.o(z), A2.scale)
    x4 = m.layers[3](xo)
    lg = m.head(x4).float()
    return F.cross_entropy(lg.reshape(-1, VOCAB), TGT.reshape(-1)).item()


def pattern_from(subset):
    s1 = sum(blk1[p] for p in subset); s2 = sum(blk2[p] for p in subset)
    return (s1 * s2) / DH**2 * MASK


PAIRS = list(itertools.product(SRC, SRC))
CE0 = run_with_pattern(pattern_from(PAIRS))          # full = gate ΔCE ~ 0 vs real model
# real-model CE for reference
with torch.no_grad():
    CEreal = F.cross_entropy(m(IDX).float().reshape(-1, VOCAB), TGT.reshape(-1)).item()
res = {'ce_real': round(CEreal, 4), 'ce_full_blocks': round(CE0, 4), 'gate': [g1, g2],
       'frob_mass': {}, 'single_block_dce': {}, 'cumulative_dce': {}}
print(f'CE real {CEreal:.4f}; CE full-blocks {CE0:.4f} (gate ΔCE {CE0-CEreal:+.2e})', flush=True)

# Frobenius mass per block (masked), summed over both branches
mask_b = MASK.bool()
frob = {}
for p in PAIRS:
    fm = (blk1[p].abs() * mask_b).pow(2).sum().sqrt().item() + (blk2[p].abs() * mask_b).pow(2).sum().sqrt().item()
    frob[p] = fm
tot = sum(frob.values())
print('\nFrobenius mass per source-interaction block (fraction of total):', flush=True)
for p in sorted(PAIRS, key=lambda q: -frob[q]):
    res['frob_mass'][f'{p[0]}x{p[1]}'] = round(frob[p] / tot, 4)
    print(f'  {p[0]}x{p[1]}: {frob[p]/tot:.3f}', flush=True)

# single-block ablation: keep ONLY this block, ΔCE
print('\nkeep ONLY one source-interaction block, ΔCE:', flush=True)
for p in sorted(PAIRS, key=lambda q: -frob[q]):
    d = run_with_pattern(pattern_from([p])) - CE0
    res['single_block_dce'][f'{p[0]}x{p[1]}'] = round(d, 4)
    print(f'  {p[0]}x{p[1]} only: ΔCE {d:+.4f}', flush=True)

# cumulative: add blocks by Frobenius rank, ΔCE vs #blocks kept
print('\ncumulative (add blocks by Frobenius rank), ΔCE vs #blocks:', flush=True)
order = sorted(PAIRS, key=lambda q: -frob[q])
for n in range(1, 10):
    d = run_with_pattern(pattern_from(order[:n])) - CE0
    res['cumulative_dce'][n] = round(d, 4)
    print(f'  top-{n} blocks {[f"{a}x{b}" for a,b in order[:n]]}: ΔCE {d:+.4f}', flush=True)
json.dump(res, open(f'{OUT}/toy_qk1_interactions.json', 'w'), indent=2)
print('\nqk1 interactions done -> toy_qk1_interactions.json', flush=True)
