"""Layer-1 QK in the INTERACTION basis: low-rank-reduce the QK maps (Logan 2026-07-20,
follows F14). F14 showed M is high-rank in the VARIANCE (PCA) basis, but the interaction-
sparse basis is the QK-singular basis. Reducing attn2's query/key projection matrices
(q1,k1,q2,k2) to rank r IS decomposing every source (E,A,M) in that interaction basis and
keeping r atoms — the M_q^T (W_q^T W_k) M_k interaction then has rank <= r. Measure end-to-
end ΔCE vs r and the description length vs the regime-1 raw baseline (~2.1 Mbit).
Gate: r=D reproduces the model. Toy block2, real TinyStories.
"""
import json, sys, math
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
buf = np.stack([val[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(48)]).astype(np.int64)
B = torch.from_numpy(buf).to(DEV); IDX, TGT = B[:, :-1], B[:, 1:]
A2 = m.layers[2]
QK_NAMES = ['q1', 'k1', 'q2', 'k2']
W0 = {n: getattr(A2, n).weight.data.clone() for n in QK_NAMES}


@torch.no_grad()
def ce():
    return F.cross_entropy(m(IDX).float().reshape(-1, VOCAB), TGT.reshape(-1)).item()


def lowrank(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


CE0 = ce()
res = {'baseline_ce': round(CE0, 4), 'raw_qk_Mbit': round(4 * D * D * 32 / 1e6, 2), 'sweep': {}}
print(f'baseline CE {CE0:.4f}; raw QK {4*D*D*32/1e6:.2f} Mbit (4 x {D}x{D} x 32). low-rank sweep:', flush=True)
for r in [2, 4, 8, 16, 32, 64, 128]:
    for n in QK_NAMES:
        getattr(A2, n).weight.data.copy_(lowrank(W0[n], r))
    dce = ce() - CE0
    bits = 4 * r * (D + D) * 32                    # rank-r factors for 4 matrices
    res['sweep'][r] = {'dce': round(dce, 4), 'Mbit': round(bits / 1e6, 3)}
    print(f'  r={r:3d}: ΔCE {dce:+.4f}   {bits/1e6:.3f} Mbit  ({100*bits/(4*D*D*32):.0f}% of raw)', flush=True)
    for n in QK_NAMES:
        getattr(A2, n).weight.data.copy_(W0[n])
    json.dump(res, open(f'{OUT}/toy_qk1_lowrank.json', 'w'), indent=2)
print('\nqk1 lowrank done -> toy_qk1_lowrank.json', flush=True)
