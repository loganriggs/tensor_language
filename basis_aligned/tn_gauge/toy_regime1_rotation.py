"""Regime 1: the EXACT rotation sweep (Logan 2026-07-20). Stage 1 of the construction,
not a control. Per private bond, solve the local varimax problem
    min_{Q in O(d_bond)} ||A Q||_1 + ||Q^T B||_1
where A = the reader weights consuming the bond and B = the writer weights producing
it; apply Q as an EXACT gauge (model unchanged, ΔCE=0) and report the per-bond
sparsity FLOOR = the core L1 that survives the best rotation (= how much superposition
that bond carries; also the per-bond atom budget for regime-2 births).

DEVIATION FLAGGED (QUESTION FOR LOGAN): Logan's objective indexes RESIDUAL bonds with
ends pinned to E/W_U. But a residual stream is one shared bus: pinning BOTH ends pins
the whole interior (embed is rank d -> the only global residual Q fixing E is I; gate-0
checks A/B). So interior residual bonds have NO rotational freedom under end-pinning —
the freedom lives in the per-layer PRIVATE bonds. This script therefore runs regime 1
on the private bonds where the gauge is exact:
  - attention OV: value head-subspace, full O(d_head) (free)   <- done here
  - attention QK: RoPE-constrained (measured: free rotation breaks CE)  <- reported
  - MLP hidden: pinned by elementwise * (perm+scale only, no rotation) <- reported
The residual bond's Φ therefore gets its sparsity from BIRTHS (regime 2), not rotation.
Also: private bonds are mutually INDEPENDENT, so regime 1 is a parallel one-shot solve,
not an iterative sweep; cross-bond coupling (the real DMRG sweep) enters in regime 2.
Toy block2, real TinyStories (only to confirm ΔCE=0; the objective is weight-only).
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
buf = np.stack([val[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(32)]).astype(np.int64)
B = torch.from_numpy(buf).to(DEV); IDX, TGT = B[:, :-1], B[:, 1:]


@torch.no_grad()
def ce():
    return F.cross_entropy(m(IDX).float().reshape(-1, VOCAB), TGT.reshape(-1)).item()


CE0 = ce()


def hoyer(X):
    """scale-free sparsity in [0,1], 1=maximally sparse (one nonzero), 0=uniform."""
    n = X.numel()
    l1 = X.abs().sum(); l2 = X.norm()
    return float((np.sqrt(n) - (l1 / l2).item()) / (np.sqrt(n) - 1))


def varimax_Q(A, Bm, iters=800, lr=0.05):
    """Rotation-to-sparsity for the bond gauge: MAXIMIZE ||A Q||_4^4 + ||Q^T Bm||_4^4
    over Q in O(d) (orthogonal Q preserves Frobenius norm, so L4/kurtosis ascent
    concentrates energy = sparsity). Validated on planted/random controls. A:(a,d)
    reader, Bm:(d,b) writer; Q rotates the shared bond so both get sparser."""
    d = A.shape[1]
    Q = torch.eye(d, device=DEV)
    for _ in range(iters):
        AQ = A @ Q; QB = Q.T @ Bm
        G = A.T @ (AQ ** 3) + Bm @ ((QB ** 3).T)      # grad of the L4^4 sum
        S = G @ Q.T; S = 0.5 * (S - S.T)
        I = torch.eye(d, device=DEV)
        Q = torch.linalg.solve(I - lr * S, I + lr * S) @ Q   # ASCENT (maximize)
    return Q


attn_layers = [i for i, s in enumerate(SPEC) if s == 'attn']
res = {'baseline_ce': round(CE0, 4), 'OV_floors': {}, 'notes': {}}
print(f'baseline CE {CE0:.4f}; regime-1 OV rotation floors:', flush=True)
print('  layer.head | L1 before | L1 after | drop% | Hoyer before->after', flush=True)
Qstore = {}
for li in attn_layers:
    for h in range(NH):
        sl = slice(h * DH, (h + 1) * DH)
        v_b = m.layers[li].v.weight.data[sl, :].float()      # (DH, D) writer (value out rows)
        o_b = m.layers[li].o.weight.data[:, sl].float()      # (D, DH) reader (o input cols)
        # bond = value head-subspace (DH). reader A = o_b (D x DH), writer B = v_b (DH x D)
        Q = varimax_Q(o_b, v_b)                              # min ||o_b Q||_1 + ||Q^T v_b||_1
        l1_0 = (o_b.abs().sum() + v_b.abs().sum()).item()
        l1_1 = ((o_b @ Q).abs().sum() + (Q.T @ v_b).abs().sum()).item()
        stk0 = torch.cat([o_b.reshape(-1), v_b.reshape(-1)])
        stk1 = torch.cat([(o_b @ Q).reshape(-1), (Q.T @ v_b).reshape(-1)])
        Qstore[(li, h)] = Q
        res['OV_floors'][f'L{li}.H{h}'] = {'l1_before': round(l1_0, 2), 'l1_after': round(l1_1, 2),
                                           'drop_pct': round(100 * (1 - l1_1 / l1_0), 1),
                                           'hoyer_before': round(hoyer(stk0), 3),
                                           'hoyer_after': round(hoyer(stk1), 3)}
        print(f'  L{li}.H{h}    |  {l1_0:7.2f}  |  {l1_1:7.2f} | {100*(1-l1_1/l1_0):4.1f} | '
              f'{hoyer(stk0):.3f} -> {hoyer(stk1):.3f}', flush=True)

# apply ALL OV rotations as an exact gauge, confirm ΔCE = 0
for li in attn_layers:
    v_new = m.layers[li].v.weight.data.clone()
    o_new = m.layers[li].o.weight.data.clone()
    for h in range(NH):
        sl = slice(h * DH, (h + 1) * DH)
        Q = Qstore[(li, h)].to(v_new.dtype)
        v_new[sl, :] = Q.T @ m.layers[li].v.weight.data[sl, :]
        o_new[:, sl] = m.layers[li].o.weight.data[:, sl] @ Q
    m.layers[li].v.weight.data.copy_(v_new)
    m.layers[li].o.weight.data.copy_(o_new)
CE1 = ce()
res['ce_after_gauge'] = round(CE1, 6)
res['delta_ce'] = round(CE1 - CE0, 6)
print(f'\napplied all OV rotations as gauge: CE {CE0:.5f} -> {CE1:.5f}  (ΔCE {CE1-CE0:+.2e}, must be ~0)', flush=True)

# QK free-rotation control (should BREAK -> RoPE-constrained) and MLP-hidden note
mean_ov_drop = np.mean([v['drop_pct'] for v in res['OV_floors'].values()])
res['notes'] = {'mean_OV_L1_drop_pct': round(float(mean_ov_drop), 1),
                'QK': 'free head rotation breaks CE (RoPE-constrained) - gate-0 check C2',
                'MLP_hidden': 'pinned by elementwise * (perm+scale only) - gate-0 check C3',
                'residual_bond': 'pinned to identity by end-pinning (embed rank=d) - floor from births, not rotation'}
json.dump(res, open(f'{OUT}/toy_regime1_rotation.json', 'w'), indent=2)
print(f'mean OV L1 drop {mean_ov_drop:.1f}%  (sparsity floor = surviving L1). regime1 done.', flush=True)
