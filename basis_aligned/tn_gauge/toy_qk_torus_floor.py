"""Regime 1, query/key bond: the RoPE-constrained rotation floor (Logan 2026-07-20).
Completes regime 1 (OV done; QK is the other private bond). A rotation R on a head's
query/key subspace preserves the raw dot product, but is a gauge ONLY if it commutes
with RoPE. For rotate-half RoPE (planes (i,i+16), 16 distinct frequencies), the
commuting subgroup is a 16-angle TORUS: one 2D rotation per frequency plane. So the QK
gauge freedom is 16 angles/head/branch (vs OV's full O(d_head)=O(32)).

Optimize the torus angles to sparsify c_q,c_k head-blocks (L4 ascent), report the floor,
and VERIFY exactness (apply to the model, ΔCE=0). GATED by a planted control: a matrix
made torus-sparse must be recovered.
Toy block2, real TinyStories (only for the ΔCE=0 check; objective is weight-only).
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
HALF = DH // 2
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


def apply_torus(X, theta):
    """rotate the DH axis (dim 0) by a per-plane 2D rotation; planes (i, i+HALF)."""
    a, b = X[:HALF], X[HALF:]
    c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    return torch.cat([c * a - s * b, s * a + c * b], 0)


def hoyer(X):
    n = X.numel(); return float((np.sqrt(n) - (X.abs().sum() / X.norm()).item()) / (np.sqrt(n) - 1))


def fit_torus(Cq, Ck, iters=1500, lr=0.05):
    theta = torch.zeros(HALF, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(iters):
        obj = -(apply_torus(Cq, theta) ** 4).sum() - (apply_torus(Ck, theta) ** 4).sum()  # maximize L4
        opt.zero_grad(); obj.backward(); opt.step()
    return theta.detach()


# ---- GATE: planted control (a torus-sparse matrix must be recovered) ----
th0 = (torch.rand(HALF, device=DEV) * 2 - 1) * 1.2
Sq = torch.zeros(DH, D, device=DEV); Sq[torch.randint(0, DH, (D,), device=DEV), torch.arange(D, device=DEV)] = torch.randn(D, device=DEV)
Aq = apply_torus(Sq, -th0)            # observed = torus-rotated-away sparse; fit should undo it
l1_before = Aq.abs().sum().item()
th = fit_torus(Aq, Aq)
l1_after = apply_torus(Aq, th).abs().sum().item()
opt = Sq.abs().sum().item()
ctrl_ok = l1_after <= 1.03 * opt        # recovered the KNOWN planted optimum
print(f'CONTROL planted torus-sparse: L1 {l1_before:.1f} -> {l1_after:.1f} '
      f'(opt {Sq.abs().sum().item():.1f}) {"PASS" if ctrl_ok else "FAIL"}', flush=True)
# negative control: random (little torus structure)
Ar = torch.randn(DH, D, device=DEV)
thr = fit_torus(Ar, Ar)
print(f'CONTROL random: L1 drop {100*(1-apply_torus(Ar,thr).abs().sum().item()/Ar.abs().sum().item()):.1f}% (small)', flush=True)
if not ctrl_ok:
    print('CONTROL FAILED -> optimizer suspect, not reporting floor'); sys.exit(1)

CE0 = ce()
attn_layers = [i for i, s in enumerate(SPEC) if s == 'attn']
res = {'baseline_ce': round(CE0, 4), 'torus_dim': HALF, 'floors': {}}
print(f'\nbaseline CE {CE0:.4f}; QK torus floor ({HALF} angles/head/branch):', flush=True)
print('  layer.head.branch | L1 drop% | Hoyer before->after', flush=True)
store = {}
for li in attn_layers:
    for h in range(NH):
        sl = slice(h * DH, (h + 1) * DH)
        for bn, (qn, kn) in [('b1', ('q1', 'k1')), ('b2', ('q2', 'k2'))]:
            Cq = getattr(m.layers[li], qn).weight.data[sl, :].float()
            Ck = getattr(m.layers[li], kn).weight.data[sl, :].float()
            th = fit_torus(Cq, Ck)
            store[(li, h, bn)] = th
            l0 = (Cq.abs().sum() + Ck.abs().sum()).item()
            l1 = (apply_torus(Cq, th).abs().sum() + apply_torus(Ck, th).abs().sum()).item()
            hb = hoyer(torch.cat([Cq.reshape(-1), Ck.reshape(-1)]))
            ha = hoyer(torch.cat([apply_torus(Cq, th).reshape(-1), apply_torus(Ck, th).reshape(-1)]))
            res['floors'][f'L{li}.H{h}.{bn}'] = {'l1_drop_pct': round(100 * (1 - l1 / l0), 2),
                                                 'hoyer_before': round(hb, 3), 'hoyer_after': round(ha, 3)}
    print(f'  L{li}: mean drop '
          f'{np.mean([res["floors"][k]["l1_drop_pct"] for k in res["floors"] if k.startswith(f"L{li}.")]):.2f}%', flush=True)

# ---- verify exactness: apply all torus rotations to q,k of both branches, ΔCE=0 ----
for li in attn_layers:
    for bn, (qn, kn) in [('b1', ('q1', 'k1')), ('b2', ('q2', 'k2'))]:
        for nm in (qn, kn):
            W = getattr(m.layers[li], nm).weight.data
            Wn = W.clone()
            for h in range(NH):
                sl = slice(h * DH, (h + 1) * DH)
                Wn[sl, :] = apply_torus(W[sl, :].float(), store[(li, h, bn)]).to(W.dtype)
            W.copy_(Wn)
CE1 = ce()
res['gauge_delta_ce'] = round(CE1 - CE0, 6)
mdrop = np.mean([v['l1_drop_pct'] for v in res['floors'].values()])
res['mean_drop_pct'] = round(float(mdrop), 2)
json.dump(res, open(f'{OUT}/toy_qk_torus_floor.json', 'w'), indent=2)
print(f'\nexact-gauge check: ΔCE {CE1-CE0:+.2e} (must be ~0)', flush=True)
print(f'mean QK torus L1 drop {mdrop:.2f}%  (OV was 7%; torus is only {HALF} angles). done.', flush=True)
