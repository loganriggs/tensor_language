"""Step 4: Ethan's data-conditioned weight reduction of the most important matrix.

W = layer-0 c_v [1152 x 1152] (chosen by s4a: the block-0 value term v1 carries
0.89 of the patching effect at the top heads; layer-8 lamb=4.0 re-broadcasts it).

X = actual task inputs to W: rms_norm(residual) entering layer-0 attention over
500 clean increment prompts x 8 positions = 4000 positions (>=3000).
Y = W @ X.T; SVD; truncate rank r; W'_r = Y_r @ pinv(X.T, rcond=1e-4).
Substitute; measure (a) task accuracy + margin on held-out 10 stimuli,
(b) general CE on FineWeb rows 500-519 length 128. Control: data-free SVD
truncation of W at the same r.

NOTE: layer-0 c_v output (v1) is re-mixed into ALL 18 layers via lamb, so this
matrix is globally load-bearing — general damage is the interesting axis.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/increment')
from common import get_model, forward, build_stimuli, OUT

torch.manual_seed(0)
m, cfg = get_model()
D = cfg['n_embd']
S = torch.load(f'{OUT}/stimuli.pt')
held_clean = S['clean'][30:].cuda()
held_ca, held_xa = S['clean_ans'][30:].cuda(), S['corr_ans'][30:].cuda()

FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy')
                      .astype(np.int64))[500:520, :128].cuda()

# ---- collect X ------------------------------------------------------------
big = build_stimuli(500, seed=1)
Xrows = []
with torch.no_grad():
    for i in range(0, 500, 8):
        b = big['clean'][i:i+8].cuda()
        cch = {}
        forward(m, b, cache=cch)
        Xrows.append(F.rms_norm(cch[('resid_a', 0)], (D,)).reshape(-1, D).cpu())
X = torch.cat(Xrows, 0)                       # [4000, 1152]
print(f"X: {tuple(X.shape)}, matrix rank ~ {torch.linalg.matrix_rank(X.cuda(), rtol=1e-4).item()}", flush=True)

W = m.transformer.h[0].attn.c_v.weight.detach().clone()   # [d_out, d_in]
W_orig = W.clone()
Xg = X.cuda()
Y = W @ Xg.T                                   # [d_out, n]
U, Sv, Vt = torch.linalg.svd(Y, full_matrices=False)
pinvXT = torch.linalg.pinv(Xg.T, rcond=1e-4)   # [n, d_in]
Uw, Sw, Vtw = torch.linalg.svd(W, full_matrices=False)


def eval_all():
    with torch.no_grad():
        lg = forward(m, held_clean)[:, -1].float()
        n = torch.arange(10, device='cuda')
        acc = (lg.argmax(-1) == held_ca).float().mean().item()
        marg = (lg[n, held_ca] - lg[n, held_xa]).mean().item()
        ce_tot, cnt = 0.0, 0
        for i in range(0, 20, 8):
            b = FW[i:i+8]
            lgf = forward(m, b[:, :-1]).float()
            ce = F.cross_entropy(lgf.reshape(-1, lgf.shape[-1]), b[:, 1:].reshape(-1))
            ce_tot += ce.item() * b[:, 1:].numel(); cnt += b[:, 1:].numel()
    return acc, marg, ce_tot / cnt


acc0, marg0, ce0 = eval_all()
print(f"baseline (unmodified): held task acc {acc0:.2f}  margin {marg0:.3f}  fineweb CE {ce0:.4f}", flush=True)

res = {'baseline': {'task_acc': acc0, 'margin': marg0, 'fineweb_ce': round(ce0, 4)},
       'W': 'layer0.attn.c_v', 'n_positions': int(X.shape[0]), 'sweep': {}}
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
for r in RANKS:
    entry = {}
    # data-conditioned
    Yr = U[:, :r] * Sv[:r] @ Vt[:r]
    Wr = Yr @ pinvXT
    m.transformer.h[0].attn.c_v.weight.data.copy_(Wr)
    acc, marg, ce = eval_all()
    entry['data'] = {'task_acc': acc, 'margin': round(marg, 3), 'fineweb_ce': round(ce, 4)}
    # data-free SVD control
    Wf = Uw[:, :r] * Sw[:r] @ Vtw[:r]
    m.transformer.h[0].attn.c_v.weight.data.copy_(Wf)
    accf, margf, cef = eval_all()
    entry['free'] = {'task_acc': accf, 'margin': round(margf, 3), 'fineweb_ce': round(cef, 4)}
    m.transformer.h[0].attn.c_v.weight.data.copy_(W_orig)
    res['sweep'][str(r)] = entry
    print(f"r={r:4d}: DATA acc {acc:.2f} marg {marg:+7.3f} CE {ce:.3f} | "
          f"FREE acc {accf:.2f} marg {margf:+7.3f} CE {cef:.3f}", flush=True)

# minimal r for >=90% retention (task accuracy >= 0.9*baseline)
def min_r(kind):
    for r in RANKS:
        if res['sweep'][str(r)][kind]['task_acc'] >= 0.9 * acc0:
            return r
    return None
res['min_r_90pct_task'] = {'data': min_r('data'), 'free': min_r('free')}
print("minimal r for >=90% task accuracy retention:", res['min_r_90pct_task'])
json.dump(res, open(f'{OUT}/s4_weightred.json', 'w'), indent=2)
print('saved s4_weightred.json')
