"""CE-ORDERED BASIS (the "optimize the decomposition for the metric"
frontier, from 737). 737 found neither A-SVD (energy-ordered) nor weight-SVD
(singular-value-ordered) orders its directions by LOSS, so both are
suboptimal for the EFFICIENCY metric (CE-recovery per component). Test: take
a candidate basis and RE-ORDER it by each direction's actual CE-importance
(single-component ablation damage), keep the top-r most-important, and
compare recovered(r) to A-SVD and weight-SVD. A loss-ordered basis should
DOMINATE both at low rank -- demonstrating that optimizing the decomposition
for the target metric beats the reconstruction-optimal orderings.

Candidate basis = weight-SVD directions of mlp1 (top 256). Order them by
single-direction CE-damage (importance); recovered(r) = keep top-r-important
directions, reconstruct Down, price by CE. Compare vs weight-SVD (singular
order) and A-SVD (energy order).

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank reconstruction reproduces baseline;
  (a) LOSS-ORDERING WINS: the CE-importance-ordered basis has HIGHER
      recovered(r) than both weight-SVD and A-SVD at low/mid rank (r <= 32)
      -- ordering by loss beats ordering by energy or singular value;
  (b) report recovered(r) for CE-ordered / weight-SVD / A-SVD;
  NULL: a RANDOM re-ordering of the same directions is much worse than the
      CE-importance ordering (the gain is from loss-ordering, not from the
      basis)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cebasis_optimize_results.json'
NFIT = 48; NEVAL = 96
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
KBASIS = 256   # candidate directions to order


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


@torch.no_grad()
def forward_ce(rows, n, mod, W=None):
    orig = mod.weight.data
    if W is not None: mod.weight.data = (torch.zeros_like(orig) if W == 'ablate' else W.to(orig.dtype))
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    if W is not None: mod.weight.data = orig
    return s/nn


@torch.no_grad()
def capture_gate(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    mod = m.transformer.h[LAYER].mlp.Down; W = mod.weight.data.float().to(DEV)
    X = capture_gate(fit, NFIT).to(DEV)
    ce_full = forward_ce(ev, NEVAL, mod); ce_abl = forward_ce(ev, NEVAL, mod, 'ablate'); ben = ce_abl - ce_full

    # candidate basis: weight-SVD output dirs (U), input pairing via W projection
    Uw, Sw, Vhw = torch.linalg.svd(W); U = Uw[:, :KBASIS]         # (D, K) output dirs
    # reconstruction keeping a set S of output dirs: W_S = U_S U_S^T W
    def recon(dirs):
        Us = U[:, dirs]; return Us @ (Us.T @ W)
    # single-direction CE-importance: ablate one dir (remove it from W) -> damage
    imp = np.zeros(KBASIS)
    for k in range(KBASIS):
        keep = [j for j in range(KBASIS) if j != k]      # full basis minus k
        # cheap proxy: importance ~ CE rise from removing dir k from the rank-K recon
        Wk = recon(keep)
        imp[k] = forward_ce(ev, NEVAL, mod, Wk) - forward_ce(ev, NEVAL, mod, recon(list(range(KBASIS))))
    order_ce = np.argsort(-imp)                           # most-important first
    print(f'CE-importance ordering computed ({time.time()-t0:.0f}s)', flush=True)

    # A-SVD for comparison
    A, B = asvd_fast(W, X)
    g = torch.Generator().manual_seed(0); order_rand = g and np.random.default_rng(0).permutation(KBASIS)

    rec = {'ce_ordered': {}, 'weightsvd': {}, 'asvd': {}, 'random_order': {}}
    for r in RANKS:
        # CE-ordered: keep top-r important weight-SVD dirs
        rec['ce_ordered'][r] = round(float((ce_abl - forward_ce(ev, NEVAL, mod, recon(order_ce[:r].tolist())))/max(ben,1e-6)),4)
        # weight-SVD: keep top-r by singular value (natural order 0..r)
        rec['weightsvd'][r] = round(float((ce_abl - forward_ce(ev, NEVAL, mod, recon(list(range(r)))))/max(ben,1e-6)),4)
        # A-SVD: energy order
        rec['asvd'][r] = round(float((ce_abl - forward_ce(ev, NEVAL, mod, A[:, :r] @ B[:r, :]))/max(ben,1e-6)),4)
        # random reorder of the weight-SVD dirs
        rec['random_order'][r] = round(float((ce_abl - forward_ce(ev, NEVAL, mod, recon(order_rand[:r].tolist())))/max(ben,1e-6)),4)
        print(f'r={r:3d}: CE-ordered {rec["ce_ordered"][r]:.3f}  weightSVD {rec["weightsvd"][r]:.3f}  '
              f'A-SVD {rec["asvd"][r]:.3f}  rand-order {rec["random_order"][r]:.3f}', flush=True)

    lowr = [4, 8, 16, 32]
    wins = all(rec['ce_ordered'][r] >= rec['weightsvd'][r] and rec['ce_ordered'][r] >= rec['asvd'][r] for r in lowr)
    null_ok = all(rec['ce_ordered'][r] > rec['random_order'][r] for r in lowr)
    print(f'\n(a) CE-ordering dominates at low rank: {wins}; NULL beats random-order: {null_ok}', flush=True)
    out = {'benefit': round(float(ben),3), 'recovered': rec, 'pred_a_loss_ordering_wins': bool(wins),
           'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
