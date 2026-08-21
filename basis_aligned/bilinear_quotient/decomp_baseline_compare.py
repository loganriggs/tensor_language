"""DECOMP BASELINE COMPARE (user: run a weight-SVD baseline; how do we
measure which decomposition is BETTER?). PRIMARY METRIC = FUNCTIONAL
EFFICIENCY: substitute the rank-r reconstruction of a layer's Down weight
into the live model and measure CE-recovery = (CE_ablate - CE_r)/(CE_ablate
- CE_full). The better basis has a HIGHER recovered(r) at each rank and a
LOWER r80. Compare three decompositions of the SAME layer:
  (1) A-SVD  -- data-conditioned (SVD of response W@X pulled back via X+);
  (2) weight-SVD -- plain SVD of W (the BASELINE; ignores the data);
  (3) random rank-r projection (lower-bound control).
Run on mlp0 (low-rank r80~8) and mlp1 (high-rank r80~128). The A-SVD - weight
-SVD recovered gap = the value of data-conditioning; that gap IS the "in what
way better" answer, in nats of CE per component.

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank all three reproduce baseline; random is the floor;
  (a) A-SVD > weight-SVD at fixed rank: A-SVD recovers MORE of the loss at
      each rank r (data-conditioning packs task-loss into fewer directions),
      so r80(A-SVD) <= r80(weight-SVD). Report the recovered(r) curves and
      r80 for both layers; the gap quantifies "better";
  (b) report recovered(r) for A-SVD / weight-SVD / random on mlp0 and mlp1;
  NULL: random projection recovers ~0 at low rank."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'decomp_baseline_compare_results.json'
NFIT = 48; NEVAL = 96
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
LAYERS = [0, 1]


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


@torch.no_grad()
def forward_ce(rows, n, mod=None, W=None):
    orig = None
    if mod is not None:
        orig = mod.weight.data
        mod.weight.data = (torch.zeros_like(orig) if W == 'ablate' else W.to(orig.dtype)) if W is not None else orig
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    if orig is not None: mod.weight.data = orig
    return s/nn


@torch.no_grad()
def capture_gate(rows, n, layer):
    cap = []
    h = m.transformer.h[layer].mlp.Down.register_forward_pre_hook(
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
    ce_full = forward_ce(ev, NEVAL)
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g)); Q = Q.to(DEV)

    out = {'baseline_ce': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        mod = m.transformer.h[L].mlp.Down; W = mod.weight.data.float().to(DEV)
        X = capture_gate(fit, NFIT, L).to(DEV)
        ce_abl = forward_ce(ev, NEVAL, mod, 'ablate'); ben = ce_abl - ce_full
        # A-SVD
        A, B = asvd_fast(W, X)
        # weight-SVD
        Uw, Sw, Vhw = torch.linalg.svd(W, full_matrices=False)
        rec = {'asvd': {}, 'weightsvd': {}, 'random': {}}
        for r in RANKS:
            Wr_a = A[:, :r] @ B[:r, :]
            rec['asvd'][r] = round(float((ce_abl - forward_ce(ev, NEVAL, mod, Wr_a))/max(ben,1e-6)),4)
            Wr_w = (Uw[:, :r] * Sw[:r]) @ Vhw[:r, :]
            rec['weightsvd'][r] = round(float((ce_abl - forward_ce(ev, NEVAL, mod, Wr_w))/max(ben,1e-6)),4)
            Qr = Q[:, :r]
            rec['random'][r] = round(float((ce_abl - forward_ce(ev, NEVAL, mod, Qr @ (Qr.T @ W)))/max(ben,1e-6)),4)
        def r80(tbl):
            for r in RANKS:
                if tbl[r] >= 0.80: return r
            return None
        out['layers'][L] = {'benefit': round(float(ben),3), 'recovered': rec,
                            'r80_asvd': r80(rec['asvd']), 'r80_weightsvd': r80(rec['weightsvd'])}
        print(f'mlp{L} (benefit {ben:.3f}): r80 A-SVD {r80(rec["asvd"])}  weight-SVD {r80(rec["weightsvd"])}', flush=True)
        for r in [4, 8, 32, 128]:
            print(f'   r={r:3d}: A-SVD {rec["asvd"][r]:.3f}  weight-SVD {rec["weightsvd"][r]:.3f}  '
                  f'random {rec["random"][r]:.3f}', flush=True)
        del X

    # verdict: A-SVD better = higher recovered at each rank
    wins = 0; tot = 0
    for L in LAYERS:
        for r in RANKS:
            tot += 1
            if out['layers'][L]['recovered']['asvd'][r] >= out['layers'][L]['recovered']['weightsvd'][r]:
                wins += 1
    pa = wins >= 0.8 * tot
    print(f'\n(a) A-SVD >= weight-SVD at {wins}/{tot} (rank,layer) points: {pa}', flush=True)
    out['asvd_wins'] = wins; out['total_points'] = tot; out['pred_a_asvd_better'] = bool(pa)
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
