"""RSPD MLP17 FUNCTIONAL RANK -- the method the user proposed in Q5,
applied properly to the flagship circuit. The recent circuit isolation
used behavior-conditioned covariance removal (a cruder tool that only
finds rank-1 additive biases). This instead uses the REAL RSPD A-SVD
primitive -- SVD of the response (X @ W.T) pulled back through pinv(X.T),
i.e. the user's SVD(WX)*X.pinv() = 'the closest weight NECESSARY for this
specific dataset' -- on mlp17's Down layer (1152 x 4608), whose real input
X is the bilinear gate Left(x)*Right(x).

Critically, per the 588 lesson (entropy-based effective_rank != task-loss
rank), each rank-r A-SVD surrogate is priced by REAL CROSS-ENTROPY:
substitute the rank-r Down weight into the live model and measure CE. This
finds the smallest r that recovers 80% of mlp17's loss-benefit -- the
'smallest r that gives 20/80' the user asked for -- and compares to a
random rank-r projection (does the data-conditioning matter?).

Definitions (per token, mean over a fresh corpus):
  CE_full   = baseline (real Down).
  CE_ablate = Down output = just Down_bias (data-dependent part removed).
  benefit   = CE_ablate - CE_full  (how much mlp17's Down helps).
  CE_r      = with the rank-r A-SVD Down surrogate W_r = A[:, :r] @ B[:r].
  recovered(r) = (CE_ablate - CE_r) / benefit.

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank A-SVD surrogate reproduces baseline CE
      (recovered ~ 1.0, |CE_full - CE_fullrank| < 0.01);
  (a) LOW 20/80 RANK: the smallest r with recovered(r) >= 0.80 is FAR below
      full rank -- r80 <= 128 (< ~11% of Down's 1152 out-rank) -- mlp17's
      data-conditioned functional core is low-rank (the 20/80 exists);
  (b) report recovered(r) for A-SVD and random projection across the sweep;
  NULL/CONTROL: a random rank-r projection of W needs a substantially
      HIGHER r for the same 80% (random r80 >= 2x the A-SVD r80) -- the
      A-SVD core is genuinely aligned to the data, not a generic low-rank
      artifact of W."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace/rspd')
from rspd.asvd import generate_lowrank_approximation

import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_mlp17_functional_rank_results.json'
NFRESH = 24
NCAP = 12            # rows used to fit A-SVD (gate activations captured)
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


@torch.no_grad()
def ce_with_down(fresh, Wsub):
    """CE over the corpus with mlp17.Down.weight temporarily set to Wsub
    (or None = real baseline; 'ablate' = zero weight, keep bias)."""
    mlp = m.transformer.h[17].mlp
    orig = mlp.Down.weight.data
    if Wsub == 'ablate':
        mlp.Down.weight.data = torch.zeros_like(orig)
    elif Wsub is not None:
        mlp.Down.weight.data = Wsub.to(orig.dtype).to(orig.device)
    ce_s = 0.0; n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean')
        ce_s += float(ce) * idx.shape[0]; n += idx.shape[0]
    mlp.Down.weight.data = orig
    return ce_s / n


@torch.no_grad()
def capture_gate(fresh):
    """Capture mlp17's real Down-input (the bilinear gate) over NCAP rows."""
    cap = []
    mlp = m.transformer.h[17].mlp
    h = mlp.Down.register_forward_pre_hook(lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, NCAP, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    W = m.transformer.h[17].mlp.Down.weight.data.float().cpu()   # (1152, 4608)

    X = capture_gate(fresh)                                       # (Ncap_tok, 4608)
    print(f'gate X {tuple(X.shape)}, Down W {tuple(W.shape)}', flush=True)

    # A-SVD: user's SVD(WX)*X.pinv(). target = X @ W.T (batch-first response).
    A_fac, B_fac = generate_lowrank_approximation(W, X, target=X @ W.T)
    maxr = A_fac.shape[1]
    print(f'A-SVD factors: A {tuple(A_fac.shape)}, B {tuple(B_fac.shape)}, maxr {maxr}',
          flush=True)

    ce_full = ce_with_down(fresh, None)
    ce_ablate = ce_with_down(fresh, 'ablate')
    benefit = ce_ablate - ce_full
    print(f'CE_full {ce_full:.4f}  CE_ablate {ce_ablate:.4f}  benefit {benefit:.4f}',
          flush=True)

    # full-rank A-SVD sanity
    W_fullrank = (A_fac @ B_fac).float()
    ce_fullrank = ce_with_down(fresh, W_fullrank)
    p0 = abs(ce_fullrank - ce_full) < 0.01
    print(f'(0) full-rank A-SVD CE {ce_fullrank:.4f} (|d|<0.01 vs {ce_full:.4f}): '
          f"{'HELD' if p0 else 'FAILED'}", flush=True)

    # random-projection null: project W onto a random r-dim subspace of its
    # output space (orthonormal Q_r), W_rand_r = Q_r Q_r^T W.
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))

    rows = {'asvd': {}, 'random': {}}
    for r in RANKS:
        if r > maxr:
            break
        W_r = (A_fac[:, :r] @ B_fac[:r, :]).float()
        ce_r = ce_with_down(fresh, W_r)
        rec = (ce_ablate - ce_r) / benefit
        rows['asvd'][r] = [round(ce_r, 4), round(float(rec), 4)]
        Qr = Q[:, :r]
        W_rand = (Qr @ (Qr.T @ W)).float()
        ce_rand = ce_with_down(fresh, W_rand)
        rec_rand = (ce_ablate - ce_rand) / benefit
        rows['random'][r] = [round(ce_rand, 4), round(float(rec_rand), 4)]
        print(f'r={r:4d}: A-SVD CE {ce_r:.4f} recovered {rec:.3f} | '
              f'random CE {ce_rand:.4f} recovered {rec_rand:.3f}', flush=True)

    def r80(tbl):
        for r in RANKS:
            if r in tbl and tbl[r][1] >= 0.80:
                return r
        return None
    r80_a = r80(rows['asvd']); r80_r = r80(rows['random'])
    pa = r80_a is not None and r80_a <= 128
    null_ok = (r80_r is None) or (r80_a is not None and r80_r >= 2 * r80_a)
    print(f'\n20/80 rank: A-SVD r80={r80_a}  random r80={r80_r}', flush=True)
    print(f'(a) low functional rank (A-SVD r80<=128): {pa}', flush=True)
    print(f'NULL random needs >=2x rank (or never): {null_ok}', flush=True)

    out = {'ce_full': round(ce_full, 4), 'ce_ablate': round(ce_ablate, 4),
           'benefit': round(benefit, 4), 'ce_fullrank_asvd': round(ce_fullrank, 4),
           'recovered_by_rank': rows, 'r80_asvd': r80_a, 'r80_random': r80_r,
           'maxr': int(maxr), 'down_out_dim': D,
           'pred_0': bool(p0), 'pred_a_low_rank': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
