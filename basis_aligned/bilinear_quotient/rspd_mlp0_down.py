"""RSPD MLP0 DOWN -- apply the REAL rspd library (github.com/ThatE10/rspd,
cloned by the user to /workspace/rspd after my own reimplementation in
578/579) to mlp0's Down layer, superseding that reimplementation.

578 built my own ablation-damage-covariance clustering because the repo
was unreachable; 579 applied it to mlp0's hidden units and found real
but weakly-named clusters. Reading the actual rspd source shows it is a
more precise, different method than my guess: a closed-form (Eckart-
Young) low-rank functional core AB = [WX]_r X^+ (README Problem 1, no
gradient descent), and RECURSIVE CIRCUIT ISOLATION -- repeatedly
re-decomposing and re-clustering DATA subsets by the shape of their
per-datum truncation-loss curves (rspd.circuit_isolation.
erank_circuit_isolation) until each branch bottoms out. This directly
answers the user's "get both the clusters and the minimal weights
required to run that computation": each recovered leaf circuit IS a
cluster of datapoints (idx) together with its minimal rank-r weight
surrogate (A, B), hierarchically nested via parent pointers.

Applied here to Down (mlp0's write matrix, 1152x4608) as W and mlp0's
hidden-unit activations h = (Lx)(Rx) over real FineWeb data as X -- a
genuine linear layer/activation pair (no nonlinearity between X and
WX), matching the README's contract exactly, unlike attention's
bilinear QK form which needs the two sides cached separately (README
sec 6) and is left for a follow-up.

r_min is NOT independently calibrated (the toy experiments calibrate it
by measuring known-pure-shape effective ranks first, which we cannot do
without ground truth) -- set from the ROOT's own effective rank
(measured here) as a first pass, flagged as a limitation rather than a
tuned value.

REGISTERED PREDICTIONS:
  (0) SANITY: the root circuit's own per-datum truncation loss at full
      rank (r = r_max) is < 1e-3 (relative to the response norm) --
      confirms the A-SVD pipeline reproduces Down exactly at full rank;
  (a) NONTRIVIAL RECURSION: the recursion produces >= 3 leaf circuits
      (not just the root) -- real structure exists to refine into;
  (b) RANK COMPRESSION: at least one leaf circuit has recovered rank
      less than half the root's effective rank -- some data subset is
      genuinely lower-dimensional than the whole, i.e. a real "minimal
      weight" simplification exists;
  (c) STABILITY (no ground truth available, same logic as 579's real
      test): rerun the recursion on an independent data half; report
      whether the number of leaves and the root's effective rank are
      similar order of magnitude across halves -- a coarse but real
      reproducibility check;
  NULL: reject-labeled leaves (HDBSCAN density noise, not a real
      cluster) should have HIGHER recovered rank on average than
      structured leaves -- reject data is by construction less
      compressible."""
import json, sys, time, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace/rspd')
from rspd.circuit_isolation import erank_circuit_isolation, recovered_weight, Circuit
from rspd.erank import effective_rank
from rspd.mrank import per_datum_truncation_losses

import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_mlp0_down_results.json'
NFRESH = 48
NSAMP = 2000


@torch.no_grad()
def capture(fresh):
    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight.float()
    cap = {}
    hk = mlp.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X', a_[0]))
    hs = []
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        X = cap['X'].float()
        h = (X @ L.T) * (X @ R.T)
        hs.append(h.reshape(-1, h.shape[-1]).cpu())
    hk.remove()
    H = torch.cat(hs, dim=0)
    return H, Dw.cpu()


def leaf_summary(circuits):
    leaves = [c for c in circuits if c.leaf]
    return leaves


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    H, Dw = capture(fresh)
    tok_flat = fresh[:, :256].reshape(-1)
    Nfull = H.shape[0]
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(Nfull, generator=g)[:NSAMP]
    Hs = H[perm]
    tok_s = tok_flat[perm]
    gi_s = perm
    N = Hs.shape[0]
    print(f'{N} samples, W (Down) shape {tuple(Dw.shape)}', flush=True)

    W = Dw  # (1152, 4608) = (out, in)
    X = Hs  # (N, 4608) batch-first

    # (0) sanity: full-rank per-datum loss on a small probe
    Lfull = per_datum_truncation_losses(X[:200], W)
    resp_norm = (X[:200] @ W.T).norm(dim=1).clamp_min(1e-9)
    rel = (Lfull[-1] / resp_norm).mean().item()
    p0 = rel < 1e-3
    print(f'(0) full-rank relative loss {rel:.2e}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'full-rank sanity failed', 'rel': rel},
                   open(OUT, 'w'), indent=1)
        return

    root_erank = effective_rank(W @ X.T)
    r_min = max(2.0, root_erank / 2)
    print(f'root effective rank {root_erank:.2f} -> r_min={r_min:.2f} '
          f'(first pass, not independently calibrated)', flush=True)

    def run(Xsub, tag):
        circuits = erank_circuit_isolation(
            Xsub, W, r_min=r_min, b_min=30, combine_threshold=0.98,
            combine_frequency=5, max_circuits=60, cluster_method='hdbscan')
        leaves = leaf_summary(circuits)
        print(f'[{tag}] {len(circuits)} circuits, {len(leaves)} leaves',
              flush=True)
        for c in leaves:
            print(f"   leaf {c.id}: n={len(c.idx)} rank={c.rank} "
                  f"erank={c.erank:.2f} origin={c.origin}", flush=True)
        return circuits, leaves

    circuits, leaves = run(X, 'full')
    pa = len(leaves) >= 3
    print(f'(a) >= 3 leaves ({len(leaves)}): '
          f"{'HELD' if pa else 'FAILED'}", flush=True)
    min_leaf_rank = min((c.rank for c in leaves), default=None)
    pb = min_leaf_rank is not None and min_leaf_rank < root_erank / 2
    print(f'(b) min leaf rank {min_leaf_rank} < root_erank/2 '
          f"{root_erank/2:.2f}: {'HELD' if pb else 'FAILED'}", flush=True)

    # (c) stability: independent half
    h1 = X[:N // 2]
    circuits2, leaves2 = run(h1, 'half1')
    order_mag = (abs(len(leaves) - len(leaves2)) <= max(len(leaves), 3))
    root2_erank = circuits2[0].erank if circuits2 else None
    pc = (root2_erank is not None and
          abs(root2_erank - root_erank) < 0.5 * root_erank)
    print(f'(c) half-data root erank {root2_erank} vs full {root_erank:.2f}, '
          f"leaf counts {len(leaves)} vs {len(leaves2)}: "
          f"{'HELD' if pc else 'FAILED'}", flush=True)

    # NULL: reject leaves have higher rank than structured leaves
    reject_ranks = [c.rank for c in leaves if c.origin == 'bisect']
    struct_ranks = [c.rank for c in leaves if c.origin != 'bisect']
    null_ok = (not reject_ranks or not struct_ranks or
               np.mean(reject_ranks) >= np.mean(struct_ranks))
    print(f'NULL: reject-origin ranks {reject_ranks} vs structured '
          f"{struct_ranks}: {'ok' if null_ok else 'CHECK'}", flush=True)

    # interpret the 3 largest leaves by example context
    def context_for(idx_local):
        gi = gi_s[idx_local].tolist()
        out = []
        for gi_ in gi[:8]:
            r_, p_ = gi_ // 256, gi_ % 256
            back = fresh[r_, max(0, p_ - 10):p_ + 1].tolist()
            pre = cl.enc().decode(back)
            out.append(pre)
        return out

    top_leaves = sorted(leaves, key=lambda c: c.rank)
    leaf_report = []
    for c in top_leaves:
        AB = recovered_weight(c)
        ex = context_for(c.idx)
        print(f"\nleaf {c.id} n={len(c.idx)} rank={c.rank} "
              f"weight_shape={AB.shape} examples:", flush=True)
        for e in ex:
            print(f"   ...{e!r}", flush=True)
        leaf_report.append({'id': c.id, 'n': len(c.idx), 'rank': c.rank,
                             'erank': c.erank, 'origin': c.origin,
                             'examples': ex})

    out = {'N': N, 'root_erank': root_erank, 'r_min': r_min,
           'pred_0': bool(p0), 'n_circuits': len(circuits),
           'n_leaves': len(leaves), 'pred_a': bool(pa),
           'min_leaf_rank': min_leaf_rank, 'pred_b': bool(pb),
           'half_root_erank': root2_erank, 'half_n_leaves': len(leaves2),
           'pred_c': bool(pc), 'reject_ranks': reject_ranks,
           'struct_ranks': struct_ranks, 'null_ok': bool(null_ok),
           'top_leaves': leaf_report, 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
