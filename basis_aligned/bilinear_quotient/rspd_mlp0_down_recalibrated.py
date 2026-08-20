"""RSPD MLP0 DOWN RECALIBRATED -- redo 580's circuit isolation on
mlp0's Down layer with a properly task-loss-calibrated r_min, closing
the methodological gap 590 flagged.

580 (the first circuit-isolation run in this arc) had to guess r_min
as half of the root's raw spectral-entropy effective rank -- an
admittedly uncalibrated first pass, since no independent task-loss
anchor existed yet. 589/590 then established the right way to do
this: use the ledger's own independently-measured task-loss rank as
r_min. The ledger already has that number for mlp0 too, from the same
passage that gave attn0's 16: "attn0's own write needs only 16
directions... where mlp0 needs 64" (both to get under 0.10 nats).
This redoes 580 with r_min=64 instead of the guessed value, for full
consistency with how 590 treated attn0.

Question: does mlp0's bulk-vs-special-case split (580: two ~500-rank
generic clusters, three much-smaller-rank special clusters, out of a
687-effective-rank root) survive a fair r_min, or was the earlier
5-leaf split partly an artifact of the uncalibrated, much higher
r_min=343.9 used before (which would have let MORE structure through
as "still worth splitting" than a properly calibrated 64 might)?

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank relative loss < 1e-3 -- VOIDS on failure;
  (a) STRUCTURE SURVIVES: with r_min=64 (much lower than 580's
      343.9), the recursion still produces >= 3 leaves -- mlp0's
      heterogeneity is not an artifact of a loose r_min;
  (b) MORE REFINEMENT, NOT LESS: a lower r_min should let the
      recursion go DEEPER (more leaves, or leaves closer to the true
      64-direction floor) than 580's run -- report the leaf count and
      ranks directly against 580's [485, 39, 531, 70, 21] for a
      direct comparison;
  (c) STABILITY: rerun on an independent data half, as in 580;
  NULL: as in 580, reject-origin leaves have higher rank than
      structured leaves."""
import json, sys, time, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace/rspd')
from rspd.circuit_isolation import erank_circuit_isolation, recovered_weight
from rspd.erank import effective_rank
from rspd.mrank import per_datum_truncation_losses

import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_mlp0_down_recalibrated_results.json'
NFRESH = 48
NSAMP = 2000
R_MIN = 64.0


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
    return torch.cat(hs, dim=0), Dw.cpu()


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
    gi_s = perm
    N = Hs.shape[0]
    print(f'{N} samples, W (Down) shape {tuple(Dw.shape)}', flush=True)

    W = Dw
    X = Hs

    L = per_datum_truncation_losses(X[:200], W)
    resp_norm = (X[:200] @ W.T).norm(dim=1).clamp_min(1e-9)
    rel = (L[-1] / resp_norm).mean().item()
    p0 = rel < 1e-3
    print(f'(0) full-rank relative loss {rel:.2e}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'sanity failed', 'rel': rel}, open(OUT, 'w'), indent=1)
        return

    root_erank = effective_rank(W @ X.T)
    print(f'root effective rank {root_erank:.2f} -> r_min={R_MIN} '
          f'(ledger task-loss anchor, vs 580\'s guessed 343.9)', flush=True)

    def run(Xsub, tag):
        circuits = erank_circuit_isolation(
            Xsub, W, r_min=R_MIN, b_min=30, combine_threshold=0.98,
            combine_frequency=5, max_circuits=60, cluster_method='hdbscan')
        leaves = [c for c in circuits if c.leaf]
        print(f'[{tag}] {len(circuits)} circuits, {len(leaves)} leaves',
              flush=True)
        for c in leaves:
            print(f"   leaf {c.id}: n={len(c.idx)} rank={c.rank} "
                  f"erank={c.erank:.2f} origin={c.origin}", flush=True)
        return circuits, leaves

    circuits, leaves = run(X, 'full')
    ranks = sorted((c.rank for c in leaves), reverse=True)
    pa = len(leaves) >= 3
    print(f'(a) >= 3 leaves ({len(leaves)}, ranks {ranks}): '
          f"{'HELD' if pa else 'FAILED'}", flush=True)
    print(f"(b) vs 580's [485, 39, 531, 70, 21]: {ranks}", flush=True)

    h1 = X[:N // 2]
    circuits2, leaves2 = run(h1, 'half1')
    root2_erank = circuits2[0].erank if circuits2 else None
    pc = (root2_erank is not None and
          abs(root2_erank - root_erank) < 0.5 * root_erank)
    print(f'(c) half-data root erank {root2_erank} vs full {root_erank:.2f}: '
          f"{'HELD' if pc else 'FAILED'}", flush=True)

    reject_ranks = [c.rank for c in leaves if c.origin == 'bisect']
    struct_ranks = [c.rank for c in leaves if c.origin != 'bisect']
    null_ok = (not reject_ranks or not struct_ranks or
               np.mean(reject_ranks) >= np.mean(struct_ranks))
    print(f'NULL: reject ranks {reject_ranks} vs structured {struct_ranks}: '
          f"{'ok' if null_ok else 'CHECK'}", flush=True)

    def context_for(idx_local):
        gi = gi_s[idx_local].tolist()
        out = []
        for gi_ in gi[:6]:
            r_, p_ = gi_ // 256, gi_ % 256
            back = fresh[r_, max(0, p_ - 10):p_ + 1].tolist()
            out.append(cl.enc().decode(back))
        return out

    top_leaves = sorted(leaves, key=lambda c: c.rank)
    leaf_report = []
    for c in top_leaves:
        ex = context_for(c.idx)
        print(f"\nleaf {c.id} n={len(c.idx)} rank={c.rank} examples:", flush=True)
        for e in ex:
            print(f"   ...{e!r}", flush=True)
        leaf_report.append({'id': c.id, 'n': len(c.idx), 'rank': c.rank,
                            'erank': c.erank, 'origin': c.origin,
                            'examples': ex})

    out = {'N': N, 'root_erank': root_erank, 'r_min': R_MIN,
           'pred_0': bool(p0), 'n_leaves': len(leaves), 'leaf_ranks': ranks,
           'pred_a': bool(pa), 'half_root_erank': root2_erank,
           'half_n_leaves': len(leaves2), 'pred_c': bool(pc),
           'reject_ranks': reject_ranks, 'struct_ranks': struct_ranks,
           'null_ok': bool(null_ok), 'leaves': leaf_report,
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
