"""RSPD ATTN0 CPROJ ISOLATION -- recursive circuit isolation on the
one component this program now has a VALIDATED, precisely-calibrated
r_min for, unlike 580's uncalibrated first attempt on mlp0's Down.

589 established, via a clean cross-tool agreement, that attn0's real
c_proj input (over real FineWeb data) needs rank ~16 to keep task
cost under the ledger's own 0.10-nat bar, and that RSPD's closed-form
rank-16 surrogate reproduces that almost exactly (0.0943 nats). 580's
first circuit-isolation run (on mlp0's Down) had to guess r_min as
half the root's raw effective rank -- an uncalibrated, admittedly
first-pass number. Here r_min can be set from a REAL, task-validated
anchor: leaves are expected to bottom out somewhere near rank 16 (a
"clean, single-behaviour" subset of data), not at an arbitrary
fraction of the root's spectral entropy.

Question: does the recursion split attn0's real c_proj-input data
into DIFFERENT token/context classes needing different effective
ranks (the way 580 found for mlp0 -- two generic bulk clusters near
the root rank, plus small much-lower-rank special-case clusters)? If
attn0 really is close to a uniform bigram table (254), the prediction
is the OPPOSITE of mlp0's finding: little to no refinement, because
a genuine bigram table should look similarly low-rank (~16) almost
everywhere, not split into bulk-vs-special-case the way mlp0's more
heterogeneous MLP computation did.

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank per-datum loss is < 1e-3 relative to the
      response norm -- VOIDS on failure;
  (a) HOMOGENEITY (the real prediction, following from 254/589):
      the recursion produces FEW leaves (<=3) and/or the leaves'
      ranks cluster tightly around 16 (max/min leaf rank ratio < 3x)
      -- attn0 should look uniformly low-rank, unlike mlp0's bulk-
      vs-special-case split (580);
  (b) IF IT DOES split (prediction (a) fails), report what
      distinguishes the low-rank vs high-rank subsets by reading
      real examples, exactly as 580 did for mlp0's rank-39
      sentence-final cluster;
  (c) STABILITY: rerun on an independent data half; report whether
      leaf count and root effective rank are similar order of
      magnitude (580's coarse reproducibility check, repeated here);
  NULL: reject-origin leaves (if any) have higher rank than
      structured leaves -- reject data should be less compressible."""
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
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_attn0_cproj_isolation_results.json'
NFRESH = 48
NSAMP = 2000
R_MIN = 16.0


@torch.no_grad()
def capture_cproj_input(fresh):
    at = m.transformer.h[0].attn
    cap = []
    tok = []
    hk = at.c_proj.register_forward_pre_hook(
        lambda mo_, a_: cap.append(a_[0].detach().float().reshape(-1, D).cpu()))
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        tok.append(idx.cpu().reshape(-1))
    hk.remove()
    return torch.cat(cap, dim=0), torch.cat(tok, dim=0)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    at = m.transformer.h[0].attn
    W = at.c_proj.weight.float().cpu()

    fresh = cl.fineweb_rows(NFRESH)
    X_full, tok_full = capture_cproj_input(fresh)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(X_full.shape[0], generator=g)[:NSAMP]
    X = X_full[perm]
    tok = tok_full[perm]
    print(f'{X.shape[0]} samples, W (c_proj) shape {tuple(W.shape)}', flush=True)

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
    print(f'root effective rank {root_erank:.2f} (r_min={R_MIN}, '
          f'calibrated from 589\'s task-loss finding)', flush=True)

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
    ranks = [c.rank for c in leaves]
    pa = len(leaves) <= 3 or (max(ranks) / max(min(ranks), 1) < 3)
    print(f'(a) homogeneity ({len(leaves)} leaves, ranks {ranks}): '
          f"{'HELD' if pa else 'FAILED -- attn0 splits like mlp0 did'}",
          flush=True)

    # (b) if it splits, read real examples for the lowest/highest-rank leaves
    if not pa:
        by_rank = sorted(leaves, key=lambda c: c.rank)
        for c in [by_rank[0], by_rank[-1]]:
            toks = [cl.d1(int(tok[i])) for i in c.idx[:10].tolist()]
            print(f'   leaf {c.id} (rank {c.rank}) example current tokens: {toks}',
                  flush=True)

    h1 = X[:X.shape[0] // 2]
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

    out = {'N': X.shape[0], 'root_erank': root_erank, 'r_min': R_MIN,
           'pred_0': bool(p0), 'n_leaves': len(leaves), 'leaf_ranks': ranks,
           'pred_a': bool(pa), 'half_root_erank': root2_erank,
           'half_n_leaves': len(leaves2), 'pred_c': bool(pc),
           'reject_ranks': reject_ranks, 'struct_ranks': struct_ranks,
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
