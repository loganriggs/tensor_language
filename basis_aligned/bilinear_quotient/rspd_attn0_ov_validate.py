"""RSPD ATTN0 OV VALIDATE -- apply the real rspd library (580) to a
component this program already understands EXACTLY, as ground truth
for the tool rather than another exploratory application.

Established facts (254, 14586-14589, 14796): attn0 is an exact bigram
table -- its input is exactly rms_norm(wte) (the token embedding, no
context), so its output is exactly a function of (current token,
attended token); and its own write needs only ~16 directions (measured
elsewhere in this ledger by a different method -- SVD truncation of
Down-style output projection). This is a rare case in this program
where the "right answer" for a low-rank recovery tool is already
known, so it's a chance to check RSPD is finding real structure
(agreeing with independently-established facts) rather than an
artifact of the tool.

The OV circuit for one attention head is (c_proj_h @ c_v_h) -- "if the
head attended fully to a position, what would it write". Combined
across all 9 heads it is W = c_proj.weight @ c_v.weight (1152x1152) --
a genuine, fixed LINEAR weight (unlike mlp0's Down, this is the
literal nn.Linear-to-nn.Linear composition, no elementwise nonlinearity
between them within the OV path itself). X = the token embedding
table (attn0's value input is exactly the embedding, established
exactly), so SVD(WX) here directly answers "how many directions of
attn0's write does the vocabulary's own geometry actually exercise" --
the README's central claim (data-conditioned rank, not weight-alone
rank) tested on a component whose answer this program can already
grade.

REGISTERED PREDICTIONS:
  (0) IDENTITY: full-rank reconstruction is exact (< 1e-4 relative
      error) -- sanity;
  (a) BALLPARK MATCH: the effective rank (Roy-Vetterli, continuous)
      of W's response on real token embeddings is < 64 -- an order-
      of-magnitude check against the ledger's independently-measured
      ~16 (different method, so exact agreement isn't expected, but
      the same order of magnitude would be a real cross-method
      confirmation);
  (b) STRUCTURE IS IN THE EMBEDDINGS, NOT JUST THE WEIGHT: the
      effective rank on REAL token embeddings is substantially lower
      (<= 60%) than on random Gaussian directions of matched count and
      norm -- if the low rank were just a property of W itself, both
      would come out similar; a real gap means the vocabulary's own
      geometry is what's compact;
  (c) RECURSIVE STRUCTURE: run the recursive circuit isolation on
      real token embeddings and report the leaf circuits -- do
      different token classes (frequent vs rare, alphabetic vs
      punctuation) get isolated as needing different ranks? No bar,
      report the split and a few example tokens per leaf;
  NULL: (b) IS the null in effect -- restated here as the registered
      check that random-direction rank is not lower than real-
      embedding rank (the opposite of the claim would mean the tool
      is not sensitive to real structure at all)."""
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
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_attn0_ov_validate_results.json'
NVOCAB = 4000


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    at = m.transformer.h[0].attn
    Wv = at.c_v.weight.float().cpu()
    Wo = at.c_proj.weight.float().cpu()
    W = Wo @ Wv  # (1152, 1152), the combined OV map

    rows = cl.rows()
    cnt = torch.bincount(rows.reshape(-1),
                         minlength=m.transformer.wte.weight.shape[0])
    freq = cnt.argsort(descending=True)[:NVOCAB]
    E_full = m.transformer.wte.weight.float().cpu()
    X = F.rms_norm(E_full[freq], (D,))  # (NVOCAB, 1152), exactly attn0's real input
    print(f'X shape {tuple(X.shape)}, W shape {tuple(W.shape)}', flush=True)

    # (0) sanity
    L = per_datum_truncation_losses(X[:200], W)
    resp_norm = (X[:200] @ W.T).norm(dim=1).clamp_min(1e-9)
    rel = (L[-1] / resp_norm).mean().item()
    p0 = rel < 1e-4
    print(f'(0) full-rank relative loss {rel:.2e}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'sanity failed', 'rel': rel}, open(OUT, 'w'), indent=1)
        return

    real_erank = effective_rank(W @ X.T)
    pa = real_erank < 64
    print(f'(a) real-embedding effective rank {real_erank:.2f} (bar <64, '
          f'ledger independently measured ~16): '
          f"{'HELD' if pa else 'FAILED'}", flush=True)

    g = torch.Generator().manual_seed(11)
    Xrand = torch.randn(X.shape[0], X.shape[1], generator=g)
    Xrand = Xrand / Xrand.norm(dim=1, keepdim=True) * X.norm(dim=1, keepdim=True).mean()
    rand_erank = effective_rank(W @ Xrand.T)
    pb = real_erank <= 0.6 * rand_erank
    print(f'(b) random-direction effective rank {rand_erank:.2f}, real/random '
          f'ratio {real_erank/rand_erank:.3f} (bar <=0.6): '
          f"{'HELD' if pb else 'FAILED'}", flush=True)

    r_min = max(2.0, real_erank / 2)
    circuits = erank_circuit_isolation(
        X, W, r_min=r_min, b_min=30, combine_threshold=0.98,
        combine_frequency=5, max_circuits=60, cluster_method='hdbscan')
    leaves = [c for c in circuits if c.leaf]
    print(f'(c) {len(circuits)} circuits, {len(leaves)} leaves '
          f'(r_min={r_min:.2f})', flush=True)
    leaf_report = []
    for c in sorted(leaves, key=lambda c: c.rank):
        toks = [cl.d1(int(freq[i])) for i in c.idx[:8].tolist()]
        print(f'   leaf {c.id}: n={len(c.idx)} rank={c.rank} '
              f'erank={c.erank:.2f} examples={toks}', flush=True)
        leaf_report.append({'id': c.id, 'n': len(c.idx), 'rank': c.rank,
                            'erank': c.erank, 'examples': toks})

    out = {'real_erank': real_erank, 'rand_erank': rand_erank,
           'pred_0': bool(p0), 'pred_a': bool(pa), 'pred_b': bool(pb),
           'n_circuits': len(circuits), 'n_leaves': len(leaves),
           'leaves': leaf_report, 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
