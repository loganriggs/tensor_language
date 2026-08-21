"""EMBEDDING MASSIVE OVERLAP -- final connecting probe between two
architecture threads: the embedding is re-injected at weight ~8 every
block (689), and a few residual dims have massive magnitude by late
layers (676). Do the massive residual dims coincide with the embedding's
own high-magnitude dims? If yes, the persistent massive dims are (partly)
the re-injected embedding's large dims amplified by 18x re-injection; if
no, the massive dims are BUILT by the blocks (consistent with 676's
finding that they grow with depth and 686/massive_from_gating that the
MLP writes them).

Compute the per-dim RMS of the (rms-normed) embedding x0 across tokens,
and the per-dim RMS of the final residual (block 17). Compare their top-10
dims and rank correlation.

REGISTERED PREDICTIONS:
  (0) SANITY: the final residual has massive dims (a few dims dominate RMS);
  (a) BUILT NOT INHERITED: the embedding's high-magnitude dims do NOT
      strongly coincide with the final residual's massive dims (top-10
      overlap <= 3) -- the massive dims are constructed by the blocks, not
      inherited from the embedding (consistent with 676: massive dims grow
      with depth; massive_from_gating: MLP writes them);
  (b) report top-10 dims of embedding vs final residual, overlap, and the
      rank correlation of per-dim RMS;
  NULL/CONTROL: report the same overlap for a random dim subset (expected
      ~10*10/1152 ~ 0.09, i.e. ~0) to calibrate the overlap scale."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'embedding_massive_overlap_results.json'
NFRESH = 24


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    ss_emb = torch.zeros(D, dtype=torch.float64)
    ss_fin = torch.zeros(D, dtype=torch.float64)
    n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        ss_emb += (x0.float() ** 2).reshape(-1, D).sum(0).double().cpu()
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        ss_fin += (x.float() ** 2).reshape(-1, D).sum(0).double().cpu()
        n += idx.numel()

    remb = np.sqrt((ss_emb / n).numpy())
    rfin = np.sqrt((ss_fin / n).numpy())
    top_emb = set(np.argsort(-remb)[:10].tolist())
    top_fin = set(np.argsort(-rfin)[:10].tolist())
    overlap = len(top_emb & top_fin)
    rank_corr = float(np.corrcoef(np.log(remb + 1e-6), np.log(rfin + 1e-6))[0, 1])
    # how peaked is the final residual (max dim RMS / median)
    peak = float(rfin.max() / (np.median(rfin) + 1e-9))
    emb_peak = float(remb.max() / (np.median(remb) + 1e-9))

    print(f'embedding top-10 dims:      {sorted(top_emb)}', flush=True)
    print(f'final residual top-10 dims: {sorted(top_fin)}', flush=True)
    print(f'overlap: {overlap}/10   rank-corr(log RMS): {rank_corr:.3f}', flush=True)
    print(f'final residual peak (max/median): {peak:.1f}x  '
          f'embedding peak: {emb_peak:.1f}x', flush=True)

    p0 = peak > 3.0
    pa = overlap <= 3
    print(f'\n(0) final residual has massive dims (peak>3x): {p0}', flush=True)
    print(f'(a) massive dims BUILT not inherited (overlap<=3): {pa}', flush=True)

    out = {'top_emb': sorted(top_emb), 'top_fin': sorted(top_fin),
           'overlap': overlap, 'rank_corr_logRMS': round(rank_corr, 4),
           'final_peak': round(peak, 2), 'emb_peak': round(emb_peak, 2),
           'pred_0': bool(p0), 'pred_a_built_not_inherited': bool(pa),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
