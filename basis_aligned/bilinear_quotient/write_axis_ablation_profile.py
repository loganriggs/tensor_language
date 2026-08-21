"""WRITE AXIS ABLATION PROFILE -- the CAUSAL writer profile, resolving
623's newline tension.

623 found the LINEAR write-axis-projection profile is a valid writer-
localizer only when it passes a position-specificity null: ARTICLE
passed (block 0 dominant writer), NEWLINE failed (late/diffuse, null
failed -> confounded). So the newline writer is unresolved by the
linear method. This runs the CAUSAL version, extending 616 (which
ablated the mlp17 newline cluster) to all 18 blocks: mean-ablate each
block's CONTRIBUTION and measure the drop in P(class) at class-target
positions.

Ablation = mean-fill the block's residual contribution: for block L,
delta = x_out - x_in; replace with x_in + mean_over_positions(delta),
removing the block's position-specific writing while preserving its
mean. Re-run to the output and measure P(class). Drop = baseline P(class)
- ablated P(class) at class-target positions. Done for each of 18 blocks
x {newline, article}.

REGISTERED PREDICTIONS:
  (0) SANITY: total effect is nonzero -- at least one block ablation
      changes P(class) at class positions by >1% relative;
  (a) ARTICLE early writer (causal): ablating block 0 drops P(article)
      more than ablating block 17 -- confirming 614/623 causally;
  (b) NEWLINE resolution: does an EARLY block (0) drop P(newline) more
      than a LATE block (17)? 616 (cluster ablation) said mlp0 > mlp17.
      Registered guess: block 0 drop > block 17 drop for newline too --
      i.e. the causal writer is early even though the linear profile
      (623) pointed late (confounded). This is the decisive test;
  (c) the full 18-block causal drop profile for both classes;
  NULL: the per-block P(class) drop is larger at class-target positions
      than the same block's effect on P(class) at non-class positions
      (the block writes the class specifically where it is needed)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'write_axis_ablation_profile_results.json'
NFRESH = 48
CLASSES = {'newline': [198, 628], 'article': [257, 281, 262, 383]}
NB = 18


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    masks = {c: np.isin(nxt, toks) for c, toks in CLASSES.items()}

    def forward_pnl(ablate_block):
        """Return per-class P(class) as (NFRESH*T,) arrays; ablate_block
        is None (baseline) or a block index whose contribution is
        mean-filled."""
        outp = {c: torch.zeros(NFRESH, T) for c in CLASSES}
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous(); B = bb.shape[0]
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for li, blk in enumerate(m.transformer.h):
                x_in = x
                x, v1 = blk(x, v1, x0)
                if ablate_block is not None and li == ablate_block:
                    delta = x - x_in
                    x = x_in + delta.mean(dim=(0, 1), keepdim=True)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            p = F.softmax(lg, dim=-1)
            for c, toks in CLASSES.items():
                outp[c][i:i + B] = p[..., toks].sum(-1).cpu()
        return {c: outp[c].reshape(-1).numpy() for c in CLASSES}

    base = forward_pnl(None)
    base_at = {c: float(base[c][masks[c]].mean()) for c in CLASSES}
    print(f'baseline P at class positions: '
          f'{ {c: round(v,5) for c,v in base_at.items()} }', flush=True)

    drops = {c: [] for c in CLASSES}          # drop at class positions
    drops_other = {c: [] for c in CLASSES}    # effect at non-class positions
    for L in range(NB):
        ab = forward_pnl(L)
        for c in CLASSES:
            m_c = masks[c]
            d_at = float(base[c][m_c].mean() - ab[c][m_c].mean())
            d_ot = float(base[c][~m_c].mean() - ab[c][~m_c].mean())
            drops[c].append(round(d_at, 5))
            drops_other[c].append(round(d_ot, 5))
        print(f'  block {L:2d}: '
              + '  '.join(f'{c} drop {drops[c][-1]:+.5f}' for c in CLASSES),
              flush=True)

    out = {'baseline_P_at_class': {c: round(v, 5) for c, v in base_at.items()},
           'classes': {}}
    for c in CLASSES:
        dr = drops[c]
        order = sorted(range(NB), key=lambda b: -abs(dr[b]))
        top3 = [(b, dr[b]) for b in order[:3]]
        b0_gt_b17 = abs(dr[0]) > abs(dr[17])
        # NULL: block-of-max-drop is more specific at class than non-class
        agg_at = float(np.abs(dr).sum())
        agg_ot = float(np.abs(drops_other[c]).sum())
        null_ok = agg_at > agg_ot
        out['classes'][c] = {
            'drops_at_class': dr, 'drops_at_other': drops_other[c],
            'top3_blocks': top3, 'block0_drop': dr[0], 'block17_drop': dr[17],
            'block0_gt_block17': bool(b0_gt_b17),
            'agg_drop_class': round(agg_at, 5), 'agg_drop_other': round(agg_ot, 5),
            'null_ok': bool(null_ok)}
        print(f'\n{c}: top-3 blocks by |drop| {top3}; block0 {dr[0]:+.5f} vs '
              f'block17 {dr[17]:+.5f} -> block0>block17 {b0_gt_b17}', flush=True)
        print(f'  NULL agg|drop| class {agg_at:.5f} > other {agg_ot:.5f}: '
              f'{"ok" if null_ok else "CHECK"}', flush=True)

    pa = out['classes']['article']['block0_gt_block17']
    pb = out['classes']['newline']['block0_gt_block17']
    p0 = any(abs(d) > 0.01 * base_at[c] for c in CLASSES for d in drops[c])
    print(f'\n(0) nonzero {p0}; (a) article block0>block17 {pa}; '
          f'(b) newline block0>block17 {pb}', flush=True)
    out['pred_0'] = bool(p0)
    out['pred_a_article_early'] = bool(pa)
    out['pred_b_newline_early'] = bool(pb)
    out['runtime_s'] = time.time() - t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
