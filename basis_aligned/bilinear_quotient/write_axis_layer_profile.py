"""WRITE AXIS LAYER PROFILE -- which layers BUILD the class write axis?

619-622 established that the causal handle for a class is the WRITE axis
W_U[class] (the unembedding row), orthogonal to the read probe. 622
showed pushing it at the FINAL residual steers P(class) to ~95%. This
asks the mechanistic next question: across the 18 blocks, WHERE does the
residual's component along W_U[class] get built? Which blocks are the
WRITERS of each class?

Method: w_c = unit-normalized mean of the class's unembedding rows. Run
the model tracking the running residual x; record x . w_c after the
embedding and after each of the 18 blocks. The per-block INCREMENT
(x_after - x_before) . w_c is that block's contribution to the class
write axis. Measured at CLASS-target positions (where the class is the
next token) vs NON-class positions. This gives a full 18-block writer
profile per class, extending 616 (mlp0 writes newline more than mlp17,
by ablation) to a direct linear per-depth decomposition.

REGISTERED PREDICTIONS:
  (0) SANITY: at the final residual, x . w_c is larger at class-target
      positions than at non-class positions (the write axis is more
      populated when the class is coming) -- VOIDS if not;
  (a) EARLY WRITER (newline): the largest per-block increment to
      W_U[newline] is at an EARLY block, and block 0's increment
      exceeds block 17's -- consistent with 616 (mlp0 > mlp17 for
      newline). Report the top-3 writer blocks;
  (b) EARLY WRITER (article): same early-writer pattern (614 traced the
      article writer to mlp0). Report the top-3 writer blocks;
  (c) the full 18-block write-increment profile for both classes, plus
      the embedding's own projection;
  NULL: at NON-class positions the per-block increments onto W_U[c] are
      much smaller in aggregate -- writing along the class axis is
      specific to when that class is the next token, not a constant
      per-block drift."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'write_axis_layer_profile_results.json'
NFRESH = 48
CLASSES = {'newline': [198, 628], 'article': [257, 281, 262, 383]}
NB = 18


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    WU = m.lm_head.weight.detach().float()

    ws = {}
    for c, toks in CLASSES.items():
        w = sum(WU[t] / WU[t].norm() for t in toks)
        ws[c] = (w / w.norm()).to(DEV)

    steps = NB + 1                       # embedding + 18 blocks
    proj = {c: torch.zeros(steps, NFRESH, T) for c in CLASSES}
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for c in CLASSES:
            proj[c][0, i:i + B] = (x @ ws[c]).cpu()
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            for c in CLASSES:
                proj[c][li + 1, i:i + B] = (x @ ws[c]).cpu()

    nxt = fresh[:, 1:257].reshape(-1).numpy()

    out = {'classes': {}}
    for c, toks in CLASSES.items():
        P = proj[c].reshape(steps, -1)           # (19, NF*T)
        isc = np.isin(nxt, toks)
        # (0) sanity: final proj at class vs non-class
        final_class = float(P[-1, isc].mean())
        final_other = float(P[-1, ~isc].mean())
        p0 = final_class > final_other
        # increments per block
        inc = P[1:] - P[:-1]                      # (18, NF*T)
        inc_class = inc[:, isc].mean(1).numpy()   # (18,)
        inc_other = inc[:, ~isc].mean(1).numpy()
        emb_class = float(P[0, isc].mean())
        # top writers by |increment at class positions|
        order = np.argsort(-np.abs(inc_class))
        top3 = [(int(b), round(float(inc_class[b]), 4)) for b in order[:3]]
        b0_gt_b17 = abs(inc_class[0]) > abs(inc_class[17])
        # NULL: aggregate |increment| at class vs non-class
        agg_class = float(np.abs(inc_class).sum())
        agg_other = float(np.abs(inc_other).sum())
        null_ok = agg_class > agg_other

        out['classes'][c] = {
            'final_proj_class': round(final_class, 4),
            'final_proj_other': round(final_other, 4),
            'sanity_class_gt_other': bool(p0),
            'emb_proj_class': round(emb_class, 4),
            'inc_class': [round(float(v), 4) for v in inc_class],
            'inc_other': [round(float(v), 4) for v in inc_other],
            'top3_writer_blocks': top3,
            'block0_gt_block17': bool(b0_gt_b17),
            'agg_inc_class': round(agg_class, 4),
            'agg_inc_other': round(agg_other, 4),
            'null_ok': bool(null_ok)}
        print(f'\n{c}: final proj class {final_class:+.3f} vs other '
              f'{final_other:+.3f} (sanity {p0})', flush=True)
        print(f'  top-3 writer blocks (block, inc): {top3}', flush=True)
        print(f'  block0 {inc_class[0]:+.4f} vs block17 {inc_class[17]:+.4f} '
              f'-- block0>block17: {b0_gt_b17}', flush=True)
        print(f'  per-block inc at class: '
              f'{[round(float(v),3) for v in inc_class]}', flush=True)
        print(f'  NULL agg|inc| class {agg_class:.3f} > other {agg_other:.3f}: '
              f'{"ok" if null_ok else "CHECK"}', flush=True)

    pa = out['classes']['newline']['block0_gt_block17']
    pb = out['classes']['article']['block0_gt_block17']
    print(f'\n(a) newline block0>block17: {pa}; (b) article block0>block17: {pb}',
          flush=True)
    out['pred_a_newline_early'] = bool(pa)
    out['pred_b_article_early'] = bool(pb)
    out['runtime_s'] = time.time() - t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
