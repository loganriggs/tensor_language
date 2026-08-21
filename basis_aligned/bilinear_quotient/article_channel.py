"""ARTICLE CHANNEL -- is there ONE article channel in the residual
stream: a single fixed direction the whole stack writes into and the
unembedding reads out? Synthesizes 608 (depth profile), 612 (low-rank
subspace), 599 (parallel redundancy), and the reusable-components
question.

The article READOUT direction is fixed by the unembedding, no fitting:
    d = W_U[' a'] + W_U[' an'] - W_U[' the'] - W_U[' The]   (1152-dim,
the residual direction that most raises P(a/an)-P(the/The) at the
output). If every article-relevant layer WRITES into d (its output
projects +d at a/an-target positions and -d at the-target positions),
then d IS a single reusable article channel -- one direction read
directly by the unembedding, written by many layers. This is the
concrete candidate for "the reusable component" the program has been
circling.

Two measurements:
  PROFILE: for each layer L, the mean projection of its MLP output
    onto d at a/an-target vs the-target positions. A layer that writes
    the channel has proj(a/an) > proj(the); the gap is its channel
    contribution. Predict this tracks 608's causal depth profile
    (mlp0/mlp1 front, mlp15-17 late, empty middle).
  SUBSPACE CONTAINMENT: what fraction of d's norm lies inside mlp0's
    top-16 output-PCA subspace (612's article subspace). If large, the
    16-dim low-rank subspace CONTAINS the readout direction -- which
    would explain why 16 directions capture the article decision (they
    include d).

REGISTERED PREDICTIONS:
  (0) SANITY: d well-defined; >= 200 a/an-target and >= 200 the-target
      positions -- VOIDS on failure;
  (a) SHARED CHANNEL: the front article layers (mlp0, mlp1) have a
      POSITIVE channel gap (proj at a/an-target > proj at the-target),
      and the gap for the middle layers (mlp6-10) is much smaller --
      the article machinery writes a common direction d, front-loaded;
  (b) SUBSPACE CONTAINS READOUT: >= 40% of d's norm lies in mlp0's
      top-16 output-PCA subspace (612) -- vs ~16/1152 = 1.4% expected
      for a random direction, so a large enrichment means the low-rank
      subspace is built around the readout direction;
  (c) THE PROFILE (no bar): report the per-layer channel gap for all
      18 layers, and whether it matches 608's causal profile;
  NULL: an unrelated readout direction (P(' he')-P(' the'), a
      different token contrast) has a much smaller channel gap at the
      article-writing layers than d does -- the layers write d
      specifically, not any unembedding direction."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
NL = 18
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_channel_results.json'
NFRESH = 64
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383
TOK_HE = 339  # ' he' for the null contrast


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    WU = m.lm_head.weight.float()          # (V, D)
    d = (WU[TOK_A] + WU[TOK_AN] - WU[TOK_THE] - WU[TOK_THE2])
    d = (d / d.norm()).to(DEV)             # article readout direction
    dnull = (WU[TOK_HE] - WU[TOK_THE])
    dnull = (dnull / dnull.norm()).to(DEV)

    nxt = fresh[:, 1:257].reshape(-1)
    aan = ((nxt == TOK_A) | (nxt == TOK_AN)).numpy()
    the = ((nxt == TOK_THE) | (nxt == TOK_THE2)).numpy()
    print(f'{aan.sum()} a/an-target, {the.sum()} the-target positions',
          flush=True)
    p0 = aan.sum() >= 200 and the.sum() >= 200
    if not p0:
        json.dump({'void': 'too few positions'}, open(OUT, 'w'), indent=1)
        return

    # capture every MLP output; also mlp0 output for the PCA subspace
    caps = {li: [] for li in range(NL)}
    mlp0_out = []
    hooks = []
    for li in range(NL):
        mlp = m.transformer.h[li].mlp
        hooks.append(mlp.register_forward_hook(
            (lambda li: lambda mo, i_, o_: caps[li].append(
                o_.detach().float().reshape(-1, D).cpu()))(li)))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    for h in hooks:
        h.remove()
    outs = {li: torch.cat(caps[li], 0) for li in range(NL)}

    aan_t = torch.tensor(aan)
    the_t = torch.tensor(the)
    dcpu = d.cpu()
    dnull_cpu = dnull.cpu()

    profile = {}
    profile_null = {}
    for li in range(NL):
        proj = outs[li] @ dcpu                    # (N,) projection onto d
        gap = float(proj[aan_t].mean() - proj[the_t].mean())
        projn = outs[li] @ dnull_cpu
        gapn = float(projn[aan_t].mean() - projn[the_t].mean())
        profile[li] = round(gap, 3)
        profile_null[li] = round(gapn, 3)
        print(f'  L{li:>2}: channel gap {gap:+.3f}  (null dir {gapn:+.3f})',
              flush=True)

    # subspace containment: mlp0's top-16 output PCA
    O0 = outs[0]
    mu0 = O0.mean(0)
    U, S, Vt = torch.linalg.svd(O0 - mu0, full_matrices=False)
    V16 = Vt[:16]                                 # (16, D)
    d_in16 = float((V16 @ dcpu).pow(2).sum())     # ||proj of d onto subspace||^2
    # d is unit-norm, so d_in16 is the fraction of d's norm^2 in the subspace
    pb = d_in16 >= 0.40
    print(f'\n(b) fraction of readout direction d in mlp0 top-16 subspace: '
          f'{d_in16:.3f} (random ~0.014): {"HELD" if pb else "FAILED"}',
          flush=True)

    front = np.mean([profile[l] for l in (0, 1)])
    mid = np.mean([abs(profile[l]) for l in (6, 7, 8, 9, 10)])
    pa = profile[0] > 0 and profile[1] > 0 and front > mid
    print(f'(a) front layers (mlp0 {profile[0]}, mlp1 {profile[1]}) write +d, '
          f'front mean {front:.3f} > middle mean {mid:.3f}: '
          f'{"HELD" if pa else "FAILED"}', flush=True)
    # null: the layers that write d most should write dnull much less
    writers = sorted(range(NL), key=lambda l: -profile[l])[:3]
    d_at_writers = np.mean([profile[l] for l in writers])
    dn_at_writers = np.mean([profile_null[l] for l in writers])
    null_ok = abs(dn_at_writers) < 0.5 * abs(d_at_writers)
    print(f'(c) top channel writers: {writers}', flush=True)
    print(f'NULL: at top writers, d gap {d_at_writers:.3f} vs null-dir gap '
          f'{dn_at_writers:.3f}: {"ok" if null_ok else "CHECK"}', flush=True)

    out = {'pred_0': bool(p0), 'channel_gap': profile,
           'null_dir_gap': profile_null, 'd_fraction_in_mlp0_top16': d_in16,
           'pred_a': bool(pa), 'pred_b': bool(pb),
           'top_writers': writers, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
