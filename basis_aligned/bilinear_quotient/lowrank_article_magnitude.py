"""LOWRANK ARTICLE MAGNITUDE -- test the 653 additive-vs-conditional
boundary on a NEW behavior. 636 found the front MLP carries the article
"magnitude" (whether to predict an article AT ALL), separate from the
a/an-vs-the choice (routing). Is that magnitude an ADDITIVE bias
(rank-1 isolable by removal, like the block-17 calibrator) or a
CONDITIONAL computation (no removable linear carrier, like the newline
routing)?

Recipe (650/652): w_art = cov(residual-after-block-2, article-follows
indicator). Remove it (rank-1, then top-r) from the post-front residual
and measure P(article) at article-target positions; controls = random
removals. Prediction discriminates the two regimes.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline P(article) at article-target positions is
      substantial (the magnitude is really there);
  (a) DISCRIMINATE THE REGIME (the finding): report whether removing the
      rank-1 (and top-8) w_art collapses P(article) toward the
      all-of-mlp0-2 removed floor. Registered guess: article-magnitude
      is CONDITIONAL like routing (context decides whether a noun phrase
      is starting), so rank-1 removal will NOT collapse it (<25%) and
      it will pattern with routing, not the calibrator;
  (b) SPECIFICITY: whatever w_art does, a random rank-1 removal does
      less;
  (c) report P(article) for baseline / remove top-r (1,4,8) / random /
      the front-attention-ablated reference;
  NULL: random-direction removals cluster near baseline."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'lowrank_article_magnitude_results.json'
NFRESH = 48
ART = [257, 281, 262, 383]
REMOVE_AFTER = 2


@torch.no_grad()
def forward(fresh, remove_Q, art_mask, capture=False):
    part = torch.zeros(NFRESH, T)
    cap = [] if capture else None
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li == REMOVE_AFTER:
                if capture:
                    cap.append(x.detach().float().reshape(-1, D).cpu())
                if remove_Q is not None:
                    x = x - (x @ remove_Q) @ remove_Q.T
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        part[i:i + B] = (p @ art_mask).cpu().view(B, T)
    part = part.reshape(-1).numpy()
    return (part, torch.cat(cap, 0).numpy()) if capture else part


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    art_mask = torch.zeros(V);
    for t in ART:
        art_mask[t] = 1.0
    art_mask = art_mask.to(DEV)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    art_tgt = np.isin(nxt, ART)
    print(f'{art_tgt.sum()} article-target positions', flush=True)

    base, X2 = forward(fresh, None, art_mask, capture=True)
    base_at = float(base[art_tgt].mean())

    # w_art: top-r directions in post-front residual tracking article-follows
    ind = art_tgt.astype(np.float64)
    Xc = X2 - X2.mean(0); ic = ind - ind.mean()
    dirs = []; R = Xc.copy()
    for _ in range(8):
        w = R.T @ ic; w = w / (np.linalg.norm(w) + 1e-9)
        dirs.append(w); R = R - (R @ w)[:, None] * w[None, :]
    Dfull = np.stack(dirs, 1)

    def Q(r):
        q, _ = np.linalg.qr(Dfull[:, :r]); return torch.tensor(q, dtype=torch.float32, device=DEV)

    curve = {}
    for r in [1, 4, 8]:
        p = forward(fresh, Q(r), art_mask)
        curve[r] = round(float(p[art_tgt].mean()), 5)
        print(f'remove top-{r}: P(article) {curve[r]:.4f} '
              f'(lost {100*(1-curve[r]/base_at):.0f}%)', flush=True)
    g = np.random.default_rng(0); rand = []
    for s in range(3):
        rr = g.standard_normal(D); rr /= np.linalg.norm(rr)
        p = forward(fresh, torch.tensor(rr, dtype=torch.float32, device=DEV)[:, None], art_mask)
        rand.append(float(p[art_tgt].mean()))
    rand_lost = [1 - r / base_at for r in rand]

    print(f'\nbaseline P(article) {base_at:.4f}', flush=True)
    print(f'remove random-1 lost {[f"{100*x:.0f}%" for x in rand_lost]}', flush=True)

    lost1 = 1 - curve[1] / base_at; lost8 = 1 - curve[8] / base_at
    p0 = base_at > 0.03
    # regime: additive if rank-1 removal collapses (>=50%), conditional if not (<25%)
    regime = ('additive-bias (rank-1 isolable)' if lost1 >= 0.5
              else 'conditional (no low-rank carrier)' if lost8 < 0.25
              else 'intermediate')
    pb = lost1 > np.mean(rand_lost)
    null_ok = np.std(rand_lost) < 0.15
    print(f'\n(0) magnitude present: {p0} (P {base_at:.3f})', flush=True)
    print(f'(a) REGIME: {regime} (rank-1 lost {100*lost1:.0f}%, top-8 lost '
          f'{100*lost8:.0f}%)', flush=True)
    print(f'(b) w_art > random: {pb}; NULL random tight: {null_ok}', flush=True)

    out = {'baseline_P_article': round(base_at, 5),
           'remove_topr_P': {str(k): v for k, v in curve.items()},
           'remove_random1_lost': [round(float(x), 4) for x in rand_lost],
           'rank1_lost': round(float(lost1), 4), 'top8_lost': round(float(lost8), 4),
           'regime': regime, 'pred_0': bool(p0), 'pred_b_specific': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
