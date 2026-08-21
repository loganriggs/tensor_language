"""ARTICLE STEERING DECOMP -- does 619's "supervised probe steers
BACKWARDS" reversal GENERALIZE to the flagship article circuit, or is
it specific to newline?

619: adding the supervised newline readout direction d_newline to the
final residual DECREASES P(newline) -- the probe is anti-aligned with
the causal/steering axis. This repeats the exact test for the ARTICLE
readout (the model's most fully-traced circuit, 614): compute
d_article = mean(mlp17 output | next token is an article) - generic,
steer along it, and decompose whether the article LOGIT rises or falls.

If article ALSO reverses -> "decoding direction != causal direction"
is a GENERAL property of these supervised readout directions (a strong
methodological result). If article steers FORWARD (adding d_article
raises P(article)) -> the reversal is specific to newline, and the two
circuits differ in how their readout aligns with the causal axis.

Articles: ' a'(257) ' an'(281) ' the'(262) ' The'(383).

REGISTERED PREDICTIONS:
  (0) IDENTITY: alpha=0 is the clean forward pass -- sanity;
  (a) GENERALIZATION (the test): does P(article) rise or fall with
      alpha? Registered guess: it also FALLS (reversal generalizes),
      because the same rms_norm/softmax-competition mechanism applies
      to any supervised mlp17-output readout direction added at the
      residual;
  (b) MECHANISM: article raw logit trend (M1 anti-alignment if it
      falls; M2 competition if it rises while P falls);
  (c) ALIGNMENT: cos(d_article, mean article unembedding row) -- sign
      predicts M1;
  NULL: a random matched-norm direction moves the article logit far
      less systematically than d_article."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_steering_decomp_results.json'
NFRESH = 48
ART = [257, 281, 262, 383]
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)

    mlp17 = m.transformer.h[17].mlp
    cap = []
    hk = mlp17.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0)
    nxt = fresh[:, 1:257].reshape(-1)
    isart = np.isin(nxt.numpy(), ART)
    n_art = int(isart.sum())
    print(f'{n_art} article-target positions', flush=True)
    d = O[torch.tensor(isart)].mean(0) - O.mean(0)
    d = (d / d.norm()).to(DEV)

    WU = m.lm_head.weight.detach().float()
    rows = WU[ART]
    rm = rows.sum(0); rm = rm / rm.norm()
    cosm = float((d @ rm.to(DEV)).cpu())
    per = {int(t): round(float((d @ (WU[t] / WU[t].norm()).to(DEV)).cpu()), 3)
           for t in ART}
    print(f'(c) cos(d, mean W_U[article]) {cosm:+.3f}; per-row {per}', flush=True)

    resid_norm = None

    def run(direction, alpha):
        nonlocal resid_norm
        part = torch.zeros(NFRESH, T)
        art_logit = torch.zeros(NFRESH, T)
        mean_logit = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            B = bb.shape[0]
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            if resid_norm is None:
                resid_norm = float(x.norm(dim=-1).mean())
            xs = x + alpha * 0.25 * resid_norm * direction[None, None]
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(xs, (D,))) / 30)).float()
            p = F.softmax(lg, dim=-1)
            part[i:i + B] = p[..., ART].sum(-1).cpu()
            art_logit[i:i + B] = lg[..., ART].mean(-1).cpu()
            mean_logit[i:i + B] = lg.mean(-1).cpu()
        return (float(part.mean()), float(art_logit.mean()),
                float(mean_logit.mean()))

    curve = {}
    for a in ALPHAS:
        pa, al, ml = run(d, a)
        curve[a] = {'P_article': round(pa, 5), 'art_logit': round(al, 4),
                    'mean_logit': round(ml, 4)}
        print(f'  alpha {a:+.1f}: P(art) {pa:.5f}  art_logit {al:+.4f}  '
              f'mean_logit {ml:+.4f}', flush=True)

    ps = [curve[a]['P_article'] for a in ALPHAS]
    als = [curve[a]['art_logit'] for a in ALPHAS]
    p_slope = ps[-1] - ps[0]
    l_slope = als[-1] - als[0]
    reversed_ = p_slope < 0
    if l_slope < 0:
        mechanism = 'M1 anti-alignment (article logit falls)'
    elif p_slope < 0:
        mechanism = 'M2 softmax competition (logit rises, P falls)'
    else:
        mechanism = 'FORWARD (P rises -- reversal does NOT generalize)'
    print(f'\n(a) reversal generalizes: {reversed_} '
          f'(P {ps[0]:.4f}->{ps[-1]:.4f}); (b) {mechanism} '
          f'(logit {als[0]:+.3f}->{als[-1]:+.3f})', flush=True)

    rand_slopes = []
    for s in range(3):
        gg = torch.Generator(device=DEV).manual_seed(s)
        r = torch.randn(D, generator=gg, device=DEV); r = r / r.norm()
        rr = [run(r, a)[1] for a in ALPHAS]
        rand_slopes.append(round(rr[-1] - rr[0], 4))
    null_ok = abs(l_slope) > max(abs(x) for x in rand_slopes)
    print(f'NULL: d article-logit slope {l_slope:+.4f} vs random '
          f'{rand_slopes}: {"ok" if null_ok else "CHECK"}', flush=True)

    out = {'n_article_pos': n_art, 'cos_d_WUart_mean': cosm, 'cos_per_row': per,
           'curve': {str(k): v for k, v in curve.items()},
           'P_slope': p_slope, 'art_logit_slope': l_slope,
           'reversal_generalizes': bool(reversed_), 'mechanism': mechanism,
           'random_logit_slopes': rand_slopes, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
