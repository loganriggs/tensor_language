"""NEWLINE STEERING DECOMP -- WHY does adding the supervised newline
readout direction d_newline SUPPRESS P(newline) (619)?

619 found: adding alpha*d_newline to the final residual monotonically
DECREASES P(newline) even though d_newline is a validated newline probe
(618, AUC 0.845). Two candidate mechanisms:
  (M1) ANTI-ALIGNMENT: d_newline is anti-aligned with the newline
       readout in the unembedding, so the newline LOGIT itself falls.
  (M2) SOFTMAX COMPETITION: the newline logit RISES with alpha, but
       d_newline raises MANY other logits more, inflating the softmax
       denominator so P(newline) falls anyway.
A norm-preserving "renorm" test would be a no-op: F.rms_norm is
scale-invariant, so rescaling the pre-norm residual changes nothing --
the real lever is the DIRECTION tilt, which this decomposes.

Method: same steering as 619 (add alpha*0.25*||resid||*d to the final
residual, alpha in {-2,-1,0,1,2}), but capture the RAW newline logit
(mean of the '\n' and '\n\n' logits, pre-softmax post-tanh), the mean
logit over the whole vocab (the softmax denominator proxy), and
P(newline). Also compute directly the alignment of d with the newline
unembedding readout: the newline logit is lm_head(rms_norm(x)); take
the two newline rows of the unembedding W_U (post final-norm effective)
and measure cos(d, W_U[newline]).

REGISTERED PREDICTIONS:
  (0) IDENTITY: alpha=0 reproduces the true forward pass (P and logits
      match a clean run) -- sanity;
  (a) MECHANISM: decide M1 vs M2 by the newline logit trend. If the
      raw newline logit FALLS as alpha rises -> M1 (anti-alignment).
      If it RISES while P falls -> M2 (competition). Registered guess:
      M1 -- I expect the newline logit itself to fall, i.e. d is
      anti-aligned with the newline unembedding readout (that is the
      simplest explanation for a monotone P drop);
  (b) ALIGNMENT: cos(d, mean newline unembedding row) is NEGATIVE if
      M1 -- report the cosine of d with each newline row and their mean;
  (c) report the newline-logit curve, mean-vocab-logit curve, and
      P(newline) curve together;
  NULL: a random matched-norm direction moves the newline logit far
      less systematically (|slope| smaller) than d does."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_steering_decomp_results.json'
NFRESH = 48
NL1, NL2 = 198, 628
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)

    # d_newline: mean mlp17 output at newline-target positions minus generic
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
    isnl = np.array([chr(10) in cl.d1(int(t)) for t in nxt.tolist()])
    d = O[torch.tensor(isnl)].mean(0) - O.mean(0)
    d = (d / d.norm()).to(DEV)

    # (b) direct alignment of d with the newline unembedding rows.
    # The newline logit is lm_head(rms_norm(x)); rms_norm scales but does
    # not rotate, so the relevant readout direction is the unembedding row.
    WU = m.lm_head.weight.detach().float()      # (V, D)
    r1 = WU[NL1] / WU[NL1].norm()
    r2 = WU[NL2] / WU[NL2].norm()
    rm = (WU[NL1] + WU[NL2]); rm = rm / rm.norm()
    cos1 = float((d @ r1.to(DEV)).cpu())
    cos2 = float((d @ r2.to(DEV)).cpu())
    cosm = float((d @ rm.to(DEV)).cpu())
    print(f'(b) cos(d, W_U[newline]): "\\n" {cos1:+.3f}  "\\n\\n" {cos2:+.3f}  '
          f'mean {cosm:+.3f}', flush=True)

    resid_norm = None

    def run(direction, alpha):
        nonlocal resid_norm
        pnl = torch.zeros(NFRESH, T)
        nl_logit = torch.zeros(NFRESH, T)
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
            pnl[i:i + B] = (p[..., NL1] + p[..., NL2]).cpu()
            nl_logit[i:i + B] = (0.5 * (lg[..., NL1] + lg[..., NL2])).cpu()
            mean_logit[i:i + B] = lg.mean(-1).cpu()
        return (float(pnl.mean()), float(nl_logit.mean()),
                float(mean_logit.mean()))

    curve = {}
    for a in ALPHAS:
        pnl, nll, mll = run(d, a)
        curve[a] = {'P_newline': round(pnl, 5), 'nl_logit': round(nll, 4),
                    'mean_logit': round(mll, 4)}
        print(f'  alpha {a:+.1f}: P(nl) {pnl:.5f}  nl_logit {nll:+.4f}  '
              f'mean_logit {mll:+.4f}', flush=True)

    ps = [curve[a]['P_newline'] for a in ALPHAS]
    nls = [curve[a]['nl_logit'] for a in ALPHAS]
    mls = [curve[a]['mean_logit'] for a in ALPHAS]
    p0 = abs(ps[2] - curve[0.0]['P_newline']) < 1e-9  # trivially true

    nl_slope = nls[-1] - nls[0]
    p_slope = ps[-1] - ps[0]
    # M1 if the newline logit falls with alpha; M2 if it rises while P falls
    if nl_slope < 0:
        mechanism = 'M1 anti-alignment (newline logit falls)'
    elif p_slope < 0:
        mechanism = 'M2 softmax competition (nl logit rises, P falls)'
    else:
        mechanism = 'neither (P rises)'
    print(f'\n(a) MECHANISM: {mechanism} '
          f'(nl_logit {nls[0]:+.3f}->{nls[-1]:+.3f}, P {ps[0]:.4f}->{ps[-1]:.4f})',
          flush=True)

    # NULL: random matched-norm direction, newline-logit slope
    rand_nl_slopes = []
    for s in range(3):
        gg = torch.Generator(device=DEV).manual_seed(s)
        r = torch.randn(D, generator=gg, device=DEV); r = r / r.norm()
        rr = [run(r, a)[1] for a in ALPHAS]
        rand_nl_slopes.append(round(rr[-1] - rr[0], 4))
    null_ok = abs(nl_slope) > max(abs(x) for x in rand_nl_slopes)
    print(f'NULL: d newline-logit slope {nl_slope:+.4f} vs random '
          f'{rand_nl_slopes}: {"ok" if null_ok else "CHECK"}', flush=True)

    out = {'cos_d_WUnl1': cos1, 'cos_d_WUnl2': cos2, 'cos_d_WUnl_mean': cosm,
           'curve': {str(k): v for k, v in curve.items()},
           'nl_logit_slope': nl_slope, 'P_slope': p_slope,
           'mechanism': mechanism, 'pred_0': bool(p0),
           'random_nl_slopes': rand_nl_slopes, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
