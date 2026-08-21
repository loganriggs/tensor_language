"""NEWLINE STEERING -- causally validate the supervised readout
direction from 618: is d_newline a functional steering direction, not
just a correlational probe?

618 found the supervised class-readout directions (mean output at
class positions minus generic) are real out-of-sample readouts
(newline AUC 0.845) and span a ~5-dim subspace. This tests whether
d_newline is CAUSAL: add alpha * d_newline to the FINAL residual
(after all 18 blocks, before the final rms_norm + unembedding), and
measure whether P(newline) moves monotonically. If yes, the supervised
direction is a functional steering vector -- the readout basis is
causal, not merely a probe.

d_newline is computed exactly as in 618 (fit on all positions here,
since we test causally, not for generalization), then added at the
residual scaled to a fraction of the residual norm.

REGISTERED PREDICTIONS:
  (0) IDENTITY: alpha=0 leaves P(newline) unchanged -- sanity;
  (a) MONOTONIC STEERING: mean P(newline) increases monotonically as
      alpha goes -2 -> +2 -- adding d_newline raises newline
      probability, the direction causally controls the readout;
  (b) SPECIFICITY: at alpha=+2, d_newline raises P(newline) more than
      a RANDOM matched-norm direction (3 draws) does;
  (c) report the P(newline) curve and per-alpha CE cost;
  NULL: the random direction does NOT monotonically raise P(newline)
      -- steering is specific to d_newline, not any residual push."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_steering_results.json'
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

    # typical final-residual norm for scaling the steering magnitude
    resid_norm = None

    def run(direction, alpha):
        nonlocal resid_norm
        pnl = torch.zeros(NFRESH, T)
        ce = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
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
            ce[i:i + B] = F.cross_entropy(lg.view(-1, lg.size(-1)), tg,
                                          reduction='none').view(B, T).cpu()
        return float(pnl.mean()), float(ce.mean())

    base_p, base_ce = run(d, 0.0)
    curve = {}
    for a in ALPHAS:
        pnl, ce = run(d, a)
        curve[a] = {'P_newline': round(pnl, 5), 'CE': round(ce, 4)}
        print(f'  alpha {a:+.1f}: P(newline) {pnl:.5f}  CE {ce:.4f}', flush=True)

    p0 = abs(curve[0.0]['P_newline'] - base_p) < 1e-9
    ps = [curve[a]['P_newline'] for a in ALPHAS]
    monotonic = all(ps[i] <= ps[i + 1] + 1e-6 for i in range(len(ps) - 1))
    pa = monotonic and ps[-1] > ps[0]
    print(f'\n(0) identity {p0}; (a) monotonic increase {pa} '
          f'(P {ps[0]:.4f} -> {ps[-1]:.4f})', flush=True)

    # (b) random directions at alpha=+2
    g = torch.Generator(device=DEV).manual_seed(0)
    rand_p2 = []
    rand_mono = []
    for s in range(3):
        gg = torch.Generator(device=DEV).manual_seed(s)
        r = torch.randn(D, generator=gg, device=DEV)
        r = r / r.norm()
        pr = [run(r, a)[0] for a in ALPHAS]
        rand_p2.append(round(pr[-1], 5))
        rand_mono.append(all(pr[i] <= pr[i + 1] + 1e-6 for i in range(len(pr) - 1))
                         and pr[-1] > pr[0])
    pb = ps[-1] > max(rand_p2)
    null_ok = not any(rand_mono)
    print(f'(b) d_newline P@+2 {ps[-1]:.5f} > random {rand_p2}: '
          f'{"HELD" if pb else "FAILED"}', flush=True)
    print(f'NULL: random directions monotonic? {rand_mono} '
          f'(want all False): {"ok" if null_ok else "CHECK"}', flush=True)

    out = {'base_P_newline': base_p, 'curve': {str(k): v for k, v in curve.items()},
           'pred_0': bool(p0), 'monotonic': bool(monotonic), 'pred_a': bool(pa),
           'random_P_at_alpha2': rand_p2, 'pred_b': bool(pb),
           'random_monotonic': rand_mono, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
