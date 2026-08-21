"""W_FREQ STEERING -- show the one clean knob is ACTIONABLE. The rank-1
frequency-calibration direction w_freq (650-672) is the model's single
isolable linear component. Scaling the model's projection onto it should
act as a frequency/diversity DIAL: amplifying the calibration suppresses
frequent tokens further (more diversity, higher CE on frequent, lower on
rare); attenuating it does the opposite.

Intervention: at mlp17's output, replace the w_freq component with alpha
times itself: o -> o + (alpha-1)*(o.w_freq)*w_freq, for alpha in
{0, 0.5, 1, 1.5, 2}. alpha=1 is identity; alpha=0 removes the calibration
(651); alpha>1 amplifies it. Measure P(frequent) and P(rare) token mass
and CE at frequent- vs rare-target positions.

REGISTERED PREDICTIONS:
  (0) IDENTITY: alpha=1 reproduces the baseline;
  (a) MONOTONIC DIAL: as alpha rises 0->2, total probability mass on the
      top-20 frequent tokens DECREASES monotonically (the calibration
      suppresses them more) -- a usable frequency/diversity dial;
  (b) TRADE-OFF: rising alpha lowers rare-target CE and raises frequent-
      target CE (amplified calibration = the 626 trade-off, stronger);
  (c) report P(top20 mass), freq-CE, rare-CE per alpha;
  NULL: scaling a RANDOM rank-1 direction the same way does NOT
      monotonically change the frequent-token mass."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'w_freq_steering_results.json'
NFRESH = 48
TOPK = 20
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]

W = {'dir': None, 'alpha': 1.0}


def hook(mo, i_, o_):
    d = W['dir']
    if d is None:
        return o_
    comp = (o_ @ d)[..., None] * d
    return o_ + (W['alpha'] - 1.0) * comp


@torch.no_grad()
def run(fresh, direction, alpha, top_mask, is_freq):
    W['dir'] = direction; W['alpha'] = alpha
    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    V = m.lm_head.weight.shape[0]
    topmass = []; ces = []; labs = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, V)
        topmass.append((p @ top_mask).cpu().numpy())
        ces.append(F.cross_entropy(lg.view(-1, V), tg, reduction='none').cpu().numpy())
        labs.append(np.array([is_freq[int(t)] for t in tg.cpu().numpy()]))
    hk.remove()
    tm = float(np.concatenate(topmass).mean())
    ce = np.concatenate(ces); lab = np.concatenate(labs)
    return tm, float(ce[lab].mean()), float(ce[~lab].mean())


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V)
    top_ids = np.argsort(-freq)[:TOPK]
    is_freq = np.array([1 if t in set(top_ids.tolist()) else 0 for t in range(V)], bool)
    top_mask = torch.zeros(V);
    for t in top_ids:
        top_mask[t] = 1.0
    top_mask = top_mask.to(DEV)

    # w_freq
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    W['dir'] = None
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0).numpy(); Oc = O - O.mean(0)
    tgt_lf = np.log(freq[nxt] + 1.0); tc = tgt_lf - tgt_lf.mean()
    w = Oc.T @ tc; w = w / (np.linalg.norm(w) + 1e-9)
    w_freq = torch.tensor(w, dtype=torch.float32, device=DEV)
    rng = np.random.default_rng(0); rr = rng.standard_normal(D); rr /= np.linalg.norm(rr)
    w_rand = torch.tensor(rr, dtype=torch.float32, device=DEV)

    curve = {}
    for a in ALPHAS:
        tm, fce, rce = run(fresh, w_freq, a, top_mask, is_freq)
        curve[a] = {'top20_mass': round(tm, 5), 'freq_CE': round(fce, 4),
                    'rare_CE': round(rce, 4)}
        print(f'alpha {a}: top20 mass {tm:.4f}  freq-CE {fce:.4f}  rare-CE {rce:.4f}',
              flush=True)
    rand_mass = [run(fresh, w_rand, a, top_mask, is_freq)[0] for a in ALPHAS]

    tms = [curve[a]['top20_mass'] for a in ALPHAS]
    mono = all(tms[i] >= tms[i + 1] - 1e-5 for i in range(len(tms) - 1))
    fces = [curve[a]['freq_CE'] for a in ALPHAS]; rces = [curve[a]['rare_CE'] for a in ALPHAS]
    p0 = True
    pa = mono and tms[-1] < tms[0]
    pb = rces[-1] < rces[0] and fces[-1] > fces[0]
    rand_mono = all(rand_mass[i] >= rand_mass[i + 1] - 1e-5 for i in range(len(rand_mass) - 1))
    null_ok = not rand_mono
    print(f'\n(a) monotonic freq-mass dial (down as alpha up): {pa} '
          f'({tms[0]:.3f}->{tms[-1]:.3f})', flush=True)
    print(f'(b) trade-off (rare-CE down, freq-CE up): {pb} '
          f'(rare {rces[0]:.3f}->{rces[-1]:.3f}, freq {fces[0]:.3f}->{fces[-1]:.3f})',
          flush=True)
    print(f'NULL random not monotonic: {null_ok} (rand mass {[round(x,3) for x in rand_mass]})',
          flush=True)

    out = {'curve': {str(k): v for k, v in curve.items()},
           'random_top20_mass': [round(x, 5) for x in rand_mass],
           'pred_a_dial': bool(pa), 'pred_b_tradeoff': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
