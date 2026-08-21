"""MASSIVE DC VS SIGNAL -- the massive dims have a large DC offset AND
large signal (679). Which part is functional? Two interventions on the
top massive mlp17-output dims:
  (A) REMOVE DC: subtract each dim's per-position mean (keep the
      variation, remove the constant offset).
  (B) REMOVE SIGNAL: replace each dim with its per-position mean (mean-
      fill, = 677: keep the constant, remove the variation).
Measure CE for each. If the DC offset is an inert constant (absorbed by
rms_norm / the readout bias), (A) barely changes CE while (B) hurts. If
the constant is functional, (A) also hurts.

REGISTERED PREDICTIONS:
  (0) SANITY: chosen dims are massive;
  (a) SIGNAL MATTERS (from 677): removing the signal (mean-fill, B)
      raises CE substantially;
  (b) DC ROLE: report whether removing just the DC offset (A) changes CE
      -- registered guess: the DC offset is largely inert (rms_norm makes
      the stream scale/shift-tolerant), so (A) hurts far less than (B);
  (c) report dCE for A, B, and a random-dim control;
  NULL: doing A/B on random median-magnitude dims barely changes CE."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'massive_dc_vs_signal_results.json'
NFRESH = 48

W = {'dims': None, 'mode': None, 'means': None}


def hook(mo, i_, o_):
    if W['dims'] is None:
        return o_
    o2 = o_.clone()
    for k, d in enumerate(W['dims']):
        if W['mode'] == 'remove_dc':
            o2[..., d] = o_[..., d] - W['means'][k]        # keep variation, drop offset
        elif W['mode'] == 'remove_signal':
            o2[..., d] = W['means'][k]                      # keep offset, drop variation
    return o2


@torch.no_grad()
def ce(fresh, dims, mode, means):
    W['dims'] = dims; W['mode'] = mode; W['means'] = means
    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    V = m.lm_head.weight.shape[0]; tot = 0.0; n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        tot += float(F.cross_entropy(lg.view(-1, V), tg, reduction='sum')); n += tg.numel()
    hk.remove()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    # rank dims by mlp17 output RMS + get per-dim means
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    W['dims'] = None
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0).numpy(); rms = np.sqrt((O ** 2).mean(0))
    order = np.argsort(-rms)
    top = order[:8].tolist()
    med = order[D // 2 - 4:D // 2 + 4].tolist()
    means = O.mean(0)
    top_means = [float(means[d]) for d in top]
    med_means = [float(means[d]) for d in med]
    print(f'top dims {top} means {[round(x,0) for x in top_means]}', flush=True)

    base = ce(fresh, None, None, None)
    a = ce(fresh, top, 'remove_dc', top_means)
    b = ce(fresh, top, 'remove_signal', top_means)
    ra = ce(fresh, med, 'remove_dc', med_means)
    rb = ce(fresh, med, 'remove_signal', med_means)
    print(f'baseline CE {base:.4f}', flush=True)
    print(f'top remove-DC     {a:.4f} (dCE {a-base:+.4f})', flush=True)
    print(f'top remove-signal {b:.4f} (dCE {b-base:+.4f})', flush=True)
    print(f'median remove-DC {ra:.4f} ({ra-base:+.4f}); remove-signal {rb:.4f} '
          f'({rb-base:+.4f})', flush=True)

    d_dc = a - base; d_sig = b - base
    p0 = float(rms[top[0]]) > 5 * float(np.median(rms))
    pa = d_sig > 0.05
    pb = d_dc < 0.5 * d_sig
    null_ok = abs(ra - base) < 0.05 and abs(rb - base) < 0.05
    print(f'\n(0) massive: {p0}', flush=True)
    print(f'(a) signal matters (remove-signal dCE>0.05): {pa}', flush=True)
    print(f'(b) DC largely inert (remove-DC dCE < 0.5x remove-signal): {pb}', flush=True)
    print(f'NULL median dims barely matter: {null_ok}', flush=True)

    out = {'top_dims': top, 'top_means': [round(x, 1) for x in top_means],
           'baseline_CE': round(base, 4), 'remove_DC_dCE': round(d_dc, 4),
           'remove_signal_dCE': round(d_sig, 4),
           'median_remove_DC_dCE': round(ra - base, 4),
           'median_remove_signal_dCE': round(rb - base, 4),
           'pred_0': bool(p0), 'pred_a_signal': bool(pa), 'pred_b_dc_inert': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
