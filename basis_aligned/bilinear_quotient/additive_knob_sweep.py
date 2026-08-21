"""ADDITIVE KNOB SWEEP -- the clean CAUSAL close of "is frequency the only
additive knob" (redoing the 659 cosine catalog, which was confounded,
with the removal test instead). For each candidate token property, remove
its behavior-conditioned rank-1 direction from mlp17 output and check for
a calibrator-style CE TRADE-OFF along that property axis (removal helps
the high-property targets and hurts the low, or vice versa), specific vs
random.

Candidates (target-token properties): frequency (known knob, positive
control), token length, is-capitalized. Metric: split targets by the
property (high/low), measure dCE_high and dCE_low on removing w_prop. An
additive knob shows OPPOSITE-signed dCE (a trade-off); a non-knob shows
same-signed or negligible dCE. Report vs a random-direction removal.

REGISTERED PREDICTIONS:
  (0) FREQUENCY IS A KNOB (positive control): removing w_freq gives
      opposite-signed dCE (freq-target down, rare-target up) -- the 626
      trade-off;
  (a) FREQUENCY DOMINATES: frequency's trade-off magnitude (|dCE_high| +
      |dCE_low|) is the largest of the candidates -- it is the primary
      additive axis;
  (b) report each property's dCE_high/dCE_low and trade-off flag; note
      that length/capitalization correlate with frequency, so any knob
      they show is likely the frequency axis re-expressed (657);
  NULL: removing a random rank-1 direction gives negligible, non-trade-off
      dCE for every property."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'additive_knob_sweep_results.json'
NFRESH = 48

W = {'dir': None}


def hook(mo, i_, o_):
    if W['dir'] is None:
        return o_
    d = W['dir']
    return o_ - (o_ @ d)[..., None] * d


@torch.no_grad()
def ce_all(fresh, remove_dir):
    W['dir'] = remove_dir
    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    V = m.lm_head.weight.shape[0]
    ces = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ces.append(F.cross_entropy(lg.view(-1, V), tg, reduction='none').cpu().numpy())
    hk.remove()
    return np.concatenate(ces)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)
    strs = [cl.d1(int(t)) for t in nxt]
    props = {
        'frequency': np.log(freq[nxt] + 1.0),
        'length': np.array([len(s.strip()) for s in strs], float),
        'is_capitalized': np.array([1.0 if s.strip()[:1].isupper() else 0.0 for s in strs]),
    }
    # capture mlp17 output for behavior-conditioned dirs
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

    base = ce_all(fresh, None)
    rng = np.random.default_rng(0); rr = rng.standard_normal(D); rr /= np.linalg.norm(rr)
    rand_ce = ce_all(fresh, torch.tensor(rr, dtype=torch.float32, device=DEV))

    out = {'properties': {}}
    for name, p in props.items():
        hi = p >= np.median(p); lo = ~hi
        w = Oc.T @ (p - p.mean()); w = w / (np.linalg.norm(w) + 1e-9)
        ce = ce_all(fresh, torch.tensor(w, dtype=torch.float32, device=DEV))
        d_hi = float(ce[hi].mean() - base[hi].mean())
        d_lo = float(ce[lo].mean() - base[lo].mean())
        rd_hi = float(rand_ce[hi].mean() - base[hi].mean())
        rd_lo = float(rand_ce[lo].mean() - base[lo].mean())
        tradeoff = (d_hi * d_lo) < 0            # opposite signs
        mag = abs(d_hi) + abs(d_lo)
        out['properties'][name] = {'dCE_high': round(d_hi, 4), 'dCE_low': round(d_lo, 4),
                                   'tradeoff': bool(tradeoff), 'magnitude': round(mag, 4),
                                   'rand_dCE_high': round(rd_hi, 4), 'rand_dCE_low': round(rd_lo, 4)}
        print(f'{name:15s} dCE_high {d_hi:+.4f} dCE_low {d_lo:+.4f}  '
              f'trade-off {tradeoff}  mag {mag:.4f}  (rand {rd_hi:+.3f}/{rd_lo:+.3f})',
              flush=True)

    P = out['properties']
    p0 = P['frequency']['tradeoff']
    pa = P['frequency']['magnitude'] == max(P[n]['magnitude'] for n in props)
    null_ok = abs(P['frequency']['rand_dCE_high']) < 0.3 * abs(P['frequency']['dCE_high'])
    print(f'\n(0) frequency is a knob (trade-off): {p0}', flush=True)
    print(f'(a) frequency dominates (largest magnitude): {pa}', flush=True)
    print(f'NULL random negligible: {null_ok}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_freq_dominates': bool(pa),
                'null_ok': bool(null_ok), 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
