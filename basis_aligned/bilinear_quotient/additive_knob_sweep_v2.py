"""ADDITIVE KNOB SWEEP v2 -- fix 674's binary-property bug (median split
put all mass in 'high' for 0/1 properties) and add candidates. For each
target-token property, remove its behavior-conditioned rank-1 direction
from mlp17 output and test for a calibrator-style CE TRADE-OFF (opposite-
signed dCE on high vs low groups), specific vs random. Binary properties
are split by ==1; continuous by median.

Candidates: frequency (positive control), length, is_capitalized,
is_punct. Only frequency is expected to be a genuine additive knob (650-
673); the rest either show no trade-off or only the frequency axis
re-expressed (657).

REGISTERED PREDICTIONS:
  (0) FREQUENCY IS A KNOB (control): opposite-signed dCE, largest
      magnitude;
  (a) NO OTHER INDEPENDENT KNOB: length, is_capitalized, is_punct either
      show no trade-off or a magnitude far below frequency's;
  (b) report dCE_high/dCE_low, trade-off flag, magnitude per property;
  NULL: random rank-1 removal gives negligible dCE for every property."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'additive_knob_sweep_v2_results.json'
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
    V = m.lm_head.weight.shape[0]; ces = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
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
    cur = fresh[:, :256].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)
    strs = [cl.d1(int(t)) for t in nxt]
    curstrs = [cl.d1(int(t)) for t in cur]
    # (value_array, is_binary)
    props = {
        'frequency': (np.log(freq[nxt] + 1.0), False),
        'length': (np.array([len(s.strip()) for s in strs], float), False),
        'is_capitalized': (np.array([1.0 if s.strip()[:1].isupper() else 0.0 for s in strs]), True),
        'is_punct': (np.array([1.0 if (s.strip() and all(not c.isalnum() for c in s.strip()))
                               else 0.0 for s in strs]), True),
        'after_quote': (np.array([1.0 if '"' in cs else 0.0 for cs in curstrs]), True),
    }
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
    for name, (p, is_bin) in props.items():
        hi = (p == 1) if is_bin else (p >= np.median(p))
        lo = ~hi
        if hi.sum() < 20 or lo.sum() < 20:
            out['properties'][name] = {'skipped': f'hi={int(hi.sum())} lo={int(lo.sum())}'}
            print(f'{name:15s} skipped (hi {int(hi.sum())} lo {int(lo.sum())})', flush=True)
            continue
        w = Oc.T @ (p - p.mean()); w = w / (np.linalg.norm(w) + 1e-9)
        ce = ce_all(fresh, torch.tensor(w, dtype=torch.float32, device=DEV))
        d_hi = float(ce[hi].mean() - base[hi].mean())
        d_lo = float(ce[lo].mean() - base[lo].mean())
        rd = float(rand_ce[hi].mean() - base[hi].mean())
        tradeoff = (d_hi * d_lo) < 0
        mag = abs(d_hi) + abs(d_lo)
        out['properties'][name] = {'dCE_high': round(d_hi, 4), 'dCE_low': round(d_lo, 4),
                                   'tradeoff': bool(tradeoff), 'magnitude': round(mag, 4),
                                   'rand_dCE_high': round(rd, 4),
                                   'n_hi': int(hi.sum()), 'n_lo': int(lo.sum())}
        print(f'{name:15s} dCE_high {d_hi:+.4f} dCE_low {d_lo:+.4f}  '
              f'trade-off {tradeoff}  mag {mag:.4f}  (rand {rd:+.3f})', flush=True)

    P = out['properties']
    valid = {n: v for n, v in P.items() if 'magnitude' in v}
    p0 = valid.get('frequency', {}).get('tradeoff', False)
    freq_mag = valid.get('frequency', {}).get('magnitude', 0)
    pa = all(freq_mag >= 2 * valid[n]['magnitude'] for n in valid if n != 'frequency')
    null_ok = all(abs(valid[n]['rand_dCE_high']) < 0.05 for n in valid)
    print(f'\n(0) frequency knob: {p0}; (a) no other independent knob (freq >= 2x): {pa}',
          flush=True)
    print(f'NULL random negligible: {null_ok}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_only_freq': bool(pa), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
