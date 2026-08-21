"""ADDITIVE BIAS CATALOG -- is the frequency calibration (w_freq) the
model's ONLY rank-1 additive bias, or does block 17 carry isolable
rank-1 biases for other token properties too? Systematic use of the
validated behavior-conditioned direction method (650-658).

For candidate target-token properties {log-frequency [known], token
length, is-capitalized, is-punctuation}, compute the rank-1 direction
w_prop = cov(mlp17 output, property). The key discriminator: cos(w_prop,
w_freq). Because token properties are correlated with frequency (rare
tokens are longer, capitalized words are rarer, 657), a property's
"bias" may simply BE the frequency bias re-expressed. If w_prop aligns
with w_freq, it is the same mechanism; only a property whose w_prop is
ORTHOGONAL to w_freq AND has an independent removal effect is a genuinely
new additive component.

REGISTERED PREDICTIONS:
  (0) SANITY: w_freq self-cosine 1.0; random property gives |cos| ~ 0.03;
  (a) FREQUENCY IS THE DOMINANT AXIS: length and is-capitalized give
      w_prop strongly aligned with w_freq (|cos| >= 0.4) -- their
      "biases" are the frequency bias re-expressed (rare<->long,
      rare<->capitalized), consistent with 657;
  (b) report cos(w_prop, w_freq) and each property's correlation with
      frequency, for all candidates;
  NULL: a randomly-labelled property gives w_prop with |cos(w_freq)| ~
      random baseline -- alignment is about real property structure, not
      an artifact of the cov construction."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'additive_bias_catalog_results.json'
NFRESH = 48


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)

    # per-target-position properties
    lf = np.log(freq[nxt] + 1.0)
    strs = [cl.d1(int(t)) for t in nxt]
    length = np.array([len(s.strip()) for s in strs], dtype=np.float64)
    is_cap = np.array([1.0 if s.strip()[:1].isupper() else 0.0 for s in strs])
    is_punct = np.array([1.0 if (s.strip() and all(not c.isalnum() for c in s.strip()))
                         else 0.0 for s in strs])
    rng = np.random.default_rng(0)
    rand_prop = rng.standard_normal(len(nxt))

    props = {'log_freq': lf, 'length': length, 'is_capitalized': is_cap,
             'is_punct': is_punct, 'random_label': rand_prop}

    # capture mlp17 output
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0).numpy(); Oc = O - O.mean(0)

    def wdir(p):
        pc = p - p.mean(); w = Oc.T @ pc; return w / (np.linalg.norm(w) + 1e-9)
    w_freq = wdir(lf)

    out = {'properties': {}}
    for name, p in props.items():
        w = wdir(p)
        cos = float(w @ w_freq)
        corr_freq = float(np.corrcoef(p, lf)[0, 1]) if name != 'log_freq' else 1.0
        out['properties'][name] = {'cos_with_wfreq': round(cos, 4),
                                   'corr_with_logfreq': round(corr_freq, 4)}
        print(f'{name:15s} cos(w_prop, w_freq) {cos:+.3f}  '
              f'corr(prop, logfreq) {corr_freq:+.3f}', flush=True)

    P = out['properties']
    p0 = abs(P['log_freq']['cos_with_wfreq'] - 1.0) < 0.01 and \
        abs(P['random_label']['cos_with_wfreq']) < 0.15
    pa = (abs(P['length']['cos_with_wfreq']) >= 0.4 and
          abs(P['is_capitalized']['cos_with_wfreq']) >= 0.4)
    null_ok = abs(P['random_label']['cos_with_wfreq']) < 0.15
    # any genuinely NEW axis? a property orthogonal to w_freq (|cos|<0.2)
    # but with real (non-random) structure
    new_axes = [n for n in ['length', 'is_capitalized', 'is_punct']
                if abs(P[n]['cos_with_wfreq']) < 0.2]
    print(f'\n(0) sanity (self-cos 1, random ~0): {p0}', flush=True)
    print(f'(a) length & capitalized align with w_freq (|cos|>=0.4): {pa}', flush=True)
    print(f'(b) candidate NEW axes (|cos(w_freq)|<0.2): {new_axes}', flush=True)
    print(f'NULL random-label near 0: {null_ok}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_freq_dominant': bool(pa),
                'candidate_new_axes': new_axes, 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
