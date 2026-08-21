"""LAYERWISE WFREQ REMOVAL -- close the calibration-uniqueness question
with the validated CAUSAL removal method (not the confounded cosine of
659). Is block 17 the UNIQUE layer with an isolable rank-1 frequency-
calibration component, or do others have one?

For each layer L, w_freq_L = cov(mlp-L output, target log-freq). Remove
it (rank-1) from layer L's mlp output and measure the frequency
calibration signature: does removal help frequent-target CE and hurt
rare-target CE (the 626 calibrator sign), like block 17? A layer with an
isolable rank-1 frequency calibration shows dCE_freq < 0 < dCE_rare on
removal; writer layers do not.

REGISTERED PREDICTIONS:
  (0) SANITY: block 17 shows the calibrator removal sign (freq CE drops,
      rare CE rises) -- reproduces 650/651;
  (a) BLOCK 17 IS (NEAR-)UNIQUE: report which layers show the calibrator
      removal signature (dCE_freq < 0 and dCE_rare > 0) and its
      magnitude; registered guess: only block 17 (and at most a weak
      block 5, per 627) -- calibration is localized to the readout layer;
  (b) report per-layer dCE_freq, dCE_rare on rank-1 w_freq removal;
  NULL: removing a RANDOM rank-1 direction from block 17 does not produce
      the calibrator sign (specificity, from 651)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layerwise_wfreq_removal_results.json'
NFRESH = 48
TOPK = 20

W = {'dir': None, 'layer': None}


def make_hook(li):
    def hook(mo, i_, o_):
        if W['layer'] == li and W['dir'] is not None:
            d = W['dir']
            return o_ - (o_ @ d)[..., None] * d
        return o_
    return hook


@torch.no_grad()
def ce_split(fresh, is_freq):
    V = m.lm_head.weight.shape[0]
    ces = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ces[i:i + B] = F.cross_entropy(lg.view(-1, V), tg, reduction='none').view(B, T).cpu()
    ce = ces.reshape(-1).numpy()
    return float(ce[is_freq].mean()), float(ce[~is_freq].mean())


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V)
    top = set(np.argsort(-freq)[:TOPK].tolist())
    is_freq = np.array([t in top for t in nxt])
    tgt_lf = np.log(freq[nxt] + 1.0); tc = tgt_lf - tgt_lf.mean()
    NL = len(m.transformer.h)

    # hooks on all mlps (inactive unless W['layer'] matches)
    handles = [m.transformer.h[li].mlp.register_forward_hook(make_hook(li))
               for li in range(NL)]

    # per-layer w_freq: capture each mlp output once (no removal)
    W['layer'] = None; W['dir'] = None
    base_f, base_r = ce_split(fresh, is_freq)
    # capture all mlp outputs in one pass
    caps = {li: [] for li in range(NL)}
    hks = [m.transformer.h[li].mlp.register_forward_hook(
        (lambda li: lambda mo, i_, o_: caps[li].append(
            o_.detach().float().reshape(-1, D).cpu()))(li)) for li in range(NL)]
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    for h in hks:
        h.remove()

    rows = {}
    for li in range(NL):
        O = torch.cat(caps[li], 0).numpy(); Oc = O - O.mean(0)
        w = Oc.T @ tc; w = w / (np.linalg.norm(w) + 1e-9)
        W['layer'] = li; W['dir'] = torch.tensor(w, dtype=torch.float32, device=DEV)
        f, r = ce_split(fresh, is_freq)
        W['layer'] = None; W['dir'] = None
        dcef = f - base_f; dcer = r - base_r
        calib = dcef < 0 < dcer
        rows[li] = {'dCE_freq': round(dcef, 4), 'dCE_rare': round(dcer, 4),
                    'calibrator_sign': bool(calib)}
        print(f'L{li:2d}: dCE_freq {dcef:+.4f}  dCE_rare {dcer:+.4f}  '
              f'{"CALIBRATOR" if calib else ""}', flush=True)

    # NULL: random rank-1 removal at block 17
    g = np.random.default_rng(0); rr = g.standard_normal(D); rr /= np.linalg.norm(rr)
    W['layer'] = 17; W['dir'] = torch.tensor(rr, dtype=torch.float32, device=DEV)
    rf, rr_ = ce_split(fresh, is_freq); W['layer'] = None; W['dir'] = None
    for h in handles:
        h.remove()

    calibs = [li for li in range(NL) if rows[li]['calibrator_sign']]
    p0 = rows[17]['calibrator_sign']
    pa = calibs == [17] or (set(calibs) <= {5, 17})
    null_ok = not (rf - base_f < 0 < rr_ - base_r)
    print(f'\n(0) block17 calibrator sign: {p0}', flush=True)
    print(f'(a) calibrator layers: {calibs} (guess only 17, maybe 5): {pa}', flush=True)
    print(f'NULL random-1 at L17 no calibrator sign: {null_ok} '
          f'(dCEf {rf-base_f:+.4f}, dCEr {rr_-base_r:+.4f})', flush=True)

    out = {'baseline_freq': round(base_f, 4), 'baseline_rare': round(base_r, 4),
           'layers': rows, 'calibrator_layers': calibs,
           'random_L17_dCE_freq': round(rf - base_f, 4),
           'random_L17_dCE_rare': round(rr_ - base_r, 4),
           'pred_0': bool(p0), 'pred_a_unique': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
