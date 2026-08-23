"""UNIFY the readout-override (§976) with the frequency-calibration mechanism (FINDINGS item 3: block 17 suppresses
tokens proportional to log-frequency). §976 showed the readout moves OFF the front's high-frequency function-word
default toward the specific content word. If that override IS frequency de-biasing, then the logit CHANGE across
the readout (L15 -> final) should be NEGATIVELY correlated with token log-frequency: the readout suppresses
frequent tokens and boosts rare (content) ones. Measure the per-token mean logit change from L15 to the final
layer (via logit lens) and correlate with corpus log-frequency. Contrast with the front (L0->L2) change.

REGISTERED PREDICTIONS:
  (0) SANITY: the L15->final logit change is non-trivial (the readout moves logits, per §944).
  (a) READOUT = FREQUENCY DE-BIAS: the per-token L15->final logit change is NEGATIVELY correlated with log-freq
      (frequent tokens suppressed, rare boosted) -> the readout-override toward content IS frequency de-biasing,
      unifying §976 with the frequency calibrator (item 3); the FRONT change is not strongly anti-frequency;
  (b) report corr(logit-change, log-freq) for readout (L15->final) and front (L0->L2)."""
import json, time, sys, torch
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_freq_debias_results.json'
NEVAL = 160; SEQ = 256; L_SRC = 15


def readout(x): return m.lm_head(F.rms_norm(x, (D,)))


def forward_capture(idx, layers):
    cap = {}; hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): cap[L] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mk(L)))
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    for h in hs: h.remove()
    return cap, x


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    V = int(m.lm_head.weight.shape[0])
    tgt_all = blocks.cpu().numpy()[:, 1:].reshape(-1); ct = Counter(tgt_all); Ntot = len(tgt_all)
    logfreq = np.log(np.array([ct.get(int(t), 0) for t in range(V)]) / Ntot + 1e-12)
    # accumulate mean logit at L0, L2, L15, final over all positions
    sum_l = {0: torch.zeros(V, device=DEV), 2: torch.zeros(V, device=DEV), L_SRC: torch.zeros(V, device=DEV), 'final': torch.zeros(V, device=DEV)}
    npos = 0
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous()
        cap, xf = forward_capture(idx, [0, 2, L_SRC])
        for L in [0, 2, L_SRC]:
            lg = readout(cap[L]).float().reshape(-1, V); sum_l[L] += lg.sum(0)
        lgf = readout(xf).float().reshape(-1, V); sum_l['final'] += lgf.sum(0); npos += lgf.shape[0]
    mean_l = {k: (v/npos).cpu().numpy() for k, v in sum_l.items()}
    # only tokens that appear (have meaningful freq)
    seen = np.array([ct.get(int(t), 0) > 0 for t in range(V)])
    lf = logfreq[seen]
    readout_change = (mean_l['final'] - mean_l[L_SRC])[seen]
    front_change = (mean_l[2] - mean_l[0])[seen]
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    out = {'n_tokens': int(seen.sum()), 'n_pos': npos,
           'corr_readout_change_vs_logfreq': round(corr(readout_change, lf), 4),
           'corr_front_change_vs_logfreq': round(corr(front_change, lf), 4),
           'readout_change_std': round(float(readout_change.std()), 3), 'front_change_std': round(float(front_change.std()), 3)}
    out['pred_a_readout_freq_debias'] = bool(out['corr_readout_change_vs_logfreq'] < -0.2 and out['corr_readout_change_vs_logfreq'] < out['corr_front_change_vs_logfreq'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"corr(L15->final logit change, log-freq) = {out['corr_readout_change_vs_logfreq']} (readout)", flush=True)
    print(f"corr(L0->L2 logit change,   log-freq) = {out['corr_front_change_vs_logfreq']} (front)", flush=True)
    print(f"(a) readout override = frequency de-bias (negative corr, more than front): {out['pred_a_readout_freq_debias']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
