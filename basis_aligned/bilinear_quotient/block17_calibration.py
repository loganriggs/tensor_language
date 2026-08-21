"""BLOCK17 CALIBRATION -- is block 17's suppression (624) a frequency
calibration? Does it push down COMMON tokens across the board?

624 found the readout layer (block 17) causally SUPPRESSES newline and
article (ablating it RAISES their probability). Hypothesis: block 17 is
a last-layer calibrator that pushes down high-frequency tokens (which
the earlier layers over-predict), so its suppression should scale with
token base frequency.

Method: track the mean logit of every vocabulary token over all
positions, with and without block L's contribution (mean-filled). The
per-token change delta_t = (mean logit with L removed) - (baseline) is
how much removing L raises/lowers each token. If block 17 suppresses
frequent tokens, delta_t is POSITIVE for frequent tokens (removing the
suppressor lets them rise) and correlates with token frequency.
Compared against block 1 (an early WRITER, 624) and a middle block
(near-null) as controls.

REGISTERED PREDICTIONS:
  (0) SANITY: removing block 17 raises the mean logit of the newline
      and article tokens (delta_t > 0 for them) -- consistent with 624;
  (a) CALIBRATION: over tokens with >= 20 occurrences, corr(log token
      frequency, delta_t for block 17) is POSITIVE and >= 0.3 --
      block 17 suppresses common tokens more, so they rise more when it
      is removed;
  (b) CONTRAST: block 1 (an early writer) does NOT show the same
      positive frequency correlation -- its correlation is lower than
      block 17's (a writer's removal lowers what it writes, not a
      frequency-calibration);
  (c) report the three correlations (block 1, 17, and a middle block)
      and example high/low-frequency token deltas;
  NULL: the middle block's frequency correlation is near zero (|corr|
      < 0.2) -- calibration is specific to the readout layer, not any
      block."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'block17_calibration_results.json'
NFRESH = 48
NL = [198, 628]
ART = [257, 281, 262, 383]
BLOCKS = {'block1': 1, 'block9': 9, 'block17': 17}


@torch.no_grad()
def mean_logit_per_token(fresh, ablate_block):
    V = m.lm_head.weight.shape[0]
    acc = torch.zeros(V, dtype=torch.float64)
    n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x_in = x
            x, v1 = blk(x, v1, x0)
            if ablate_block is not None and li == ablate_block:
                delta = x - x_in
                x = x_in + delta.mean(dim=(0, 1), keepdim=True)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        lg = lg.reshape(-1, V)
        acc += lg.sum(0).double().cpu()
        n += lg.shape[0]
    return (acc / n).numpy()


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)

    # token base frequency (as next-tokens in the corpus)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    V = m.lm_head.weight.shape[0]
    freq = np.bincount(nxt, minlength=V).astype(float)

    base = mean_logit_per_token(fresh, None)
    deltas = {}
    for name, L in BLOCKS.items():
        ab = mean_logit_per_token(fresh, L)
        deltas[name] = ab - base            # >0 means removing L raises token
        print(f'{name}: computed', flush=True)

    # (0) sanity on newline/article for block17
    d17 = deltas['block17']
    nl_d = float(d17[NL].mean()); art_d = float(d17[ART].mean())
    p0 = nl_d > 0 and art_d > 0
    print(f'(0) block17 removed: newline delta {nl_d:+.4f}, article delta '
          f'{art_d:+.4f} (want >0): {p0}', flush=True)

    # frequency correlation over tokens with >=20 occurrences
    keep = freq >= 20
    lf = np.log(freq[keep])
    corrs = {name: round(pearson(lf, deltas[name][keep]), 4) for name in BLOCKS}
    print(f'(a/b/c) corr(log-freq, delta) per block: {corrs}', flush=True)

    pa = corrs['block17'] >= 0.3
    pb = corrs['block1'] < corrs['block17']
    null_ok = abs(corrs['block9']) < 0.2
    print(f'(a) block17 corr>=0.3: {pa}; (b) block1 < block17: {pb}; '
          f'NULL block9 ~0: {"ok" if null_ok else "CHECK"}', flush=True)

    # example tokens: highest-frequency kept tokens and their block17 delta
    kept_idx = np.where(keep)[0]
    top_freq = kept_idx[np.argsort(-freq[kept_idx])[:12]]
    examples = [{'tok': cl.d1(int(t)), 'freq': int(freq[t]),
                 'block17_delta': round(float(d17[t]), 4)} for t in top_freq]
    print('  top-frequency tokens (tok, freq, block17 delta):', flush=True)
    for e in examples:
        print(f'    {e["tok"]!r:14} freq {e["freq"]:5d}  delta {e["block17_delta"]:+.4f}',
              flush=True)

    out = {'newline_delta17': nl_d, 'article_delta17': art_d, 'pred_0': bool(p0),
           'freq_corrs': corrs, 'pred_a_block17_calib': bool(pa),
           'pred_b_block1_contrast': bool(pb), 'null_ok': bool(null_ok),
           'n_tokens_kept': int(keep.sum()), 'examples': examples,
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
