"""CALIBRATOR CE PROFILE -- cross-validate 627's frequency-correlation
metric against the causal CE trade-off (626). Is block 5 a GENUINE
second calibrator?

627 found two calibrators by corr(log-freq, delta): block 17 (+0.64)
and block 5 (+0.36); the rest write or are neutral. The frequency
correlation is a proxy; the ground truth for "calibrator" is the causal
CE trade-off from 626: a real calibrator, when removed, LOWERS CE at
frequent-target positions (it was suppressing correct frequent tokens)
and RAISES CE at rare-target positions (it was suppressing frequent
competitors). A writer, when removed, RAISES CE at both.

This runs the 626 CE split for a set of blocks spanning the profile:
block 5 (weak calibrator), block 6 (neutral +0.14), block 17 (strong
calibrator, reference), block 8 (strong writer, control). If block 5
shows the calibrator signature (freq-target dCE < rare-target dCE, and
freq-target dCE notably less positive or negative), the corr metric is
validated and block 5 is a real second calibrator.

REGISTERED PREDICTIONS:
  (0) SANITY: block 17 reproduces 626 (freq-target dCE < 0 < rare-
      target dCE);
  (a) BLOCK 5 CALIBRATOR: block 5's freq-target dCE is markedly lower
      than its rare-target dCE (the calibrator asymmetry), and lower
      than a writer's -- confirming block 5 calibrates;
  (b) WRITER CONTROL: block 8 (corr -0.30) RAISES CE at both freq- and
      rare-target positions (same sign) -- a writer;
  (c) report freq/rare/all dCE for all four blocks, and the calibrator
      asymmetry (rare dCE - freq dCE) per block;
  NULL: the asymmetry (rare - freq dCE) is LARGE for the calibrators
      (5, 17) and near zero or negative for the writer (8)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'calibrator_ce_profile_results.json'
NFRESH = 48
TOPK = 20
BLOCKS = {'block5': 5, 'block6': 6, 'block8': 8, 'block17': 17}


@torch.no_grad()
def ce_per_position(fresh, ablate_block):
    V = m.lm_head.weight.shape[0]
    ces = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x_in = x
            x, v1 = blk(x, v1, x0)
            if ablate_block is not None and li == ablate_block:
                delta = x - x_in
                x = x_in + delta.mean(dim=(0, 1), keepdim=True)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ce = F.cross_entropy(lg.view(-1, V), tg, reduction='none').view(B, T)
        ces[i:i + B] = ce.cpu()
    return ces.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    V = m.lm_head.weight.shape[0]
    freq = np.bincount(nxt, minlength=V)
    top_tokens = set(np.argsort(-freq)[:TOPK].tolist())
    is_freq = np.array([t in top_tokens for t in nxt])

    base = ce_per_position(fresh, None)
    bf, br = float(base[is_freq].mean()), float(base[~is_freq].mean())
    print(f'baseline CE freq-target {bf:.4f}  rare-target {br:.4f}', flush=True)

    out = {'baseline': {'freq': round(bf, 4), 'rare': round(br, 4)}, 'blocks': {}}
    for name, L in BLOCKS.items():
        ab = ce_per_position(fresh, L)
        d_all = float(ab.mean() - base.mean())
        d_f = float(ab[is_freq].mean() - bf)
        d_r = float(ab[~is_freq].mean() - br)
        asym = d_r - d_f
        out['blocks'][name] = {'dCE_all': round(d_all, 4), 'dCE_freq': round(d_f, 4),
                               'dCE_rare': round(d_r, 4), 'asymmetry': round(asym, 4)}
        print(f'{name}: dCE all {d_all:+.4f}  freq {d_f:+.4f}  rare {d_r:+.4f}  '
              f'asymmetry(rare-freq) {asym:+.4f}', flush=True)

    b17 = out['blocks']['block17']; b5 = out['blocks']['block5']
    b8 = out['blocks']['block8']
    p0 = b17['dCE_freq'] < 0 < b17['dCE_rare']
    pa = b5['dCE_freq'] < b5['dCE_rare'] and b5['asymmetry'] > b8['asymmetry']
    pb = b8['dCE_freq'] > 0 and b8['dCE_rare'] > 0
    null_ok = b5['asymmetry'] > 0.1 and b17['asymmetry'] > 0.1 and b8['asymmetry'] < b5['asymmetry']
    print(f'\n(0) block17 calibrator signature: {p0}', flush=True)
    print(f'(a) block5 calibrates (asym {b5["asymmetry"]:+.4f} > writer '
          f'{b8["asymmetry"]:+.4f}): {pa}', flush=True)
    print(f'(b) block8 writer (raises CE both): {pb}', flush=True)
    print(f'NULL asymmetry large for calibrators, small/neg for writer: '
          f'{"ok" if null_ok else "CHECK"}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_block5_calibrates': bool(pa),
                'pred_b_block8_writer': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
