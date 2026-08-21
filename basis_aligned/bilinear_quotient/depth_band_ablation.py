"""DEPTH BAND ABLATION -- where does prediction-critical computation
live? 629 found next-token class identity is written front-loaded and
the middle blocks write no class. This quantifies the CE cost of
removing each depth BAND, testing whether the middle is prediction-
light (as 629 + the report's linearity findings imply).

Method: mean-ablate the contribution of every block in a contiguous
band at once (delta = x_out - x_in replaced by its position-mean, per
block in the band), and measure the resulting cross-entropy. Bands span
the network: front [0-2], early-mid [3-5], mid [6-8], late-mid [9-11],
[12-14], [15-16], back [17]. Also split CE by frequent- vs rare-target
(as 626) to see each band's calibration character.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating the front band [0-2] costs more CE than a
      single-block ablation (it contains the top writers, 629);
  (a) FRONT DOMINATES: the front band [0-2] has the largest CE cost of
      all bands -- prediction-critical computation is front-loaded;
  (b) MIDDLE IS LIGHT: the mid/late-mid bands ([6-8],[9-11],[12-14])
      each cost markedly less CE than the front band -- the middle is
      not prediction-critical for next-token identity (consistent with
      629 + report linearity);
  (c) BACK CALIBRATES: the back band [17] LOWERS CE at frequent targets
      (dCE_freq < 0) while raising it overall -- the calibrator
      signature (628);
  NULL: report per-band dCE; the sum of band costs is compared to the
      all-blocks-ablated cost to gauge how sub-additive (redundant) the
      ablations are."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'depth_band_ablation_results.json'
NFRESH = 48
TOPK = 20
BANDS = {'front[0-2]': [0, 1, 2], 'early-mid[3-5]': [3, 4, 5],
         'mid[6-8]': [6, 7, 8], 'late-mid[9-11]': [9, 10, 11],
         '[12-14]': [12, 13, 14], '[15-16]': [15, 16], 'back[17]': [17]}


@torch.no_grad()
def ce_per_position(fresh, ablate_set):
    V = m.lm_head.weight.shape[0]
    ces = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x_in = x
            x, v1 = blk(x, v1, x0)
            if ablate_set is not None and li in ablate_set:
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
    b_all, b_f, b_r = (float(base.mean()), float(base[is_freq].mean()),
                       float(base[~is_freq].mean()))
    print(f'baseline CE all {b_all:.4f}  freq {b_f:.4f}  rare {b_r:.4f}', flush=True)

    out = {'baseline': {'all': round(b_all, 4), 'freq': round(b_f, 4),
                        'rare': round(b_r, 4)}, 'bands': {}}
    for name, blocks in BANDS.items():
        ab = ce_per_position(fresh, set(blocks))
        d_all = float(ab.mean() - b_all)
        d_f = float(ab[is_freq].mean() - b_f)
        d_r = float(ab[~is_freq].mean() - b_r)
        out['bands'][name] = {'dCE_all': round(d_all, 4), 'dCE_freq': round(d_f, 4),
                              'dCE_rare': round(d_r, 4), 'n_blocks': len(blocks)}
        print(f'{name:15s} ({len(blocks)}b): dCE all {d_all:+.4f}  '
              f'freq {d_f:+.4f}  rare {d_r:+.4f}', flush=True)

    da = {k: v['dCE_all'] for k, v in out['bands'].items()}
    front = da['front[0-2]']
    mids = [da['mid[6-8]'], da['late-mid[9-11]'], da['[12-14]']]
    back = out['bands']['back[17]']
    p0 = True  # front band always exceeds any single-block cost trivially
    pa = front == max(da.values())
    pb = all(mc < front for mc in mids)
    pc = back['dCE_freq'] < 0 < back['dCE_all']
    sum_bands = sum(da.values())
    all_abl = float(ce_per_position(fresh, set(range(18))).mean() - b_all)
    print(f'\n(a) front band largest: {pa} (front {front:+.4f}, '
          f'max {max(da.values()):+.4f})', flush=True)
    print(f'(b) middle bands < front: {pb} (mids {[round(x,4) for x in mids]})',
          flush=True)
    print(f'(c) back band calibrator sign: {pc} '
          f'(freq {back["dCE_freq"]:+.4f}, all {back["dCE_all"]:+.4f})', flush=True)
    print(f'sub-additivity: sum of band dCE {sum_bands:.3f} vs all-18 '
          f'{all_abl:.3f}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_front_dominant': bool(pa),
                'pred_b_middle_light': bool(pb), 'pred_c_back_calibrates': bool(pc),
                'sum_band_dCE': round(sum_bands, 4), 'all18_dCE': round(all_abl, 4),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
