"""BLOCK17 CALIBRATION CE -- is block 17's frequency suppression (625)
a NET-BENEFICIAL calibration? Test the trade-off it must make.

625 showed block 17 suppresses common tokens in proportion to their
frequency (corr +0.64). If that is genuine calibration (the earlier
"writer" blocks over-predict frequent tokens, and block 17 trims them),
it should make a specific TRADE-OFF:
  - at positions whose TARGET is a RARE token, suppressing the frequent
    COMPETITORS helps -> removing block 17 RAISES CE there;
  - at positions whose TARGET is a FREQUENT token, suppressing the
    (correct) frequent token hurts -> removing block 17 LOWERS CE there.
A net-beneficial calibrator has the first effect outweigh the second.

Method: measure per-position cross-entropy with vs without a block's
contribution (mean-filled). Split positions by whether the target next
token is in the top-20 most frequent tokens (FREQUENT-target) or not
(RARE-target). Report the CE change at each. Block 1 (an early writer,
625 corr -0.28) is the control -- a writer should raise CE at both.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating block 17 changes overall CE measurably;
  (a) TRADE-OFF at RARE targets: removing block 17 RAISES CE at
      rare-target positions (its suppression of frequent competitors
      was helping);
  (b) TRADE-OFF at FREQUENT targets: removing block 17 LOWERS CE at
      frequent-target positions (its suppression of the correct
      frequent token was hurting) -- opposite sign to (a), the
      calibration trade-off;
  (c) NET: report the overall CE change from removing block 17 (net
      beneficial if positive, i.e. removing it raises CE);
  NULL/CONTROL: block 1 (a writer) RAISES CE at BOTH frequent- and
      rare-target positions (same sign) -- it writes content, it does
      not make the calibration trade-off."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'block17_calibration_ce_results.json'
NFRESH = 48
TOPK = 20
BLOCKS = {'block1': 1, 'block17': 17}


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
    is_freq_target = np.array([t in top_tokens for t in nxt])
    print(f'{is_freq_target.sum()} frequent-target, {(~is_freq_target).sum()} '
          f'rare-target positions', flush=True)

    base = ce_per_position(fresh, None)
    base_all = float(base.mean())
    base_freq = float(base[is_freq_target].mean())
    base_rare = float(base[~is_freq_target].mean())
    print(f'baseline CE: all {base_all:.4f}  freq-target {base_freq:.4f}  '
          f'rare-target {base_rare:.4f}', flush=True)

    out = {'baseline_CE': {'all': base_all, 'freq_target': base_freq,
                           'rare_target': base_rare}, 'blocks': {}}
    for name, L in BLOCKS.items():
        ab = ce_per_position(fresh, L)
        d_all = float(ab.mean() - base_all)
        d_freq = float(ab[is_freq_target].mean() - base_freq)
        d_rare = float(ab[~is_freq_target].mean() - base_rare)
        out['blocks'][name] = {'dCE_all': round(d_all, 4),
                               'dCE_freq_target': round(d_freq, 4),
                               'dCE_rare_target': round(d_rare, 4)}
        print(f'{name} removed: dCE all {d_all:+.4f}  freq-target {d_freq:+.4f}  '
              f'rare-target {d_rare:+.4f}', flush=True)

    b17 = out['blocks']['block17']
    b1 = out['blocks']['block1']
    p0 = abs(b17['dCE_all']) > 0.001
    pa = b17['dCE_rare_target'] > 0                 # removing 17 hurts rare targets
    pb = b17['dCE_freq_target'] < 0                 # removing 17 helps freq targets
    # control: block 1 same sign at both (a writer)
    ctrl_ok = (b1['dCE_freq_target'] > 0) and (b1['dCE_rare_target'] > 0)
    print(f'\n(0) {p0}; (a) block17 raises CE at rare targets: {pa}; '
          f'(b) block17 lowers CE at freq targets: {pb}', flush=True)
    print(f'(c) NET block17 dCE {b17["dCE_all"]:+.4f} '
          f'({"net-beneficial calibrator" if b17["dCE_all"]>0 else "net-harmful"})',
          flush=True)
    print(f'CONTROL block1 raises CE at both (writer): {ctrl_ok} '
          f'(freq {b1["dCE_freq_target"]:+.4f}, rare {b1["dCE_rare_target"]:+.4f})',
          flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_rare_hurt': bool(pa),
                'pred_b_freq_help': bool(pb),
                'net_beneficial': bool(b17['dCE_all'] > 0),
                'control_block1_writer': bool(ctrl_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
