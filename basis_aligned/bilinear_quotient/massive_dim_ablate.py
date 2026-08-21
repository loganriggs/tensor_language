"""MASSIVE DIM ABLATE -- are the massive-activation dims (676: persistent
645, 990; block-17 top 981, 990, 645, 329, 992) CAUSAL calibration
channels, or inert high-magnitude sinks? 676 found 88% of the frequency-
calibration direction w_freq lives on the outlier dims. Test: mean-ablate
the top-K massive dims in mlp17's output and measure the freq/rare CE
trade-off (the calibration signature, 626). If they carry the
calibration, ablating them helps frequent-target CE and hurts rare-target
CE, like removing w_freq (651). Control: mean-ablate K random dims.

REGISTERED PREDICTIONS:
  (0) SANITY: the top massive dims have far larger RMS than random dims
      (reproduces 676);
  (a) CAUSAL CALIBRATION: mean-ablating the top-K massive dims of mlp17
      output produces the calibrator trade-off -- freq-target CE drops,
      rare-target CE rises (like w_freq removal, 651);
  (b) SPECIFIC: K random dims do NOT produce the trade-off (they hurt
      both or neither);
  (c) report freq/rare dCE for top-K massive vs K random, K in {3,8};
  NULL: random-dim ablation is not a calibrator trade-off."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'massive_dim_ablate_results.json'
NFRESH = 48
TOPK = 20

W = {'dims': None}


def hook(mo, i_, o_):
    if W['dims'] is None:
        return o_
    o2 = o_.clone()
    mean = o_.mean(dim=(0, 1), keepdim=True)
    for d in W['dims']:
        o2[..., d] = mean[..., d]
    return o2


@torch.no_grad()
def ce_split(fresh, dims, is_freq):
    W['dims'] = dims
    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    V = m.lm_head.weight.shape[0]; ces = []; labs = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ces.append(F.cross_entropy(lg.view(-1, V), tg, reduction='none').cpu().numpy())
        labs.append(np.array([is_freq[int(t)] for t in tg.cpu().numpy()]))
    hk.remove()
    ce = np.concatenate(ces); lab = np.concatenate(labs)
    return float(ce[lab].mean()), float(ce[~lab].mean())


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V)
    top = set(np.argsort(-freq)[:TOPK].tolist())
    is_freq = np.array([1 if t in top else 0 for t in range(V)], bool)

    # rank dims by mlp17 OUTPUT RMS
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
    O = torch.cat(cap, 0).numpy()
    rms = np.sqrt((O ** 2).mean(0))
    order = np.argsort(-rms)
    top_dims = order[:8].tolist()
    print(f'top massive mlp17-output dims: {top_dims} '
          f'(RMS {[round(float(rms[d]),1) for d in top_dims]}, median {np.median(rms):.1f})',
          flush=True)

    base_f, base_r = ce_split(fresh, None, is_freq)
    rng = np.random.default_rng(0)
    out = {'top_dims': top_dims, 'base_freq': round(base_f, 4), 'base_rare': round(base_r, 4),
           'conds': {}}
    for K in [3, 8]:
        tf, tr = ce_split(fresh, top_dims[:K], is_freq)
        rdims = rng.choice(D, size=K, replace=False).tolist()
        rf, rr = ce_split(fresh, rdims, is_freq)
        out['conds'][f'top{K}'] = {'dCE_freq': round(tf - base_f, 4),
                                   'dCE_rare': round(tr - base_r, 4)}
        out['conds'][f'rand{K}'] = {'dCE_freq': round(rf - base_f, 4),
                                    'dCE_rare': round(rr - base_r, 4)}
        print(f'K={K}: top-massive dCE_freq {tf-base_f:+.4f} dCE_rare {tr-base_r:+.4f} | '
              f'random dCE_freq {rf-base_f:+.4f} dCE_rare {rr-base_r:+.4f}', flush=True)

    c8 = out['conds']['top8']; r8 = out['conds']['rand8']
    p0 = float(rms[top_dims[0]]) > 5 * float(np.median(rms))
    pa = c8['dCE_freq'] < 0 < c8['dCE_rare']
    pb = not (r8['dCE_freq'] < 0 < r8['dCE_rare'])
    null_ok = pb
    print(f'\n(0) massive dims real: {p0}', flush=True)
    print(f'(a) top-massive ablation = calibrator trade-off: {pa}', flush=True)
    print(f'(b) random dims not a trade-off: {pb}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_causal_calib': bool(pa),
                'pred_b_specific': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
