"""V8-NO-MASK CONTROL (added for the go/no-go assessment): V8 landed at +0.115
nats vs V1 - outside the 0.05-0.08 budget - and section 108 shows the V6 dilation
mask alone cost +0.128, so the mask is the suspected cost driver. This control
trains V8nm = V5 subspace-partitioned writes + the SAME group-lasso read penalty
(coeff from qk_v8.json's sweep) but with FULL visibility (no dilation mask), same
lr, same nonzero write init, same 4122-step budget. It answers:
  (a) does dropping the mask return the model to the V5-level budget (~+0.05)?
  (b) does the group penalty ALONE convert V5's null/0.29 wiring agreement into
      readable wiring (the section-108 hypothesis), without the mask's help?
Measures held CE (paired vs V1, bf16 convention) and the wiring block: causal
consumption graph over all 169 visible pairs (fp32, 96 held seqs), weight-support
Spearman (all write pairs / effectual > 0.005 / top-10 precision), per-block
totals. Saves qk_v8_nomask.pt / qk_v8_nomask_heldloss.npy; results merged into
qk_v8.json under 'nomask'.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch

import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_deeproute_train_2 as R2
from qk_deeproute_train import DEPTH

QK = Q.QK
EFFECTUAL = 0.005

# register the control variant: sum kernel + V5 projections, NO mask, nonzero init
R.KERNEL['V8nm'] = 'sum'
R.PROJ.add('V8nm')
R.WRITE_INIT.add('V8nm')


def make_v8nm():
    torch.manual_seed(Q.SEED)
    return V8T.V8Route('V8nm', DEPTH).to('cuda')


def load_v8nm():
    ck = torch.load(f'{QK}/qk_v8_nomask.pt', map_location='cuda',
                    weights_only=False)
    m = make_v8nm()
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


def main():
    Q.gpu_guard(min_free=7000)
    out = json.load(open(f'{QK}/qk_v8.json'))
    LR = out['lrsweep']['chosen']
    COEFF = out['coeffsweep']['chosen']

    if not os.path.exists(f'{QK}/qk_v8_nomask.pt'):
        print(f"==== training V8nm control (lr {LR}, group coeff {COEFF}) ====",
              flush=True)
        V8T.train_v8(LR, COEFF, V8T.STEPS, factory=make_v8nm,
                     save_stem='qk_v8_nomask')

    # ---- CE paired vs V1 ----
    lv = np.load(f'{QK}/qk_v8_nomask_heldloss.npy')
    l1 = np.load(f'{QK}/qk_deeproute_heldloss_V1.npy')
    dd = lv - l1
    ds = dd.reshape(len(Q.HELD), Q.T).mean(1)
    nm = {'held_ce': float(lv.mean()),
          'minus_V1': float(dd.mean()),
          'minus_V1_se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
          'minus_V1_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds))),
          'lr': LR, 'group_coeff': COEFF}
    print(json.dumps(nm, indent=1), flush=True)

    # ---- wiring agreement ----
    model, ck = load_v8nm()
    base, means = R2.base_and_means(model)
    nm['base_ce_fp32_abl'] = round(base, 5)
    dce = R2.consumption_graph(model, base, means)
    totals = {}
    for j in range(DEPTH):
        totals[j] = round(sum(dce[li].get(si, 0.0) for li in dce
                              for si in (1 + 2 * j, 2 + 2 * j) if si in dce[li]), 5)
    nm['per_block_total_consumption'] = {str(j): totals[j] for j in totals}
    nm['dead_blocks_below_0.001'] = [j for j in totals if totals[j] < 0.001]
    scores, _ = R2.weight_wiring(model, 'V8nm')
    wp = [(li, si) for li in scores for si in scores[li]]
    sup = [scores[li][si] for li, si in wp]
    cau = [dce[li][si] for li, si in wp]
    nm['wiring_n_pairs'] = len(wp)
    nm['wiring_spearman_all'] = round(R2.spearman(sup, cau), 4)
    eff = [k for k in range(len(wp)) if cau[k] > EFFECTUAL]
    nm['wiring_n_effectual'] = len(eff)
    nm['wiring_spearman_effectual'] = round(
        R2.spearman([sup[k] for k in eff], [cau[k] for k in eff]), 4)
    top_sup = set(np.argsort(sup)[::-1][:10].tolist())
    top_cau = set(np.argsort(cau)[::-1][:10].tolist())
    nm['wiring_top10_precision'] = len(top_sup & top_cau) / 10.0
    nm['consumption_top15'] = [
        {'consumer': ('readout' if li == DEPTH else f'block{li}'),
         'source': R2.stream_name(si), 'dce': round(v, 5)}
        for li, si, v in sorted([(li, si, dce[li][si]) for li in dce
                                 for si in dce[li]], key=lambda p: -p[2])[:15]]
    print(f"V8nm wiring: n {len(wp)}, all {nm['wiring_spearman_all']}, "
          f"effectual ({len(eff)}) {nm['wiring_spearman_effectual']}, "
          f"top10 {nm['wiring_top10_precision']}", flush=True)
    print(f"V8nm dead blocks: {nm['dead_blocks_below_0.001']}", flush=True)

    out = json.load(open(f'{QK}/qk_v8.json'))
    out['nomask'] = nm
    json.dump(out, open(f'{QK}/qk_v8.json', 'w'), indent=2)
    print('saved qk_v8.json (nomask)', flush=True)


if __name__ == '__main__':
    main()
