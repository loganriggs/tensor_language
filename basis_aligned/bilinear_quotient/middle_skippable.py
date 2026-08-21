"""IS bilin18's INERT MIDDLE SKIPPABLE? (§812 barbell capstone). §812 showed layers 6-11
contribute only 0.49 nats of per-component loss-benefit (vs the front's 10.4) — nearly
inert. Direct causal test: ablate ALL of layers 6-11's components at once (zero their
outputs) and measure the CE cost. If the middle is genuinely a quiet transport region, the
simultaneous cost should be small — the model is functionally front+back. Controls:
matched-count nulls ablating the same NUMBER of components taken from the FRONT (0-5) or
BACK (12-17), which §812 says matter far more, so those should cost much more.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating everything raises CE a lot; full-model CE reproduces;
  (a) SKIPPABLE: ablating all 12 middle components (layers 6-11) simultaneously costs
      LITTLE CE (< 0.4 nats; may exceed the 0.49 per-component sum a little via interaction,
      but stays small) -> the middle is largely skippable;
  (b) NULL/CONTROL: ablating 12 FRONT components (layers 0-5) costs FAR more (several nats),
      and 12 BACK components (12-17) more than the middle -> the low middle cost is real,
      not a generic 'ablating 12 things is cheap' effect;
  (c) report middle cost as a fraction of the front cost."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_skippable_results.json'
NEVAL = 160
ABL = {'set': set()}   # set of (which,L) to zero


def mk_hook(which, L):
    def hook(mo, i_, o_):
        if (which, L) not in ABL['set']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_
        z = torch.zeros_like(y)
        return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


def comps_in(layers):
    return {(w, L) for L in layers for w in ('attn', 'mlp')}


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hooks = []
    for L in range(18):
        for w in ('attn', 'mlp'):
            comp = m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn
            hooks.append(comp.register_forward_hook(mk_hook(w, L)))
    ABL['set'] = set(); ce_full = ce_on(rows, NEVAL)
    ABL['set'] = comps_in(range(18)); ce_all = ce_on(rows, NEVAL)
    ABL['set'] = comps_in(range(6, 12)); ce_mid = ce_on(rows, NEVAL)        # 12 middle components
    ABL['set'] = comps_in(range(0, 6)); ce_front = ce_on(rows, NEVAL)       # 12 front components
    ABL['set'] = comps_in(range(12, 18)); ce_back = ce_on(rows, NEVAL)      # 12 back components
    ABL['set'] = set()
    for h in hooks: h.remove()
    mid = ce_mid - ce_full; front = ce_front - ce_full; back = ce_back - ce_full
    out = {'ce_full': round(ce_full, 4), 'ce_ablate_all': round(ce_all, 4),
           'middle_6_11_cost': round(mid, 4), 'front_0_5_cost': round(front, 4), 'back_12_17_cost': round(back, 4),
           'middle_over_front': round(mid/max(front, 1e-6), 4), 'middle_over_back': round(mid/max(back, 1e-6), 4),
           'pred_a_skippable': bool(mid < 0.4), 'pred_b_control': bool(front > 2*mid and back > mid),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'CE_full {ce_full:.3f} | ablate-all {ce_all:.3f}', flush=True)
    print(f'MIDDLE(6-11) cost {mid:.3f} | FRONT(0-5) cost {front:.3f} | BACK(12-17) cost {back:.3f}', flush=True)
    print(f'middle/front {out["middle_over_front"]} | middle/back {out["middle_over_back"]}', flush=True)
    print(f'(a) middle skippable (<0.4): {out["pred_a_skippable"]} | (b) controls cost more: {out["pred_b_control"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
