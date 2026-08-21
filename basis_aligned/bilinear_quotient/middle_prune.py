"""HOW COMPRESSIBLE IS THE REDUNDANT MIDDLE? (§813/815 quantification). §813 showed the
middle (layers 6-11) is redundant: removing all of it costs 1.93 nats but the per-component
sum is only 0.49 (super-additive). Redundancy predicts you can drop SOME middle layers
cheaply. Measure the prune curve: ablate the k cheapest middle layers (both attn+mlp),
k=0..6, dropping in order of increasing per-layer individual benefit, and report CE cost
vs k. A concave/accelerating curve = genuine redundancy (early drops nearly free, cost
accelerates as the redundancy is exhausted). Control: dropping the k cheapest FRONT layers
(0-5) should cost far more at every k.

REGISTERED PREDICTIONS:
  (0) SANITY: k=0 reproduces full CE; k=6 (whole middle) reproduces ~1.93 nats;
  (a) REDUNDANT/CONCAVE: dropping the 2-3 cheapest middle layers costs little (< 0.4 nats),
      with cost accelerating toward k=6 -> the middle is compressible, several layers are
      spare;
  (b) CONTROL: pruning the cheapest FRONT layers costs much more per layer (front is not
      redundant transport);
  (c) report the full cost-vs-k curve for middle and front."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_prune_results.json'
NEVAL = 160
ABL = {'set': set()}


def mk_hook(w, L):
    def hook(mo, i_, o_):
        if (w, L) not in ABL['set']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
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


def layer_set(layers): return {(w, L) for L in layers for w in ('attn', 'mlp')}


def prune_curve(rows, ce_full, band_layers):
    # rank band layers by individual (whole-layer) ablation benefit, cheapest first
    perlayer = []
    for L in band_layers:
        ABL['set'] = {('attn', L), ('mlp', L)}; perlayer.append((ce_on(rows, NEVAL) - ce_full, L))
    order = [L for _, L in sorted(perlayer)]
    curve = []
    for k in range(0, len(band_layers) + 1):
        ABL['set'] = layer_set(order[:k]); curve.append(round(ce_on(rows, NEVAL) - ce_full, 4))
    ABL['set'] = set()
    return {'drop_order_cheapest_first': order, 'cost_vs_k': curve}


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hooks = []
    for L in range(18):
        for w in ('attn', 'mlp'):
            comp = m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn
            hooks.append(comp.register_forward_hook(mk_hook(w, L)))
    ABL['set'] = set(); ce_full = ce_on(rows, NEVAL)
    mid = prune_curve(rows, ce_full, list(range(6, 12)))
    front = prune_curve(rows, ce_full, list(range(0, 6)))
    for h in hooks: h.remove()
    # concavity check for middle: cost of first 3 drops vs last 3
    mc = mid['cost_vs_k']; first3 = mc[3] - mc[0]; last3 = mc[6] - mc[3]
    out = {'ce_full': round(ce_full, 4), 'middle': mid, 'front': front,
           'middle_first3_cost': round(first3, 4), 'middle_last3_cost': round(last3, 4),
           'pred_a_concave': bool(mc[3] < 0.4 and last3 > first3),
           'pred_b_front_costlier': bool(front['cost_vs_k'][3] > 2*mc[3]),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'CE_full {ce_full:.3f}', flush=True)
    print(f'MIDDLE prune cost vs k (cheapest first, drop order {mid["drop_order_cheapest_first"]}): {mc}', flush=True)
    print(f'FRONT  prune cost vs k (drop order {front["drop_order_cheapest_first"]}): {front["cost_vs_k"]}', flush=True)
    print(f'middle first-3-drops {first3:.3f} vs last-3 {last3:.3f} | (a) concave/redundant {out["pred_a_concave"]} | (b) front costlier {out["pred_b_front_costlier"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
