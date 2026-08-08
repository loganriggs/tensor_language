"""LOCATE the induction that appears at depth 2, width 256.

The synthetic battery says +0.084 nats there (floor 0.017) and the per-head
sweep says dropping LAYER-0 HEAD 1 removes 87% of it.  A "we found the
induction head" claim needs more than one ablation, so this file asks the
three questions that would falsify it:

  1. Is layer-0 head 1 a PREVIOUS-TOKEN head?  Measured as the share of its
     attention-pattern mass at distance 1, against every other head and
     against the model's own distance profile.
  2. Does layer-1's selection actually READ that head?  Measured by deleting
     only head 1's write from layer 1's READ vector (the residual is left
     untouched, so nothing else changes) and re-running the battery.  If the
     induction survives, the circuit story is wrong.
  3. Is the pair the whole circuit?  Joint ablation of the named layer-0 and
     layer-1 heads versus their individual ablations.

Nothing here names a sign.  Every number is a behavioural delta produced by an
intervention on the real model.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import tf_interp as I1
import tf_interp2 as I2

HERE = os.path.dirname(os.path.abspath(__file__))
_rms = I2._rms


@torch.no_grad()
def distance_profile_per_head(D, n_seq=32, T=256):
    """Share of |pattern| mass at each distance, per head, per layer."""
    arr = I2.tf_corpus.load_split(D.V, 'held', n_seq, tok=D.cfg.tok)
    x = torch.from_numpy(arr[:, :T]).to(D.dev)
    out = {}
    ar = torch.arange(T, device=D.dev)
    dist = (ar[:, None] - ar[None, :])
    tot = {}
    for a in range(0, x.shape[0], 8):
        b = x[a:a + 8]
        P = D.run(b)
        for li in range(D.L):
            hn = P['read'][li]
            pat = D._pat_layer0(b) if li == 0 else D._pat_from(li, hn)
            ap = pat.abs()
            for d in (0, 1, 2, 3, 4, 8, 16):
                m = (dist == d).float()
                k = (li, d)
                tot[k] = tot.get(k, 0.0) + (ap * m).sum((0, 2, 3))
            k = (li, 'all')
            tot[k] = tot.get(k, 0.0) + ap.sum((0, 2, 3))
    for li in range(D.L):
        allm = tot[(li, 'all')]
        for d in (0, 1, 2, 3, 4, 8, 16):
            out[f'l{li}_share_at_distance_{d}'] = [
                float(v) for v in (tot[(li, d)] / allm).cpu()]
    return out


@torch.no_grad()
def battery_with(D, attn=None, read_minus=None, mlp_read_minus=None, seeds=5):
    """The induction battery with an intervention, using the folded pipeline.

    `read_minus=(li, h)`      : delete layer-0 head h's write from layer li's
                                Q/K/V READ only; the residual is untouched.
    `mlp_read_minus=(li, h)`  : delete it from what MLP li SQUARES only.
    These two exhaust the routes by which a layer-0 head can reach layer 1."""
    def fwd(z):
        if read_minus is None and mlp_read_minus is None:
            return D.readout(D.run(z, attn=attn)['r'])
        if read_minus is not None:
            li, h = read_minus
            keep = D.run(z, attn={0: ('keep', h)})['A'][0]

            def rd(P):
                pre = P['e']
                for j in range(li):
                    pre = pre + P['A'][j] + P['M'][j]
                return _rms(pre - keep, D.Ws)
            return D.readout(D.run(z, attn=attn, reads={li: rd})['r'])
        li, h = mlp_read_minus
        keep = D.run(z, attn={0: ('keep', h)})['A'][0]
        return D.readout(D.run(z, attn=attn,
                               mlp_reads={li: lambda P, x: x - keep})['r'])
    r = [I1.induction_battery(D, seed=s, model=fwd) for s in range(seeds)]
    return {'induction_score_mean': float(np.mean([q['induction_score']
                                                   for q in r])),
            'induction_score_sd': float(np.std([q['induction_score']
                                                for q in r], ddof=1)),
            'bag_score_mean': float(np.mean([q['bag_score'] for q in r]))}


def main(stem, l0=1, l1=15):
    D = I2.DeepFold(stem)
    out = {'stem': stem, 'named_layer0_head': l0, 'named_layer1_head': l1}
    out['distance_profile'] = distance_profile_per_head(D)
    out['baseline'] = battery_with(D)
    out[f'drop_l0_head{l0}'] = battery_with(D, attn={0: ('drop', l0)})
    out[f'drop_l1_head{l1}'] = battery_with(D, attn={1: ('drop', l1)})
    out['drop_both'] = battery_with(D, attn={0: ('drop', l0), 1: ('drop', l1)})
    out['layer1_read_minus_l0_head'] = battery_with(D, read_minus=(1, l0))
    out['mlp0_input_minus_l0_head'] = battery_with(D, mlp_read_minus=(0, l0))
    out['mlp1_input_minus_l0_head'] = battery_with(D, mlp_read_minus=(1, l0))
    # a control: the same read-deletion for a head that is NOT the named one
    ctrl = (l0 + 1) % D.H
    out['control_layer1_read_minus_other_head'] = battery_with(
        D, read_minus=(1, ctrl))
    out['control_head'] = ctrl
    # and the KL cost of each, so 'kills the induction' is not confused with
    # 'breaks the model'
    kl = {}
    for nm, at in (('drop_l0', {0: ('drop', l0)}), ('drop_l1', {1: ('drop', l1)}),
                   ('drop_both', {0: ('drop', l0), 1: ('drop', l1)})):
        acc, n = 0.0, 0
        for x, y in I1.held_batches(D, 32, 256, 8):
            ref = D.readout(D.run(x)['r'])
            lp = F.log_softmax(ref.float(), -1)
            p = lp.exp()
            q = F.log_softmax(D.readout(D.run(x, attn=at)['r']).float(), -1)
            acc += float((p * (lp - q)).sum())
            n += y.numel()
        kl[nm] = acc / n
    out['kl_cost'] = kl
    json.dump(out, open(f'{HERE}/{stem}_induction_circuit.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k != 'distance_profile'}, indent=2))
    dp = out['distance_profile']
    print('layer0 share at distance 1 per head:',
          [round(v, 3) for v in dp['l0_share_at_distance_1']])
    print('layer1 share at distance 1 per head:',
          [round(v, 3) for v in dp['l1_share_at_distance_1']])
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'tf_vanilla_d2_w256_b8192_s0',
         int(a[1]) if len(a) > 1 else 1, int(a[2]) if len(a) > 2 else 15)
