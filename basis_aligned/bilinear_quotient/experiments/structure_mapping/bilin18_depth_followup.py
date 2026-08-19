"""Three things the depth profile left unresolved.

`bilin18_depth_profile.py` replaced each of bilin18's 18 bilinear MLPs by R output
directions carrying rank-k forms and scored the model. Only layers 0 and 16 reached
5% of their own delete-cost. Three problems with reading that as "layer 17 is unique
and nothing else compresses":

1. LAYER 17 "FAILED" ITS OWN RESULT, which cannot be right. The discrepancy is a
   bookkeeping one: bilin18_layer17.py measured damage from the PROJECTION baseline
   (0.7% at R=4, k=2) while the profile measures it from the UNTOUCHED model, and
   layer 17's projection-to-4-directions step alone costs 8.8%. Both numbers are
   correct and they answer different questions. The profile's R came from a fixed
   "90% of output variance" rule, which for layer 17 gives R=4 -- too few. R should be
   chosen by what it costs, not by a variance rule picked in advance. So: sweep R and
   report the actual Pareto frontier of (parameters, damage) for the three layers that
   look promising.

2. THE RELATIVE TOLERANCE IS BRUTAL FOR THE MIDDLE LAYERS. Deleting layer 12's
   quadratic part costs 0.024 nats. Requiring a replacement within 5% of that means
   matching the full model to 0.0012 nats -- a far harsher absolute bar than the same
   5% imposes on layer 1 (0.283 nats). Reported alongside the absolute damage, the
   middle layers may not be "incompressible" so much as "already nearly free".

3. IF EACH MIDDLE LAYER IS NEARLY FREE, ARE THEY JOINTLY FREE? Layers 2-15 cost
   0.024-0.52 nats each to delete, summing to about 1.5. Ablations notoriously do not
   add: the rest of the network compensates when one layer goes, and cannot when all of
   them do. So delete the quadratic part of all fourteen at once and compare the joint
   cost against the sum of the individual ones. This is the cheapest available test of
   whether the middle of this network is genuinely doing little, or doing something
   collectively that no single-layer ablation can see.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language')
from tier2_model import load_elriggs, eval_ce
from bilin18_identifiable import mlp_inputs, form_for_direction
from bilin18_whitened import sqrtm_psd, truncate
from bilin18_layer17 import Truncated
from bilin18_depth_profile import out_pcs_full

DEV = 'cuda'
N_FIT = 6000
PARETO_LAYERS = (0, 16, 17)
RS = (4, 8, 16, 32, 64)
KS = (2, 4, 8, 16)
MID = tuple(range(2, 16))


def build(model, tokens, li, R):
    mlp = model.transformer.h[li].mlp
    V, mu, ev = out_pcs_full(model, tokens, li)
    P = V[:R]
    X = mlp_inputs(model, tokens, (li,), N_FIT)[li].to(DEV)
    S = X.T @ X / X.shape[0]
    Sh, Sih = sqrtm_psd(S)
    d = P.shape[1]
    bias = mlp.Down_bias.detach().float() if hasattr(mlp, 'Down_bias') \
        else torch.zeros(d, device=DEV)
    forms = torch.stack([form_for_direction(mlp, P[p]) for p in range(R)])
    return mlp, P, forms, (mu - bias).float(), bias.float(), Sh, Sih


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    d = cfg['n_embd']
    orig = 3 * d * 4608
    base = eval_ce(model, tokens, batch=4)
    out = {'ce_baseline': base, 'orig_params_per_mlp': orig}
    print(f'baseline CE {base:.4f}   (one MLP = {orig:,} parameters)\n')

    # ---------- 1. Pareto frontier for the promising layers ----------
    print('== 1. sweeping R instead of fixing it by a variance rule ==')
    print(f"  {'layer':>5} {'R':>4} {'k':>3} {'CE':>8} {'damage vs delete':>17} "
          f"{'params':>9} {'compression':>12}")
    out['pareto'] = {}
    for li in PARETO_LAYERS:
        mlp = model.transformer.h[li].mlp
        orig_fw = mlp.forward
        rows = []
        dead = None
        for R in RS:
            mlp_, P, forms, mu_q, bias, Sh, Sih = build(model, tokens, li, R)

            def ce_with(F):
                mlp.forward = Truncated(P.float(), F.float(), mu_q, bias).to(DEV).forward
                try:
                    return eval_ce(model, tokens, batch=4)
                finally:
                    mlp.forward = orig_fw

            if dead is None:
                dead = ce_with(torch.zeros_like(forms))
            span = max(dead - base, 1e-6)
            for k in KS:
                Fw = torch.stack([Sih @ truncate(Sh @ forms[p] @ Sh, k) @ Sih
                                  for p in range(R)])
                ce = ce_with(Fw)
                np_ = R * d + R * k * d + R * k
                rows.append({'R': R, 'k': k, 'ce': ce, 'damage': (ce - base) / span,
                             'params': np_, 'compression': orig / np_})
                del Fw
            del forms
            torch.cuda.empty_cache()
        # Pareto front: cheapest configuration at or under each damage level
        best5 = min((r for r in rows if r['damage'] <= 0.05),
                    key=lambda r: r['params'], default=None)
        best10 = min((r for r in rows if r['damage'] <= 0.10),
                     key=lambda r: r['params'], default=None)
        out['pareto'][li] = {'delete_cost': dead - base, 'rows': rows,
                             'best_at_5pct': best5, 'best_at_10pct': best10}
        for tag, r in (('<=5%', best5), ('<=10%', best10)):
            if r:
                print(f"  {li:>5} {r['R']:>4} {r['k']:>3} {r['ce']:>8.4f} "
                      f"{100*r['damage']:>15.1f}% {r['params']:>9,} "
                      f"{r['compression']:>11.0f}x   cheapest at {tag}")
            else:
                print(f"  {li:>5}   --  --       --  {'':>16} {'':>9} "
                      f"{'':>12}   nothing at {tag}")
        print(flush=True)

    # ---------- 2 & 3. individual vs joint ablation of the middle ----------
    print('== 2/3. the middle of the network: individually nearly free, jointly? ==')
    prof = json.load(open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                          'bilin18_depth_profile_results.json'))
    indiv = {li: prof['layers'][str(li)]['delete_cost'] for li in MID}
    ssum = sum(indiv.values())
    print(f"  deleting each of layers {MID[0]}-{MID[-1]} on its own costs "
          f"{min(indiv.values()):.4f}-{max(indiv.values()):.4f} nats")
    print(f"  sum of the individual costs: {ssum:.4f} nats")

    saved = {}
    for li in MID:
        m = model.transformer.h[li].mlp
        saved[li] = m.forward
        V, mu, ev = out_pcs_full(model, tokens, li)
        bias = m.Down_bias.detach().float() if hasattr(m, 'Down_bias') \
            else torch.zeros(d, device=DEV)
        P = V[:1]
        m.forward = Truncated(P.float(), torch.zeros(1, d, d, device=DEV),
                              (mu - bias).float(), bias.float()).to(DEV).forward
    ce_joint = eval_ce(model, tokens, batch=4)
    for li, fw in saved.items():
        model.transformer.h[li].mlp.forward = fw
    joint = ce_joint - base
    out['mid_individual_costs'] = indiv
    out['mid_sum_individual'] = ssum
    out['mid_joint_cost'] = joint
    out['superadditivity'] = joint / ssum
    print(f"  deleting ALL FOURTEEN at once costs:  {joint:.4f} nats "
          f"(CE {ce_joint:.4f})")
    print(f"  ratio joint/sum = {joint/ssum:.2f}x")
    if joint / ssum > 1.5:
        v = ('the middle layers do something COLLECTIVELY that no single-layer ablation '
             'sees -- removing them together costs far more than the sum of removing '
             'them one at a time')
    elif joint / ssum < 0.8:
        v = ('the middle layers are redundant with each other -- the network compensates '
             'for the whole group nearly as well as for any one of them')
    else:
        v = ('the middle layers are close to independent: joint damage tracks the sum of '
             'individual damage')
    out['verdict'] = v
    print(f"  -> {v}")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_depth_followup_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
