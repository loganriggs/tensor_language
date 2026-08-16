"""Does layer 17's treatment work anywhere else? A compression profile of all 18 MLPs.

§8.4 recorded the honest limit of the layer-17 result: it worked because layer 17 is
nearly rank-4, and "whether the same treatment produces anything legible in the middle
of the network is untested and is the natural next step." This is that step, run on all
18 layers of the real model.

For each layer the bilinear MLP is REPLACED by

    y(x)  ~=  sum_{p<R} P_p * ( x^T M_p^{(k)} x )  +  mu_perp  +  Down_bias

with P the principal directions of that layer's own MLP output (R chosen to hold 90% of
its output variance, so each layer gets the budget it needs rather than a fixed one) and
each form truncated to rank k in the Lambda-weighted metric. Cross-entropy is measured
on held-out text after the swap.

Everything is reported against that layer's own scale, because the layers are not
equally important: deleting layer 3's quadratic part and deleting layer 17's cost very
different amounts, and a replacement that gives back 99% of a small thing is not the
same achievement as one that gives back 99% of a large thing. So each layer's damage is
divided by the cost of deleting it outright.

The two numbers that matter per layer: R (how many output directions carry the layer)
and the compression ratio at a fixed damage tolerance. Layer 17 sets the bar at R=4,
k=2, 0.7% damage, 1150x compression. The question is whether anything else comes close.
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

DEV = 'cuda'
N_FIT = 6000
R_CAP = 48
KS = (2, 4, 8, 16, 32)
TOL = 0.05          # damage tolerance: 5% of the cost of deleting the layer


@torch.no_grad()
def out_pcs_full(model, tokens, li, nb=32):
    store = []
    h = model.transformer.h[li].mlp.register_forward_hook(
        lambda m, i, o: store.append(o.detach().reshape(-1, o.shape[-1]).float()))
    for i in range(0, nb, 4):
        b = tokens[i:i + 4].to(DEV)
        model(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    h.remove()
    Y = torch.cat(store, 0)
    mu = Y.mean(0)
    _, Sv, V = torch.linalg.svd(Y - mu, full_matrices=False)
    ev = (Sv ** 2) / (Sv ** 2).sum()
    return V, mu, ev


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    d = cfg['n_embd']
    base = eval_ce(model, tokens, batch=4)
    out = {'ce_baseline': base, 'tol': TOL, 'r_cap': R_CAP, 'layers': {}}
    print(f'baseline CE {base:.4f}\n')
    print(f"  {'layer':>5} {'R':>4} {'delete cost':>12} {'best (R,k)':>12} "
          f"{'damage':>8} {'params':>10} {'compression':>12}")

    orig_params = 3 * d * cfg.get('n_embd_mlp', 4608)   # Left, Right, Down
    for li in range(cfg['n_layer']):
        mlp = model.transformer.h[li].mlp
        orig_forward = mlp.forward
        V, mu, ev = out_pcs_full(model, tokens, li)
        cum = ev.cumsum(0)
        R = min(int((cum < 0.90).sum()) + 1, R_CAP)
        P = V[:R]
        X = mlp_inputs(model, tokens, (li,), N_FIT)[li].to(DEV)
        S = X.T @ X / X.shape[0]
        Sh, Sih = sqrtm_psd(S)
        bias = mlp.Down_bias.detach().float() if hasattr(mlp, 'Down_bias') \
            else torch.zeros(d, device=DEV)
        forms = torch.stack([form_for_direction(mlp, P[p]) for p in range(R)])

        def ce_with(F):
            mlp.forward = Truncated(P.float(), F.float(), (mu - bias).float(),
                                    bias.float()).to(DEV).forward
            try:
                return eval_ce(model, tokens, batch=4)
            finally:
                mlp.forward = orig_forward

        ce_dead = ce_with(torch.zeros_like(forms))
        ce_proj = ce_with(forms)
        span = max(ce_dead - base, 1e-6)
        rec = {'R': R, 'ce_dead': ce_dead, 'delete_cost': ce_dead - base,
               'ce_project_only': ce_proj,
               'damage_project_only': (ce_proj - base) / span,
               'output_var_top4': float(cum[3]), 'ks': {}}
        best = None
        for k in KS:
            if k > R_CAP * 8:
                continue
            Fw = torch.stack([Sih @ truncate(Sh @ forms[p] @ Sh, k) @ Sih
                              for p in range(R)])
            ce = ce_with(Fw)
            dmg = (ce - base) / span
            rec['ks'][k] = {'ce': ce, 'damage': dmg}
            if best is None and dmg <= TOL:
                best = (R, k, dmg, R * d + R * k * d + R * k)
            del Fw
        rec['best'] = best
        comp = orig_params / best[3] if best else None
        rec['compression'] = comp
        out['layers'][li] = rec
        bs = f'({best[0]},{best[1]})' if best else 'none'
        dm = f'{100*best[2]:.1f}%' if best else '  --'
        pr = f'{best[3]:,}' if best else '--'
        cp = f'{comp:.0f}x' if comp else '--'
        print(f"  {li:>5} {R:>4} {ce_dead-base:>12.4f} {bs:>12} {dm:>8} {pr:>10} "
              f"{cp:>12}", flush=True)
        del forms, X, S, Sh, Sih
        torch.cuda.empty_cache()

    ok = [li for li, r in out['layers'].items() if r['best']]
    out['summary'] = {'n_layers_compressible': len(ok), 'n_layers': cfg['n_layer'],
                      'orig_params_per_mlp': orig_params}
    print(f"\n{len(ok)}/{cfg['n_layer']} layers reach {100*TOL:.0f}% damage within "
          f"R<={R_CAP}, k<={max(KS)}")
    if ok:
        cs = [out['layers'][li]['compression'] for li in ok]
        print(f"  compression among those: median {sorted(cs)[len(cs)//2]:.0f}x, "
              f"best {max(cs):.0f}x (layer "
              f"{max(ok, key=lambda l: out['layers'][l]['compression'])})")
    hard = [li for li, r in out['layers'].items() if not r['best']]
    if hard:
        print(f"  layers that do NOT compress at this budget: {hard}")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_depth_profile_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
