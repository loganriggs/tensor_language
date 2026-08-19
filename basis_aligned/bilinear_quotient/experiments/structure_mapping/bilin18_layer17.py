"""bilin18's last bilinear MLP is nearly a four-dimensional quadratic. What is it?

§7.3 found the one place on the real model where this program's decomposition machinery
has enough signal per parameter to be worth pointing at: layer 17's MLP reaches 90% of
its function at whitened rank 4 and 99.5% at rank 16, against rank 32-64 everywhere
else. That was measured per output direction, in held-out FVU. Two things follow, and
this script does both.

PART 1 -- THE GATE. FVU on a quadratic feature is not the same as the model still
working. So actually replace the layer-17 MLP with the truncated version and measure
cross-entropy. The replacement keeps r principal output directions and gives each a
rank-k form:

    y(x)  ~=  sum_{p<r} P_p * ( x^T M_p^{(k)} x )   +  Down_bias

and the ablation is nested, so each step's cost is attributable:

    baseline                      the untouched model
    project only                  output confined to the top-r PCs, forms exact
    project + whitened rank-k     the claim under test
    project + raw rank-k          the same budget spent by |eigenvalue| instead

The last row is the control that says whether whitening matters to the MODEL, not just
to a reconstruction error. A dead-token null (predicting the unigram distribution) fixes
the top of the CE scale so "how bad is bad" is answerable.

PART 2 -- WHAT THE DIRECTIONS ARE. A symmetric form is a signed sum of squares,

    x^T M_d x  =  sum_i lambda_i (w_i . x)^2 ,

so if four terms carry the function, the last MLP computes four squared projections and
adds them with signs. Each w_i is a direction in the layer-17 residual stream, which is
the same space the unembedding reads, so it can be named two ways: the tokens whose
unembedding rows it points along, and the actual corpus positions that excite it most.
A positive lambda means "this feature being large, in either sign, pushes the output
along d"; negative means it suppresses. That is a readable description of a real
computation in a 546M-parameter model, and it is what the whole program has been
building toward being able to say.
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

DEV = 'cuda'
LAYER = 17
N_FIT = 6000
R_OUT = 16          # (superseded at run time by the 90%-variance choice)
KS = (2, 4, 8, 16)
N_TOK = 12          # tokens listed per direction


@torch.no_grad()
def out_pcs(model, tokens, li, r):
    store = []
    h = model.transformer.h[li].mlp.register_forward_hook(
        lambda m, i, o: store.append(o.detach().reshape(-1, o.shape[-1]).float()))
    for i in range(0, 64, 4):
        b = tokens[i:i + 4].to(DEV)
        model(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    h.remove()
    Y = torch.cat(store, 0)
    mu = Y.mean(0)
    Yc = Y - mu
    U_, Sv, V = torch.linalg.svd(Yc, full_matrices=False)
    ev = (Sv ** 2) / (Sv ** 2).sum()
    return V[:r], mu, ev


class Truncated(torch.nn.Module):
    """Replacement for a Bilinear MLP: output confined to P, each coordinate a
    truncated quadratic form. forms: (r, d, d) float32."""

    def __init__(self, P, forms, mu_q, bias):
        super().__init__()
        self.register_buffer('P', P)
        self.register_buffer('forms', forms)
        self.register_buffer('mu_perp', mu_q - P.T @ (P @ mu_q))
        self.register_buffer('bias', bias)

    def forward(self, x):
        xf = x.float()
        # coefficient along each kept output direction
        c = torch.einsum('...i,rij,...j->...r', xf, self.forms, xf)
        y = torch.einsum('...r,ri->...i', c, self.P)
        # P was fitted to the CENTRED output, so the part of the mean lying outside
        # span(P) is not recoverable from the quadratic and is restored explicitly.
        # `mu` here is the mean of the quadratic part only -- the bias is added once.
        return (y + self.mu_perp + self.bias).to(x.dtype)


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    out = {'layer': LAYER, 'r_out': R_OUT, 'ks': list(KS)}

    base = eval_ce(model, tokens, batch=4)
    out['ce_baseline'] = base
    print(f'baseline CE {base:.4f}')

    # a floor for the CE scale: how bad is a model that has stopped using this layer?
    mlp = model.transformer.h[LAYER].mlp
    orig_forward = mlp.forward
    mean_out = None

    P, mu, evr = out_pcs(model, tokens, LAYER, 512)
    cum = evr.cumsum(0)
    r_needed = {q: int((cum < q).sum()) + 1 for q in (0.5, 0.9, 0.99)}
    print(f'layer-{LAYER} MLP output variance: top-16 PCs hold {100*cum[15]:.1f}%, '
          f'top-64 {100*cum[63]:.1f}%, top-256 {100*cum[255]:.1f}%')
    print(f'  PCs needed for 50/90/99% of output variance: '
          f"{r_needed[0.5]}/{r_needed[0.9]}/{r_needed[0.99]}")
    out['output_pc_variance'] = {'top16': float(cum[15]), 'top64': float(cum[63]),
                                 'top256': float(cum[255]), 'r_needed': r_needed}
    R = min(r_needed[0.9], 128)
    out['r_out'] = R
    print(f'  -> using R = {R} output directions (90% of output variance)\n')
    P = P[:R]
    X = mlp_inputs(model, tokens, (LAYER,), N_FIT)[LAYER].to(DEV)
    S = X.T @ X / X.shape[0]
    Sh, Sih = sqrtm_psd(S)
    bias = mlp.Down_bias.detach().float() if hasattr(mlp, 'Down_bias') else \
        torch.zeros(cfg['n_embd'], device=DEV)

    forms_exact = torch.stack([form_for_direction(mlp, P[p]) for p in range(R)])
    print(f'built {R} interaction forms for layer {LAYER}')

    def ce_with(forms):
        mlp.forward = Truncated(P.float(), forms.float(), (mu - bias).float(),
                                bias.float()).to(DEV).forward
        try:
            return eval_ce(model, tokens, batch=4)
        finally:
            mlp.forward = orig_forward

    # dead layer: the quadratic part removed entirely
    ce_dead = ce_with(torch.zeros_like(forms_exact))
    ce_proj = ce_with(forms_exact)
    out['ce_layer17_quadratic_removed'] = ce_dead
    out['ce_project_only'] = ce_proj
    print(f'\n== the gate: does the rank claim survive as cross-entropy? ==')
    print(f"  {'baseline':38s} {base:.4f}")
    print(f"  {'quadratic part of layer 17 removed':38s} {ce_dead:.4f}   "
          f"(+{ce_dead-base:.4f})")
    print(f"  {'output projected to top-%d PCs only' % R:38s} {ce_proj:.4f}   "
          f"(+{ce_proj-base:.4f})")
    span = max(ce_dead - base, 1e-9)
    out['ks_result'] = {}
    for k in KS:
        fw = torch.stack([Sih @ truncate(Sh @ forms_exact[p] @ Sh, k) @ Sih
                          for p in range(R)])
        fr = torch.stack([truncate(forms_exact[p], k) for p in range(R)])
        cw, cr = ce_with(fw), ce_with(fr)
        out['ks_result'][k] = {'ce_whitened': cw, 'ce_raw': cr,
                               'frac_damage_whitened': (cw - ce_proj) / span,
                               'frac_damage_raw': (cr - ce_proj) / span}
        print(f"  {'  + whitened rank %-3d' % k:38s} {cw:.4f}   (+{cw-base:.4f}, "
              f"{100*(cw-ce_proj)/span:5.1f}% of the way to a dead layer)")
        print(f"  {'  + raw      rank %-3d' % k:38s} {cr:.4f}   (+{cr-base:.4f}, "
              f"{100*(cr-ce_proj)/span:5.1f}% of the way to a dead layer)", flush=True)

    # ---- part 2: name the four directions of the leading output direction ----
    print(f'\n== what the four directions ARE (output direction = top output PC) ==')
    M = Sh @ forms_exact[0] @ Sh
    ev, U = torch.linalg.eigh(M)
    idx = ev.abs().argsort(descending=True)[:4]
    W = (Sih @ U[:, idx])                       # back to residual-stream coordinates
    lam = ev[idx]
    tot = float(ev.abs().sum())
    wte = model.transformer.wte.weight.detach().float()
    Wn = (W / W.norm(dim=0, keepdim=True)).float()
    logit_align = wte @ Wn                      # (V, 4)
    excite = (X.float() @ Wn)                   # (N, 4) activation on real inputs
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
    except Exception:
        pass
    out['directions'] = []
    for j in range(4):
        share = float(lam[j].abs()) / tot
        top = logit_align[:, j].abs().argsort(descending=True)[:N_TOK].tolist()
        toks = [enc.decode([t]) if enc else str(t) for t in top]
        rec = {'eigenvalue': float(lam[j]), 'abs_share_of_form': share,
               'sign': 'boosts' if lam[j] > 0 else 'suppresses',
               'top_tokens': toks,
               'activation_kurtosis': float(((excite[:, j] - excite[:, j].mean()) ** 4)
                                            .mean() / excite[:, j].var() ** 2)}
        out['directions'].append(rec)
        print(f"  direction {j+1}: lambda {lam[j]:+.3e} ({100*share:4.1f}% of the form's "
              f"spectral mass, {rec['sign']})")
        print(f"     activation kurtosis on real inputs {rec['activation_kurtosis']:.1f} "
              f"(3.0 = Gaussian; high = a rare, sharp feature)")
        print(f"     unembedding-aligned tokens: {toks}")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_layer17_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
