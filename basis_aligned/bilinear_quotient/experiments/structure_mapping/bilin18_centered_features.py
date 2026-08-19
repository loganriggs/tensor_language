"""Test #1 from BILIN18_CONNECTION.md §4, in the form the code actually calls for.

`qk_analytic_bilin.py` expands each block's MLP about the data mean mu = E[xhat]:

    mlp(xhat) = Down[(L.mu) o (R.mu)] + Down_bias        constant
              + Down[diag(L.mu) R + diag(R.mu) L] d      linear in d = xhat - mu
              + Down h(d),   h(d) = (L d) o (R d)        the pure quadratic remainder

and approximates the last term by Eckart-Young: SVD of Down C^(1/2) with C = E[h h^T],
truncated to k. §2.4 registered a test -- re-rank those features by curvature rather
than by singular value, on the grounds that T3 makes size uninformative about which
features must stay curved.

Reading the code changes what that test should be. The ranking is already in the
data-weighted metric (C^(1/2) is exactly the whitening §7 validated), so "ranked by raw
size" was the wrong description of it. But the same reading exposes a straightforward
defect that T3's logic points at directly:

    C = E[h h^T] is the UNCENTERED second moment, and the constant term does NOT
    contain Down E[h].

d is centred by construction, but h is QUADRATIC in d, so E[h] is not zero -- it is the
contraction of d's covariance through L and R. The model therefore carries an unmodelled
constant, and Eckart-Young, being asked to reproduce a non-zero-mean quantity, must spend
singular directions representing it. Rank spent on a constant is rank not spent on
variation, and the constant is free to store: Down E[h] is 1152 numbers folded into the
existing bias.

THE TEST. Same pipeline, same budgets, one change:

    current    C = E[h h^T],                 constant = Down[(L.mu) o (R.mu)] + Down_bias
    centred    C = E[(h-hbar)(h-hbar)^T],    constant = the above + Down hbar
                                             and the features see (h - hbar)

Scored by held-out cross-entropy delta at matched k, with the exact-k gate that the
original script uses. PREDICTION, registered before running: the centred version wins,
by most at small k, and the size of the win tracks the share of the quadratic term's
energy that is pure constant -- which is reported alongside so the mechanism is checkable
rather than inferred.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0)
torch.set_num_threads(8)
DEV = 'cuda'
BLOCKS = [0, 1, 5, 7]
KS = [8, 32, 128]
B0 = 6

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); FH = 4608
# NOTE: the original script used /workspace/tensor_language/data_fineweb_tokens.npy,
# which is not on this box. Substituted the pile-10k sample built for the other bilin18
# runs. Both arms of the comparison see identical data, so the current-vs-centred
# contrast is unaffected; only the absolute dCE values are not comparable to the
# fineweb numbers in qk_analytic_bilin.json.
FW = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                'bilin18_eval_tokens_large.pt')
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[300:452, :128].to(DEV)
APPROX = {}


@torch.no_grad()
def fwd(idx, use=None, collect=None, acc=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D,))

        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        if collect is not None and li == collect:
            acc.append(xhat.detach())
        if use is not None and li == use:
            mu, W, bc, A, Bf, k, hbar = APPROX[li]
            d = xhat - mu
            mo = bc + d @ W
            if k > 0:
                hd = mlp.Left(d) * mlp.Right(d)
                if hbar is not None:
                    hd = hd - hbar
                mo = mo + (hd @ Bf.T) @ A.T
        else:
            mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(B, T - 1)


def held(**kw):
    return torch.cat([fwd(HELD[i:i + B0], **kw) for i in range(0, HELD.shape[0], B0)])


def main():
    t0 = time.time()
    base = held(); BASE = float(base.mean())
    print(f'base CE {BASE:.4f} on {base.numel()} held-out tokens\n')
    out = {'base_ce': BASE, 'blocks': BLOCKS, 'ks': KS, 'cells': {}}

    for li in BLOCKS:
        blk = m.transformer.h[li]; mlp = blk.mlp
        Lw, Rw = mlp.Left.weight.detach(), mlp.Right.weight.detach()
        Dw, db = mlp.Down.weight.detach(), mlp.Down_bias.detach()
        # pass 1: mean of xhat
        s = torch.zeros(D, device=DEV, dtype=torch.float64); n = 0
        for i in range(0, TRAIN.shape[0], B0):
            acc = []; fwd(TRAIN[i:i + B0], collect=li, acc=acc)
            z = acc[0].reshape(-1, D).double(); s += z.sum(0); n += z.shape[0]
        mu = (s / n).float()
        Lmu, Rmu = Lw @ mu, Rw @ mu
        W = ((Dw * Lmu[None, :]) @ Rw + (Dw * Rmu[None, :]) @ Lw).T.contiguous()
        bc = Dw @ (Lmu * Rmu) + db
        # pass 2: hbar and the uncentered second moment
        hs = torch.zeros(FH, device=DEV, dtype=torch.float64)
        C = torch.zeros(FH, FH, device=DEV, dtype=torch.float32); n = 0
        for i in range(0, TRAIN.shape[0], B0):
            acc = []; fwd(TRAIN[i:i + B0], collect=li, acc=acc)
            dd = acc[0] - mu
            h = (mlp.Left(dd) * mlp.Right(dd)).reshape(-1, FH)
            C += h.T @ h; hs += h.double().sum(0); n += h.shape[0]
        C /= n
        hbar = (hs / n).float()
        Ccen = C - torch.outer(hbar, hbar)
        # how much of the quadratic term is pure constant?
        const_energy = float((Dw @ hbar).pow(2).sum())
        tot_energy = float(torch.einsum('ij,jk,ik->', Dw, C, Dw))
        frac = const_energy / max(tot_energy, 1e-30)
        print(f'block {li}: the omitted constant Down.E[h] carries '
              f'{100*frac:.1f}% of the quadratic term\'s energy')

        def fit(Cm, k):
            ev, Qv = torch.linalg.eigh(Cm.double())
            ev = ev.clamp_min(0); keep = ev > 1e-10 * float(ev.max())
            Qk, evk = Qv[:, keep], ev[keep]
            G = (Dw.double() @ Qk) * evk.sqrt()[None, :]
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)
            kk = min(k, S.shape[0])
            A = U[:, :kk].float()
            Bf = ((S[:kk, None] * Vh[:kk]) @ (Qk * (1.0 / evk.sqrt())[None, :]).T).float()
            return A, Bf

        cell = {'const_energy_fraction': frac, 'k': {}}
        print(f"  {'k':>5} {'current dCE':>13} {'centred dCE':>13} {'improvement':>13}")
        for k in KS:
            A, Bf = fit(C, k)
            APPROX[li] = (mu, W, bc, A, Bf, k, None)
            d_cur = float((held(use=li) - base).mean())
            A2, Bf2 = fit(Ccen, k)
            APPROX[li] = (mu, W, bc + Dw @ hbar, A2, Bf2, k, hbar)
            d_cen = float((held(use=li) - base).mean())
            APPROX.pop(li, None)
            cell['k'][k] = {'dce_current': d_cur, 'dce_centered': d_cen,
                            'improvement': d_cur - d_cen}
            print(f"  {k:>5} {d_cur:>+13.4f} {d_cen:>+13.4f} {d_cur-d_cen:>+13.4f}",
                  flush=True)
        out['cells'][li] = cell
        print()
        del C, Ccen
        torch.cuda.empty_cache()

    wins = [(li, k, c['k'][k]['improvement']) for li, c in out['cells'].items()
            for k in KS]
    nwin = sum(w > 0 for _, _, w in wins)
    out['summary'] = {'n_win': nwin, 'n_total': len(wins),
                      'mean_improvement': sum(w for _, _, w in wins) / len(wins),
                      'best': max(wins, key=lambda t: t[2])}
    print(f"centring wins at {nwin}/{len(wins)} (block, k) cells; mean improvement "
          f"{out['summary']['mean_improvement']:+.4f} nats, best "
          f"{out['summary']['best'][2]:+.4f} at block {out['summary']['best'][0]} "
          f"k={out['summary']['best'][1]}")
    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_centered_features_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
