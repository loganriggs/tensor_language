"""The §25 protocol applied to layers 0 and 1: how much of each leader's form is
distribution-robust computation?

§25 established, for layer 16: surrogate == full form on fineweb (in-distribution),
surrogate BEATS full form on pile (shifted) -- the remainder is distribution-specific.
The same split for the other two verified leaders. Prior expectations, registered:
layer 1's surrogate repaired 92% on pile; if its missing 8% is distribution-specific
too, it should close on fineweb. Layer 0's repaired only 66%, and its leader is a
token-identity feature; token identity does not shift between corpora the way document
mixture does, so the gap should NOT close -- the 34% should be real in-distribution
structure the rank-1 surrogate genuinely lacks.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_layer0_battery as B0mod
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from bilin18_source_folding import forward_tracked

OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_robustness_split_results.json')


def build_surrogate(li, Xh, d0):
    mlp = m.transformer.h[li].mlp
    M = form_for_direction(mlp, d0).float()
    c = torch.einsum('ni,ij,nj->n', Xh, M, Xh)
    S = (Xh.T @ Xh / Xh.shape[0]).double()
    ev, U = torch.linalg.eigh(S)
    kd = ev > 1e-8 * ev.max()
    Sih = (U[:, kd] * ev[kd].rsqrt()) @ U[:, kd].T
    Sh = (U[:, kd] * ev[kd].sqrt()) @ U[:, kd].T
    Mw = Sh @ M.double() @ Sh
    ew, Uw = torch.linalg.eigh(Mw)
    u = (Sih @ Uw[:, ew.abs().argmax()]).float(); u = u / u.norm()
    p2 = (Xh @ u) ** 2
    co = torch.linalg.lstsq(torch.stack([p2, torch.ones_like(p2)], 1),
                            c[:, None]).solution.squeeze()
    return d0, float(c.mean()), u, float(co[0]), float(co[1])


def hooks_for(pack):
    d0, cbar, u, a_, b_ = pack

    def hd(xhat, mo):
        c = mo.float() @ d0
        return mo + ((cbar - c)[..., None] * d0).to(mo.dtype)

    def hs(xhat, mo):
        c = mo.float() @ d0
        chat = a_ * (xhat.float() @ u) ** 2 + b_
        return mo + ((chat - c)[..., None] * d0).to(mo.dtype)
    return hd, hs


def ce_on(li, tokens, hook):
    B0mod.LI = li
    B0mod.COEFF_FN = hook
    try:
        tot, n = 0.0, 0
        for i in range(0, tokens.shape[0], 6):
            ce = B0mod.fwd_hook(tokens[i:i + 6].to(DEV))
            tot += float(ce.sum()); n += ce.numel()
        return tot / n
    finally:
        B0mod.COEFF_FN = None


def main():
    t0 = time.time()
    fine = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                      'fineweb_eval_tokens.pt')
    pile = FW[452:512, :257]
    out = {'layers': {}}

    # ----- layer 0 leader -----
    Y = B0mod.collect0.__wrapped__ if False else None
    accs = []
    for i in range(0, 300, 6):
        acc = []
        fwd(FW[i:i + 6, :513].to(DEV), collect=0, acc=acc)
        accs.append(acc[0])
    Y0 = torch.cat(accs)
    _, _, Vh0 = torch.linalg.svd((Y0 - Y0.mean(0)).float(), full_matrices=False)
    Q0 = orth(Vh0[:32].T)
    phi0 = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                      'bilin18_layer0_battery_results_phi.pt').mean(1)
    X0 = []
    for i in range(0, 96, 6):
        r_ = None
        import bilin18_layer0_battery as _b
        _b.LI = 0
        E_, A_, Xh_, Ah_ = _b.tracked0(FW[i:i + 6, :513].to(DEV))
        X0.append(Xh_)
    X0 = torch.cat(X0)
    pack0 = build_surrogate(0, X0, Q0[:, int(phi0.argmax())].float())

    # ----- layer 1 leader -----
    accs = []
    for i in range(0, 300, 6):
        acc = []
        fwd(FW[i:i + 6, :513].to(DEV), collect=1, acc=acc)
        accs.append(acc[0])
    Y1 = torch.cat(accs)
    _, _, Vh1 = torch.linalg.svd((Y1 - Y1.mean(0)).float(), full_matrices=False)
    Q1 = orth(Vh1[:32].T)
    X1 = []
    for i in range(0, 96, 6):
        p_, xh_, _ = forward_tracked(FW[i:i + 6, :513].to(DEV))
        X1.append(xh_)
    X1 = torch.cat(X1)
    pack1 = build_surrogate(1, X1, Q1[:, 0].float())

    print(f"  {'layer':>5} {'corpus':>9} {'baseline':>9} {'delete-b':>9} "
          f"{'surr-b':>9} {'repair':>8}")
    for li, pack in ((0, pack0), (1, pack1)):
        hd, hs = hooks_for(pack)
        rec = {}
        for tag, toks in (('pile', pile), ('fineweb', fine)):
            b_ = ce_on(li, toks, None)
            d_ = ce_on(li, toks, hd)
            s_ = ce_on(li, toks, hs)
            rep = 1 - (s_ - b_) / max(d_ - b_, 1e-9)
            rec[tag] = {'baseline': b_, 'delete_minus_base': d_ - b_,
                        'surr_minus_base': s_ - b_, 'repair': rep}
            print(f"  {li:>5} {tag:>9} {b_:>9.4f} {d_-b_:>+9.4f} {s_-b_:>+9.4f} "
                  f"{100*rep:>7.1f}%", flush=True)
        out['layers'][li] = rec

    print('\nlayer 16 (from §25): pile repair >100% (beats baseline), fineweb ~100%')
    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
