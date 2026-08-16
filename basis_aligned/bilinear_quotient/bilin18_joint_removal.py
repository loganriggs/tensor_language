"""Test #3: is "irreducibly distributed" a fact about MLP1, or about PCA?

§74 of RESULTS_l0_mdl.md reports the signature that defines the repo's boundary result:
MLP1's top-32 SVD output directions removed TOGETHER cost 0.161 nats, while the SUM of
the 32 individual removals is only 0.039 -- so individual directions account for 24% of
the effect and 76% appears only jointly. Read as: the hub is irreducibly distributed.

A5 in this program showed that exact signature -- parts individually poor, jointly
complete -- is what PCA produces when the TRUE parts overlap, even when those parts are
individually clean and causally separable. With planted parts at mutual cosine 0.64, PCA
and a sparse dictionary had identical reconstruction error at every budget while
identifying the planted parts at 0.87/0.39/0.53 (PCA) against 0.96/0.98/1.00
(dictionary). PCA's failure was invisible in error and only showed up against ground
truth. So the 24% may be a property of the basis rather than of the layer.

WHAT §78 ALREADY SETTLED, so this does not redo it. A bounded sparse-dictionary red-team
found a dictionary crosses the NAMEABILITY boundary (23/32 monosemantic against SVD's
0/32) and not the CAUSAL one (0/32 individually load-bearing; all 1212 active features
jointly reach 2.15% of the full-layer knockout). Decisive on its own terms.

WHAT IT DID NOT MEASURE is §74's actual statistic. §78 compared individual z-scores, and
a cumulative fraction against the FULL-LAYER knockout (5.57 nats) -- a different
denominator from §74's top-32 joint removal (0.161 nats). The ratio "sum of solo removals
/ joint removal at matched count" is the number A5 predicts on, and it is not in either
section. That is what this measures.

FOUR ARMS, all at 32 directions, all mean-ablated identically:
    svd32        §74's own directions, to reproduce its 24% as a gate
    dict32       a 32-atom sparse dictionary fit to the same output -- matched count
    dict4096     the top 32 atoms of a 4096-feature dictionary -- §78's actual object
    rot32        a random rotation inside svd32's span -- §74's basis-independence control,
                 which should reproduce svd32 if the signature is about the subspace

Removed energy is reported for every arm, because a fraction is only comparable at
matched energy and the arms' spans are not identical.

REGISTERED PREDICTION: A5 says the dictionary arms show a substantially higher
individually-attributable fraction. §74's own red-team says they will not (its rotations
gave 78% joint-only, unchanged). These make opposite predictions, which is the point.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0)
torch.set_num_threads(8)
DEV = 'cuda'
LAYER = 1
NDIR = 32
B0 = 6

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                'bilin18_eval_tokens_large.pt')
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[300:452, :128].to(DEV)
PATCH = {}      # li -> (Q, mean_coeff)  ablate the span of Q (orthonormal columns)


@torch.no_grad()
def fwd(idx, collect=None, acc=None):
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
        mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        if collect is not None and li == collect:
            acc.append(mo.detach().reshape(-1, D))
        if li in PATCH:
            Q, cbar = PATCH[li]
            c = mo @ Q                     # (..., r) coefficients
            mo = mo - (c - cbar) @ Q.T     # mean-ablate the span
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(B, T - 1)


def held():
    return torch.cat([fwd(HELD[i:i + B0]) for i in range(0, HELD.shape[0], B0)])


def collect_out(idx_set, li):
    acc = []
    for i in range(0, idx_set.shape[0], B0):
        fwd(idx_set[i:i + B0], collect=li, acc=acc)
    return torch.cat(acc, 0)


def _sae_once(Yn, n_feat, l1, steps, lr):
    n, d = Yn.shape
    gg = torch.Generator(device=DEV).manual_seed(0)
    W = torch.nn.Parameter(torch.randn(d, n_feat, device=DEV, generator=gg) / d ** 0.5)
    Wd = torch.nn.Parameter(torch.randn(n_feat, d, device=DEV, generator=gg)
                            / n_feat ** 0.5)
    b = torch.nn.Parameter(torch.zeros(n_feat, device=DEV))
    opt = torch.optim.Adam([W, Wd, b], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for t in range(steps):
        idx = torch.randint(0, n, (4096,), device=DEV, generator=gg)
        y = Yn[idx]
        a = F.relu(y @ W + b)
        loss = (a @ Wd - y).pow(2).sum(1).mean() + l1 * a.abs().sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        a = F.relu(Yn @ W + b)
        usage = (a > 0).float().mean(0) * Wd.norm(dim=1)
        atoms = Wd / Wd.norm(dim=1, keepdim=True).clamp_min(1e-8)
        l0 = float((a > 0).float().sum(1).mean())
        fvu = float((a @ Wd - Yn).pow(2).sum() / Yn.pow(2).sum())
    return atoms.detach(), usage.detach(), l0, fvu


def fit_sae(Y, n_feat, target_l0, steps=4000, lr=3e-3):
    """L1 sparse autoencoder on Y. The L1 weight is searched rather than fixed: a
    penalty tuned for one scaling silently collapses the dictionary under another (the
    first version of this script used 2.5 and got L0 = 0.1, i.e. no dictionary at all).
    Returns unit-norm decoder atoms at the sparsity closest to target_l0."""
    Yc = Y - Y.mean(0, keepdim=True)
    Yn = Yc / Yc.norm(dim=1).mean()
    best = None
    for l1 in (0.003, 0.01, 0.03, 0.1, 0.3):
        atoms, usage, l0, fvu = _sae_once(Yn, n_feat, l1, steps, lr)
        if best is None or abs(l0 - target_l0) < abs(best[2] - target_l0):
            best = (atoms, usage, l0, fvu, l1)
    return best


def orth(Adirs):
    """Orthonormal basis for the span of the columns of Adirs (d, r)."""
    Q, _ = torch.linalg.qr(Adirs)
    return Q[:, :Adirs.shape[1]]


def score(Q, Ybar, base):
    """joint removal of span(Q), and the sum of the solo removals of each column."""
    cbar = Ybar @ Q
    PATCH[LAYER] = (Q, cbar)
    joint = float((held() - base).mean())
    PATCH.pop(LAYER)
    solo = 0.0
    per = []
    for j in range(Q.shape[1]):
        q = Q[:, j:j + 1]
        PATCH[LAYER] = (q, Ybar @ q)
        dj = float((held() - base).mean())
        PATCH.pop(LAYER)
        per.append(dj); solo += dj
    return joint, solo, per


def main():
    t0 = time.time()
    base = held(); BASE = float(base.mean())
    print(f'base CE {BASE:.4f} on {base.numel()} held-out tokens')
    Ytr = collect_out(TRAIN, LAYER)
    Ybar = Ytr.mean(0)
    Yc = Ytr - Ybar
    tot_energy = float(Yc.pow(2).sum())
    print(f'MLP{LAYER} output collected: {tuple(Ytr.shape)}\n')
    out = {'base_ce': BASE, 'layer': LAYER, 'n_dir': NDIR, 'arms': {}}

    # --- build the four direction sets ---
    _, Sv, Vh = torch.linalg.svd(Yc, full_matrices=False)
    svd32 = Vh[:NDIR].T                                   # (d, 32) orthonormal
    g = torch.Generator(device=DEV).manual_seed(0)
    Rm = torch.linalg.qr(torch.randn(NDIR, NDIR, device=DEV, generator=g))[0]
    rot32 = svd32 @ Rm                                    # same span, rotated basis
    a32, u32, l0a, fvua, p1 = fit_sae(Ytr, NDIR, target_l0=8)
    dict32 = a32[u32.argsort(descending=True)[:NDIR]].T
    a4k, u4k, l0b, fvub, p2 = fit_sae(Ytr, 4096, target_l0=40)
    dict4096 = a4k[u4k.argsort(descending=True)[:NDIR]].T
    out['dictionaries'] = {'d32': {'l0': l0a, 'fvu': fvua, 'l1': p1},
                           'd4096': {'l0': l0b, 'fvu': fvub, 'l1': p2}}
    print(f'dictionaries fit: 32-atom L0 {l0a:.1f} FVU {fvua:.3f} (l1={p1}) | '
          f'4096-atom L0 {l0b:.1f} FVU {fvub:.3f} (l1={p2})')
    if fvua > 0.5 or fvub > 0.5:
        print('  WARNING: a dictionary explains under half the variance; the arm built '
              'from it is weak evidence')
    print()

    arms = {'svd32': svd32, 'rot32': rot32, 'dict32': dict32, 'dict4096': dict4096}
    print(f"  {'arm':>10} {'removed energy':>15} {'joint dCE':>11} {'sum of solos':>13} "
          f"{'attributable':>13}")
    for name, Adirs in arms.items():
        Q = orth(Adirs)
        removed = float((Yc @ Q).pow(2).sum()) / tot_energy
        joint, solo, per = score(Q, Ybar, base)
        frac = solo / joint if abs(joint) > 1e-9 else float('nan')
        out['arms'][name] = {'removed_energy': removed, 'joint_dce': joint,
                             'solo_sum': solo, 'attributable_fraction': frac,
                             'rank': int(Q.shape[1]), 'per_direction': per,
                             'max_solo': max(per)}
        print(f"  {name:>10} {100*removed:>14.1f}% {joint:>+11.4f} {solo:>+13.4f} "
              f"{100*frac:>12.1f}%", flush=True)

    sv = out['arms']['svd32']['attributable_fraction']
    dm = max(out['arms']['dict32']['attributable_fraction'],
             out['arms']['dict4096']['attributable_fraction'])
    out['a5_prediction_held'] = bool(dm > 1.5 * sv)
    print(f"\n§74 reproduced at {100*sv:.0f}% attributable (it reported 24% on its own "
          f"data and directions)")
    print(f"best dictionary arm: {100*dm:.0f}% attributable")
    print(f"A5's prediction (dictionary substantially higher): "
          f"{'HELD' if out['a5_prediction_held'] else 'FAILED'}")
    print(f"§74's own basis-independence control (rot32): "
          f"{100*out['arms']['rot32']['attributable_fraction']:.0f}%")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_joint_removal_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
