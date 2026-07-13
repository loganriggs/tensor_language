"""Shared machinery for the basis-aligned bilinear program.

Model is a plain dict of tensors so masks/pruning/folding stay trivial:
    p = {'E': (d_model,d_in), 'L': (d_h,d_model), 'R': (d_h,d_model),
         'D': (d_model,d_h),  'U': (n_out,d_model)}
'E' and 'U' are optional (thread 2 has neither).
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- model

def forward(p, x):
    z = x @ p['E'].T if 'E' in p else x
    h = (z @ p['L'].T) * (z @ p['R'].T)
    u = h @ p['D'].T
    return u @ p['U'].T if 'U' in p else u


def fold(p):
    """Fold embedding into L,R and unembedding into D. Detached copies."""
    E = p.get('E')
    Lf = (p['L'] @ E if E is not None else p['L']).detach()
    Rf = (p['R'] @ E if E is not None else p['R']).detach()
    Df = (p['U'] @ p['D'] if 'U' in p else p['D']).detach()
    return {'Lf': Lf, 'Rf': Rf, 'Df': Df}


def interaction(p):
    """Per-output symmetric quadratic form B[c] with y_c = x^T B_c x.

    Fully basis-invariant to rotations inserted at the embedding or
    unembedding interface (it only sees the folded function).
    """
    f = fold(p)
    B = torch.einsum('ck,ki,kj->cij', f['Df'], f['Lf'], f['Rf'])
    return 0.5 * (B + B.transpose(1, 2))


# ---------------------------------------------------------------- metrics

def hoyer(w):
    """Hoyer sparsity in [0,1]: 0 = flat/dense, 1 = one-hot."""
    w = w.detach().flatten().double()
    n = w.numel()
    l2 = w.norm()
    if l2 == 0:
        return 1.0
    return float((math.sqrt(n) - w.abs().sum() / l2) / (math.sqrt(n) - 1))


def near_zero_frac(w, rel=1e-3):
    """Fraction of entries below rel * max|w|."""
    a = w.detach().abs()
    m = a.max()
    if m == 0:
        return 1.0
    return float((a < rel * m).float().mean())


def block_score(B, pairs):
    """Fraction of |B_c| mass inside class c's ground-truth block.

    pairs[c] = (i, j): class c's block spans rows/cols {i, j}.
    """
    total = B.abs().sum()
    if total == 0:
        return 0.0
    inside = 0.0
    for c, (i, j) in enumerate(pairs):
        idx = torch.tensor([i, j], device=B.device)
        inside += B[c][idx][:, idx].abs().sum()
    return float(inside / total)


def sparsity_report(p, rel=1e-3):
    rep = {}
    for k, w in p.items():
        rep[k] = {'hoyer': hoyer(w), 'zero_frac': near_zero_frac(w, rel)}
    f = fold(p)
    for k, w in f.items():
        rep[k] = {'hoyer': hoyer(w), 'zero_frac': near_zero_frac(w, rel)}
    return rep


# ---------------------------------------------------------------- data

def block_data(batch, device, n_blocks=4):
    """One block active per sample; y_c = x_{2c} * x_{2c+1} on the active class."""
    k = torch.randint(0, n_blocks, (batch,), device=device)
    a = torch.randn(batch, device=device)
    b = torch.randn(batch, device=device)
    x = torch.zeros(batch, 2 * n_blocks, device=device)
    r = torch.arange(batch, device=device)
    x[r, 2 * k] = a
    x[r, 2 * k + 1] = b
    y = torch.zeros(batch, n_blocks, device=device)
    y[r, k] = a * b
    return x, y


def squares_data(batch, m, p_active, device):
    """Sparse features, each independently active w.p. p_active, value U(-1,1).

    Target: elementwise square of the input.
    """
    active = torch.rand(batch, m, device=device) < p_active
    x = (torch.rand(batch, m, device=device) * 2 - 1) * active
    return x, x ** 2


# ---------------------------------------------------------------- training

def _apply_masks(p, masks):
    with torch.no_grad():
        for k in masks:
            p[k].mul_(masks[k])


def train(p, data_fn, steps, lr=3e-3, l1=0.0, sparse_keys=(), masks=None,
          train_keys=None, log_every=0):
    """Adam on MSE (+ optional L1 on sparse_keys). Masked entries stay zero."""
    train_keys = list(p) if train_keys is None else train_keys
    for k in p:
        p[k].requires_grad_(k in train_keys)
    opt = torch.optim.Adam([p[k] for k in train_keys], lr=lr)
    masks = masks or {}
    for step in range(steps):
        x, y = data_fn()
        loss = F.mse_loss(forward(p, x), y)
        if l1 > 0:
            loss = loss + l1 * sum(p[k].abs().sum() for k in sparse_keys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        _apply_masks(p, masks)
        if log_every and step % log_every == 0:
            print(f'  step {step:6d}  loss {loss.item():.3e}')
    for k in p:
        p[k].requires_grad_(False)
    return p


@torch.no_grad()
def eval_fvu(p, data_fn, n_batches=8):
    """Fraction of variance unexplained: MSE / Var[y]."""
    se, yy, n = 0.0, [], 0
    for _ in range(n_batches):
        x, y = data_fn()
        se += ((forward(p, x) - y) ** 2).sum().item()
        yy.append(y)
        n += y.numel()
    y = torch.cat(yy)
    var = y.var().item()
    return (se / n) / var


def _snapshot(p, masks):
    return ({k: v.detach().clone() for k, v in p.items()},
            {k: v.clone() for k, v in masks.items()})


def _restore(p, masks, snap):
    ps, ms = snap
    with torch.no_grad():
        for k in p:
            p[k].copy_(ps[k])
    for k in masks:
        masks[k].copy_(ms[k])


def remaining_frac(masks, sparse_keys):
    tot = sum(masks[k].numel() for k in sparse_keys)
    return sum(masks[k].sum().item() for k in sparse_keys) / tot


def _prune(p, masks, sparse_keys, prune_frac):
    """Zero + mask the bottom prune_frac of the REMAINING weights (global)."""
    vals = torch.cat([(p[k].abs() + torch.where(masks[k] > 0, 0.0, float('inf')))
                      .flatten() for k in sparse_keys])
    n_rem = int(sum(masks[k].sum().item() for k in sparse_keys))
    n_cut = int(prune_frac * n_rem)
    if n_cut == 0:
        return 0
    thresh = vals.kthvalue(n_cut).values
    with torch.no_grad():
        for k in sparse_keys:
            cut = (p[k].abs() <= thresh) & (masks[k] > 0)
            masks[k][cut] = 0.0
            p[k][cut] = 0.0
    return n_cut


def iterated_sparsify(p, data_fn, eval_fn, sparse_keys, l1=1e-3, lr=1e-3,
                      steps_per_iter=1500, prune_frac=0.15, degrade_fvu=0.01,
                      max_iters=40, verbose=True):
    """The user-specified protocol: [L1-train -> prune] repeated; on the first
    iterate whose post-train error exceeds degrade_fvu, revert to the previous
    (good) iterate and finetune WITHOUT L1. Returns (p, masks, history)."""
    masks = {k: torch.ones_like(p[k]) for k in sparse_keys}
    hist = []
    good = _snapshot(p, masks)
    for it in range(max_iters):
        train(p, data_fn, steps_per_iter, lr=lr, l1=l1,
              sparse_keys=sparse_keys, masks=masks)
        fvu = eval_fn(p)
        frac = remaining_frac(masks, sparse_keys)
        hist.append({'iter': it, 'frac_remaining': frac, 'fvu': fvu, 'phase': 'l1'})
        if verbose:
            print(f'  iter {it:2d}  remaining {frac:6.1%}  fvu {fvu:.3e}')
        if fvu > degrade_fvu:
            _restore(p, masks, good)
            train(p, data_fn, steps_per_iter, lr=lr, masks=masks)
            fvu = eval_fn(p)
            frac = remaining_frac(masks, sparse_keys)
            hist.append({'iter': it, 'frac_remaining': frac, 'fvu': fvu,
                         'phase': 'revert+finetune'})
            if verbose:
                print(f'  DEGRADED -> revert, finetune: remaining {frac:6.1%}  fvu {fvu:.3e}')
            break
        good = _snapshot(p, masks)
        _prune(p, masks, sparse_keys, prune_frac)
    else:
        train(p, data_fn, steps_per_iter, lr=lr, masks=masks)
        hist.append({'iter': max_iters, 'frac_remaining': remaining_frac(masks, sparse_keys),
                     'fvu': eval_fn(p), 'phase': 'final_finetune'})
    return p, masks, hist


# ---------------------------------------------------------------- misc

def random_orthogonal(n, device, dtype=torch.float32, seed=None):
    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
    q = q * torch.sign(torch.diagonal(r))
    return q.to(device=device, dtype=dtype)


def init_params(d_in, d_model, d_h, n_out, device, seed, scale=1.0,
                embed=True, unembed=True):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)

    def mat(a, b):
        return (torch.randn(a, b, generator=g) * scale / math.sqrt(b)).to(device)

    p = {'L': mat(d_h, d_model), 'R': mat(d_h, d_model), 'D': mat(d_model, d_h)}
    if embed:
        p['E'] = mat(d_model, d_in)
    if unembed:
        p['U'] = mat(n_out, d_model)
    return p
