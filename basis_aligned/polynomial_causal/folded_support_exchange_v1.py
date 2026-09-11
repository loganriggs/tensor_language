"""Exact one-reader quadratic and best single support exchange, weights only.

Fixed partner, writer and other products. No joint/global sparse optimum claim.
Schur-complement selection extends shared_dictionary_ols_v1/OLS completion.
Writers and targets must use the same Euclidean output metric (e.g. U-whitened).
"""
import torch


@torch.no_grad()
def quadratic(native, a, b, w, j, basis, basis_gram=None):
    na, nb, nw = native
    partner = b[j]
    out = w[:, j] @ w
    half_gradient = .5 * ((out * (b @ partner)) @ a
                         + (out * (a @ partner)) @ b)
    nc = w[:, j] @ nw
    half_gradient -= .5 * ((nc * (nb @ partner)) @ na
                           + (nc * (na @ partner)) @ nb)
    scale = .5 * w[:, j].square().sum()
    ha = scale * (partner.square().sum() * a[j] + (a[j] @ partner) * partner)
    rhs = basis @ (ha - half_gradient)
    bp = basis @ partner
    if basis_gram is None:
        basis_gram = basis @ basis.T
    gram = scale * (partner.square().sum() * basis_gram + bp[:, None] * bp[None, :])
    return gram, rhs


@torch.no_grad()
def best_exchange(gram, rhs, support):
    """Refit current support, then find best nonsingular one-drop/one-add swap.

Returns conditional coefficients and gain in unnormalized squared tensor error.
The gain excludes improvement from the initial same-support refit.
"""
    support = list(support)
    ss = gram[support][:, support]
    chol = torch.linalg.cholesky((ss + ss.T) / 2)
    inverse = torch.cholesky_inverse(chol)
    beta = torch.cholesky_solve(rhs[support, None], chol).flatten()
    v = inverse @ gram[support]
    diagonal = gram.diag() - (gram[support] * v).sum(0)
    correlation = rhs - gram[:, support] @ beta
    invdiag = inverse.diag()
    removed_diag = diagonal[None, :] + v.square() / invdiag[:, None]
    removed_corr = correlation[None, :] + beta[:, None] * v / invdiag[:, None]
    threshold = gram.diag().max() * 1e-12
    scores = removed_corr.square() / removed_diag.clamp_min(threshold)
    scores -= beta.square()[:, None] / invdiag[:, None]
    scores[removed_diag <= threshold] = -torch.inf
    scores[:, support] = -torch.inf
    flat = int(scores.argmax())
    drop, add = divmod(flat, len(gram))
    gain = float(scores[drop, add])
    if gain <= 0:
        return support, beta, dict(gain=0., best_available_gain=gain, changed=False)
    new_support = support.copy()
    new_support[drop] = add
    new_ss = gram[new_support][:, new_support]
    new_chol = torch.linalg.cholesky((new_ss + new_ss.T) / 2)
    values = torch.cholesky_solve(rhs[new_support, None], new_chol).flatten()
    replay_gain = float(rhs[new_support] @ values - rhs[support] @ beta)
    return new_support, values, dict(gain=gain, replay_gain=replay_gain,
                                    changed=True, dropped=support[drop], added=add)
