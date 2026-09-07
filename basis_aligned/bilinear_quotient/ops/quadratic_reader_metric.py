"""Gauge-invariant metric for bilinear-MLP quadratic reader subspaces."""
# BQGATE: LIBRARY
from __future__ import annotations


def hidden_metric(left, right):
    """Materialize the PSD hidden coefficient metric for repeated small-subspace comparisons."""
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("left/right must have identical [hidden,input] shapes")
    return .5 * ((left @ left.T) * (right @ right.T) + (left @ right.T) * (right @ left.T))


def gram_from_hidden_metric(coefficients_a, hidden_gram, coefficients_b=None):
    b = coefficients_a if coefficients_b is None else coefficients_b
    if hidden_gram.shape != (coefficients_a.shape[0], coefficients_a.shape[0]) or b.shape[0] != coefficients_a.shape[0]:
        raise ValueError("hidden metric and coefficient row dimensions disagree")
    return coefficients_a.T @ hidden_gram @ b


def quadratic_gram(torch, left, right, coefficients_a, coefficients_b=None, *, block_size=256):
    """Return Frobenius Grams of symmetric L^T diag(a) R forms without materializing them."""
    b = coefficients_a if coefficients_b is None else coefficients_b
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("left/right must have identical [hidden,input] shapes")
    if coefficients_a.ndim != 2 or b.ndim != 2 or coefficients_a.shape[0] != left.shape[0] or b.shape[0] != left.shape[0]:
        raise ValueError("coefficient matrices must have hidden rows")
    output = torch.zeros((coefficients_a.shape[1], b.shape[1]), device=left.device,
                         dtype=torch.promote_types(coefficients_a.dtype, b.dtype))
    left, right, coefficients_a, b = (x.to(output.dtype) for x in (left, right, coefficients_a, b))
    for start in range(0, left.shape[0], block_size):
        stop = min(start + block_size, left.shape[0]); li, ri = left[start:stop], right[start:stop]
        metric_block = .5 * ((li @ left.T) * (ri @ right.T) + (li @ right.T) * (ri @ left.T))
        output += coefficients_a[start:stop].T @ (metric_block @ b)
    return output


def principal_cosines(torch, gram_a, gram_b, cross_gram, *, relative_tol=1e-10):
    """Principal cosines after quotienting numerically null directions in two PSD metric Grams."""
    def whitener(gram):
        gram = .5 * (gram + gram.T); values, vectors = torch.linalg.eigh(gram)
        cutoff = values.max().clamp_min(0) * relative_tol
        keep = values > cutoff
        if not bool(keep.any()): return vectors[:, :0]
        return vectors[:, keep] / values[keep].sqrt().unsqueeze(0)
    wa, wb = whitener(gram_a), whitener(gram_b)
    if wa.shape[1] == 0 or wb.shape[1] == 0: return cross_gram.new_zeros((0,))
    return torch.linalg.svdvals(wa.T @ cross_gram @ wb).clamp(0, 1)
