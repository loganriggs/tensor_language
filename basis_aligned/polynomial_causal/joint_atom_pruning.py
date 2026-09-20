"""Exact removal accounting for an implicit sum of symmetric tensor atoms.

Native atom k is C[:,k] outer sym(A[k] outer B[k]). A subset retains the
original coefficients; no activation fitting, scalar refit or row selection.
"""
import torch


def atom_gram(C, A, B):
    cross = A @ B.T
    return (C.T @ C) * ((A @ A.T)*(B @ B.T) + cross*cross.T) / 2


def removal_energy(gram, kept):
    removed = torch.ones(len(gram), dtype=gram.dtype, device=gram.device)
    removed[kept] = 0
    return removed @ gram @ removed


def greedy_removal_order(gram):
    """Remove the atom with smallest exact incremental squared error each step.

    This is a greedy fixed-dictionary method, not a global sparse optimum.
    Increments can be negative because removing cancelling atoms can reduce
    error. Runtime O(n²), workspace O(n²); no factor rescaling changes choices.
    """
    n = len(gram)
    scores = gram.diag().clone()
    removed = torch.zeros(n, dtype=torch.bool, device=gram.device)
    order = []
    for _ in range(n):
        i = int(scores.masked_fill(removed, torch.inf).argmin())
        order.append(i)
        removed[i] = True
        scores += 2 * gram[:, i]
    return torch.tensor(order, device=gram.device)


def subset_error_bound(gram, budgets):
    """Spectral bound for ANY native-atom subset, even with free scalar refit.

    For unit-atom Gram K and omitted coefficient vector d, error² >=
    lambda_min(K)*||d||². Every omitted native atom has coefficient ||atom||.
    Sum the smallest n-m norms to bound every size-m support. Numerical
    eigenvalues are diagnostics, not interval-arithmetic certificates.
    """
    diagonal = gram.diag()
    if not bool((diagonal > 0).all()):
        raise ValueError('remove zero atoms before spectral analysis')
    norms = diagonal.sqrt()
    normalized = gram/norms[:, None]/norms[None, :]
    normalized = (normalized+normalized.T)/2
    eigenvalues = torch.linalg.eigvalsh(normalized)
    minimum = eigenvalues[0].clamp_min(0)
    energies = diagonal.sort().values.cumsum(0)
    total = gram.sum()
    bounds = {}
    for retained in budgets:
        omitted = len(gram)-retained
        if omitted < 0 or retained < 0: raise ValueError('invalid budget')
        bounds[retained] = float((minimum*energies[omitted-1]/total).sqrt()) if omitted else 0.
    return eigenvalues, bounds
