"""Output-sharing quadratic blocks with new signed-square input features.

For an orthonormal output frame W, each scalar quadratic is diagonalized
independently. Its signed eigenfeatures need not be native MLP products.
Across blocks the atoms are orthogonal through W; within blocks through the
eigenvectors. Largest squared eigenvalues therefore give optimal support for
this fixed spectral dictionary. This is not optimal over output rotations.
"""
import torch


def quadratic_blocks(C, A, B, W):
    coefficients = W.T @ C
    raw = torch.einsum('ak,ki,kj->aij', coefficients, A, B)
    return (raw+raw.transpose(-1, -2))/2


def execute_component(x, readers, coefficients, writer, *, h=None, rms_eps=0.):
    """Extracted numerator component; h supplies the actual RMSNorm state."""
    value = (x @ readers).square() @ coefficients
    if h is not None:
        denominator = h.square().mean(-1)+rms_eps
        if rms_eps < 0 or not bool((denominator > 0).all()):
            raise ValueError('invalid RMS denominator')
        value = value/denominator
    return value[..., None]*writer


def allocate_squares(eigenvalues, budget):
    if budget < 0 or budget > eigenvalues.numel(): raise ValueError('invalid square budget')
    indices = eigenvalues.flatten().square().topk(budget).indices
    active = indices//eigenvalues.shape[-1]
    energy = eigenvalues.flatten()[indices].double().square().sum()
    return indices, active.unique(), energy
