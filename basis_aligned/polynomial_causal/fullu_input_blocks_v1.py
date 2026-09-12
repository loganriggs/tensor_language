"""Exact all-output commutator contractions; symmetric inputs, no text fitting.

G is the Gram matrix of output writers, A_j=sym(l_j r_j^T).
No vocabulary-by-d-by-d tensor is materialized.
"""
import json
from pathlib import Path
import torch


def sandwich(l, r, g, x):
    """Sum_v S_v X S_v where S_v=sum_j W_vj A_j and G=W^T W."""
    lx, rx = l @ x, r @ x
    a = l.T @ ((g * (rx @ l.T)) @ r)
    b = l.T @ ((g * (rx @ r.T)) @ l)
    c = r.T @ ((g * (lx @ l.T)) @ r)
    return (a + a.T + b + c) / 4


def commutator(l, r, g, x, k=None):
    if k is None:
        k = sandwich(l, r, g, torch.eye(l.shape[1], dtype=l.dtype, device=l.device))
    return k @ x + x @ k - 2 * sandwich(l, r, g, x)


def partition_metrics(l, r, g, projector, k=None):
    """Exact cross-block energy and each side's share of incident energy.

    A quiet block can have small global leakage without meaningful decoupling.
    normalized_cut divides cross energy by both sides' incident energy.
    """
    if k is None:
        k = sandwich(l, r, g, torch.eye(l.shape[1], dtype=l.dtype, device=l.device))
    total = torch.trace(k)
    incident = torch.trace(projector @ k)
    cross = incident - torch.trace(projector @ sandwich(l, r, g, projector))
    return dict(offblock_energy_fraction=float(2 * cross / total),
                incident_fraction=float(incident / total),
                normalized_cut=float(cross / incident + cross / (total - incident)))


def controls():
    torch.set_default_dtype(torch.float64)
    rng = torch.Generator().manual_seed(120422)
    l, r = [torch.randn(11, 7, generator=rng) for _ in range(2)]
    w = torch.randn(9, 11, generator=rng)
    atoms = (l[:, :, None] * r[:, None, :] + r[:, :, None] * l[:, None, :]) / 2
    forms = torch.einsum('vj,jab->vab', w, atoms)
    x = torch.randn(7, 7, generator=rng); x = (x + x.T) / 2
    g = w.T @ w
    explicit = torch.einsum('vij,jk,vkl->il', forms, x, forms)
    error = float((sandwich(l, r, g, x) - explicit).norm() / explicit.norm())
    energy = ((forms @ x - x @ forms) ** 2).sum()
    energy_error = float(abs((x * commutator(l, r, g, x)).sum() - energy) / energy)
    q = torch.linalg.qr(torch.randn(7, 7, generator=rng)).Q
    p = q[:, :3] @ q[:, :3].T
    metrics = partition_metrics(l, r, g, p)
    offblock = p @ forms @ (torch.eye(7) - p)
    expected = float(2 * offblock.square().sum() / forms.square().sum())
    partition_error = abs(metrics['offblock_energy_fraction'] - expected)
    # An almost silent coordinate attached only through cross terms: tiny
    # global leakage, but every bit of that coordinate's signal crosses blocks.
    eps = 1e-4
    quiet_forms = torch.tensor([[[1., eps], [eps, 0.]], [[2., -eps], [-eps, 0.]]])
    quiet_l = torch.tensor([[1., 0.], [1., 0.]])
    quiet_r = torch.tensor([[1., 0.], [0., 2.]])
    quiet_w = torch.tensor([[1., eps], [2., -eps]])
    quiet = partition_metrics(quiet_l, quiet_r, quiet_w.T @ quiet_w, torch.diag(torch.tensor([0., 1.])))
    assert torch.allclose(torch.einsum('vj,jab->vab', quiet_w,
        (quiet_l[:, :, None]*quiet_r[:, None, :] + quiet_r[:, :, None]*quiet_l[:, None, :])/2), quiet_forms)
    passed = max(error, energy_error, partition_error) < 1e-12 and quiet['offblock_energy_fraction'] < 1e-7 and quiet['normalized_cut'] > .99
    return dict(sandwich_relative_error=error, commutator_energy_relative_error=energy_error,
                partition_absolute_error=partition_error, quiet_direction=quiet, all_passed=passed)


if __name__ == '__main__':
    result = controls()
    Path(__file__).with_name('FULLU_INPUT_BLOCKS_V1_CONTROL.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    assert result['all_passed']
