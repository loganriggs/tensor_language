"""Exact diagonal gauge balancing within mixed-radix linear maps.

For adjacent stages S,T, replace them with C S,T C^-1. Each channel
update minimizes a^2*c^2+b^2/c^2 at c=sqrt(b/a). No function or CP
component changes. L-BFGS history must not be reused blindly after this map.
"""
import torch


def row_norm(weight):
    return weight.square().sum(-1).sqrt().permute(0, 2, 1).reshape(-1)


def col_norm(weight):
    return weight.square().sum(-2).sqrt().permute(0, 2, 1).reshape(-1)


def imbalance(model):
    return max(float((row_norm(a).clamp_min(1e-100).log()
                      - col_norm(b).clamp_min(1e-100).log()).abs().max())
               for a, b in zip(model.stages[:-1], model.stages[1:]))


@torch.no_grad()
def boundary(left, right):
    a, b = row_norm(left), col_norm(right)
    assert (a > 0).all() and (b > 0).all(), 'Zero channel needs separate handling'
    scale = (b / a).sqrt()
    o, s, q, r = left.shape
    left.mul_(scale.reshape(o, q, s).permute(0, 2, 1).unsqueeze(-1))
    o, s, q, r = right.shape
    right.div_(scale.reshape(o, r, s).permute(0, 2, 1).unsqueeze(-2))


@torch.no_grad()
def balance(model, max_sweeps=200, tolerance=1e-6):
    history = []
    initial = sum(float(p.square().sum()) for p in model.parameters())
    before = imbalance(model)
    for sweep in range(max_sweeps):
        indices = list(range(len(model.stages)-1))
        for j in indices + indices[::-1]:
            boundary(model.stages[j], model.stages[j+1])
        error = imbalance(model)
        energy = sum(float(p.square().sum()) for p in model.parameters())
        history.append(dict(sweep=sweep+1, energy=energy, imbalance=error))
        if error <= tolerance:
            break
    return dict(initial_energy=initial, final_energy=energy,
                energy_reduction_factor=initial/energy,
                initial_imbalance=before, final_imbalance=error,
                sweeps=sweep+1, converged=error <= tolerance, history=history)
