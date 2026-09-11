"""Full-dimensional orthogonal reader dictionary by L4 maximization.

MSP (Zhai et al., JMLR 2020): A <- polar((A Y)^3 Y^T).
Y columns are weight vectors, not token activations. Finite-sample fitting
and projected stationarity do not establish the sparse generative model.
"""
import time
import torch


def polar(matrix):
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def statistics(a, y):
    z = a @ y
    objective = z.pow(4).sum() / y.shape[1]
    gradient = 4 * (z.pow(3) @ y.T) / y.shape[1]
    inner = a.T @ gradient
    tangent = gradient - a @ ((inner + inner.T) / 2)
    return objective, gradient, float(tangent.norm() / objective.abs().clamp_min(1e-30))


@torch.no_grad()
def fit(y, seed, max_steps=2000, max_seconds=120, callback=None):
    generator = torch.Generator(device=y.device).manual_seed(seed)
    a = polar(torch.randn(y.shape[0], y.shape[0], dtype=y.dtype, device=y.device, generator=generator))
    started = time.perf_counter()
    history = []
    converged = False
    stop = 'step_limit'
    objective, gradient, relative = statistics(a, y)
    history.append(dict(step=0, objective=float(objective), relative_stationarity=relative, seconds=0.))
    for step in range(1, max_steps + 1):
        a = polar(gradient)
        objective, gradient, relative = statistics(a, y)
        row = dict(step=step, objective=float(objective), relative_stationarity=relative,
                   seconds=time.perf_counter()-started)
        assert row['objective'] >= history[-1]['objective'] - 1e-10 * max(abs(row['objective']),1)
        history.append(row)
        progress = abs(row['objective']-history[-6]['objective'])/max(abs(row['objective']),1e-30) if len(history)>=6 else None
        row['five_step_relative_change'] = progress
        converged = relative <= 1e-6 and progress is not None and progress <= 1e-9
        if callback is not None and (step % 10 == 0 or converged):
            callback(row)
        if converged:
            stop = 'converged'
            break
        if row['seconds'] >= max_seconds:
            stop = 'time_limit'
            break
    return a, dict(history=history, final=history[-1], converged=converged,
                   stop=stop, seconds=time.perf_counter()-started,
                   orthogonality_error=float((a@a.T-torch.eye(a.shape[0],device=a.device,dtype=a.dtype)).norm()))


def topk_energy(a, y, k):
    z = a @ y
    return float(z.square().topk(k, dim=0).values.sum()/y.square().sum())
