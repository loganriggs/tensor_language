"""Exact joint writer/core block minimization in fixed orthonormal input frames.

The rank-four squared-nuclear proximal solve is globally optimal for ONE
conditional block. Cyclic updates are not a global multi-block guarantee.
"""
import json
import math
from pathlib import Path

import torch


def pack(matrix):
    i, j = torch.triu_indices(matrix.shape[-1], matrix.shape[-1], device=matrix.device)
    scale = torch.where(i == j, 1., math.sqrt(2.))
    return matrix[..., i, j] * scale


def unpack(vector, rank):
    i, j = torch.triu_indices(rank, rank, device=vector.device)
    scale = torch.where(i == j, 1., math.sqrt(2.))
    matrix = vector.new_zeros(*vector.shape[:-1], rank, rank)
    matrix[..., i, j] = vector / scale
    matrix[..., j, i] = vector / scale
    return matrix


def shrink(singular, penalty, outputs=4):
    s = singular[:outputs]
    alpha = penalty / outputs
    for k in range(1, len(s) + 1):
        threshold = alpha * s[:k].sum() / (1 + alpha * k)
        if k == len(s) or s[k] <= threshold:
            return (s - threshold).clamp_min(0)
    raise AssertionError('No active set')


def solve_block(residual, penalty):
    left, s, right = torch.linalg.svd(residual, full_matrices=False)
    t = shrink(s, penalty)
    h = residual.new_tensor([[1, 1, 1, 1], [1, -1, 1, -1],
                             [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
    writer = (left[:, :4] * t.sqrt()) @ h
    core = (right[:4].T * t.sqrt()) @ h
    norms = core.norm(dim=0)
    safe = norms.clamp_min(torch.finfo(core.dtype).tiny)
    core = core / safe
    writer = writer * norms
    # All-zero optimum: choose legal unit cores; writer remains zero.
    core[:, norms == 0] = 0
    core[0, norms == 0] = 1
    return writer, core, s[:5]


def project_core(core, overlap):
    matrix = unpack(core.T, overlap.shape[0])
    return pack(overlap.T @ matrix @ overlap).T


def native_targets(l, r, down, bank):
    targets = []
    for frame in bank:
        a, b = l @ frame, r @ frame
        symmetric = (a[:, :, None] * b[:, None, :] + b[:, :, None] * a[:, None, :]) / 2
        targets.append(down @ pack(symmetric))
    return torch.stack(targets)


def conditional_residual(targets, bank, writers, cores, group):
    residual = targets[group].clone()
    for other in range(len(bank)):
        if other != group:
            transported = project_core(cores[other], bank[other].T @ bank[group])
            residual -= writers[other] @ transported.T
    return residual


def local_cost(residual, writer, core, penalty):
    value = writer @ core.T
    energy = (writer.square().sum(0) * core.square().sum(0)).sum()
    return value.square().sum() - 2 * (residual * value).sum() + penalty * energy


def full_cost(targets, bank, writers, cores, penalty, total):
    energy = writers[0].new_zeros(())
    cross = energy.clone()
    regularizer = energy.clone()
    for g in range(len(bank)):
        a = writers[g] @ cores[g].T
        cross += (a * targets[g]).sum()
        regularizer += (writers[g].square().sum(0) * cores[g].square().sum(0)).sum()
        for h in range(len(bank)):
            transported = project_core(cores[h], bank[h].T @ bank[g])
            energy += ((writers[g].T @ writers[h]) * (cores[g].T @ transported)).sum()
    residual = 1 + (energy - 2 * cross) / total
    return residual + penalty * regularizer / total, residual, regularizer / total


def sweep(targets, bank, writers, cores, penalty):
    maximum_increase = 0.
    for g in range(len(bank)):
        residual = conditional_residual(targets, bank, writers, cores, g)
        old = local_cost(residual, writers[g], cores[g], penalty)
        w, c, _ = solve_block(residual, penalty)
        new = local_cost(residual, w, c, penalty)
        maximum_increase = max(maximum_increase, float(new-old))
        writers[g], cores[g] = w, c
    return maximum_increase


def gaps(targets, bank, writers, cores, penalty):
    result = []
    for g in range(len(bank)):
        residual = conditional_residual(targets, bank, writers, cores, g)
        w, c, _ = solve_block(residual, penalty)
        result.append(float(local_cost(residual, writers[g], cores[g], penalty)
                            - local_cost(residual, w, c, penalty)))
    return result


def control():
    import numpy as np
    from scipy.optimize import minimize
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(535)
    d, k, o, n = 9, 4, 7, 13
    bank = torch.linalg.qr(torch.randn(2, d, k)).Q
    l, r, down = torch.randn(n, d), torch.randn(n, d), torch.randn(o, n)
    tensor = torch.einsum('ok,ki,kj->oij', down, l, r)
    tensor = (tensor + tensor.transpose(-1, -2)) / 2
    targets = native_targets(l, r, down, bank)
    target_error = max(float((targets[g] - pack(bank[g].T @ tensor @ bank[g])).norm()
                            / targets[g].norm()) for g in range(2))
    writers = torch.randn(2, o, 4)
    cores = torch.randn(2, k*(k+1)//2, 4)
    def dense():
        fit = torch.zeros_like(tensor)
        for g in range(2):
            q = bank[g] @ unpack(cores[g].T, k) @ bank[g].T
            fit += torch.einsum('om,mij->oij', writers[g], q)
        reg = (writers.square().sum(1) * cores.square().sum(1)).sum()
        return ((tensor-fit).square().sum() + .01*reg) / tensor.square().sum()
    total = tensor.square().sum()
    cost_error = abs(float(full_cost(targets, bank, writers, cores, .01, total)[0]-dense()))
    before = dense()
    increase = sweep(targets, bank, writers, cores, .01)
    after = dense()
    cost_error = max(cost_error, abs(float(full_cost(targets, bank, writers, cores, .01, total)[0]-after)))
    residual = torch.randn(o, k*(k+1)//2)
    errors = []
    for penalty in (0., .01, 1., 100.):
        w, c, s = solve_block(residual, penalty)
        expected = torch.linalg.svdvals(residual)[:4].numpy()
        alpha = penalty/4
        fun = lambda x: float(np.sum((x-expected)**2) + alpha*x.sum()**2)
        jac = lambda x: 2*(x-expected) + 2*alpha*x.sum()
        ref = minimize(fun, np.maximum(expected/(1+penalty), 0), jac=jac,
                       bounds=[(0, None)]*4, method='L-BFGS-B',
                       options={'gtol':1e-12,'ftol':1e-15,'maxiter':1000})
        ours = shrink(torch.from_numpy(expected), penalty).numpy()
        errors.append(abs(fun(ours)-fun(ref.x))/max(1., abs(fun(ref.x))))
    planted = torch.randn(o, 3) @ torch.randn(3, k*(k+1)//2)
    w, c, _ = solve_block(planted, 0.)
    recovery = float((w@c.T-planted).norm()/planted.norm())
    result = dict(target_projection_error=target_error, dense_cost_error=cost_error,
                  qp_reference_error=max(errors), planted_recovery_error=recovery,
                  before=float(before), after=float(after), maximum_local_increase=increase,
                  scope='Exact conditional solve controls; no joint/global recovery guarantee')
    assert max(target_error,cost_error,max(errors),recovery) < 1e-9
    assert after < before and increase < 1e-8
    with Path(__file__).with_name('CONDITIONAL_BLOCK_SVD_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    control()
