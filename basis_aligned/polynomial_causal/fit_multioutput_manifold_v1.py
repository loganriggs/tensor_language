"""Variable-projection PR+ descent on overlapping quadratic block terms.

Within-block row frames are Stiefel; normalized symmetric cores are spheres.
No mutual orthogonality between blocks. Conditional writers are exact.
"""
import json
import math
import time
import torch


def tangent(point, vector):
    b, c = point
    gb, gc = vector
    inner = gb @ b.transpose(-1, -2)
    return (gb - ((inner + inner.transpose(-1, -2)) / 2) @ b,
            gc - c * (gc * c).sum(-1, keepdim=True))


def retract(point, direction, step):
    b, c = [x + step * d for x, d in zip(point, direction)]
    q, r = torch.linalg.qr(b.transpose(-1, -2), mode='reduced')
    signs = torch.where(r.diagonal(dim1=-2, dim2=-1) >= 0, 1., -1.)
    return ((q * signs.unsqueeze(-2)).transpose(-1, -2),
            torch.nn.functional.normalize(c, dim=-1))


def dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def assign(model, point):
    with torch.no_grad():
        model.bank.copy_(point[0]); model.core.copy_(point[1])


def fit(model, objective, seconds=240, max_steps=100000, state=None):
    state = state or {}
    if 'model' in state:
        model.load_state_dict(state['model'])
    prev_g = state.get('previous_gradient')
    prev_p = state.get('previous_direction')
    step_hint = state.get('step_hint')
    iteration = state.get('iteration', 0)
    start_iteration = iteration
    history = list(state.get('history', []))
    start = time.perf_counter()
    backtracks = restarts = evaluations = 0
    converged = False
    reason = 'budget_limit'
    max_increase = 0.

    def evaluate(gradient=True):
        nonlocal evaluations
        evaluations += 1
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(gradient):
            loss, residual, energy, writer, gram, reg = objective.terms(model)
            if gradient:
                loss.backward()
                g = tangent((model.bank.detach(), model.core.detach()),
                            (model.bank.grad.detach(), model.core.grad.detach()))
            else:
                g = None
        if not torch.isfinite(loss):
            raise ArithmeticError('Nonfinite block objective')
        return loss.detach(), residual.detach(), energy.detach(), writer.detach(), gram.detach(), reg.detach(), g

    result = evaluate()
    initial_loss = float(result[0])

    def diagnose():
        nonlocal converged
        loss, residual, energy, writer, gram, reg, g = result
        captured = max(1 - float(residual), 1e-12)
        b, c = model.bank.detach(), model.core.detach()
        orth = float((b @ b.transpose(-1, -2) - torch.eye(model.rank, device=b.device, dtype=b.dtype)).abs().max())
        sphere = float((c.norm(dim=-1) - 1).abs().max())
        row = dict(iteration=iteration, optimization_loss=float(loss),
                   squared_relative_error=float(residual), captured_energy_fraction=captured,
                   component_energy_over_native_total=float(energy),
                   relative_stationarity=max(float(g[0].norm()) * math.sqrt(model.groups * model.rank), float(g[1].norm()) * math.sqrt(model.groups * model.outputs)) / captured,
                   gradient_max_abs=max(float(x.abs().max()) for x in g),
                   orthogonality_error=orth, sphere_error=sphere,
                   gram_diagonal_error=float((gram.diag() - 1).abs().max()),
                   regularized_gram_condition=float(torch.linalg.cond(reg)))
        if not all(math.isfinite(v) for v in row.values()) or max(orth, sphere, row['gram_diagonal_error']) > 1e-10:
            raise ArithmeticError('Invalid block manifold state')
        if row['regularized_gram_condition'] > (model.groups * model.outputs + objective.penalty) / objective.penalty * (1 + 1e-8):
            raise ArithmeticError('Regularized Gram exceeds theoretical condition bound')
        if history and history[-1]['iteration'] == iteration:
            history[-1] = row
        else:
            history.append(row)
        if len(history) >= 5:
            row['relative_progress_five_checks'] = abs(row['optimization_loss'] - history[-5]['optimization_loss']) / captured
            converged = (row['relative_progress_five_checks'] <= 1e-5 and
                         row['relative_stationarity'] <= 1e-4 and row['gradient_max_abs'] <= 1e-7)
        print(json.dumps(row), flush=True)

    diagnose()
    while time.perf_counter() - start < seconds and iteration - start_iteration < max_steps and not converged:
        point = (model.bank.detach().clone(), model.core.detach().clone())
        g = result[-1]
        direction = tuple(-x for x in g)
        if prev_g is not None:
            transported = tangent(point, prev_g)
            beta = max(0., min(10., float(dot(g, tuple(x-y for x,y in zip(g, transported))) / dot(prev_g, prev_g).clamp_min(1e-30))))
            direction = tuple(-x + beta*y for x,y in zip(g, tangent(point, prev_p)))
            if float(dot(g, direction)) > -.1 * float(dot(g, g)):
                direction = tuple(-x for x in g); restarts += 1
        slope = float(dot(g, direction))
        capture = max(1-float(result[1]), 1e-12)
        hint = .01 * capture / max(-slope, 1e-30) if step_hint is None else step_hint
        step = min(hint, 1e6, 1 / max(math.sqrt(float(dot(direction, direction))), 1e-30))
        old = float(result[0]); accepted = False
        for _ in range(25):
            assign(model, retract(point, direction, step))
            trial = evaluate(False)
            if float(trial[0]) <= old + 1e-4 * step * slope:
                accepted = True; break
            backtracks += 1; step *= .5
        if not accepted:
            assign(model, point); reason = 'line_search_failed'; break
        prev_g = tuple(x.clone() for x in g)
        prev_p = tuple(x.clone() for x in direction)
        step_hint = step * 1.5
        iteration += 1
        result = evaluate()
        max_increase = max(max_increase, float(result[0]) - old)
        if max_increase > 1e-10:
            raise ArithmeticError('Block objective increased')
        if iteration % 5 == 0:
            diagnose()
    if history[-1]['iteration'] != iteration:
        diagnose()
    if converged:
        reason = 'projected_stationarity_and_plateau'
    return dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},
                writer=result[3], previous_gradient=prev_g, previous_direction=prev_p,
                step_hint=step_hint, iteration=iteration, history=history, converged=converged,
                terminal_reason=reason, initial_loss=initial_loss, maximum_objective_increase=max_increase,
                chunk_seconds=time.perf_counter()-start, chunk_evaluations=evaluations,
                chunk_backtracks=backtracks, chunk_direction_restarts=restarts)
