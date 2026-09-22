"""Bounded profiled optimization; checkpoint every valid line-search evaluation."""
import torch

class EvaluationBudget(Exception):
    pass

def fit(params, evaluate, max_iter=250, max_evaluations=500, report=None):
    optimizer = torch.optim.LBFGS(params, lr=1, max_iter=max_iter,
        max_eval=max_evaluations, tolerance_grad=1e-12,
        tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")
    calls = 0
    best = early = None
    initial = normalizer = None
    normal = 0.0
    history = []
    budget_stop = False
    def closure():
        nonlocal calls, best, early, initial, normalizer, normal
        if calls >= max_evaluations:
            raise EvaluationBudget()
        optimizer.zero_grad()
        loss, factors, coefficients, info = evaluate(params)
        assert torch.isfinite(loss)
        value = float(loss.detach())
        if initial is None:
            initial = value
            normalizer = max(abs(value), 1e-10)
        (loss / normalizer).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        calls += 1
        normal = max(normal, info["normal_residual"])
        if best is None or value < best[0]:
            best = (value, calls, [f.detach().clone() for f in factors], coefficients.detach().clone())
        if calls <= 251:
            early = best
        if calls == 1 or calls % 25 == 0:
            row = dict(evaluation=calls, objective=value)
            history.append(row)
            if report is not None:
                report(row)
        return loss / normalizer
    try:
        optimizer.step(closure)
    except EvaluationBudget:
        budget_stop = True
    return dict(best=best, early=early, initial=initial, normalizer=normalizer,
        normal=normal, history=history, evaluations=calls, budget_stop=budget_stop,
        iterations=int(optimizer.state[params[0]].get("n_iter", 0)))
