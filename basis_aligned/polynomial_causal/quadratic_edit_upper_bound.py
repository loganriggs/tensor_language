"""LP outer relaxation of a bounded quadratic edit; no native-model bound."""
import numpy as np
from scipy.optimize import linprog


def upper_bound(g, h, reference, multiplier=1., budget=.08):
    """Bound multiplier * number effect under the selector's modal constraints."""
    n = g.shape[-1]
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    width = n + len(pairs)
    effects = np.zeros((len(g), width))
    effects[:, :n] = -g
    h = (h + h.transpose(0, 2, 1)) / 2
    for k, (i, j) in enumerate(pairs, n):
        effects[:, k] = -h[:, i, j] * (.5 if i == j else 1.)
    sign = np.sign(-g[0] @ reference) or 1.
    constraints, rhs = [], []
    for modal in effects[1:]:
        for s in [-1., 1.]:
            constraints.append(s*modal-budget*sign*effects[0])
            rhs.append(0.)
    bounds = [(-1., 1.)] * n
    for k, (i, j) in enumerate(pairs, n):
        bounds.append((0., 1.) if i == j else (-1., 1.))
        if i == j:
            for t in [-1., -.5, 0., .5, 1.]:
                row = np.zeros(width); row[i] = 2*t; row[k] = -1.
                constraints.append(row); rhs.append(t*t)
        else:
            # Four McCormick inequalities over [-1,1]^2.
            for si, sj, sk in [(1,1,-1),(-1,-1,-1),(1,-1,1),(-1,1,1)]:
                row = np.zeros(width); row[i] = si; row[j] = sj; row[k] = sk
                constraints.append(row); rhs.append(1.)
    matrix, rhs = np.array(constraints), np.array(rhs)
    objective = multiplier * effects[0]
    scale = max(np.linalg.norm(objective), 1e-30)
    solution = linprog(-objective/scale, A_ub=matrix, b_ub=rhs,
                       bounds=bounds, method='highs')
    assert solution.success, solution.message
    # Any nonnegative inequality multipliers give an upper bound after
    # maximizing the stationarity residual over the explicit variable box.
    dual = np.maximum(-solution.ineqlin.marginals*scale, 0.)
    residual = objective - matrix.T @ dual
    lower, upper = np.array(bounds).T
    value = float(rhs @ dual + np.maximum(residual*lower, residual*upper).sum())
    return dict(upper_bound=value, relaxed_primal=float(objective @ solution.x),
                dual_gap=float(value-objective @ solution.x),
                primal_violation=float(max(0., (matrix @ solution.x-rhs).max())),
                variables=width, inequalities=len(rhs))


def controls():
    # A box-linear objective with zero modal rows has an exact known maximum.
    rng = np.random.default_rng(41)
    errors = []
    for n in [2, 5, 6]:
        g = np.zeros((4, n)); g[0] = rng.normal(size=n)
        h = np.zeros((4, n, n))
        b = upper_bound(g, h, -g[0])
        errors.append(abs(b['upper_bound']-abs(g[0]).sum()))
    # Positive and negative squares catch diagonal sign and factor-of-two bugs.
    for coefficient, expected in [(1., 1.), (-1., 0.)]:
        g = np.zeros((4, 2)); h = np.zeros((4, 2, 2))
        h[0, 0, 0] = -2*coefficient
        errors.append(abs(upper_bound(g,h,np.ones(2))['upper_bound']-expected))
    # Every genuine random product is feasible in the product envelopes.
    x = rng.uniform(-1, 1, (1000, 2)); a, b = x.T; product = a*b
    violation = float(max((a+b-product-1).max(),(-a-b-product-1).max(),
                          (a-b+product-1).max(),(-a+b+product-1).max(),0.))
    assert max(errors) < 1e-10 and violation == 0.
    return dict(max_known_optimum_error=max(errors), product_envelope_violation=violation)


if __name__ == '__main__':
    import json
    print(json.dumps(controls()))
