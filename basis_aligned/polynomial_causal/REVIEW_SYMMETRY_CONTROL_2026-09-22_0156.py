"""Bounded FP64 CPU review control; no native artifacts or helpers are modified."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'direct_tensor_match'))
from conditional_quartic_cp import construct, evaluate

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
torch.manual_seed(220922)
factors = [torch.tensor([[1., 0.]])] * 3 + [torch.tensor([[0., 1.]])]
C = torch.ones(1, 1)
S = torch.eye(2)
V = torch.tensor([[1.], [0.]])
mu = torch.tensor([1., 2.])
program = construct(factors, C, S, mu, V)
grid = torch.linspace(-4, 4, 41)
x = torch.stack([grid, torch.zeros_like(grid)], 1)
g = evaluate(program, x).flatten()
exact = 2 * grid**3
full = construct(factors, C, S, mu, torch.eye(2))
test = torch.randn(101, 2)
full_error = float((evaluate(full, test).flatten() - test[:, 0]**3 * test[:, 1]).abs().max())
gauge = [f.clone() for f in factors]
gauge[0] *= -7
gauge[1] /= -7
gauge_error = float((evaluate(construct(gauge, C, S, mu, V), x).flatten() - g).abs().max())
# Independent conditional integration, exact for the degree-one discarded slot.
nodes, weights = np.polynomial.hermite.hermgauss(12)
nodes = torch.from_numpy(nodes * np.sqrt(2))
weights = torch.from_numpy(weights / np.sqrt(np.pi))
integrated = (grid[:, None]**3 * (2 + nodes[None, :]) * weights).sum(1)
conditional_error = float((g - integrated).abs().max())
centered = construct(factors, C, S, torch.zeros(2), V)
centered_error = float(evaluate(centered, x).abs().max())
# For the symmetric mixture .5 N(mu,I)+.5 N(-mu,I), posterior component
# probability conditional on y=x1 is sigmoid(2*y), hence E[f|y]=2*y^3*tanh(y).
posterior = torch.sigmoid(2 * grid)
mixture = 2 * grid**3 * torch.tanh(grid)
independent_mixture = posterior * integrated + (1-posterior) * (-integrated)
mixture_error = float((mixture-independent_mixture).abs().max())

def risks(order):
    z, w = np.polynomial.hermite.hermgauss(order)
    z, w = z*np.sqrt(2), w/np.sqrt(np.pi)
    # Integrate the discarded conditional variance analytically; integrate y
    # independently by Gauss-Hermite. These are absolute squared L2 errors.
    out = {}
    for law in ['shifted', 'symmetric_mixture']:
        totals = dict(shifted_conditional=0., naive_even_projection=0., mixture_conditional=0.)
        signs = [1] if law == 'shifted' else [-1, 1]
        for sign in signs:
            y = sign + z
            target_mean = 2*sign*y**3
            variance = y**6
            predictions = [2*y**3, np.zeros_like(y), 2*y**3*np.tanh(y)]
            for key, pred in zip(totals, predictions):
                totals[key] += float(w @ (variance+(target_mean-pred)**2))/len(signs)
        out[law] = totals
    return out

r128, r192 = risks(128), risks(192)
integration_drift = max(abs(r128[l][k]-r192[l][k]) for l in r128 for k in r128[l])
assert max(full_error, gauge_error, conditional_error, centered_error, mixture_error) < 1e-10
assert integration_drift < 1e-7
assert r192['shifted']['shifted_conditional'] < r192['shifted']['naive_even_projection']
assert r192['symmetric_mixture']['mixture_conditional'] < r192['symmetric_mixture']['naive_even_projection']
assert r192['symmetric_mixture']['naive_even_projection'] < r192['symmetric_mixture']['shifted_conditional']
result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), status='PASS',
    scope='planted quartic x1^3*x2; no native or text causal evidence',
    precision='float64 CPU', seed=220922, actual_text_context_cells=0,
    errors=dict(full_rank=full_error, reciprocal_gauge=gauge_error,
                conditional_quadrature=conditional_error, centered_parity=centered_error,
                mixture_posterior=mixture_error, risk_quadrature_drift=integration_drift),
    risks_absolute_squared=r192,
    consequence='Parity restoration is law-dependent. Symmetric-mixture conditional expectation uses posterior weights, not uniform even projection; it is generally non-polynomial.',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256((ROOT/'direct_tensor_match/conditional_quartic_cp.py').read_bytes()).hexdigest())
destination = Path(__file__).with_suffix('.json')
with destination.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(result, indent=2))
