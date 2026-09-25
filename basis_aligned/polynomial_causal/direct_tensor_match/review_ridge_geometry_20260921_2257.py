"""Read-only native Gram audit; unique review receipt, no fitting/native jobs.

Preregistered checks: native Grams must be positive definite; report (not gate
away) lambda/(lambda_min+lambda), the uniform relative function-norm shrinkage
bound versus the unregularized fixed-feature optimum. Planted exact quadrature
Gram agreement and covariant-penalty gauge replay must be <1e-10. Reset identity
ridge is expected to change a planted function by >1%; no native rescue claimed.
"""
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from noncentral_gaussian_cp import gram

P = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
start = time.monotonic()
stats = torch.load(P/'EXPANDED_INPUT_GEOMETRY_V1.pt', weights_only=True, map_location='cpu')
rows = []
sources = ['EXPANDED_INPUT_GEOMETRY_V1.pt', 'noncentral_gaussian_cp.py',
           'QUARTIC_CP512_SEED1001_V2.pt', 'QUARTIC_CP512_SEED1002_V2.pt']
for law in ['covariance_zero', 'covariance_shifted', 'second_zero']:
    S = torch.linalg.cholesky(stats['second_moment' if law == 'second_zero' else 'covariance'].double())
    mu = stats['mean'].double() if law == 'covariance_shifted' else torch.zeros(1152)
    for seed in [1001, 1002]:
        src = torch.load(P/f'QUARTIC_CP512_SEED{seed}_V2.pt', weights_only=True, map_location='cpu')
        f = [a.double() for a in src['factors']]
        fw, b = [a@S for a in f], [a@mu for a in f]
        G = gram(fw, b, fw, b)
        eig = torch.linalg.eigvalsh((G+G.T)/2)
        row = dict(law=law, seed=seed, min_eigenvalue=float(eig[0]),
                   max_eigenvalue=float(eig[-1]), mean_diagonal=float(G.diag().mean()),
                   relative_ridge=1e-6/float(G.diag().mean()),
                   uniform_function_shrinkage_bound=1e-6/(float(eig[0])+1e-6))
        assert eig[0] > 0
        rows.append(row)

# Independent full-degree Gaussian quadrature on a planted two-atom quartic.
torch.manual_seed(2257)
f = [torch.randn(2, 2) for _ in range(4)]
mu = torch.tensor([1., -.3]); b = [a@mu for a in f]
G = gram(f, b, f, b)
nodes, weights = np.polynomial.hermite.hermgauss(5)
ix = torch.tensor(list(itertools.product(range(5), repeat=2)))
x = torch.tensor(nodes*2**.5)[ix]+mu
w = torch.tensor(weights/np.sqrt(np.pi))[ix].prod(1)
phi = torch.ones(25, 2)
for a in f: phi *= x@a.T
Gq = phi.T@(w[:, None]*phi)
quad_error = float((G-Gq).norm()/G.norm())
target = torch.tensor([[1., -2.]])
X = target@G
ridge = 1e-6
solve = lambda g, cross, penalty: torch.linalg.solve(g+ridge*penalty, cross.T).T
c = solve(G, X, torch.eye(2))
A = torch.diag(torch.tensor([1e-5, 100.]))
Gp, Xp = A@G@A, X@A
reset = solve(Gp, Xp, torch.eye(2))@A
transported = solve(Gp, Xp, A@A)@A
rel = lambda v: float(((v-c)@G@(v-c).T).clamp_min(0).sqrt()/ (c@G@c.T).sqrt())
unregularized = torch.linalg.solve(G, X.T).T
planted = dict(quadrature_error=quad_error, reset_identity_function_change=rel(reset),
               transported_penalty_function_change=rel(transported),
               full_capacity_coefficient_recovery=float((unregularized-target).norm()/target.norm()))
assert quad_error < 1e-10 and planted['transported_penalty_function_change'] < 1e-10
assert planted['reset_identity_function_change'] > .01
result = dict(utc=datetime.now(timezone.utc).isoformat(), native=rows, planted=planted,
              source_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in sources},
              seconds=time.monotonic()-start,
              scope='Native fixed CP Gram conditioning only, no native refit or causal test. Bound is relative to the unregularized projection function, not teacher error. Gauge toy uses unnormalized factors; native original-coordinate normalization fixes that gauge.')
out=P/'REVIEW_RIDGE_GEOMETRY_20260921_2257.json'
with out.open('x') as h: json.dump(result,h,indent=2); h.write('\n')
print(json.dumps(result,indent=2))
