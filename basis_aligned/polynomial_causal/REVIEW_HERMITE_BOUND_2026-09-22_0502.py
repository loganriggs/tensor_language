"""Bounded CPU control: Gaussian degree-four conditional-error sandwich.

Prediction before execution: exact Hermite energies and independent 5-point
tensor Gaussian quadrature agree to 1e-11 relative; E <= D <= 4E.
Original native spectra are read only; their plug-in values are NOT certificates.
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import numpy as np
from numpy.polynomial.hermite_e import hermegauss, hermeval

P = Path(__file__).resolve().parent
nodes, w = hermegauss(5)
w = w / math.sqrt(2 * math.pi)
grid = np.array(list(itertools.product(range(5), repeat=3)))
x = nodes[grid]
qw = np.prod(w[grid], axis=1)
indices = [a for a in itertools.product(range(5), repeat=3) if sum(a) <= 4]
rng = np.random.default_rng(22090502)
weights = np.array([0.3, 2.0])

def basis(a):
    result = np.ones(len(x))
    for j, k in enumerate(a):
        result *= hermeval(x[:, j], [0] * k + [1]) / math.sqrt(math.factorial(k))
    return result

rows = []
cases = [('random', rng.normal(size=(len(indices), 2)))]
for name, index in [('linear_endpoint', (0, 1, 0)), ('quartic_endpoint', (0, 4, 0)), ('closed_port', (4, 0, 0))]:
    c = np.zeros((len(indices), 2)); c[indices.index(index)] = [1, -2]
    cases.append((name, c))
for name, coeff in cases:
    # Retain coordinate 0; conditional averaging removes terms with a1+a2>0.
    residual = np.zeros((len(x), 2))
    jac = np.zeros((len(x), 2, 2))
    exact_e = exact_d = 0.0
    for a, c in zip(indices, coeff):
        outside_degree = a[1] + a[2]
        if not outside_degree:
            continue
        energy = float(np.dot(weights, c*c))
        exact_e += energy; exact_d += outside_degree * energy
        residual += basis(a)[:, None] * c
        for j in [1, 2]:
            if a[j]:
                b = list(a); b[j] -= 1
                jac[:, :, j-1] += math.sqrt(a[j]) * basis(b)[:, None] * c
    measured_e = float(np.einsum('n,no,o->', qw, residual**2, weights))
    measured_d = float(np.einsum('n,noj,o->', qw, jac**2, weights))
    discrepancy = max(abs(measured_e-exact_e), abs(measured_d-exact_d))/max(1, exact_e, exact_d)
    assert discrepancy < 1e-11
    assert exact_e <= exact_d + 1e-12 and exact_d <= 4*exact_e + 1e-12
    rows.append(dict(case=name, error_energy=exact_e, discarded_derivative_energy=exact_d,
                     quadrature_relative_discrepancy=discrepancy))

source = P/'direct_tensor_match/NATIVE_RESIDUAL_SUBSPACE_LARGE_V1.json'
native = json.loads(source.read_text())
estimates = []
for row in native['rows']:
    for capture in row['captures']:
        if capture['rank'] in [64, 512]:
            c = capture['checking']['weighted_capture']
            estimates.append(dict(seed=row['seed'], rank=capture['rank'], checking_capture=c,
                plug_in_centered_relative_RMS_floor=math.sqrt((1-c)/4),
                status='Population formula evaluated at sampled capture; not a certified native bound'))
out = dict(prediction_pass=True, controls=rows, native_plug_in_estimates=estimates,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Gaussian polynomial conditional-mean error only; no text, causal or arbitrary-DAG bound')
target = P/'REVIEW_HERMITE_BOUND_2026-09-22_0502.json'
with target.open('x') as f:
    json.dump(out, f, indent=2); f.write('\n')
print(json.dumps(out, indent=2))
