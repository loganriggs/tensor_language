"""Bounded review control: polynomial leverage, metric gauges, and document tails.

No model execution or fitting. Existing native receipts are read only.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json
import numpy as np

P = Path(__file__).resolve().parent
rng = np.random.default_rng(210947)
d = 8
ii, jj = np.triu_indices(d)
scale = np.where(ii == jj, 1., np.sqrt(2.))

def features(x):
    return x[..., ii] * x[..., jj] * scale

# Orthonormal symmetric coefficient coordinates: <Q,xx^T>=svec(Q).psi(x).
signs = np.array(list(itertools.product([-1., 1.], repeat=d)))
axes = np.concatenate([np.eye(d), -np.eye(d)]) * np.sqrt(d)
Ps, Pa = features(signs), features(axes)
Ms, Ma = Ps.T @ Ps / len(Ps), Pa.T @ Pa / len(Pa)
assert np.allclose(signs.T @ signs / len(signs), np.eye(d))
assert np.allclose(axes.T @ axes / len(axes), np.eye(d))
assert np.allclose((signs**2).sum(1), d)
assert np.allclose((axes**2).sum(1), d)
q = np.zeros(len(ii)); q[np.flatnonzero((ii == 0) & (jj == 0))] = 1
q[np.flatnonzero((ii == 1) & (jj == 1))] = -1
assert abs(q @ Ms @ q) < 1e-12 and abs(q @ Ma @ q - 16) < 1e-12

# Coefficient ridge is an inner product, not a fabricated empirical distribution.
floor = .01
M = Ms + floor * np.eye(len(ii))
v = features(axes[0])
inv_v = np.linalg.solve(M, v)
leverage = float(v @ inv_v)
witness = inv_v / leverage
assert abs(v @ witness - 1) < 1e-12
assert abs(witness @ M @ witness - 1/leverage) < 1e-12
random_c = rng.normal(size=(100, len(ii)))
slack = leverage * np.einsum('bi,ij,bj->b', random_c, M, random_c) - (random_c @ v)**2
assert slack.min() >= -1e-9
# The exact distribution-transfer factor is the largest generalized eigenvalue.
C = np.linalg.cholesky(M)
K = np.linalg.solve(C, Ma)
K = np.linalg.solve(C, K.T).T
ev, U = np.linalg.eigh(K)
c = np.linalg.solve(C.T, U[:, -1])
ratio = float(c @ Ma @ c / (c @ M @ c))
assert abs(ratio - ev[-1]) < 1e-9

gauges = []
for seed in range(5):
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.normal(size=M.shape))
    H = Q @ np.diag(np.linspace(.5, 2., len(ii)))
    transformed = H @ M @ H.T
    vg = H @ v
    correct = float(vg @ np.linalg.solve(transformed, vg))
    wrong = float(vg @ np.linalg.solve(H @ Ms @ H.T + floor*np.eye(len(ii)), vg))
    assert abs(correct/leverage - 1) < 1e-10
    gauges.append(dict(seed=seed, congruent_relative_error=abs(correct/leverage-1),
                       reset_identity_floor_relative_change=wrong/leverage-1))

# Diagnostic only: original gates are aggregate, so document tails cannot change them.
source = P / 'DUAL_FRESH_NATIVE_V1.json'
raw = source.read_bytes(); native = json.loads(raw)
rows = []
for name in ['graph', 'isotropic_graph', 'covariance_graph', 'separate', 'isotropic_baseline']:
    for domain in ['fineweb', 'stdlib']:
        records = [r for r in native['records'] if r['candidate'] == name
                   and r['domain'] == domain and r['cohort'] == 'all']
        grouped = {}
        for r in records:
            key = (r['document'], r['selection'], r['family'], r['cohort'])
            if key not in grouped:
                grouped[key] = dict(r, error_energy=0., reference_energy=0.)
            for field in ['error_energy', 'reference_energy']:
                grouped[key][field] += r[field]
        vals = []
        for r in grouped.values():
            if r['reference_energy'] <= 0:
                continue
            error = float(np.sqrt(r['error_energy']/r['reference_energy']))
            cap = .20 if r['family'] == 'change' else .15
            vals.append((r, error, cap))
        worst = max(vals, key=lambda t: t[1]/t[2])
        bad = [r for r,e,cap in vals if e > cap]
        rows.append(dict(candidate=name, domain=domain, raw_records=len(records), evaluated_document_cells=len(vals),
                         distinct_documents=len({r['document'] for r,e,c in vals}),
                         cells_above_descriptive_cap=len(bad),
                         documents_with_any_exceedance=len({r['document'] for r in bad}),
                         worst_error=worst[1], worst_family=worst[0]['family'],
                         worst_selection=worst[0]['selection'],
                         skipped_zero_reference=sum(r['reference_energy'] <= 0 for r in grouped.values())))
out = dict(created_utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=hashlib.sha256(raw).hexdigest(), source=source.name,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    control=dict(d=d, coefficient_dimension=len(ii), same_covariance_and_radius=True,
        sign_metric_rank=int(np.linalg.matrix_rank(Ms)), axis_metric_rank=int(np.linalg.matrix_rank(Ma)),
        unregularized_shift_bound='infinite: q=x0^2-x1^2 has training zero and shift MSE16',
        floor=floor, point_leverage=leverage, extremizer_training_regularized_energy=float(witness@M@witness),
        extremizer_point_value=float(v@witness), distribution_transfer_factor=ratio,
        five_gauges=gauges, checks_pass=True),
    native_document_diagnostics=rows,
    correction='V2 aggregates repeated records by document/selection/family/cohort before computing diagnostic ratios. V1 mislabeled raw records as document cells; its tail statistics are superseded. Mathematical controls unchanged.',
    scope='One CPU review consequence. Planted coefficient-space control, not native calibration certificate. Native receipt now opened for document-tail analysis. Aggregate preregistered verdicts unchanged. Repeated selections/families are dependent cells, not independent samples. No model or primary receipt changed.')
dest=P/'REVIEW_MOMENT_CERTIFICATE_20260921_1047_V2.json'
with dest.open('x') as f:
    json.dump(out,f,indent=2); f.write('\n')
print(json.dumps(out,indent=2))
