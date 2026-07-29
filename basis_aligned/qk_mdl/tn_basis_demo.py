"""Lesson 3 (the right basis): the neuron basis is arbitrary (lesson 2), so find a REAL one —
sparse in what the layer actually reads. Two points:
(1) A computation can be DENSE in neurons but SPARSE in the right basis (few atoms per input).
(2) You must fit that basis in the geometry the weights read (the METRIC), not raw L2 — raw L2
    wastes capacity on high-variance directions the layer ignores (its kernel).
Toy: x lives in a task subspace (carries the signal) + a big nuisance subspace the readout ignores.
"""
import json
import numpy as np

rng = np.random.default_rng(0)
d, tdim, K, m, N = 16, 8, 8, 24, 3000
# task atoms live in the first tdim dims; each sample activates k=2 of them
Atrue = np.zeros((K, d)); Atrue[:, :tdim] = rng.standard_normal((K, tdim))
Atrue /= np.linalg.norm(Atrue, axis=1, keepdims=True)
S = np.zeros((N, K))
for n in range(N):
    idx = rng.choice(K, 2, replace=False); S[n, idx] = rng.uniform(0.5, 1.5, 2)
task_signal = S @ Atrue                                    # (N,d) in task subspace
nuisance = np.zeros((N, d)); nuisance[:, tdim:] = rng.standard_normal((N, d-tdim)) * 3.0  # BIG variance, ignored
X = task_signal + nuisance
# the layer's readout reads ONLY the task subspace -> metric G emphasizes those dims
Wy = np.zeros((d, 4)); Wy[:tdim] = rng.standard_normal((tdim, 4))
Y = X @ Wy                                                 # depends only on task subspace
# "neurons": a dense random mixing
Wn = rng.standard_normal((m, d)); Hn = X @ Wn.T           # (N,m) dense neuron activations

def part_ratio(v):
    p = v**2 / (v**2).sum(); return float(1/(p**2).sum())   # participation ratio (# active)

neuron_pr = float(np.mean([part_ratio(np.abs(Hn[n])) for n in range(200)]))
true_code_active = float((S[:200] > 0).sum(1).mean())

def y_r2_from_code(code):
    A = np.c_[code, np.ones(N)]
    W = np.linalg.lstsq(A, Y, rcond=1e-8)[0]
    pred = A @ W
    return 1 - ((pred-Y)**2).sum()/((Y-Y.mean(0))**2).sum()

Xc = X - X.mean(0)
# raw top-k = PCA of X (nuisance dominates variance -> picks the wrong directions)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
# metric: whiten by the geometry the readout reads (G = Wy Wy^T), then PCA (picks task directions)
G = Wy @ Wy.T
evals, evecs = np.linalg.eigh(np.clip(G, None, None))
Gh = evecs @ np.diag(np.sqrt(np.clip(evals, 0, None))) @ evecs.T
Xg = Xc @ Gh.T
_, _, Vtg = np.linalg.svd(Xg, full_matrices=False)

ks = [1, 2, 4, 6, 8, 12]
raw_r2 = [round(y_r2_from_code(Xc @ Vt[:k].T), 3) for k in ks]
metric_r2 = [round(y_r2_from_code(Xg @ Vtg[:k].T), 3) for k in ks]
# atom recovery: best cosine of each true atom to the top-K directions of each basis
def recov(dirs):
    C = np.abs(Atrue @ dirs[:K].T); return round(float(C.max(1).mean()), 3)
metric_dirs, raw_dirs = Vtg, Vt
res = {'d': d, 'neurons': m, 'K_atoms': K,
       'neuron_participation_ratio': round(neuron_pr, 1), 'true_code_active_per_input': round(true_code_active, 1),
       'ks': ks, 'raw_L2_y_r2': raw_r2, 'metric_y_r2': metric_r2,
       'atom_recovery_cos_metric': recov(metric_dirs), 'atom_recovery_cos_raw': recov(raw_dirs),
       'sample_neuron_act': [round(float(x), 2) for x in Hn[0]],
       'sample_atom_code': [round(float(x), 2) for x in S[0]]}
json.dump(res, open('tn_basis_demo.json', 'w'), indent=1)
print(f"neurons dense (participation {neuron_pr:.1f} of {m}); true code active {true_code_active:.1f} of {K}", flush=True)
print(f"y-R2 vs k:  RAW-L2 {raw_r2}  |  METRIC {metric_r2}", flush=True)
print(f"atom recovery cosine: metric {res['atom_recovery_cos_metric']} vs raw {res['atom_recovery_cos_raw']}", flush=True)
print('TN BASIS DEMO DONE', flush=True)
