"""Lesson 5 — answer the rank-vs-rotation question with a computation. Two 12x12 bond matrices your
eye can't tell apart: a full-rank ROTATION (wide, lossless — every dimension crosses, just relabeled)
and a rank-4 PROJECTION dressed with rotations so its entries look just as dense. The singular-value
spectrum tells them apart instantly; the truncation probe shows the rotation is load-bearing in every
dimension (degrades from the first cut) while the projection is flat until its true rank, then cliffs.
"""
import json
import numpy as np

rng = np.random.default_rng(0)
d = 12
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))                 # full-rank rotation (orthogonal)
U, _ = np.linalg.qr(rng.standard_normal((d, d))); V, _ = np.linalg.qr(rng.standard_normal((d, d)))
B = U[:, :4] @ V[:, :4].T                                        # rank-4
# dress with rotations so entries look dense (rank unchanged)
Ra, _ = np.linalg.qr(rng.standard_normal((d, d))); Rb, _ = np.linalg.qr(rng.standard_normal((d, d)))
B = Ra @ B @ Rb

N = 3000; X = rng.standard_normal((N, d))
def trunc_r2(Mat, r):
    Uu, S, Vt = np.linalg.svd(Mat)
    Mr = (Uu[:, :r] * S[:r]) @ Vt[:r]
    Y = X @ Mat.T; Yr = X @ Mr.T
    return 1 - ((Y - Yr)**2).sum() / (Y**2).sum()

rs = list(range(1, d + 1))
res = {'d': d,
       'sv_rotation': [round(float(s), 2) for s in np.linalg.svd(Q, compute_uv=False)],
       'sv_projection': [round(float(s), 2) for s in np.linalg.svd(B, compute_uv=False)],
       'trunc_r': rs,
       'rotation_r2': [round(trunc_r2(Q, r), 3) for r in rs],
       'projection_r2': [round(trunc_r2(B, r), 3) for r in rs],
       'rotation_matrix': [[round(float(v), 2) for v in row] for row in Q.tolist()],
       'projection_matrix': [[round(float(v), 2) for v in row] for row in B.tolist()]}
json.dump(res, open('tn_bond_rotation_demo.json', 'w'), indent=1)
print(f"rotation singular values (all ~1): {res['sv_rotation']}", flush=True)
print(f"projection singular values (4 then ~0): {res['sv_projection']}", flush=True)
print(f"rotation truncation R2: {res['rotation_r2']}", flush=True)
print(f"projection truncation R2: {res['projection_r2']}", flush=True)
print('TN BOND ROTATION DEMO DONE', flush=True)
