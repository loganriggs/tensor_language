"""Exact <=4-group coarsening screen on two frozen, unconverged fits.

A: chosen subset CP/Gram replay <=1e-9 and complete enumeration.
B: >=5/10 unstable anchors permit >=50% absolute drift-energy reduction.
C: >=5/10 permit that reduction AND <=.1 relative subset function error.
Selection and measurement use the same two fits: no holdout/causal claim.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from ll1_group_matching_v1 import group_inner
from ll1_joint_parent_graph_v2 import factors
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    root = Path(__file__).parent
    out = root / 'LL1_OPTIMIZER_COARSENING_V1.json'
    assert not out.exists()
    paths = [root / 'PROJECTED_LL1_CONVERGENCE_V3_SPECTRAL.pt',
             root / 'SHARED_READER_JOINT_FIT_V1_SPECTRAL_ORIGINAL.pt']
    old = torch.load(paths[0], weights_only=True, map_location='cpu')['parts']
    new = factors(torch.load(paths[1], weights_only=True, map_location='cpu'))
    matching = json.loads((root / 'JOINT_READER_OPTIMIZER_GROUP_COMPARISON_V1.json').read_text())
    assert all(row['old'] == row['new'] for row in matching['matching'])
    aa, bb, ab = group_inner(old, old), group_inner(new, new), group_inner(old, new)
    k = (aa + bb - ab - ab.T).numpy()
    reference = aa.numpy()
    cosine = ab.diag() / (aa.diag() * bb.diag()).sqrt()
    anchors = torch.where(cosine < .8)[0].tolist()
    assert len(anchors) == 10
    rows, worst = [], 0.
    for anchor in anchors:
        remaining = [i for i in range(64) if i != anchor]
        candidates = [[anchor, *rest] for size in range(1, 5)
                      for rest in itertools.combinations(remaining, size - 1)]
        expected = sum(math.comb(63, j) for j in range(4))
        assert len(candidates) == expected
        scored = []
        for size in range(1, 5):
            ids = np.array([s for s in candidates if len(s) == size])
            drift = k[ids[:, :, None], ids[:, None, :]].sum(axis=(1, 2))
            energy = reference[ids[:, :, None], ids[:, None, :]].sum(axis=(1, 2))
            assert np.all(energy > 0) and np.all(drift >= -1e-6)
            relative = np.sqrt(np.maximum(drift, 0) / energy)
            scored.extend((float(d), float(r), s.tolist()) for d, r, s in zip(drift, relative, ids))
        best_drift = min(scored, key=lambda x: x[0])
        best_relative = min(scored, key=lambda x: x[1])
        feasible = [s for s in scored if s[0] <= .5*k[anchor, anchor] and s[1] <= .1]
        selected = dict(minimum_drift=best_drift, minimum_relative=best_relative)
        if feasible:
            selected['joint_bar_witness'] = min(feasible, key=lambda x: (len(x[2]), x[0]))
        witnesses = {}
        for name, (drift, relative, ids) in selected.items():
            left, right = cp(*(v[ids] for v in old)), cp(*(v[ids] for v in new))
            el, er, cross = inner(left, left), inner(right, right), inner(left, right)
            replay = abs(float(el + er - 2*cross) - drift) / max(float(el), float(er))
            worst = max(worst, replay)
            witnesses[name] = dict(groups=ids, relative_error=relative,
                                   drift_over_singleton=drift/k[anchor, anchor], replay=replay)
        rows.append(dict(anchor=anchor, enumerated=expected,
                         singleton_relative_error=math.sqrt(k[anchor, anchor]/reference[anchor, anchor]),
                         half_drift=best_drift[0] <= .5*k[anchor, anchor],
                         joint_bar=bool(feasible), witnesses=witnesses))
    result = dict(pred_a=worst <= 1e-9,
                  pred_b=sum(r['half_drift'] for r in rows) >= 5,
                  pred_c=sum(r['joint_bar'] for r in rows) >= 5,
                  half_drift_count=sum(r['half_drift'] for r in rows),
                  joint_bar_count=sum(r['joint_bar'] for r in rows),
                  maximum_replay=worst, rows=rows,
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope='Exact bounded in-sample subset search; overlapping candidate merges, '
                        'not a globally consistent partition, holdout stability, or identified circuits.')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'sources')}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
