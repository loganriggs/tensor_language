"""Falsify using quartic matrix rank as a count of quadratic intermediates.

One equal-positive rank-r quadratic, squared, has flattening rank r(r+1)/2.
Pred A: spectrum agrees with analytic values to 1e-12 in each r=1,2,4,8 case.
Pred B: measured rank equals r(r+1)/2 despite one arithmetic square node.
"""
from pathlib import Path
import json
import math
import torch
from quartic_weighted_trace_v1 import fitted_action

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
rows = []
for r in (1, 2, 4, 8):
    basis = []
    for i in range(r):
        for j in range(i, r):
            e = torch.zeros(r, r)
            if i == j:
                e[i, j] = 1
            else:
                e[i, j] = e[j, i] = 1 / math.sqrt(2)
            basis.append(e)
    basis = torch.stack(basis)
    b = torch.eye(r)[None]
    n = torch.ones(1, r) / math.sqrt(r)
    actions = torch.stack([fitted_action(b, n, torch.ones(1), e) for e in basis])
    matrix = torch.einsum('aij,bij->ab', basis, actions)
    actual = torch.linalg.eigvalsh(matrix)
    expected = torch.full_like(actual, 2 / (3 * r))
    expected[-1] = (r + 2) / (3 * r)
    rows.append(dict(quadratic_rank=r, arithmetic_square_nodes=1,
                     flattening_rank=int((actual.abs() > 1e-12).sum()),
                     expected_rank=r * (r + 1) // 2,
                     eigenvalue_error=float((actual - expected).abs().max())))
result = dict(pred_a=all(x['eigenvalue_error'] <= 1e-12 for x in rows),
              pred_b=all(x['flattening_rank'] == x['expected_rank'] for x in rows),
              rows=rows,
              scope='Exact synthetic counterexample, not a native model measurement or recovery guarantee.')
out = Path(__file__).with_name('QUARTIC_FLATTENING_COMPLEXITY_V1_RESULT.json')
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert result['pred_a'] and result['pred_b']
