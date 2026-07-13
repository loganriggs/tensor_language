"""e6b: extend e6 to ~50% budgets + a noise control at matched FVU."""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6  # runs nothing extra: guard below

# reuse the already-loaded module state (E, model, eval fns) from a fresh import
# NOTE: e6 executes its whole run on import; to avoid that, we re-implement the
# tiny driver here using its functions after monkey-checking it ran.

res = json.load(open('/workspace/tensor_language/basis_aligned/e6_results.json'))

extra = [
    ('svd', 'r=512', lambda: e6.arm_svd(512)),          # ~51%
    ('kmeans', 'n=25k', lambda: e6.arm_kmeans(25600)),  # ~51%
    ('rq', 'c=1k,h=25', lambda: e6.arm_rq(1024, 25)),   # ~51%
    ('noise_control', 'fvu~0.75', lambda: (
        e6.E + (e6.E - e6.arm_svd(100)[0]).norm() / (e6.V * e6.D) ** 0.5 *
        torch.randn_like(e6.E), 0)),
]
for method, label, fn in extra:
    Ehat, params = fn()
    row = {'method': method, 'label': label, 'params': params,
           'budget': params / (e6.V * e6.D), 'fvu': e6.fvu(Ehat),
           'ce': e6.eval_ce(Ehat)}
    row['dce'] = row['ce'] - res['baseline_ce']
    res['rows'].append(row)
    print(f"{method:14s} {label:10s} budget {row['budget']:6.1%}  "
          f"fvu {row['fvu']:.4f}  CE {row['ce']:.4f}  dCE {row['dce']:+.4f}")
    del Ehat
    torch.cuda.empty_cache()

with open('/workspace/tensor_language/basis_aligned/e6_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print('updated e6_results.json')
