"""e6c: noise-scale sweep — dCE vs FVU for ADDITIVE isotropic noise, to contrast
with SUBTRACTIVE compression at the same FVU."""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6

res = json.load(open('/workspace/tensor_language/basis_aligned/e6_results.json'))
# drop the earlier single noise point; replaced by this sweep
res['rows'] = [r for r in res['rows'] if r['method'] != 'noise_control']

rms = ((e6.E - e6.ROWMEAN) ** 2).mean().sqrt()
for target_fvu in [0.1, 0.32, 0.75, 1.5, 3.0]:
    scale = rms * target_fvu ** 0.5
    torch.manual_seed(1)
    Ehat = e6.E + scale * torch.randn_like(e6.E)
    row = {'method': 'noise', 'label': f'fvu~{target_fvu}', 'params': 0,
           'budget': 0.0, 'fvu': e6.fvu(Ehat), 'ce': e6.eval_ce(Ehat)}
    row['dce'] = row['ce'] - res['baseline_ce']
    res['rows'].append(row)
    print(f"noise fvu {row['fvu']:.3f}  CE {row['ce']:.4f}  dCE {row['dce']:+.4f}")
    del Ehat
    torch.cuda.empty_cache()

with open('/workspace/tensor_language/basis_aligned/e6_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print('updated e6_results.json')
