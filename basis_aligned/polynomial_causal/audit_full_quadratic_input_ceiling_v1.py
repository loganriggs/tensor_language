"""Use saved spectra to bound any 23x4 full-quadratic frame program."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(557)
p=Path(__file__).resolve().parent
source=json.loads((p/'SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT.json').read_text())
cache=Path(source['cache']['path'])
assert hashlib.sha256(cache.read_bytes()).hexdigest()==source['cache']['sha256']
saved=torch.load(cache,map_location='cpu',weights_only=True)
forms=torch.randn(6,9,9);forms=(forms+forms.transpose(-1,-2))/2
second=(forms@forms).sum(0)
frame=torch.linalg.qr(torch.randn(9,3)).Q
inside=(frame.T@forms@frame).square().sum()
incidence=torch.trace(frame.T@second@frame)
bound=torch.linalg.eigvalsh(second)[-3:].sum()
assert inside<=incidence+1e-10 and incidence<=bound+1e-10
rows={}
for name,state in saved.items():
    s,e,v=[state[k] for k in ('second','eigenvalues','eigenvectors')]
    total=float(s.trace())
    residual=float((s@v-v*e[None,:]).norm())
    capture_bound=float(e[-92:].sum())/total
    padded=(float(e[-92:].sum())+92*residual)/total
    rows[name]=dict(rank92_bound=capture_bound,residual_padded_bound=padded,
        eigen_residual_over_total=residual/total,
        rank256_bound=float(e[-256:].sum())/total,
        minimum_input_union_for_half=int(torch.searchsorted(e.flip(0).cumsum(0),.5*total))+1)
full=rows['full']['residual_padded_bound']
result=dict(predictions={'pred_a_instrument':max(r['eigen_residual_over_total'] for r in rows.values())<=1e-10,
                        'pred_b_quarter_ceiling':full<.25,
                        'pred_c_improvement_not_excluded':full>.08634383041327387},
    metrics=rows,toy=dict(inside=float(inside),one_sided=float(incidence),upper=float(bound)),
    source_cache_sha256=source['cache']['sha256'],body_forwards=0,corpus_access=False,gpu_access=False,
    scope='Input-union support ceiling, not a tight block optimum or natural-state/circuit bound. Eigen residual padding is numerical, not interval certification.')
with (p/'FULL_QUADRATIC_INPUT_CEILING_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2));assert result['predictions']['pred_a_instrument']
