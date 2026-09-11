"""Weight-only planted joint tensor fit with standard SciPy trust regions.

Small d8, known wiring, near-identity distribution. Not native/global recovery.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from mixed_radix_bilinear_v1 import MixedRadix

torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
torch.manual_seed(613)
templates=[MixedRadix([2,2,2]) for _ in range(3)]
specs=[[(name,p.shape,p.numel()) for name,p in model.named_parameters()] for model in templates]
eye=torch.eye(8)


def flatten(models):return torch.cat([p.detach().flatten() for model in models for p in model.parameters()])


def tensor_from(flat):
    matrices=[];offset=0
    for model,spec in zip(templates,specs):
        params={}
        for name,shape,count in spec:
            params[name]=flat[offset:offset+count].reshape(shape);offset+=count
        matrices.append(torch.func.functional_call(model,params,(eye,)).T)
    a,c,b=matrices
    result=torch.einsum('ok,ki,kj->oij',b,a,c)
    return (result+result.transpose(-1,-2))/2


target=tensor_from(flatten(templates)).detach();scale=target.norm()
def residual(flat):return ((tensor_from(flat)-target)/scale).flatten()
jacobian=torch.func.jacrev(residual)
rows=[];errors=[]
for seed in (614,615,616):
    torch.manual_seed(seed)
    initial=flatten([MixedRadix([2,2,2]) for _ in range(3)])
    initial_error=float(residual(initial).norm())
    direction=torch.randn_like(initial);direction/=direction.norm();eps=1e-5
    j=jacobian(initial)
    finite=(residual(initial+eps*direction)-residual(initial-eps*direction))/(2*eps)
    errors.append(float((j@direction-finite).norm()/finite.norm()))
    def fun(x):return residual(torch.from_numpy(x)).detach().numpy()
    def jac(x):return jacobian(torch.from_numpy(x)).detach().numpy()
    start=time.perf_counter()
    fit=least_squares(fun,initial.numpy(),jac=jac,method='trf',
        max_nfev=300,gtol=1e-8,ftol=1e-12,xtol=1e-12)
    row=dict(seed=seed,initial_error=initial_error,final_error=float(np.linalg.norm(fit.fun)),
             gradient_max=float(fit.optimality),evaluations=fit.nfev,jacobians=fit.njev,
             seconds=time.perf_counter()-start,status=int(fit.status),message=str(fit.message))
    rows.append(row);print(json.dumps(row),flush=True)
result=dict(predictions={'pred_a_jacobian':max(errors)<=1e-6,
                        'pred_b_all_recovered':all(r['final_error']<=1e-5 and r['gradient_max']<=1e-7 for r in rows),
                        'pred_c_nontrivial_starts':all(r['initial_error']>.1 for r in rows)},
    starts=rows,maximum_jacobian_error=max(errors),parameters=len(initial),tensor_entries=target.numel(),
    body_forwards=0,corpus_access=False,gpu_access=False,
    scope='Small known-wiring local planted fit. No native result, unrestricted initialization guarantee, or factor identification.')
with Path(__file__).with_name('MIXED_RADIX_PLANTED_V1_RESULT.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result['predictions']));assert result['predictions']['pred_a_jacobian']
