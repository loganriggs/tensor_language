"""Replay and diagnose the failed planted start; save its point for future audits."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from mixed_radix_bilinear_v1 import MixedRadix

torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(613)
models=[MixedRadix([2,2,2]) for _ in range(3)]
spec=[[(name,p.shape,p.numel()) for name,p in m.named_parameters()] for m in models]
def flatten(ms):return torch.cat([p.detach().flatten() for m in ms for p in m.parameters()])
def tensor(flat):
    offset=0;matrices=[]
    for m,ss in zip(models,spec):
        params={}
        for name,shape,n in ss:params[name]=flat[offset:offset+n].reshape(shape);offset+=n
        matrices.append(torch.func.functional_call(m,params,(torch.eye(8),)).T)
    a,c,b=matrices;q=torch.einsum('ok,ki,kj->oij',b,a,c)
    return (q+q.transpose(-1,-2))/2
target=tensor(flatten(models)).detach();scale=target.norm()
def residual(x):return ((tensor(x)-target)/scale).flatten()
def loss(x):return residual(x).square().sum()/2
jac=torch.func.jacrev(residual);gradient=torch.func.grad(loss)
torch.manual_seed(614);initial=flatten([MixedRadix([2,2,2]) for _ in range(3)])
fit=least_squares(lambda x:residual(torch.from_numpy(x)).detach().numpy(),initial.numpy(),
    jac=lambda x:jac(torch.from_numpy(x)).detach().numpy(),method='trf',max_nfev=300,
    gtol=1e-8,ftol=1e-12,xtol=1e-12)
x=torch.from_numpy(fit.x);h=torch.func.hessian(loss)(x);h=(h+h.T)/2
values,vectors=torch.linalg.eigh(h)
torch.manual_seed(607);direction=torch.randn_like(x);direction/=direction.norm();eps=1e-5
finite=(gradient(x+eps*direction)-gradient(x-eps*direction))/(2*eps)
error=float((h@direction-finite).norm()/finite.norm())
old=float(loss(x));trials=[]
for magnitude in (1e-4,1e-3,.01,.1):
    for sign in (-1,1):
        step=sign*magnitude*max(float(x.norm()),1.)
        trials.append(dict(step=step,loss=float(loss(x+step*vectors[:,0]))))
gain=(old-min(t['loss'] for t in trials))/old
p=Path(__file__).resolve().parent
prior=json.loads((p/'MIXED_RADIX_PLANTED_V1_RESULT.json').read_text())['starts'][0]
replay=abs(float(residual(x).norm())-prior['final_error'])
cache=Path('/dev/shm/bilin18_mixed_radix_failed_seed614.pt');assert not cache.exists()
torch.save(dict(point=x,target=target,hessian=h),cache)
result=dict(predictions={'pred_a_replay_and_hessian':error<=1e-5 and replay<=1e-10 and float(gradient(x).abs().max())<=1e-7,
                        'pred_b_negative_curvature':float(values[0]) < -1e-6*float(values.abs().max()),
                        'pred_c_eigendirection_improvement':gain>=1e-4},
    replay_error=replay,hessian_fd_error=error,gradient_max=float(gradient(x).abs().max()),
    minimum_eigenvalue=float(values[0]),maximum_absolute_eigenvalue=float(values.abs().max()),
    relative_trial_improvement=gain,trials=trials,cache=str(cache),
    scope='Exact Hessian of one failed planted point; not a native or global landscape theorem.',
    body_forwards=0,corpus_access=False,gpu_access=False)
with (p/'MIXED_RADIX_CURVATURE_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2));assert result['predictions']['pred_a_replay_and_hessian']
