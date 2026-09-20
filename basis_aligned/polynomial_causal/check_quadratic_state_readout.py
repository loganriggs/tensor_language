"""Quadratic states, rank-deficient states and derivatives test the exact fold."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from quadratic_state_readout import compile_core, fields_at, execute

torch.set_num_threads(2)
torch.manual_seed(735)
states=torch.randn(5,3,7,dtype=torch.float64)
states[0,2]=states[0,0]+states[0,1]
rows=10*torch.randn(5,4,2,7,dtype=torch.float64)
core=compile_core(states,rows)
def direct(t):
    h=states[:,0]+t*states[:,1]+t*t*states[:,2]
    z=F.rms_norm(h,(7,),eps=torch.finfo(torch.float32).eps)
    scores=30*torch.tanh((z[:,None,None,:]*rows).sum(-1)/30)
    return scores[:,:,0]-scores[:,:,1]
errors=[];norm_errors=[]
for radius in [-2.,-1.,-.5,0.,.25,.5,1.,2.]:
    errors.append(float((execute(core,radius)-direct(radius)).abs().max()))
    h=states[:,0]+radius*states[:,1]+radius**2*states[:,2]
    expected=h.square().mean(-1)+torch.finfo(torch.float32).eps
    norm_errors.append(float(((fields_at(core,radius)[:,-1]-expected).abs()/expected).max()))
def derivative(fn,t,order):
    if order==1:
        return torch.autograd.functional.jacobian(fn,t,create_graph=True)
    return torch.autograd.functional.jacobian(lambda u:derivative(fn,u,order-1),t,create_graph=True)
t=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
third_error=float((derivative(lambda u:execute(core,u),t,3)-derivative(direct,t,3)).abs().max().detach())
assert max(errors)<1e-11 and max(norm_errors)<1e-12 and third_error<1e-10
out=dict(max_readout_replay=max(errors),max_relative_norm_error=max(norm_errors),
         third_derivative_error=third_error,coefficients_per_ray=sum(v.numel() for v in core.values())//len(states),
         shared_norm_features=3,numerator_degree=2,norm_degree=4,
         scope='Exact planted quadratic state curves including rank deficiency; native-state approximation not yet tested.')
(Path(__file__).parent/'QUADRATIC_STATE_READOUT_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
