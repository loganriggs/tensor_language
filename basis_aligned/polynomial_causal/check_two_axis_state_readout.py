"""Live mixed-term control for state capture and the exact shared-norm fold."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from two_axis_state_readout import basis,capture_state_coefficients,compile_core,execute

torch.set_num_threads(2);torch.manual_seed(882)
states=torch.randn(4,6,11,dtype=torch.float64)
rows=10*torch.randn(4,4,2,11,dtype=torch.float64)
calls=[]
def state_function(z):
    calls.append(1)
    return torch.einsum('bk,bkd->bd',basis(z),states)
captured,checks=capture_state_coefficients(state_function,torch.zeros(4,dtype=torch.float64))
error=float((captured-states).abs().max());assert error<1e-12 and len(calls)==3
core=compile_core(captured,rows);replay=[]
for coordinate in [(0,0),(1,0),(0,1),(1,1),(1,-1),(-1,1),(-1,-1),(.5,.5),(.5,-.5)]:
    z=torch.tensor(coordinate,dtype=torch.float64).expand(4,2);h=state_function(z)
    normalized=F.rms_norm(h,(11,),eps=torch.finfo(torch.float32).eps)
    logits=30*torch.tanh((normalized[:,None,None,:]*rows).sum(-1)/30)
    direct=logits[:,:,0]-logits[:,:,1]
    replay.append(float((direct-execute(core,coordinate)).abs().max()))
omitted=states.clone();omitted[:,4]=0
cross_delta=float((execute(core,(1,1))-execute(compile_core(omitted,rows),(1,1))).norm())
assert max(replay)<1e-11 and cross_delta>.1
out=dict(state_jet_error=error,jet_checks=checks,max_readout_replay=max(replay),
         mixed_term_omission_delta=cross_delta,coefficients_per_context=sum(v.numel() for v in core.values())//4,
         scope='Planted two-axis quadratic state; native composition not yet tested.')
(Path(__file__).parent/'TWO_AXIS_STATE_READOUT_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
