"""Known-affine upstream state has exact quadratic fields and nonpolynomial output."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from final_readout_field_program import pack_fields, evaluate_fields, evaluate_jet

torch.manual_seed(721)
h0 = torch.randn(5, 7, dtype=torch.float64)
h1 = .8 * torch.randn(5, 7, dtype=torch.float64)
rows = 10 * torch.randn(5, 4, 2, 7, dtype=torch.float64)
c0 = pack_fields(h0, rows)
c1 = torch.cat([(h1[:,None,None,:]*rows).sum(-1).flatten(1),
                2*(h0*h1).mean(-1,keepdim=True)],dim=-1)
c2 = torch.zeros_like(c0);c2[:,-1]=h1.square().mean(-1)
jet = torch.stack([c0,c1,c2],dim=1)
def direct(t):
    h=h0+t*h1
    z=F.rms_norm(h,(7,),eps=torch.finfo(torch.float32).eps)
    out=30*torch.tanh((z[:,None,None,:]*rows).sum(-1)/30)
    return out[:,:,0]-out[:,:,1]
t=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
base=direct(t)
linear=torch.autograd.functional.jacobian(direct,t)
quadratic=torch.autograd.functional.jacobian(
    lambda u:torch.autograd.functional.jacobian(direct,u,create_graph=True),t)/2
errors=[];taylor_errors=[]
for radius in [-1.,-.5,0.,.25,.5,1.]:
    expected=direct(radius);actual=evaluate_jet(jet,radius)
    errors.append(float((actual-expected).abs().max()))
    taylor_errors.append(float(((base+radius*linear+radius**2*quadratic-expected).norm()/expected.norm()).detach()))
d1=torch.autograd.functional.jacobian(lambda u:evaluate_jet(jet,u),t)
d2=torch.autograd.functional.jacobian(lambda u:torch.autograd.functional.jacobian(
    lambda v:evaluate_jet(jet,v),u,create_graph=True),t)
derivative_error=max(float((d1-linear).abs().max()),float((d2-2*quadratic).abs().max()))
bad=c0.clone();bad[:,-1]=-1.
try:
    evaluate_fields(bad)
except ValueError:
    rejected=True
else:
    rejected=False
assert max(errors)<1e-12 and derivative_error<1e-11 and rejected
out=dict(max_readout_replay=max(errors),max_derivative_error=derivative_error,
         max_quadratic_output_relative_error=max(taylor_errors),
         nonpositive_denominator_rejected=rejected,fields=9,coefficients_per_ray=27,
         scope='Planted affine upstream states only; no native model or discovered circuit.')
(Path(__file__).parent/'FINAL_READOUT_FIELD_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
