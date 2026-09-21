"""Fixed-dictionary independent root fits and final-logit sensitivity diagonals."""
import torch

def sensitivity(state,directions,denominator,unembedding,cap=30.,eps=None):
 """N x R squared norms of d logits / d root, without N x V x R storage."""
 eps=torch.finfo(state.dtype).eps if eps is None else eps;inv=(state.square().mean(-1,keepdim=True)+eps).rsqrt();a=state@directions/state.shape[-1]
 ux=state@unembedding.T;uw=unembedding@directions
 gain=(1-torch.tanh(ux*inv/cap).square()).square()
 first=gain@uw.square();cross=(gain*ux)@uw;last=(gain*ux.square()).sum(-1,keepdim=True)
 value=inv.square()*first-2*inv.pow(4)*a*cross+inv.pow(6)*a.square()*last
 return value.clamp_min(0)/denominator.square()

def fit_roots(phi,target,weights,ridge_fraction=1e-6):
 scales=phi.square().mean(0).sqrt().clamp_min(1e-30);z=phi/scales;ys=target.square().mean(0).sqrt().clamp_min(1e-30);y=target/ys;weights=weights/weights.mean(0,keepdim=True);eye=torch.eye(phi.shape[1],dtype=phi.dtype,device=phi.device);cs=[]
 for g in range(target.shape[1]):
  w=weights[:,g];c=torch.linalg.solve(z.T@(w[:,None]*z)+len(phi)*ridge_fraction*eye,z.T@(w*y[:,g]));cs.append(c/scales*ys[g])
 return torch.stack(cs)

def controls():
 from logit_directional_response import derivative
 torch.set_num_threads(2);torch.manual_seed(182);dtype=torch.float64;x=torch.randn(31,7,dtype=dtype);directions=torch.randn(7,4,dtype=dtype);u=torch.randn(17,7,dtype=dtype);den=torch.rand(31,1,dtype=dtype)+.1
 actual=sensitivity(x,directions,den,u);expected=torch.stack([derivative(x,-directions[:,g]/den,u).square().sum(-1) for g in range(4)],1);jac=float((actual-expected).norm()/expected.norm());assert jac<1e-12
 rows=[]
 for family in ['independent','shared_inputs','shared_outputs','squares','cancellation']:
  phi=torch.randn(71,6,dtype=dtype);target=torch.randn(71,4,dtype=dtype);w=torch.rand(71,4,dtype=dtype)+.001
  if family=='shared_inputs':phi[:,1]=phi[:,0]
  if family=='shared_outputs':target[:,1]=target[:,0]
  if family=='squares':phi=phi.square()
  if family=='cancellation':phi[:,1]=-phi[:,0]
  c=fit_roots(phi,target,w);scales=phi.square().mean(0).sqrt();z=phi/scales;refs=[]
  for g in range(4):
   ww=w[:,g]/w[:,g].mean();design=torch.cat([z*ww.sqrt()[:,None],(len(phi)*1e-6)**.5*torch.eye(6,dtype=dtype)]);y=torch.cat([target[:,g]*ww.sqrt(),torch.zeros(6,dtype=dtype)]);refs.append(torch.linalg.lstsq(design,y,driver='gelsd').solution/scales)
  ref=torch.stack(refs);error=float((phi@(c-ref).T).norm()/target.norm());assert error<1e-9;rows.append(dict(family=family,weighted_prediction_replay=error))
 # Exercise explicit native epsilon where it materially changes normalization.
 import torch.nn.functional as F
 small=x*1e-5;eps=torch.finfo(torch.float32).eps
 native=sensitivity(small,directions,den,u,eps=eps);refs=[]
 fn=lambda z:30*torch.tanh(F.rms_norm(z,(7,),eps=eps)@u.T/30)
 for g in range(4):
  _,jvp=torch.func.jvp(fn,(small,),(-directions[:,g]/den,));refs.append(jvp.square().sum(-1))
 expected_native=torch.stack(refs,1);eps_error=float((native-expected_native).norm()/expected_native.norm());assert eps_error<1e-12
 return dict(sensitivity_replay=jac,native_epsilon_replay=eps_error,fit_controls=rows)
if __name__=='__main__':
 import json
 from pathlib import Path
 out=controls();Path(__file__).with_name('SENSITIVE_ROOT_READOUT_CONTROLS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
