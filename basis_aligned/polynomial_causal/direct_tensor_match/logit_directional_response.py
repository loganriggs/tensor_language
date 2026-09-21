"""Analytic directional response through final RMSNorm and tanh logit cap."""
import torch
import torch.nn.functional as F

def derivative(state,direction,unembedding,cap=30.):
 eps=torch.finfo(state.dtype).eps
 inv=(state.square().mean(-1,keepdim=True)+eps).rsqrt()
 tangent=direction*inv-state*(state*direction).mean(-1,keepdim=True)*inv.pow(3)
 uncapped=(state*inv)@unembedding.T
 return (1-torch.tanh(uncapped/cap).square())*(tangent@unembedding.T)

def controls():
 torch.manual_seed(913);torch.set_num_threads(2);dtype=torch.float64;rows=[]
 for scale in [.01,1.,10.]:
  x=torch.randn(9,7,dtype=dtype)*scale;d=torch.randn_like(x);u=torch.randn(13,7,dtype=dtype)
  fn=lambda z:30*torch.tanh(F.linear(F.rms_norm(z,(7,)),u)/30)
  _,oracle=torch.func.jvp(fn,(x,),(d,));actual=derivative(x,d,u)
  step=scale*1e-5;finite=(fn(x+step*d)-fn(x-step*d))/(2*step)
  rows.append(dict(scale=scale,jvp_relative=float((actual-oracle).norm()/oracle.norm()),finite_relative=float((actual-finite).norm()/finite.norm())))
 assert max(r['jvp_relative'] for r in rows)<1e-12 and max(r['finite_relative'] for r in rows)<1e-7
 return rows
if __name__=='__main__':
 import json
 from pathlib import Path
 rows=controls();Path(__file__).with_name('LOGIT_DIRECTIONAL_RESPONSE_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
