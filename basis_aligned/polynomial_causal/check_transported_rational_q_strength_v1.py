"""Q-term omission across signed amplitudes against exact conditional generator.
A routing and scalar response errors <=.1 in every family at all five strengths.
Only unit-strength generator has the native replay from the preceding control.
"""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from head17_source_interface_v1 import CHECKPOINT
from regional_even_routing_v1 import routing
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
 rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
 p=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
 prog=torch.load(P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt',weights_only=True)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu');lam=sd['transformer.h.9.lambdas'].double();bias=sd['transformer.h.8.mlp.Down_bias'].double()
 raw=cache['raw9'].double();z=cache['z8'][0].double();a=cache['amplitude8'][0].double()[...,None];w=prog['direction'].double();J=prog['mixed_map'].double();eps=torch.finfo(torch.float32).eps
 ids=torch.zeros(z.shape[:2],dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
 for i,r in enumerate(rows):ids[i,:len(r['ids'])]=torch.tensor(r['ids']);valid[i,:len(r['ids'])]=True
 x0=F.rms_norm(sd['transformer.wte.weight'][ids],(1152,),eps=eps).double();m0=(raw[0]-lam[1]*x0)/lam[0]-z-bias
 beta=(z*w).mean(-1,keepdim=True);gamma=w.square().mean();rho=(z-a*w).square().mean(-1,keepdim=True)+eps
 pp=z@J.T-2*beta*m0;qq=J@w-2*gamma*m0
 terms=torch.stack([-a*w,-a/rho*pp,a.square()/(2*rho)*qq])*lam[0]
 def evaluate(r):
  x=F.rms_norm(r.float(),(1152,),eps=eps);g=routing(x,p,1);return g,(g@(x.double()@p['current_value_reader'])[...,None])[...,0]
 g0,s0=evaluate(raw[0]);records=[]
 def error(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 for strength in [-2.,-1.,.5,1.,2.]:
  aa=a*strength;den=(z-aa*w).square().mean(-1,keepdim=True)+eps
  tt=torch.stack([-aa*w,-aa/den*pp,aa.square()/(2*den)*qq])*lam[0]
  full=tt.sum(0);delta=full-tt[2];gf,sf=evaluate(raw[0]+full);gp,sp=evaluate(raw[0]+delta);cells=[]
  for family in sorted(set(r['family'] for r in rows)):
   ix=[i for i,r in enumerate(rows) if r['family']==family];last=[len(rows[i]['ids'])-1 for i in ix];mask=valid[ix];gm=mask[:,:,None]&mask[:,None,:]
   cells.append(dict(family=family,state_error=error(delta[ix][mask],full[ix][mask]),routing_error=error((gp[ix]-g0[ix])[gm],(gf[ix]-g0[ix])[gm]),scalar_error=error(sp[ix,last]-s0[ix,last],sf[ix,last]-s0[ix,last])))
  records.append(dict(strength=strength,cells=cells))
 A=all(max(c['routing_error'],c['scalar_error'])<=.1 for r in records for c in r['cells']);passing=[r['strength'] for r in records if all(max(c['routing_error'],c['scalar_error'])<=.1 for c in r['cells'])]
 out={'pred_a':A,'passing_strengths':passing,'records':records,'scope':'Signed strength comparison to exact conditional finite-response generator using cached pristine inputs, not native strength interventions.  full head9 routing/scalar reevaluation. Rational term grouping with native pristine MLP background and dense J. No fitted coefficients, fresh data, logit or whole-program savings claim.'}
 (P/'TRANSPORTED_RATIONAL_Q_STRENGTH_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
