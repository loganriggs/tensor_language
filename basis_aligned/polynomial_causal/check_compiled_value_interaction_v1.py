"""Direct fourmode formula and native cachedchange comparison, frozen program.
A formula<=1e-10; B cached nativevaluechange<=5%. No changed input passed to executor.
"""
from pathlib import Path
import torch,json
import torch.nn.functional as F
from compiled_value_interaction_v1 import compile_response,execute,EPS
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(980);g=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);d=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][0].double();p=compile_response(g,d);cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];z=torch.cat([cache['z8'][i,:len(r['ids'])] for i,r in enumerate(rows)]).double();a=torch.cat([cache['a8'][i,:len(r['ids'])] for i,r in enumerate(rows)]);raw=torch.stack([torch.cat([cache['r9'][s,i,:len(r['ids'])] for i,r in enumerate(rows)]) for s in [0,1]]);rho=(raw[0].square().mean(-1)+EPS).sqrt().double();target=(F.rms_norm(raw,(1152,),eps=EPS).double()@g['reader']);target=target[1]-target[0]
 def direct(z,a,rho):
  changed=z-a[...,None]*d
  def num(x):return p['downstream_gain']*((x@g['reader'])+((x@p['readers'].T).square()*p['eigenvalues']).sum(-1)/(x.square().mean(-1)+EPS))
  return (num(changed)-num(z))/rho
 rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30));pred=execute(z,a,rho,p);control=rel(pred,direct(z,a,rho));random=torch.randn(32,1152,dtype=torch.float64);testa=torch.linspace(-1,1,32,dtype=torch.float64)*random.norm(dim=-1)/d.norm();random_error=rel(execute(random,testa,torch.ones(32),p),direct(random,testa,torch.ones(32)));error=rel(pred,target)
 result=dict(pred_a=max(control,random_error)<=1e-10,pred_b=error<=.05,cached_formula_error=control,random_formula_error=random_error,native_change_error=error,tensor_scalars=sum(x.numel() for x in p.values()),tensor_bytes=sum(x.numel()*x.element_size() for x in p.values()),scope='Pristinez8+interventionamplitude+baseline9norm; no changedstate, changednorm or nativebaselinevalue input. Frozenfourmode valuechange approximation. Existingbroaderhead8removalcache; actualselective/newcue data pending. Prefix/routing generation not closed.')
 torch.save(p,P/'COMPILED_VALUE_INTERACTION_V1_PROGRAM.pt');(P/'COMPILED_VALUE_INTERACTION_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
