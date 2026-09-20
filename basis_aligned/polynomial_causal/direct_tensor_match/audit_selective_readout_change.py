"""Exact additive attribution of cached scalar1 MSE change; no causal claim."""
import json
from pathlib import Path
import torch
from extract_scalar_modes import evaluate
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 old={k:v.double() for k,v in torch.load(P/'EXTRACTED_SCALAR_INTERVENTIONS_V1.pt',weights_only=True)['program'].items()};new={k:v.double() for k,v in torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True)['program'].items()};view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);u=view['output_directions'].double()[:,1];mu=view['output_mean'].double();records=[]
 for panel in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels']:
  x=panel['rows'].double();target=(panel['targets'].double()-mu)@u;p=(x@old['A'].T)*(x@old['B'].T);h=(p@old['root_left'].T)*(p@old['root_right'].T);original=evaluate(old,x)[:,1];replacement=evaluate(new,x)[:,1];residual=target-original
  components=dict(constant=torch.full_like(target,float(new['constant'][1]-old['constant'][1])),quadratic=p@(new['quadratic_readout'][1]-old['quadratic_readout'][1]),quartic=h@(new['quartic_readout'][1]-old['quartic_readout'][1]));total=sum(components.values());replay=float((replacement-original-total).norm()/total.norm());assert replay<1e-12
  # Symmetrically allocate all cross terms: sum_i (2<r,d_i>-<d_i,d>) = oldSSE-newSSE.
  gains={name:float((2*residual*delta-delta*total).mean()) for name,delta in components.items()};gain=float((residual.square()-(target-replacement).square()).mean());assert abs(sum(gains.values())-gain)<1e-8*max(abs(gain),1)
  records.append(dict(context=panel['context'],old_mse=float(residual.square().mean()),new_mse=float((target-replacement).square().mean()),mse_gain=gain,symmetric_gain_attribution=gains,component_rms={k:float(v.square().mean().sqrt()) for k,v in components.items()},delta_replay=replay))
 result=dict(records=records,scope='Algebraic symmetric allocation of cached feature1 MSE change. Correlated components are not independent causal effects; not attribution of nonlinear native intervention gains. Reused diagnostics only.')
 (P/'SELECTIVE_READOUT_CHANGE_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
