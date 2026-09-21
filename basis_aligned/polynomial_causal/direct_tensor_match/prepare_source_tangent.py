"""Continue fixed-cost node exchange from a saved parent and accumulated directions."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from conditional_source_constraints import ConditionalSourceConstraints
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);name=sys.argv[1];tag=sys.argv[2];record=json.loads((P/name).read_text());r=record['records'][0];plan=record['plan'];edits=plan.get('replacements',[plan['replacement']] if 'replacement' in plan else []);allocation=r['allocation'];offsets=np.cumsum([0]+allocation);x=np.array(r['solver']['x']);data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];parent=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)[r['geometry']+'_inherited'];bundle=expand(parent);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);metric=ConditionalSourceConstraints(data,H,base);S=metric.S;inv=data['inverse_root'];Ks=[];banks=[]
with torch.no_grad():
 for j,n in enumerate(allocation):
  e,U=torch.linalg.eigh(S@(metric.true[2*j+1]-H[2*j+1])@S);ix=e.abs().argsort(descending=True)[:14];V=U[:,ix].clone()
  for edit in edits:
   if edit['pair']==j:V[:,edit['column']]=torch.tensor(edit['metric_direction'],dtype=H.dtype)
  a=torch.tensor(x[offsets[j]:offsets[j+1]],dtype=H.dtype);banks.append((V[:,:n].clone(),e[ix[:n]].clone(),a.clone()));Ks.append(S@H[2*j+1]@S+(V[:,:n]*(e[ix[:n]]*a))@V[:,:n].T)
K=torch.stack(Ks).requires_grad_(True);weights=torch.tensor(r['dual_weights'],dtype=H.dtype);ratios=metric.ratios(inv@K@inv);assert float((ratios-torch.tensor(r['solver']['values'],dtype=H.dtype)).abs().max())<1e-8
grad=torch.autograd.grad(weights@ratios,K)[0];options=[]
for j,(V,lam,amp) in enumerate(banks):
 for k in range(V.shape[1]):
  v=V[:,k];signed=float(lam[k]*amp[k]);g=(grad[j]+grad[j].T)/2;raw=g@v-v*(v@g@v);length=float(raw.norm());strength=2*abs(signed)*length
  if length>0 and signed!=0:options.append((strength,j,k,v.detach(),(-np.sign(signed)*raw/length).detach()))
strength,j,k,v,tangent=max(options,key=lambda t:t[0]);arms=[]
for label,angle in [('A',.01),('B',.03),('C',.1),('D',.3)]:
 vector=np.cos(angle)*v+np.sin(angle)*tangent;assert abs(float(vector.norm())-1)<1e-10
 arm=json.loads(json.dumps(plan));arm.pop('replacement',None);arm.update(replacements=edits+[dict(pair=j,column=k,metric_direction=vector.tolist(),kind='tangent',angle=angle,parent_receipt=name)],created_utc=datetime.now(timezone.utc).isoformat(),scope='Four preregistered rank-preserving tangent angles; refit amplitudes and accept only improvement versus incumbent worst fidelity at unchanged14-product cost. Opened-state proposal, no fresh validation.')
 filename=f'SOURCE_TANGENT_{tag}_{label}_PLAN.json';(P/filename).write_text(json.dumps(arm,indent=2)+'\n');arms.append(dict(label=label,angle=angle,plan=filename))
out=dict(parent_receipt=name,parent_objective=r['solver']['maximum'],pair=j,column=k,first_order_weighted_descent=strength,arms=arms,selection='Retain incumbent unless best executable arm strictly improves maximum squared normalized fidelity. Report all four arms. Original fidelity limits unchanged.')
(P/f'SOURCE_TANGENT_{tag}_PLAN.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
