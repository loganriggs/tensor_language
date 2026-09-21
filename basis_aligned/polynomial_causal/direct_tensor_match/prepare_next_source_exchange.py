"""Continue fixed-cost node exchange from a saved parent and accumulated directions."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from conditional_source_constraints import ConditionalSourceConstraints
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);name=sys.argv[1];tag=sys.argv[2];record=json.loads((P/name).read_text());r=record['records'][0];plan=record['plan'];edits=plan.get('replacements',[plan['replacement']] if 'replacement' in plan else []);allocation=r['allocation'];offsets=np.cumsum([0]+allocation);x=np.array(r['solver']['x']);data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];parent=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)[r['geometry']+'_inherited'];bundle=expand(parent);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);metric=ConditionalSourceConstraints(data,H,base);S=metric.S;inv=data['inverse_root'];Ks=[]
with torch.no_grad():
 for j,n in enumerate(allocation):
  e,U=torch.linalg.eigh(S@(metric.true[2*j+1]-H[2*j+1])@S);ix=e.abs().argsort(descending=True)[:14];V=U[:,ix].clone()
  for edit in edits:
   if edit['pair']==j:V[:,edit['column']]=torch.tensor(edit['metric_direction'],dtype=H.dtype)
  a=torch.tensor(x[offsets[j]:offsets[j+1]],dtype=H.dtype);Ks.append(S@H[2*j+1]@S+(V[:,:n]*(e[ix[:n]]*a))@V[:,:n].T)
K=torch.stack(Ks).requires_grad_(True);weights=torch.tensor(r['dual_weights'],dtype=H.dtype);ratios=metric.ratios(inv@K@inv);assert float((ratios-torch.tensor(r['solver']['values'],dtype=H.dtype)).abs().max())<1e-8
grad=torch.autograd.grad(weights@ratios,K)[0];options=[]
for j in range(3):
 e,U=torch.linalg.eigh((grad[j]+grad[j].T)/2);ix=e.abs().argmax();options.append((float(e[ix].abs()),j,U[:,ix].detach()))
strength,j,v=max(options,key=lambda t:t[0]);G=np.array(r['quadratics']['G']);b=np.array(r['quadratics']['b']);c=np.array(r['quadratics']['c']);ablations=[]
for k in range(offsets[j],offsets[j+1]):
 t=x.copy();t[k]=0;ablations.append((float((np.einsum('i,kij,j->k',t,G,t)+2*b@t+c).max()),int(k)))
_,remove=min(ablations);new=dict(pair=j,column=int(remove-offsets[j]),metric_direction=v.tolist(),kind='gradient',parent_receipt=name,derivative_magnitude=strength);plan.pop('replacement',None);plan.update(replacements=edits+[new],created_utc=datetime.now(timezone.utc).isoformat(),scope='Sequential fixed14-product graph exchange, selected by current dual-gradient and same-pair ablation. Opened-state proposal only; unchanged original fidelity/cost requirements.')
(P/f'SOURCE_EXCHANGE_{tag}_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared',tag,'pair',j,'column',new['column'],'gradient',strength)
