"""Native warm-start directional derivative and descent check; no optimizer sweep."""
from pathlib import Path
import json,torch
from free_private_varpro import FreePrivateMetric,parameters_from_private
from pairwise_reader_graph import GROUPS
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'ALTERNATING_COMPLETION_PROGRAMS_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=Q.dtype);rows=[]
for key in ('calibration_shaped_pairwise_0','native_isotropic_pairwise_26301'):
 A=S if key.startswith('calibration') else I;p=programs[key];shared=[A@torch.cat([p['input_bases'][str(a)],p['input_bases'][str(b)]],1) for a,b in GROUPS];private=[A@p['pairs'][str(j)]['private_reader'] for j in range(3)];params=[v.requires_grad_() for v in parameters_from_private(shared,private)];metric=FreePrivateMetric(A@Q@A,shared);initial=metric.loss(params)[0];gradient=torch.autograd.grad(initial,params);norm=sum(g.square().sum() for g in gradient).sqrt();direction=[-g/norm for g in gradient]
 with torch.no_grad():
  values=[]
  for step in (1e-4,1e-3,1e-2,.1,1.):values.append(dict(step=step,loss=float(metric.loss([v+step*u for v,u in zip(params,direction)])[0])))
  eps=1e-5;plus=metric.loss([v+eps*u for v,u in zip(params,direction)])[0];minus=metric.loss([v-eps*u for v,u in zip(params,direction)])[0];fd=float((plus-minus)/(2*eps));relative=abs(fd+float(norm))/float(norm);assert relative<1e-4
 rows.append(dict(key=key,initial_loss=float(initial.detach()),gradient_norm=float(norm),gradient_norms_by_parameter=[float(g.norm()) for g in gradient],finite_difference_relative_discrepancy=relative,descent_tests=values,has_verified_descent=min(v['loss'] for v in values)<float(initial.detach())))
(P/'FREE_PRIVATE_DESCENT_V1.json').write_text(json.dumps(dict(records=rows,scope='One normalized negative-gradient direction per native warmstart, five bounded lengths. Demonstrates localdescent orstationarity; not a newfit, bestglobalstep orrelaxedthreshold.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
