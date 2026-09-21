from pathlib import Path
import json,torch
from pairwise_reader_graph import GROUPS
from free_private_varpro import FreePrivateMetric,parameters_from_private
from orthogonal_private_varpro import OrthogonalPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2)
data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'ALTERNATING_COMPLETION_PROGRAMS_V1.pt',weights_only=True)
Q=torch.stack([q for pair in data['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(data['inverse_root']);rows=[]
for key in ('calibration_shaped_pairwise_0','native_isotropic_pairwise_26301'):
 transform=S if key.startswith('calibration') else torch.eye(1152,dtype=Q.dtype);program=programs[key]
 shared=[transform@torch.cat([program['input_bases'][str(a)],program['input_bases'][str(b)]],1) for a,b in GROUPS]
 private=[transform@program['pairs'][str(j)]['private_reader'] for j in range(3)]
 old=FreePrivateMetric(transform@Q@transform,shared);new=OrthogonalPrivateMetric(transform@Q@transform,shared)
 old_loss,old_states=old.loss(parameters_from_private(shared,private),dense=True)
 params=[torch.linalg.qr(v,mode='reduced').Q.requires_grad_() for v in private];loss,states=new.loss(params);dense=new.loss(params,dense=True)[0]
 discrepancy=max(abs(float(loss.detach()-old_loss)),abs(float(loss.detach()-dense.detach())));assert discrepancy<1e-9
 matrix_replay=[];denominators=[]
 for (P0,V,E,C0,A0),(P1,D,W,C1,A1) in zip(old_states,states):
  H0=P0@A0@P0.T+V@C0@V.T+P0@E@C0@V.T+V@C0@E.T@P0.T
  H1=D@C1@D.T+P1@(A1-W@C1@W.T)@P1.T
  matrix_replay.append(float((H0-H1).norm().detach()));denominators.append(float((1-torch.linalg.eigvalsh(W.T@W).square().max()).detach()))
 assert max(matrix_replay)<1e-8
 grad=torch.autograd.grad(loss,params);norm=sum(v.square().sum() for v in grad).sqrt();direction=[-v/norm for v in grad]
 with torch.no_grad():
  eps=1e-6;fd=float((new.loss([v+eps*u for v,u in zip(params,direction)])[0]-new.loss([v-eps*u for v,u in zip(params,direction)])[0])/(2*eps))
  fd_relative=abs(fd+float(norm))/float(norm);assert fd_relative<1e-4
  descent=[dict(step=t,loss=float(new.loss([v+t*u for v,u in zip(params,direction)])[0])) for t in (1e-4,1e-3,.01,.1)]
 rows.append(dict(key=key,loss=float(loss.detach()),loss_replay=discrepancy,matrix_replay=matrix_replay,min_core_denominator=denominators,gradient_norm=float(norm),finite_difference_relative=fd_relative,descent=descent))
(P/'ORTHOGONAL_PRIVATE_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Same native warm functions and core optima in equivalent full-private orthonormal coordinates. No fitting or adoption claim.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
