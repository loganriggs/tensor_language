from pathlib import Path
import json,torch
from pairwise_reader_graph import GROUPS
from free_private_varpro import FreePrivateMetric,parameters_from_private
from joint_orthogonal_private import JointOrthogonalPrivateMetric
from export_orthogonal_private import export
from pairwise_graph_assessment import Assessment
P=Path(__file__).parent;torch.set_num_threads(2)
data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'ALTERNATING_COMPLETION_PROGRAMS_V1.pt',weights_only=True)
audit=Assessment(data)
Q=torch.stack([q for pair in data['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(data['inverse_root']);rows=[]
for key in ('calibration_shaped_pairwise_0','native_isotropic_pairwise_26301'):
 transform=S if key.startswith('calibration') else torch.eye(1152,dtype=Q.dtype);program=programs[key]
 shared=[transform@torch.cat([program['input_bases'][str(a)],program['input_bases'][str(b)]],1) for a,b in GROUPS]
 private=[transform@program['pairs'][str(j)]['private_reader'] for j in range(3)]
 old=FreePrivateMetric(transform@Q@transform,shared);new=JointOrthogonalPrivateMetric(transform@Q@transform,shared)
 old_loss,old_states=old.loss(parameters_from_private(shared,private),dense=True)
 bases=[transform@program['input_bases'][str(j)] for j in range(3)];params=[(v/v.norm(dim=0)).detach().requires_grad_() for v in bases]+[torch.linalg.qr(v,mode='reduced').Q.requires_grad_() for v in private];loss,states=new.loss(params);dense=new.loss(params,dense=True)[0]
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
 
 with torch.no_grad():
  inverse=data['inverse_root'] if key.startswith('calibration') else transform
  template=dict(program,input_bases={str(j):inverse@params[j] for j in range(3)});graph,compiler=export(states,new.scales,template,transform,inverse);graph=audit.correct(graph);scores=audit.assess(graph)
  field='covariance_error' if key.startswith('calibration') else 'native_error'
  assert abs(scores[field]-float(dense.detach().sqrt()))<1e-8
  assert scores['stored_floats']==scores['physical_storage_floats']==1058124
  assert scores['source_total_multiplications']==1047648
 rows.append(dict(key=key,export_scores=scores,loss=float(loss.detach()),loss_replay=discrepancy,matrix_replay=matrix_replay,min_core_denominator=denominators,gradient_norm=float(norm),shared_gradient_norm=float(sum(g.square().sum() for g in grad[:3]).sqrt()),private_gradient_norm=float(sum(g.square().sum() for g in grad[3:]).sqrt()),finite_difference_relative=fd_relative,descent=descent))
(P/'JOINT_ORTHOGONAL_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Same native warm functions and core optima with shared dictionaries also variable. No fitting or adoption claim.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
