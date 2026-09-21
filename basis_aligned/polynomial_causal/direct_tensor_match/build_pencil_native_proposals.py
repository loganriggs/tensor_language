"""CPU native-size initialization from bounded algebraic block proposals."""
from pathlib import Path
import json,time,torch
from pencil_shared_scalable import propose
from pairwise_reader_graph import GROUPS
from joint_orthogonal_private import JointOrthogonalPrivateMetric
from pairwise_graph_assessment import Assessment
from export_orthogonal_private import export
P=Path(__file__).parent;torch.set_num_threads(2);plan=json.loads((P/'PENCIL_NATIVE_PROPOSAL_PLAN_V1.json').read_text());start=time.monotonic()
data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);parents=torch.load(P/'ALTERNATING_COMPLETION_PROGRAMS_V1.pt',weights_only=True);audit=Assessment(data);records=[];states={};graphs={}
for geometry in plan['metrics']:
 A,inv=(audit.S,data['inverse_root']) if geometry=='calibration_shaped' else (audit.I,audit.I);row=dict(geometry=geometry)
 try:
  target=A@audit.Q@A;proposal=propose(target,plan['edge_width'],plan['private_width'],plan['beam']);params=[v.detach().clone().requires_grad_() for v in proposal['bases']+[torch.linalg.qr(v,mode='reduced').Q for v in proposal['private']]]
  metric=JointOrthogonalPrivateMetric(target,[torch.cat([params[a],params[b]],1).detach() for a,b in GROUPS]);loss,components=metric.loss(params);dense=metric.loss(params,dense=True)[0];replay=abs(float((loss-dense).detach()));grads=torch.autograd.grad(loss,params);assert replay<1e-8 and all(torch.isfinite(g).all() for g in grads)
  states[geometry]=[v.detach() for v in params];row.update(loss=float(loss.detach()),loss_replay=replay,gradient_norms=[float(g.norm()) for g in grads],proposal_stats=proposal['stats'])
  template=dict(parents['calibration_shaped_pairwise_0'],input_bases={str(j):inv@params[j].detach() for j in range(3)})
  with torch.no_grad():
   graph,diag=export(components,metric.scales,template,A,inv);graph=audit.correct(graph);scores=audit.assess(graph);field='covariance_error' if geometry=='calibration_shaped' else 'native_error';assert abs(scores[field]-float(dense.detach().sqrt()))<1e-8
   assert scores['stored_floats']==scores['physical_storage_floats']==1058124 and scores['source_total_multiplications']==1047648
  graphs[geometry]=graph;row.update(instrument=True,compiler=diag,**scores)
 except (ValueError,RuntimeError,AssertionError) as error:row.update(instrument=False,failure=str(error))
 records.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('compiler','proposal_stats')}),flush=True)
torch.save(states,P/'PENCIL_NATIVE_PROPOSAL_STATES_V1.pt');torch.save(graphs,P/'PENCIL_NATIVE_PROPOSAL_PROGRAMS_V1.pt');(P/'PENCIL_NATIVE_PROPOSAL_V1.json').write_text(json.dumps(dict(plan=plan,records=records,seconds=time.monotonic()-start),indent=2)+'\n')
