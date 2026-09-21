from pathlib import Path
import json,torch
from free_private_varpro import FreePrivateMetric,parameters_from_private
from pairwise_reader_graph import GROUPS
from export_free_private import export
from pairwise_graph_assessment import Assessment
P=Path(__file__).parent;torch.set_num_threads(2);d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'ALTERNATING_COMPLETION_PROGRAMS_V1.pt',weights_only=True);audit=Assessment(d);rows=[]
for key in ('calibration_shaped_pairwise_0','native_isotropic_pairwise_26301'):
 program=programs[key];A,inv=(audit.S,d['inverse_root']) if key.startswith('calibration') else (audit.I,audit.I);shared=[A@torch.cat([program['input_bases'][str(a)],program['input_bases'][str(b)]],1) for a,b in GROUPS];private=[A@program['pairs'][str(j)]['private_reader'] for j in range(3)];metric=FreePrivateMetric(A@audit.Q@A,shared);params=[x.requires_grad_() for x in parameters_from_private(shared,private)]
 loss,states=metric.loss(params);dense=metric.loss(params,dense=True)[0];grads=torch.autograd.grad(loss,params);assert all(torch.isfinite(g).all() for g in grads);replay=abs(float((loss-dense).detach()));assert replay<1e-8
 with torch.no_grad():
  output,diag=export(states,metric.scales,program,A,inv);output=audit.correct(output);scores=audit.assess(output);field='covariance_error' if key.startswith('calibration') else 'native_error';error=abs(scores[field]-float(dense.sqrt()));assert error<1e-8
  assert scores['stored_floats']==scores['physical_storage_floats']==1058124 and scores['source_total_multiplications']==1047648
 rows.append(dict(key=key,loss_replay=replay,export_replay=error,finite_gradients=True,**scores))
(P/'FREE_PRIVATE_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');print(json.dumps(rows))
