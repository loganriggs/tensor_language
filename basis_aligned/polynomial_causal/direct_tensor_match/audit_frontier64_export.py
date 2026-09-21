"""Freeze and independently audit the passing local frontier graph."""
import json,hashlib
from pathlib import Path
import torch
from audit_pairwise_execution import components,cast
from pairwise_reader_graph import expand,price
from local_shared_reader_graph import decode
from pairwise_graph_assessment import Assessment,global_form
from global_mixed_source_graph import score
P=Path(__file__).parent;torch.set_num_threads(2)
path=P/'TWO_READ_CORRECTION_PROGRAM_FRONTIER64_V1.pt';program=torch.load(path,weights_only=True);d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'TWO_READ_CORRECTION_FRONTIER64_V1.json').read_text());assert meta['predictions']['fidelity'];ids=d['indices'];z=d['z'][ids];h=d['h'][ids]
with torch.no_grad():
 phi=components(z,h,program);phi32=components(z.float(),h.float(),cast(program)).double();drift=((phi32-phi).norm(dim=0)/(phi-phi.mean(0)).norm(dim=0)).tolist();assert max(drift)<1e-4
 bundle=expand(program);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);lin=torch.stack([bundle[str(j)][k+'_linear'] for j in range(3) for k in ('a','b')]);bias=torch.stack([bundle[str(j)][k+'_bias'] for j in range(3) for k in ('a','b')])
zz=z[:16].clone().requires_grad_(True);hh=h[:16];v=components(zz,hh,program);gradient=[]
with torch.no_grad():
 reads=torch.einsum('ni,oij,nj->no',zz,H,zz)+zz@lin.T+bias;grad=2*torch.einsum('oij,nj->noi',H,zz)+lin[None];s=(hh.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
for j in range(3):
 actual=torch.autograd.grad(v[:,j].sum(),zz,retain_graph=j<2)[0]
 with torch.no_grad():
  p=program['pairs'][str(j)];a=(hh@p['h_reader']-.5*reads[:,2*j])/s-p['alpha'];b=reads[:,2*j+1]/s-p['beta'];expected=-.5*(b/s)[:,None]*grad[:,2*j]+(a/s)[:,None]*grad[:,2*j+1];gradient.append(float((actual-expected).norm()/expected.norm()))
assert max(gradient)<1e-8
with torch.no_grad():
 assessment=Assessment(d).assess(program);baselines=torch.load(P/'FRONTIER_PAIR_BASELINES_V1.pt',weights_only=True);comparisons=[]
 for geometry in ('calibration_shaped','native_isotropic'):
  baseline=baselines[geometry+'_323'];scores=score(global_form(baseline),d);comparisons.append(dict(geometry=geometry,**scores,candidate_component_within_110percent=[a<=1.1*b for a,b in zip(assessment['per_mode_errors'],scores['per_mode_errors'])],candidate_jacobian_within_110percent=[a<=1.1*b for a,b in zip(assessment['euclidean_jacobian_errors'],scores['euclidean_jacobian_errors'])]))
 for name in ('native_error','covariance_error','source_total_multiplications'):assert abs(assessment[name]-meta['history'][-1][name])<1e-10
out=dict(program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fp32_component_drift=drift,actual_executor_gradient_replay=gradient,assessment=assessment,baselines=comparisons,source_saving=1-assessment['source_total_multiplications']/1330560,scope='Independent saved-artifact replay and FP32/autodiff checks on448opened states/16gradient rows. Immutable candidate for prospective evaluation. Source price excludes common endpoint/affine work consistently with baselines. Not full-model speedup, fresh validity or semantic circuit identification.')
(P/'FRONTIER64_EXPORT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
