"""Unpriced dense-core relaxation within each jointly learned pair's actual reader span."""
from pathlib import Path
import json,torch
from local_shared_reader_graph import expand
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=S.dtype);data=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True);meta=json.loads((P/'JOINT_OVERLAP_V1.json').read_text());ids=d['indices'];z=d['z'][ids];h=d['h'][ids];rms=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();rows=[]
for rec in meta['records']:
 program=data[rec['key']];bundle=expand(program);shaped=rec['key'].startswith('calibration_shaped');transform=S if shaped else I;inverse=d['inverse_root'] if shaped else I;components=[]
 for j in range(3):
  reader=transform@bundle[str(j)]['shared_reader'];U,s,_=torch.linalg.svd(reader,full_matrices=False);rank=int((s>1e-10*s[0]).sum());U=U[:,:rank];Qs=torch.stack(d['pairs'][j]['Qs']);T=transform@Qs@transform;cores=U.T@T@U;hat=inverse@U@cores@U.T@inverse;error=float((U@cores@U.T-T).norm()/T.norm());reads=[]
  for Q,H in zip(Qs,hat):
   delta=Q-H;linear=2*delta@d['mu'];bias=torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu'];reads.append(torch.einsum('ni,ij,nj->n',z,H,z)+z@linear+bias)
  pair=d['pairs'][j];phi=((h@pair['a']-.5*reads[0])/rms-pair['alpha'])*(reads[1]/rms-pair['beta']);truth=pair['truth'][ids];value=float((phi-truth).norm()/(truth-truth.mean()).norm());components.append(dict(mode=j+1,span_rank=rank,metric_coefficient_error=error,component_error=value,dense_symmetric_core_coefficients=rank*(rank+1)))
 aggregate=(sum(c['metric_coefficient_error']**2 for c in components)/3)**.5;actual=rec['covariance_error' if shaped else 'native_error'];assert aggregate<=actual+1e-8
 rows.append(dict(key=rec['key'],components=components,dense_core_metric_error=aggregate,actual_product_graph_metric_error=actual,scope='Optimal coefficient projection within these fixed pair input spans, but not an adopted/priced shared program. Recompiling dense pair cores would change reader maps and arithmetic.'))
out=dict(records=rows,coefficient_guards={k:1.10*meta['plan']['baseline'][k] for k in ('native_error','covariance_error')},scope='Separates reader-span restriction from fixed product topology for the overlapping edit. Uses opened448only; no native model forward or fresh claim.')
(P/'JOINT_READER_SPAN_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
