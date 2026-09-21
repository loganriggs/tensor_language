"""CPU fixed-basis graph edit + exact constrained output refit. No new native data."""
from pathlib import Path
import json,time,torch
from shared_linear_source_graph import factor,expand,source_reads,component_scalars,arithmetic
from compact_source_graph import source_reads as reference
from global_mixed_source_graph import export,score
from dual_geometry_source_metric import DualGeometrySourceMetric
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def big_program(p):
 out={k:v for k,v in p.items() if k not in ('square_reader','square_weights','square_output')}
 out['left_reader']=torch.cat([p['left_reader'],p['square_reader']],1)
 out['right_reader']=torch.cat([p['right_reader'],p['square_reader']],1)
 extra=p['product_weights'].new_zeros(len(p['square_weights']),6);extra[:,5]=p['square_weights']
 out['product_weights']=torch.cat([p['product_weights'],extra],0)
 return out

def main():
 start=time.perf_counter();plan=json.loads((P/'SHARED_LINEAR_GRAPH_PLAN_V1.json').read_text());d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
 Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);T=Q/d['scales'][:,None,None];S=torch.linalg.inv(d['inverse_root']);ids=d['indices'];z=d['z'][ids];h=d['h'][ids];rows=[];parents={};saved={}
 for label,alpha in plan['parents'].items():
  parent=torch.load(P/f'DUAL_FRESH_{label.upper()}_V1.pt',weights_only=True);m=parent['left_reader'].shape[1];v=parent['square_reader'].shape[1];joint=torch.cat([parent[k] for k in ('left_reader','right_reader','square_reader')],1);U,s,_=torch.linalg.svd(joint,full_matrices=False)
  metric=DualGeometrySourceMetric(T,S,alpha)
  def assess(p):
   raw=torch.einsum('ir,ro,jr->oij',p['left_reader'],p['product_weights'],p['right_reader']);hat=(raw+raw.transpose(-1,-2))/2;hat[5]+=(p['square_reader']*p['square_weights'])@p['square_reader'].T
   err=(hat-Q)/d['scales'][:,None,None];cov=S@err@S
   values=score(big_program(p),d)
   return dict(native_error=float((err.square().sum((-1,-2)).reshape(3,2).sum(1)/metric.energies[0]).mean().sqrt()),covariance_error=float((cov.square().sum((-1,-2)).reshape(3,2).sum(1)/metric.energies[1]).mean().sqrt()),**values)
  parents[label]=dict(**assess(parent),**arithmetic(parent))
  full=factor(parent,U);full_replay=float((source_reads(z,full)-reference(z,parent)).norm()/reference(z,parent).norm());assert full_replay<1e-10
  for rank in plan['ranks']:
   factorized=factor(parent,U[:,:rank]);p=expand(factorized);L=p['left_reader'];R=p['right_reader'];V=p['square_reader']
   _,W,w=metric.loss(L,R,V)
   extra=W.new_zeros(6,v);extra[5]=w
   big=export(S@torch.cat([L,V],1),S@torch.cat([R,V],1),torch.cat([W,extra],1),d)
   p.update(product_weights=big['product_weights'][:m].clone(),square_weights=big['product_weights'][m:,5].clone(),source_linear=big['source_linear'],source_bias=big['source_bias'])
   # Preserve the actual two-stage program, with no retained dense readers.
   factorized.update({k:p[k] for k in ('product_weights','square_weights','source_linear','source_bias')})
   actual=source_reads(z,factorized);expected=reference(z,p);execution=float((actual-expected).norm()/expected.norm());assert execution<1e-8
   component_replay=float((component_scalars(z,h,factorized)-__import__('compact_source_graph').component_scalars(z,h,p)).norm()/__import__('compact_source_graph').component_scalars(z,h,p).norm());assert component_replay<1e-8
   key=f'{label}_{rank}';saved[key]=factorized
   row=dict(key=key,parent=label,rank=rank,full_reader_control_replay=full_replay,execution_replay=execution,component_execution_replay=component_replay,relative_reader_error=float(s[rank:].norm()/s.norm()),**arithmetic(factorized),**assess(p));rows.append(row);print(json.dumps(row),flush=True)
 primary=next(r for r in rows if r['key']=='mixed_448');base=parents['mixed']
 gates=dict(pred_a_instrument=all(max(r['full_reader_control_replay'],r['execution_replay'],r['component_execution_replay'])<1e-8 for r in rows),pred_b_component_fidelity=all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),pred_c_metric_fidelity=primary['native_error']<=1.1*base['native_error'] and primary['covariance_error']<=1.1*base['covariance_error'],pred_d_total_arithmetic=primary['source_total_multiplications']<=.8*base['source_total_multiplications'] and primary['stored_floats']<base['stored_floats'])
 torch.save(saved,P/'SHARED_LINEAR_GRAPH_PROGRAMS_V1.pt');out=dict(plan=plan,parents=parents,records=rows,predictions=gates,seconds=time.perf_counter()-start,scope='SVD proposes shared linear input nodes; fixed projected readers then exact constrained output refit in each parent metric. Actual factorized execution and original component definitions retained. Opened448 diagnostic only, no fresh or semantic identification.')
 (P/'SHARED_LINEAR_GRAPH_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(gates),flush=True)
if __name__=='__main__':main()
