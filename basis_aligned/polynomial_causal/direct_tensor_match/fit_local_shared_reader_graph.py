"""Shared selected input projections + direct private bypasses; CPU screen."""
from pathlib import Path
import json,time,torch
from local_shared_reader_graph import select_blocks,factor_bundle,expand,decode,refit_pair,price,source_reads,component_scalars,product_factors
from global_mixed_source_graph import score
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def as_global(bundle):
 left=[];right=[];weights=[];linear=[];bias=[]
 for j in range(3):
  p=bundle[str(j)];L,R=product_factors(p['shared_reader'],p['product_indices']);left.append(L);right.append(R);W=L.new_zeros(L.shape[1],6);W[:,2*j:2*j+2]=p['product_weights'];weights.append(W)
  for key in ('a','b'):linear.append(p[key+'_linear']);bias.append(p[key+'_bias'])
 return dict(left_reader=torch.cat(left,1),right_reader=torch.cat(right,1),product_weights=torch.cat(weights,0),source_linear=torch.stack(linear,1),source_bias=torch.stack(bias),h_readers=torch.stack([bundle[str(j)]['h_reader'] for j in range(3)],1),alpha=torch.stack([bundle[str(j)]['alpha'] for j in range(3)]),beta=torch.stack([bundle[str(j)]['beta'] for j in range(3)]),residual_writer=bundle['0']['residual_writer'])
def main():
 start=time.perf_counter();plan=json.loads((P/'LOCAL_SHARED_READER_PLAN_V1.json').read_text());d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);baselines=torch.load(P/'COST_MATCHED_PAIR_BASELINES_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=Q.dtype);energies=[x.square().sum((-1,-2)).reshape(3,2).sum(1) for x in (Q,S@Q@S)];ids=d['indices'];z=d['z'][ids];h=d['h'][ids];rows=[];parents={};programs={}
 def assess(bundle):
  hat=torch.cat([decode(bundle[str(j)]) for j in range(3)]);E=hat-Q;errors=[E,S@E@S];return dict(native_error=float((errors[0].square().sum((-1,-2)).reshape(3,2).sum(1)/energies[0]).mean().sqrt()),covariance_error=float((errors[1].square().sum((-1,-2)).reshape(3,2).sum(1)/energies[1]).mean().sqrt()),**score(as_global(bundle),d))
 for metric in plan['metrics']:
  parent=baselines[metric+'_384'];transform=I if metric=='native_isotropic' else S
  projectors=[]
  for p in parent.values():
   U=torch.linalg.qr(p['shared_reader'],mode='reduced').Q;projectors.append(U@U.T)
  _,U=torch.linalg.eigh(sum(projectors));parent_factor=factor_bundle(parent,I[:,:0],{str(j):torch.empty(0,dtype=torch.int64) for j in range(3)})
  parents[metric]=dict(**assess(parent),**price(parent_factor));parents[metric]['index_integers']=sum(p['product_indices'].numel() for p in parent.values());original=source_reads(z,parent_factor)
  for rank,count in plan['layouts']:
   basis=U[:,-rank:];selections={};selection_records={}
   for j in range(3):selections[str(j)],selection_records[str(j)]=select_blocks(parent[str(j)],basis,transform,count)
   program=factor_bundle(parent,basis,selections);bundle=expand(program);diagnostics={}
   before=assess(bundle)
   for j in range(3):
    p=bundle[str(j)];target=Q[2*j:2*j+2];W,diag=refit_pair(p,target,transform);p['product_weights']=W;hat=decode(p)
    for key,true,qhat in zip(('a','b'),target,hat):
     delta=true-qhat;p[key+'_linear']=2*delta@d['mu'];p[key+'_bias']=torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu']
    program['pairs'][str(j)].update({k:p[k] for k in ('product_weights','a_linear','a_bias','b_linear','b_bias')});diagnostics[str(j)]=diag
   # Independent polynomial decode, not just the expanded factor executor.
   direct=[]
   for j in range(3):
    p=bundle[str(j)];direct.append(torch.einsum('ni,oij,nj->no',z,decode(p),z)+torch.stack([z@p[k+'_linear']+p[k+'_bias'] for k in ('a','b')],1))
   direct=torch.cat(direct,1);execution=float((source_reads(z,program)-direct).norm()/direct.norm());assert execution<1e-8
   key=f'{metric}_{rank}_{count}';programs[key]=program;row=dict(key=key,metric=metric,shared_width=rank,selected_columns_per_pair=count,maximum_common_input_span=rank+3*(384-count),execution_replay=execution,selection_records=selection_records,refit=diagnostics,before_refit=before,**price(program),**assess(bundle));rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('selection_records','refit','before_refit')}),flush=True)
 primary=next(r for r in rows if r['key']==plan['primary']);base=parents['calibration_shaped'];pred=dict(pred_a_instrument=all(r['execution_replay']<1e-8 for r in rows),pred_b_components=all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),pred_c_metrics=all(primary[k]<=1.1*base[k] for k in ('native_error','covariance_error')),pred_d_arithmetic=primary['source_total_multiplications']<=.8*base['source_total_multiplications'] and primary['stored_floats']<base['stored_floats'])
 out=dict(plan=plan,parents=parents,records=rows,predictions=pred,seconds=time.perf_counter()-start,scope='Graph edit with private bypasses, unchanged six source targets and three components. Shared basis from parent span projectors, block choice from weights-only coefficient perturbations, readout refit in parent geometry. Opened448diagnostics, no fresh validation.')
 torch.save(programs,P/'LOCAL_SHARED_READER_PROGRAMS_V1.pt');(P/'LOCAL_SHARED_READER_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(pred),flush=True)
if __name__=='__main__':main()
