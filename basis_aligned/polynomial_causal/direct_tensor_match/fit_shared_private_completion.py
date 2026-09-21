"""Deterministic common/private completion followed by fixed-product readout fitting."""
from pathlib import Path
import json,time,torch
from compile_shared_private_pair import compile_common_private
from pairwise_reader_graph import GROUPS,expand,source_reads,price,parameters_from_program,PairwiseOverlapMetric
from local_shared_reader_graph import decode
from pack_reader_graph_artifacts import packed
from fit_local_shared_reader_graph import as_global
from global_mixed_source_graph import score
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
plan=json.loads((P/'SHARED_PRIVATE_COMPLETION_PLAN_V1.json').read_text());start=time.monotonic()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);states=torch.load(P/'PAIRWISE_SUBSPACE_NATIVE_STATES_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=Q.dtype);records=[];outprograms={}
def correct(program):
 for j,p in enumerate(expand(program).values()):
  hat=decode(p)
  for name,true,H in zip(('a','b'),Q[2*j:2*j+2],hat):
   delta=true-H;program['pairs'][str(j)][name+'_linear']=2*delta@d['mu'];program['pairs'][str(j)][name+'_bias']=torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu']
 return packed(program)
def assess(program):
 bundle=expand(program);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);metrics=[]
 for A in (I,S):
  T=A@Q@A;E=A@(H-Q)@A;metrics.append(float((E.square().sum((-1,-2)).reshape(3,2).sum(1)/T.square().sum((-1,-2)).reshape(3,2).sum(1)).mean().sqrt()))
 z=d['z'][d['indices']];linear=torch.stack([bundle[str(j)][k+'_linear'] for j in range(3) for k in ('a','b')],1);bias=torch.stack([bundle[str(j)][k+'_bias'] for j in range(3) for k in ('a','b')]);dense=torch.einsum('ni,oij,nj->no',z,H,z)+z@linear+bias;replay=float((source_reads(z,program)-dense).norm()/dense.norm());assert replay<1e-8
 result=dict(native_error=metrics[0],covariance_error=metrics[1],execution_replay=replay,**score(as_global(bundle),d),**price(program));assert result['stored_floats']==result['physical_storage_floats']==1058124 and result['source_total_multiplications']==1047648
 return result
for key in plan['parents']:
 params=states[key];A,inv=(S,d['inverse_root']) if key.startswith('calibration') else (I,I);writer=d['residual_writer'].clone();program=dict(input_bases={str(j):(inv@params[j]).clone() for j in range(3)},pairs={});diagnostics=[]
 try:
  for j,(a,b) in enumerate(GROUPS):
   block,H,diag=compile_common_private(A@Q[2*j:2*j+2]@A,torch.cat([params[a],params[b]],1),params[3+j]);block['private_reader']=inv@block['private_reader'];pair=d['pairs'][j];block.update(h_reader=pair['a'].clone(),alpha=pair['alpha'].clone(),beta=pair['beta'].clone(),residual_writer=writer);program['pairs'][str(j)]=block;diagnostics.append(diag)
  program=correct(program);before=assess(program)
  metric=PairwiseOverlapMetric(A@Q@A,[program['pairs'][str(j)] for j in range(3)],ridge=1e-12);coeff=parameters_from_program(program,A);loss,W,_=metric.loss(coeff)
  for j in range(3):program['pairs'][str(j)]['product_weights']=W[j]
  program=correct(program);after=assess(program);field='covariance_error' if key.startswith('calibration') else 'native_error';assert after[field]<=before[field]+1e-8
  outprograms[key]=program;records.append(dict(key=key,instrument=True,before_readout_refit=before,completion=diagnostics,**after));print(key,after,flush=True)
 except (ValueError,RuntimeError) as error:
  records.append(dict(key=key,instrument=False,failure=str(error)));print(key,'COMPILER FAILURE',error,flush=True)
primary=next(r for r in records if r['key']==plan['primary']);base=plan['baseline'];valid=primary['instrument']
pred=dict(instrument=all(r['instrument'] for r in records),components=valid and all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),coefficients=valid and all(primary[k]<=1.1*base[k] for k in ('native_error','covariance_error')),arithmetic=valid and primary['source_total_multiplications']<=.8*base['source_total_multiplications'])
torch.save(outprograms,P/'SHARED_PRIVATE_COMPLETION_PROGRAMS_V1.pt');(P/'SHARED_PRIVATE_COMPLETION_V1.json').write_text(json.dumps(dict(plan=plan,records=records,predictions=pred,seconds=time.monotonic()-start,scope='Actual compiled1056product graphs with pairwise sharedlinear dictionaries. Common/private cross approximation is explicit; fixed-product output coefficients refitted. Opened448only, suppliedz/h remain.'),indent=2)+'\n');print(pred,flush=True)
