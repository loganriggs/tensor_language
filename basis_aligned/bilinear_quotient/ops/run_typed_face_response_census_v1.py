#!/usr/bin/env python3
# BQGATE:80bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a residual<=1e-4/readout<=1e-4abs,1e-3relative and anchors<=1e-5.
pred_b direct error<=.35; pred_c fixed early suffix<=.35; pred_d family cosine>=.9.
80forwards. Fixed midpoint; response attribution, not downstream intervention.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v2 import measure,CONTROL_PAIRS
from run_even_value_factorial_native_v1 import setup
from typed_face_write_atoms_v1 import native
STEM='TYPED_FACE_RESPONSE_CENSUS_V1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 doc=json.loads((P/'TYPED_FACE_PROSPECTIVE_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows);n=len(groups)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  assert n==40;print('80bodyforwards;40prefixes;19 response terms including output normalization');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');p={k:v.to('cuda') for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 state={};captures={l:[] for l in range(9,18)};handles=[]
 def callbacks(l):
  def pre(module,args):state[l]={'mixed':(module.lambdas[0]*args[0][:,-1]+module.lambdas[1]*args[2][:,-1]).detach().cpu().double()}
  def mlp(module,args,value):state[l]['mlp']=value[:,-1].detach().cpu().double()
  def post(module,args,value):
   x=value[0][:,-1].detach().cpu().double();m=state[l]['mlp'];a=x-state[l]['mixed']-m;captures[l].append((a,m,x))
  return pre,mlp,post
 for l in range(9,18):
  b=model.transformer.h[l];pre,mlp,post=callbacks(l);handles += [b.register_forward_pre_hook(pre),b.mlp.register_forward_hook(mlp),b.register_forward_hook(post)]
 def write(arm,row,donor_row,current,donor,mask):
  city=row['city_position'];return .5*native.execute(p,current,donor[:,city],row['ids'][city],donor_row['ids'][city],city,mask)
 try:m=measure(model,graph,groups,['native','midpoint'],write)
 finally:
  for handle in handles:handle.remove()
 v=expand(m['values'],mapping,6);old=torch.load(P/'TYPED_FACE_PROSPECTIVE_V1_ARTIFACT.pt',weights_only=True)['values'][[0,3]]
 anchor=float((v-old).abs().max());terms=[];names=[];gamma=1.
 scales={}
 for l in range(17,8,-1):scales[l]=gamma;gamma*=float(model.transformer.h[l].lambdas[0])
 for l in range(9,18):
  assert len(captures[l])==2*n
  for j,name in [(0,'attention'),(1,'mlp')]:
   a=torch.cat([x[j] for x in captures[l]]).reshape(2,n,1152);terms.append(scales[l]*(a[1]-a[0]));names.append(f'{name}{l}')
 final=torch.cat([x[2] for x in captures[17]]).reshape(2,n,1152);delta=final[1]-final[0];raw=torch.stack(terms);residual_error=float((raw.sum(0)-delta).norm()/delta.norm())
 eps=torch.finfo(torch.float32).eps;rho=(final.square().mean(-1,keepdim=True)+eps).sqrt()
 normalized=torch.cat([raw/rho[1][None],(final[0]*(1/rho[1]-1/rho[0]))[None]],0);names.append('output_normalization')
 contributions=torch.zeros(19,n,10,dtype=torch.float64)
 for i,row in enumerate(groups):
  pairs=row['endpoint_pairs']+CONTROL_PAIRS;ids=torch.tensor(pairs,device='cuda');W=model.lm_head.weight[ids].detach().cpu().double()
  u=torch.einsum('ad,jkd->ajk',final[:,i]/rho[:,i],W);du=u[1]-u[0];capped=30*torch.tanh(u/30)
  secant=torch.where(du.abs()>1e-12,(capped[1]-capped[0])/du,1-torch.tanh(u[0]/30).square())
  readers=secant[:,0,None]*W[:,0]-secant[:,1,None]*W[:,1]
  contributions[:,i]=normalized[:,i]@readers.T
 c=expand(contributions,mapping,6);effect=v[1]-v[0];diff=c.sum(0)-effect
 margin_abs=float(diff.abs().max());margin_rel=float(diff.norm()/effect.norm());records={};vectors={}
 for family in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==family];target=effect[idx,0];den=target.norm();parts=c[:,idx,0]
  stats={name:{'norm_ratio':float(part.norm()/den),'aligned_fraction':float(part@target/(target@target)),'mean_paired_logits':float((part[::2]-part[1::2]).mean())} for name,part in zip(names,parts)}
  early=parts[[0,1,3,5,18]].sum(0)
  records[family]={'direct_error':float((parts[0]-target).norm()/den),'fixed_early_error':float((early-target).norm()/den),'term_stats':stats}
  vectors[family]=(parts[:,::2]-parts[:,1::2]).mean(1)
 ref=vectors['line_break'];cosines={family:float(vector@ref/(vector.norm()*ref.norm())) for family,vector in vectors.items() if family!='line_break'}
 result={'pred_a':anchor<=1e-5 and residual_error<=1e-4 and margin_abs<=1e-4 and margin_rel<=1e-3 and bool(torch.isfinite(c).all()) and m['body_forwards']==80,
 'pred_b':all(x['direct_error']<=.35 for x in records.values()),'pred_c':all(x['fixed_early_error']<=.35 for x in records.values()),'pred_d':min(cosines.values())>=.9,
 'anchor_max_abs':anchor,'residual_relative_error':residual_error,'margin_max_abs':margin_abs,'margin_relative':margin_rel,'families':records,'line_break_term_vector_cosines':cosines,
 'body_forwards':m['body_forwards'],'seconds':time.perf_counter()-start,'source_shas':binding,'panel_status':'opened prospective panel','scope':'Fixed-edit finite-change response attribution, including transported module responses, final RMS and softcap secants. No independent downstream edit or selective/extracted suffix claim.'}
 torch.save({'values':v,'contributions':c,'term_names':names,'transported_raw_responses':raw,'final_states':final},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_shas','families']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
