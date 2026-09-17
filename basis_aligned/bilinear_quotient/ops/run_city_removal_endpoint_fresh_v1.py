#!/usr/bin/env python3
# BQGATE:800bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a anchors<=1e-5/write<=1e-4; pred_b targeterror<=.35/live>=1e-5;
pred_c collateral<=.5target; pred_d attenuation>=.02/positive>=.90;
pred_e>=2median/16nulls; pred_f<=.8eachbaseline.800forwards.
Fresh new-endpoint removal test; no causal-composition or complete-circuit promotion.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
from typed_face_write_atoms_v1 import native
STEM='CITY_REMOVAL_ENDPOINT_FRESH_V1';ARMS=['native','native8_midpoint','full_reentry','compiled']+[f'null:{k}' for k in range(16)]
PACKAGE=P/'extracted_circuits/typed_face_single_head_norm_v1'
sys.path.insert(0,str(PACKAGE))
import execute as compiled
import city_inherited_removal_v1 as candidate
from freeze_city_removal_endpoint_baselines_v1 import features
from head2_mlp8_cross_edit_v1 import retained_delta
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 doc=json.loads((P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print('800bodyforwards;40prefixes;native/native8/compiled');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 program={k:{name:v.cuda() for name,v in values.items()} for k,values in torch.load(PACKAGE/'program.pt',weights_only=True).items()}
 tables={k:v.cuda() for k,v in torch.load(P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_TABLES.pt',weights_only=True).items()}
 p['token_ids']=tables['token_ids'];p['first_table']=tables['first_table'][:,256:384]
 program['head8']['token_ids']=tables['token_ids'];program['head8']['first_table']=tables['first_table'][:,256:384]
 program['entry']={k:tables[k] for k in ['token_ids','initial_table','lambdas8']}
 candidate_program=program
 cache=[{} for _ in groups];state={};joint_errors=[];outside=[];fixtures=[];norm_errors=[]
 def pre8(module,args):
  if state['arm']=='native':cache[state['i']]['residual7']=args[0].clone();cache[state['i']]['mixed']=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
 def attnpre(module,args):
  if state['arm']=='native':cache[state['i']]['current']=args[0].clone()
 def attnpost(module,args,result):
  c=cache[state['i']]
  if state['arm']=='native':c['a']=result[0].clone();c['g']=c.pop('mixed')+result[0]
  elif state['arm']=='native8_midpoint':return result[0]+c['delta'].to(result[0].dtype),result[1]
  return result
 def projection_pre(module,args):
  if state['arm']=='native':cache[state['i']]['a2']=F.linear(args[0][:,:,256:384],module.weight[:,256:384])
 def pre9(module,args):
  c=cache[state['i']];x,first,x0=args;arm=state['arm']
  if arm=='native':c['x9']=x.clone()
  elif arm=='native8_midpoint':
   actual=x.double()-c['x9'].double();i=state['i'];row=groups[i];city=row['city_position']
   inputs={'post_attention8':c['g'],'delta':c['delta']}
   m=program['mlp8'];z=c['g'].double();d=c['delta'].double();L=m['left'].double();R=m['right'].double();D=m['down'].double();eps=torch.finfo(torch.float32).eps
   s0=z.square().mean(-1,keepdim=True)+eps;s1=(z+d).square().mean(-1,keepdim=True)+eps
   c['complete']=d+(((z+d)@L.T)*((z+d)@R.T)/s1-(z@L.T)*(z@R.T)/s0)@D.T
   joint_errors.append(float((c['complete']-actual).norm()/actual.norm().clamp_min(1e-30)))
   candidate_inputs={'residual7':c['residual7'],'token_ids':torch.tensor([row['ids']],device=x.device),'city':city,'destination':c['mask']}
   c['compiled']=candidate.execute(candidate_program,**candidate_inputs)
   outside.append(float(c['compiled'][:,~c['mask']].abs().max()))
   fixtures.append({'inputs':{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in inputs.items()},'expected_native_delta':actual.cpu(),'candidate_inputs':{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in candidate_inputs.items()},'expected_candidate_delta':c['compiled'].cpu()})
  elif arm=='full_reentry':return x+c['complete'].to(x.dtype),first,x0
  elif arm=='compiled':return x+c['compiled'].to(x.dtype),first,x0
  elif arm.startswith('null:'):
   row=groups[state['i']];seed=17092700+1000*int(arm.split(':')[1])+row['context_id'];generator=torch.Generator(device=x.device).manual_seed(seed)
   direction=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator);norm=c['compiled'].norm(dim=-1,keepdim=True)
   direction=(direction.double()*norm/direction.double().norm(dim=-1,keepdim=True).clamp_min(1e-30)*(1 if row['cue']=='British' else -1)).to(x.dtype)
   norm_errors.append(float(((direction.double().norm(dim=-1,keepdim=True)-norm).abs()/norm.clamp_min(1e-8)).max()));outside.append(float(direction[:,~c['mask']].abs().max()))
   return x+direction,first,x0
 handles=[model.transformer.h[8].attn.c_proj.register_forward_pre_hook(projection_pre),model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_pre_hook(attnpre),model.transformer.h[8].attn.register_forward_hook(attnpost),model.transformer.h[9].register_forward_pre_hook(pre9)]
 values=torch.zeros(20,40,10,dtype=torch.float64);count=0
 try:
  for ai,arm in enumerate(ARMS):
   state['arm']=arm
   for i,row in enumerate(groups):
    state['i']=i
    if arm=='native8_midpoint':
     c=cache[i];city=row['city_position'];mask=torch.zeros(len(row['ids']),dtype=torch.bool,device='cuda');mask[row['destination_positions']]=True;c['mask']=mask
     route=native.routing(p,c['current'],c['current'][:,city],city);value=native.inherited(p,row['ids'][city]).to(c['current'].dtype)
     c['delta']=-.5*F.linear(route[...,None]*p['mixture']*value[:,None],p['output'])*mask[None,:,None]
    ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[ai,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:
  for h in handles:h.remove()
 v=expand(values,mapping,6);e=v-v[:1];records={};baseline=json.loads((P/(STEM+'_BASELINES.json')).read_text())
 X=torch.tensor([features(r) for r in rows],dtype=torch.float64);fit=X@torch.tensor(baseline['coefficients'],dtype=torch.float64);constant=torch.tensor([baseline['cue_constants'][r['cue']] for r in rows],dtype=torch.float64)
 for family in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==family];actual=e[1,idx];prediction=e[3,idx];rms=e[:,idx].square().mean(1).sqrt();target=float(rms[3,0]);nr=rms[4:,0];ordered=nr.sort().values;median=float((ordered[7]+ordered[8])/2)
  native_pair=v[0,idx,0][::2]-v[0,idx,0][1::2];retained_pair=v[3,idx,0][::2]-v[3,idx,0][1::2];cap=native_pair>=.1;atten=(native_pair[cap]-retained_pair[cap])/native_pair[cap]
  records[family]={'target_error':float((prediction[:,0]-actual[:,0]).norm()/actual[:,0].norm()),'retained_target_rms':target,'control_over_target':(rms[3,1:]/max(target,1e-30)).tolist(),'capable_pairs':int(cap.sum()),'attenuation_positive_fraction':float((atten>0).double().mean()) if len(atten) else 0.,'attenuation_mean':float(atten.mean()) if len(atten) else 0.,'target_over_null_median':target/max(median,1e-30),'nulls_beaten':int((nr<target).sum()),'null_readout_rms':rms[4:].tolist(),'cue_constant_error':float((constant[idx]-actual[:,0]).norm()/actual[:,0].norm()),'text_ols_error':float((fit[idx]-actual[:,0]).norm()/actual[:,0].norm())}
 diff=v[2]-v[1];anchor_abs=float(diff.abs().max());anchor_rel=float(diff.norm()/v[1].norm())
 result={'pred_a':anchor_abs<=1e-4 and anchor_rel<=1e-5 and max(joint_errors)<=1e-4 and max(outside)==0 and max(norm_errors)<=1e-5 and bool(torch.isfinite(v).all()) and count==800,
 'pred_b':all(r['capable_pairs']>=18 and r['target_error']<=.35 and r['retained_target_rms']>=1e-5 for r in records.values()),'pred_c':all(max(r['control_over_target'])<=.5 for r in records.values()),
 'pred_d':all(r['attenuation_positive_fraction']>=.90 and r['attenuation_mean']>=.02 for r in records.values()),'pred_e':all(r['target_over_null_median']>=2 and r['nulls_beaten']>=16 for r in records.values()),
 'pred_f':all(r['target_error']<=.8*r['cue_constant_error'] and r['target_error']<=.8*r['text_ols_error'] for r in records.values()),
 'families':records,'full_reentry_max_abs':anchor_abs,'full_reentry_relative':anchor_rel,'max_full_joint_input_error':max(joint_errors),'max_null_norm_error':max(norm_errors),'max_outside':max(outside),'body_forwards':count,'seconds':time.perf_counter()-start,'source_shas':binding,'scope':'Fresh constructions/cities/endpoints donor-free inherited-city removal; one native input and native suffix. Candidate has more native information than text baselines. No corpus/token-only/composition claim.'}
 torch.save({'values':v,'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
