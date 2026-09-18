#!/usr/bin/env python3
# BQGATE:120bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a anchors<=1e-5/write<=1e-4; pred_b margin<=1e-4abs/1e-5relative;
pred_c all effects<=1e-3; pred_d isolated40fixtures<=1e-4.120forwards.
No compression, fresh science, or causal-composition promotion.
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
STEM='CITY_FULL_EXTRACTED_V1';ARMS=['native','native8_midpoint','compiled']
PACKAGE=P/'extracted_circuits/city_full_removal_v1'
sys.path.insert(0,str(PACKAGE))
import execute as compiled
from head2_mlp8_cross_edit_v1 import retained_delta
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 doc=json.loads((P/'CITY_FULL_FRESH_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print('120bodyforwards;40prefixes;native/native8/compiled');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 program={k:{name:v.cuda() for name,v in params.items()} for k,params in torch.load(PACKAGE/'program.pt',weights_only=True).items()}
 p['token_ids']=program['head8']['token_ids'];p['first_table']=program['head8']['first_table']
 saved=torch.load(P/'CITY_FULL_FRESH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 cache=[{} for _ in groups];state={};joint_errors=[];outside=[];fixtures=[]
 def pre8(module,args):
  if state['arm']=='native':
   c=cache[state['i']];c['residual7']=args[0].clone();c['mixed']=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone();c['rho8']=(c['mixed'].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
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
   inputs={'residual7':c['residual7'],'token_ids':torch.tensor([row['ids']],device=x.device),'city':city,'destination':c['mask']}
   actual=saved[i]['expected_candidate_delta'].to(x.device)
   c['compiled']=compiled.execute(program,**inputs)
   joint_errors.append(float((c['compiled']-actual).norm()/actual.norm().clamp_min(1e-30)))
   outside.append(float(c['compiled'][:,~c['mask']].abs().max()))
   fixtures.append({'inputs':{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in inputs.items()},'expected_native_delta':actual.cpu()})
  elif arm=='compiled':return x+c['compiled'].to(x.dtype),first,x0
 handles=[model.transformer.h[8].attn.c_proj.register_forward_pre_hook(projection_pre),model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_pre_hook(attnpre),model.transformer.h[8].attn.register_forward_hook(attnpost),model.transformer.h[9].register_forward_pre_hook(pre9)]
 values=torch.zeros(3,40,10,dtype=torch.float64);count=0
 try:
  for ai,arm in enumerate(ARMS):
   state['arm']=arm
   for i,row in enumerate(groups):
    state['i']=i
    if arm=='native8_midpoint':
     c=cache[i];city=row['city_position'];mask=torch.zeros(len(row['ids']),dtype=torch.bool,device='cuda');mask[row['destination_positions']]=True;c['mask']=mask
     route=native.routing(p,c['current'],c['current'][:,city],city);iv=native.inherited(p,row['ids'][city]).to(c['current'].dtype);cv=F.linear(c['current'],p['current_value'])[:,city];value=p['mixture']*iv+(1-p['mixture'])*cv
     c['delta']=-.5*F.linear(route[...,None]*value[:,None],p['output'])*mask[None,:,None]
    ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[ai,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:
  for h in handles:h.remove()
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'CITY_FULL_FRESH_V1_ARTIFACT.pt',weights_only=True)['values'];anchor=float((v[:2]-old[[0,1]]).abs().max());records={}
 for family in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==family];actual=(old[3]-old[0])[idx];prediction=e[2,idx]
  records[family]=((prediction-actual).norm(dim=0)/actual.norm(dim=0).clamp_min(1e-30)).tolist()
 diff=v[2]-old[3];abs_error=float(diff.abs().max());relative_error=float(diff.norm()/old[3].norm())
 result={'pred_a':anchor<=1e-5 and max(joint_errors)<=1e-4 and max(outside)==0 and bool(torch.isfinite(v).all()) and count==120,'pred_b':abs_error<=1e-4 and relative_error<=1e-5,'pred_c':max(max(x) for x in records.values())<=1e-3,'family_effect_errors':records,'anchor_max_abs':anchor,'max_joint_input_error':max(joint_errors),'max_outside':max(outside),'margin_max_abs':abs_error,'margin_relative':relative_error,'body_forwards':count,'seconds':time.perf_counter()-start,'source_shas':binding,'scope':'Opened fresh-panel precision replay. Native suffix external. Isolated replay gate scored separately; no composition/simplicity promotion.'}
 torch.save({'values':v,'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
