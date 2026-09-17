#!/usr/bin/env python3
# BQGATE:120bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a anchors<=1e-5/write<=1e-4; pred_b targeterror<=.35/live>=1e-5;
pred_c collateral<=.5target.120forwards.
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
STEM='HEAD2_MLP8_CROSS_EDIT_V1';ARMS=['native','native8_midpoint','compiled']
PACKAGE=P/'extracted_circuits/typed_face_mlp8_coupled_v1'
sys.path.insert(0,str(PACKAGE))
import execute as compiled
from head2_mlp8_cross_edit_v1 import retained_delta
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 doc=json.loads((P/'TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print('120bodyforwards;40prefixes;native/native8/compiled');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 program={k:{name:v.cuda() for name,v in values.items()} for k,values in torch.load(PACKAGE/'program.pt',weights_only=True).items()}
 cache=[{} for _ in groups];state={};joint_errors=[];outside=[];fixtures=[]
 def pre8(module,args):
  if state['arm']=='native':cache[state['i']]['mixed']=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
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
   inputs={'current8':c['current'],'donor_city8':cache[i^1]['current'][:,city],'post_attention8':c['g'],'recipient_token':row['ids'][city],'donor_token':groups[i^1]['ids'][city],'city':city,'destination':c['mask']}
   c['compiled']=compiled.execute(program,**inputs)
   joint_errors.append(float((c['compiled']-actual).norm()/actual.norm().clamp_min(1e-30)))
   c['compiled']=retained_delta(c['compiled'],c['delta'],c['a'],c['a2'],c['g'],program['mlp8'])
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
     c['delta']=.5*native.execute(p,c['current'],cache[i^1]['current'][:,city],row['ids'][city],groups[i^1]['ids'][city],city,mask)
    ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[ai,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:
  for h in handles:h.remove()
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'TYPED_FACE_NATIVE8_FRESH_V1_ARTIFACT.pt',weights_only=True)['values'][[0,3]];anchor=float((v[:2]-old).abs().max());records={}
 for family in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==family];actual=e[1,idx];prediction=e[2,idx];rms=prediction.square().mean(0).sqrt();target=float(rms[0])
  records[family]={'target_error':float((prediction[:,0]-actual[:,0]).norm()/actual[:,0].norm()),'retained_target_rms':target,'control_over_target':(rms[1:]/max(target,1e-30)).tolist()}
 result={'pred_a':anchor<=1e-5 and max(joint_errors)<=1e-4 and max(outside)==0 and bool(torch.isfinite(v).all()) and count==120,'pred_b':all(r['target_error']<=.35 and r['retained_target_rms']>=1e-5 for r in records.values()),'pred_c':all(max(r['control_over_target'])<=.5 for r in records.values()),'families':records,'anchor_max_abs':anchor,'max_full_joint_input_error':max(joint_errors),'max_outside':max(outside),'body_forwards':count,'seconds':time.perf_counter()-start,'source_shas':binding,'scope':'Opened causal screen: only other-head attention/write cross terms omitted. Full normalization/background terms retained; all later native operations recomputed. Not a context port closure or random-null selectivity confirmation.'}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
