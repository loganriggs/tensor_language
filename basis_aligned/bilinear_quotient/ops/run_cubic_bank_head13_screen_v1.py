#!/usr/bin/env python3
# BQGATE:192bodyforwards;48prefixes;4arms;600seconds;frozen banks/no fitting.
"""pred_a baseline<=1e-4, FP32/64 projection<=1e-4, live component and cue.
pred_b either bank BOTH families coverage>=.10, >=10/12 positive pairs,
control mean absolute effect <=.5 target mean absolute effect.
pred_c B plus replica removal-effect relative RMS<=.10 in BOTH families.
Null: coefficient-gain head13 projection is weak, nonspecific or restart-unstable.
Price:192bodyforwards,600seconds, two earlier frozen rank16 source banks.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from compiled_cubic_bank_head_v1 import compile_head,execute_pairs,execute_sequence
from cubic_cluster_coordinates_v1 import components
from folded_normalized_router_v1 import rotary,EPS
from regional_cue_row_check_v1 import validate
STEM='CUBIC_BANK_HEAD13_SCREEN_V1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==48
 for family in [0,1]:assert sum(r['family']==family for r in rows)==24
 audit=json.loads((P/'COMPILED_CUBIC_BANK_NATIVE_V1_AUDIT.json').read_text());assert all(audit[k] for k in ['pred_a','pred_b','pred_c'])
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('192bodyforwards; frozen physical head13.0 banks;600seconds');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(600);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();a=model.transformer.h[13].attn
 q1,k1,q2,k2=[getattr(a,n).weight.reshape(9,128,1152)[0].float() for n in ['c_q','c_k','c_q2','c_k2']]
 mu=float(a.lamb);v=torch.cat(((1-mu)*a.c_v.weight.double(),mu*model.transformer.h[0].attn.c_v.weight.double()),1).reshape(9,128,2304)[0];o=a.c_proj.weight.double().reshape(1152,9,128)[:,0]
 programs=[]
 for item in torch.load(P/'CUBIC_CLUSTER_NATIVE_V1_CHARTS.pt',weights_only=True,map_location='cpu'):
  c,m=components(item['theta'],item['chart']);c=c.cuda();m=m.cuda();programs.append((compile_head(c,m,q1,k1,q2,k2,v,o),compile_head(c,m,q1.double(),k1.double(),q2.double(),k2.double(),v,o)))
 assert len(programs)==2
 context={};rounding=[];norms=[];count=0
 def first_hook(module,args):context['first']=args[0]
 def projection_hook(module,args):context['full']=args[0][...,:128].float()@o.float().T
 def head_hook(module,args,output):
  current=args[0];arm=context['arm']
  if arm==0:
   context['current']=current.clone();context['fields']=[execute_sequence(current,context['first'],p[0]) for p in programs];norms.extend(float(z.norm()) for z in context['fields'])
   length=current.shape[1];source=torch.cat((current,context['first']),-1)[0];query=current[0,-1].expand(length,-1);rr=torch.stack([rotary(length-1,128).T@rotary(j,128) for j in range(length)]).cuda()
   for p32,p64 in programs:
    z32=execute_pairs(query,source,rr,p32);z64=execute_pairs(query.double(),source.double(),rr,p64);rounding.append(float((z32-z64).norm()/z64.norm().clamp_min(1e-30)))
   return output
  assert torch.equal(current,context['current'])
  field=context['fields'][arm-1] if arm<3 else context['full']
  return output[0]-field.to(output[0].dtype),output[1]
 handles=[model.transformer.h[0].attn.register_forward_pre_hook(first_hook),a.c_proj.register_forward_pre_hook(projection_hook),a.register_forward_hook(head_hook)]
 margins=torch.zeros(48,4,2,dtype=torch.float64)
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda')
   for arm in range(4):
    context['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,),eps=EPS);x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,),eps=EPS))/30))[0];count+=1
    margins[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();margins[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
   if i%12==11:print(json.dumps(dict(rows=i+1,seconds=time.perf_counter()-tic)),flush=True)
 finally:
  for handle in handles:handle.remove()
 assert count==192
 old=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['margins'][:,0];baseline_error=float((margins[:,0]-old).norm()/old.norm().clamp_min(1e-30));cells=[];replicas=[];capability=[]
 for family in [0,1]:
  ix=[i for i,r in enumerate(rows) if r['family']==family];native=margins[ix,0,0];contrast=native[::2]-native[1::2];capability.append(dict(family=family,mean=float(contrast.mean()),positive=int((contrast>0).sum())))
  for arm in [1,2,3]:
   changed=margins[ix,arm,0];reduction=contrast-(changed[::2]-changed[1::2]);target=float((native-changed).abs().mean());control=float((margins[ix,arm,1]-margins[ix,0,1]).abs().mean());coverage=float(reduction.mean()/contrast.mean());positive=int((reduction>0).sum())
   cells.append(dict(family=family,arm=arm,coverage=coverage,positive_pairs=positive,target_meanabs=target,control_meanabs=control,passed=coverage>=.1 and positive>=10 and control<=.5*target))
  e0=margins[ix,0,0]-margins[ix,1,0];e1=margins[ix,0,0]-margins[ix,2,0];den=((e0.square().mean()+e1.square().mean())/2).sqrt();err=(e0-e1).square().mean().sqrt()/den.clamp_min(1e-30);replicas.append(dict(family=family,reference_rms=float(den),relative_rms=float(err),passed=float(den)>=1e-5 and float(err)<=.1))
 A=baseline_error<=1e-4 and max(rounding)<=1e-4 and min(norms)>1e-8 and all(z['mean']>=.2 and z['positive']>=10 for z in capability)
 B=A and any(all(z['passed'] for z in cells if z['arm']==arm) for arm in [1,2])
 result={'pred_a':A,'pred_b':B,'pred_c':B and all(z['passed'] for z in replicas),'baseline_relative_error':baseline_error,'max_fp32_projection_error':max(rounding),'min_component_norm':min(norms),'capability':capability,'cells':cells,'replicas':replicas,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Existing48 regional prompts, frozen earlier trust-pilot cluster banks, physical O head13.0 projection removal with native prefix/norms/suffix. Not fresh/OOD, donor or sufficiency evidence. Full head descriptive: components may oppose. FP64 polynomial identity is distinct from FP32-native arithmetic.'}
 torch.save(dict(margins=margins),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
