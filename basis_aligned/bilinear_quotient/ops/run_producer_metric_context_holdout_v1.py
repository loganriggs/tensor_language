#!/usr/bin/env python3
# BQGATE:130bodyforwards,1040sequences,lengths14-24;frozen512pairvalidation,no fitting.
"""A manual/physical tail<=1e-5, exact component algebra<=1e-8, finite/bound;
B family swaps<=.10, sign>=.90, live>=4; C removal CE disagreement<=.02;
D family write error<=.05; E native capability>=.75 every family and side.
76,096 fitted values plus15,925,248 shared native producer values and background.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from quartic_frozen_native_score_v2 import score
STEM='PRODUCER_METRIC_CONTEXT_HOLDOUT_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(digest(k)==v for k,v in binding.items())
 authority=json.loads((P/(STEM+'_ROWS.json')).read_text());rows=authority['rows']
 assert hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()==authority['authority_sha256']=='4a9ff5762fc4d7d12385630ab84e67e4bf85f7329f4a3d60e3f57e7f63621f37'
 sequences=[];buckets={}
 for row in rows:
  for side in ('base','donor'):
   ids=row[side+'_ids'];assert row[side+'_prediction_position']==len(ids)-1
   i=len(sequences);sequences.append(ids);buckets.setdefault(len(ids),[]).append(i)
 assert len(rows)==512 and len(sequences)==1024 and min(buckets)==14 and max(buckets)==24
 assert sum((len(v)+7)//8 for v in buckets.values())==128
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=130,sequences=1040,maximum_length=24,fitting=False)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PORTS.pt')
 assert not out.exists() and not ap.exists();signal.alarm(600)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();last=model.transformer.h[17]
 program=torch.load(P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt',weights_only=True)
 baseline_program=torch.load(P/'QUARTIC_OUTER32_V1_PROGRAM.pt',weights_only=True)
 assert torch.equal(baseline_program['output_writers'],program['output_writers'])
 baseline_modes=baseline_program['output_readers'].cuda();baseline_weights=baseline_program['outer_weights'].cuda()
 writer=program['output_writers'].cuda();modes=program['output_readers'].cuda();weights=program['outer_weights'].cuda()
 l0,r0,d0=[getattr(model.transformer.h[16].mlp,k).weight.double() for k in ('Left','Right','Down')]
 l1,r1,d1=[getattr(last.mlp,k).weight.double() for k in ('Left','Right','Down')]
 u=model.lm_head.weight.double();uw=u@writer
 metric_readers=u.T@uw-len(u)*u.mean(0)[:,None]*uw.mean(0)[None,:]
 output_readers=d1.T@metric_readers
 matrices=[]
 for m in range(2):
  raw=l1.T@(output_readers[:,m,None]*r1);matrices.append((raw+raw.T)/2)
 scale=float(last.lambdas[0]);assert scale==program['producer_scale']
 def component(x16,pre):
  x=x16.double();producer=((x@l0.T)*(x@r0.T))@d0.T*scale
  exact_scalar=((producer@l1.T)*(producer@r1.T))@output_readers
  matrix_scalar=torch.stack([((producer@a)*producer).sum(-1) for a in matrices],-1)
  algebra=float((matrix_scalar-exact_scalar).norm()/exact_scalar.norm().clamp_min(1e-30))
  scalar=torch.stack([((producer@modes[m]).square()*weights[m]).sum(-1) for m in range(2)],-1)
  baseline_scalar=torch.stack([((producer@baseline_modes[m]).square()*baseline_weights[m]).sum(-1) for m in range(2)],-1)
  den=pre.double().square().mean(-1)+torch.finfo(torch.float32).eps
  return exact_scalar@writer.T/den[...,None],scalar@writer.T/den[...,None],baseline_scalar@writer.T/den[...,None],algebra
 captured={};count=[0,0]
 def input_hook(module,args):captured['x16']=args[0].detach()
 def counter(module,args):
  count[0]+=1;count[1]+=len(args[0]);assert count[0]<=130 and args[0].shape[1]<=24
 handles=[model.transformer.h[16].mlp.register_forward_pre_hook(input_hook),model.transformer.h[0].attn.register_forward_pre_hook(counter)]
 ports={k:torch.empty(1024,1152) for k in ('input16','pre','native_output')}
 references=torch.empty(1024,1152,dtype=torch.float64);approximations=torch.empty_like(references);baselines=torch.empty_like(references)
 controls=[];algebra_errors=[]
 def logits(h):return 30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
 try:
  first=True
  for length,indices in sorted(buckets.items()):
   for off in range(0,len(indices),8):
    selected=indices[off:off+8];tokens=torch.tensor([sequences[i] for i in selected],device='cuda')
    x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
    for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
    residual=last.lambdas[0]*x+last.lambdas[1]*x0
    attention,v1=last.attn(F.rms_norm(residual,(1152,)),v1)
    pre=residual+attention;xin=F.rms_norm(pre,(1152,));native=last.mlp(xin)
    x16=captured['x16'];ref,approx,baseline,algebra=component(x16[:,-1],pre[:,-1]);algebra_errors.append(algebra)
    for key,value in [('input16',x16),('pre',pre),('native_output',native)]:ports[key][selected]=value[:,-1].cpu()
    references[selected]=ref.cpu();approximations[selected]=approx.cpu();baselines[selected]=baseline.cpu()
    if first:
     assert len(selected)==8
     for replace in (False,True):
      hook_values={}
      def head_hook(module,args,value):hook_values['raw']=value[:,-1].detach()
      def mlp_hook(module,args,value):
       hook_values['input_error']=float((args[0]-xin).norm()/xin.norm())
       if not replace:return value
       value=value.clone();value[:,-1]+=(approx-ref).float();return value
      extra=[model.lm_head.register_forward_hook(head_hook),last.mlp.register_forward_hook(mlp_hook)]
      try:model(tokens,tokens)
      finally:
       for hook in extra:hook.remove()
      native_replaced=native[:,-1]+((approx-ref).float() if replace else 0)
      manual=logits(pre[:,-1]+native_replaced);physical=30*torch.tanh(hook_values['raw']/30)
      controls.append(dict(replace=replace,input_error=hook_values['input_error'],logit_relative_error=float((manual-physical).norm()/physical.norm())))
     first=False
 finally:
  for hook in handles:hook.remove()
 assert count==[130,1040]
 h=ports['pre']+ports['native_output'];cpu_u=model.lm_head.weight.cpu()
 result=score([references,baselines,approximations],h,rows,cpu_u)
 families=[]
 for family in sorted({r['family'] for r in rows}):
  ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
  capability=[]
  for side in (0,1):
   passed=[]
   for i in ids.tolist():
    key='base' if side==0 else 'donor';row=rows[i]
    readers=cpu_u[[row[key+'_answer_id'],row[key+'_foil_id']]]
    z=F.linear(F.rms_norm(h[2*i+side],(1152,)),readers);passed.append(float(z[0]>z[1]))
   capability.append(sum(passed)/len(passed))
  families.append(dict(family=family,write_relative_error=float((approximations[ep]-references[ep]).norm()/references[ep].norm()),native_capability=capability))
 assert all(torch.isfinite(v).all() for v in [references,approximations,*ports.values()])
 torch.save(dict(ports=ports,reference_write=references,approximate_write=approximations,baseline_write=baselines,rows_sha256=digest(P/(STEM+'_ROWS.json')),program_sha256=digest(P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt')),ap)
 result.update({'pred_a':result['pred_a'] and max(algebra_errors)<=1e-8 and all(max(r['input_error'],r['logit_relative_error'])<=1e-5 for r in controls),
  'pred_b':result['pred_b'],'pred_c':result['pred_c'],
  'pred_d':all(f['write_relative_error']<=.05 for f in families),
  'pred_e':all(min(f['native_capability'])>=.75 for f in families)})
 result.update(dict(families=families,controls=controls,maximum_algebra_error=max(algebra_errors),counts=count,artifact_sha256=digest(ap),program_sha256=digest(P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt'),rows_sha256=digest(P/(STEM+'_ROWS.json')),wall_seconds=time.perf_counter()-start,scope='Frozen paired and ordinary programs on new contextual/core constructions;32cells scored separately, existing lexical groups reused; native parents, normalization and background retained. No fitting, corpus OOD or standalone selective-circuit claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
