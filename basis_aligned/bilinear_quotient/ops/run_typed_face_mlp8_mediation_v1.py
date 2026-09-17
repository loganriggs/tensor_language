#!/usr/bin/env python3
# BQGATE:160bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a replay<=1e-5, local/joint closures<=1e-4; pred_b singles>=.05joint.
pred_c skip error<=.35; pred_d MLP8 error<=.35; pred_e interaction<=.25smaller
and additive prediction<=.10. 160 forwards, at most40local MLP8 evaluations.
Opened screen; no random-split specificity or new selectivity claim.
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
STEM='TYPED_FACE_MLP8_MEDIATION_V1';ARMS=['native','native8_midpoint','skip_only','mlp_only']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 doc=json.loads((P/'TYPED_FACE_PROSPECTIVE_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print('160bodyforwards;40prefixes;four factorial arms;local MLP8 oracle reused from joint arm');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 cache=[{} for _ in groups];state={};local_errors=[];joint_errors=[];outside=[];term_stats=[]
 mlp=model.transformer.h[8].mlp;L=mlp.Left.weight.double();R=mlp.Right.weight.double();D=mlp.Down.weight.double();eps=torch.finfo(torch.float32).eps
 def pre8(module,args):
  if state['arm']=='native':cache[state['i']]['mixed']=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
 def attnpre(module,args):
  if state['arm']=='native':cache[state['i']]['current']=args[0].clone()
 def attnpost(module,args,result):
  c=cache[state['i']]
  if state['arm']=='native':c['g']=c.pop('mixed')+result[0]
  elif state['arm']=='native8_midpoint':return result[0]+c['delta'].to(result[0].dtype),result[1]
  return result
 def mlppost(module,args,result):
  c=cache[state['i']]
  if state['arm']=='native':c['m0']=result.clone()
  elif state['arm']=='native8_midpoint':
   g=c['g'].double();dg=c['delta'].double();s0=g.square().mean(-1,keepdim=True)+eps;s1=(g+dg).square().mean(-1,keepdim=True)+eps
   lg=g@L.T;rg=g@R.T;ld=dg@L.T;rd=dg@R.T;m0=(lg*rg)@D.T/s0
   terms={'normalization':(s0/s1-1)*m0,'cross':(ld*rg+lg*rd)@D.T/s1,'quadratic':(ld*rd)@D.T/s1}
   folded=sum(terms.values());actual=result.double()-c['m0'].double();c['dm']=folded.float()
   local_errors.append(float((folded-actual).norm()/actual.norm().clamp_min(1e-30)))
   outside.append(float(folded[:,~c['mask']].abs().max()))
   term_stats.append({k:{'norm_ratio':float(v.norm()/folded.norm()),'aligned_fraction':float((v*folded).sum()/folded.square().sum())} for k,v in terms.items()})
 def pre9(module,args):
  c=cache[state['i']];x,first,x0=args;arm=state['arm']
  if arm=='native':c['x9']=x.clone()
  elif arm=='native8_midpoint':
   actual=x.double()-c['x9'].double();pred=c['delta'].double()+c['dm'].double();joint_errors.append(float((pred-actual).norm()/actual.norm().clamp_min(1e-30)))
  elif arm=='skip_only':return x+c['delta'].to(x.dtype),first,x0
  elif arm=='mlp_only':return x+c['dm'].to(x.dtype),first,x0
 handles=[model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_pre_hook(attnpre),model.transformer.h[8].attn.register_forward_hook(attnpost),mlp.register_forward_hook(mlppost),model.transformer.h[9].register_forward_pre_hook(pre9)]
 values=torch.zeros(4,40,10,dtype=torch.float64);count=0
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
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'TYPED_FACE_NATIVE8_SCOPE_V1_ARTIFACT.pt',weights_only=True)['values'][[0,2]];anchor=float((v[:2]-old).abs().max());records={}
 for family in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==family];a=e[:,idx,0];rms=e[:,idx].square().mean(1).sqrt();joint=a[1].norm();interaction=a[1]-a[2]-a[3];smaller=min(a[2].norm(),a[3].norm())
  records[family]={'target_rms_logits':rms[:,0].tolist(),'skip_error':float((a[2]-a[1]).norm()/joint),'mlp_error':float((a[3]-a[1]).norm()/joint),'joint_from_singles_error':float(interaction.norm()/joint),'interaction_over_smaller':float(interaction.norm()/smaller),'single_over_joint_rms':(rms[2:,0]/rms[1,0]).tolist(),'all_readout_rms_logits':rms.tolist()}
 result={'pred_a':anchor<=1e-5 and max(local_errors)<=1e-4 and max(joint_errors)<=1e-4 and max(outside)==0 and bool(torch.isfinite(v).all()) and count==160,
 'pred_b':all(min(r['single_over_joint_rms'])>=.05 and min(r['target_rms_logits'][2:])>=1e-5 for r in records.values()),'pred_c':all(r['skip_error']<=.35 for r in records.values()),'pred_d':all(r['mlp_error']<=.35 for r in records.values()),'pred_e':all(r['joint_from_singles_error']<=.10 and r['interaction_over_smaller']<=.25 for r in records.values()),
 'families':records,'anchor_max_abs':anchor,'max_local_response_error':max(local_errors),'max_joint_input_error':max(joint_errors),'max_outside':max(outside),'local_fold_terms':term_stats,'body_forwards':count,'extra_local_mlp_evaluations':0,'seconds':time.perf_counter()-start,'source_shas':binding,'scope':'Opened native8 factorial. Exact local response folded in FP64 and installed FP32. Native joint supplies oracle with no extra MLP forwards. No fresh/selectivity/random-split composition certification.'}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','local_fold_terms')},indent=2));signal.alarm(0)
if __name__=='__main__':main()
