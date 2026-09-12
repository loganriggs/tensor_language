#!/usr/bin/env python3
# BQGATE:150bodyforwards;48contexts<=256tokens;4suffixarms;300seconds;no fitting.
"""pred_a native/compiled replay<=1e-5,noedit and original V1 replay<=1e-4relative.
pred_b EACHnatural16-row half NL/comma mean>=.2,>=12positive,meanCE<=5;
mean replacement damages newlineCE>=.02mean and>=8positive.
pred_c EACHcandidate/naturalhalf meanabsCEchange<=.02,maxabs<=.1 given A/B.
Null: missing capability/control makes preservation inconclusive; V1 misses preserved.
Price150bodyforwards:6calibration batches+3*48test;4suffixarms;300sec,no factorfit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
from compiled_reading_head_v1 import execute_reading_head
from compiled_mixed_token_head_v1 import execute_mixed_token_head
from folded_normalized_router_v1 import rotary
STEM='SCALAR_PRODUCERS_NEWLINE_NATURAL_V2'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 payload=json.loads((P/(STEM+'_ROWS.json')).read_text());rows=payload['rows'];mean_rows=payload['mean_token_rows']
 assert len(rows)==48 and len(mean_rows)==24 and all(len(r)==256 for r in mean_rows)
 assert all(sum(r['pool']==pool and r['family']==f for r in rows)==count for pool,count in [('authored',8),('fineweb',16)] for f in range(2))
 assert max(len(r['ids']) for r in rows)<=256 and all(r['newline_id']==198 and r['comma_id']==11 for r in rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('150bodyforwards48fixedcontexts;zero/meanheadcontrol+3conditionalregionalremovals');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 from jacclust.tt_model import apply_rotary_emb
 model=load_model_fast().cuda().eval();last=model.transformer.h[17];state=model.state_dict()
 producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()};consumer={k:v.cuda() for k,v in torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()};merge=torch.load(P/'STRUCTURED_PRODUCER_SHARED_OUTPUT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['projectors'].cuda()
 scales=[]
 for layer in (8,9):
  scale=1.
  for j in range(layer+1,18):scale*=float(state[f'transformer.h.{j}.lambdas'][0])
  scales.append(scale)
 captured={};head_replay=[];subtract_replay=[];component_replay=[];branch_replay=[];logit_replay=[];ce_replay=[];ce=torch.zeros(48,6,dtype=torch.float64);margins=torch.zeros_like(ce);body_count=0;calibration=[];mean_head=None
 def hook(index,head,module,args,output):
  x=args[0];n,length,_=x.shape
  q,k,q2,k2=[getattr(module,key)(x).reshape(n,length,9,128) for key in ('c_q','c_k','c_q2','c_k2')];cos,sin=module.rotary(q)
  q,k,q2,k2=[apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin) for v in (q,k,q2,k2)]
  gamma=(torch.einsum('nthd,nshd->nhts',q,k)/128)*(torch.einsum('nthd,nshd->nhts',q2,k2)/128);gamma=gamma.masked_fill(~torch.ones(length,length,dtype=torch.bool,device='cuda').tril()[None,None],0)
  values=(1-module.lamb)*module.c_v(x).reshape(n,length,9,128)+module.lamb*args[1].reshape(n,length,9,128)
  z=torch.einsum('nhts,nshd->nthd',gamma,values);rebuilt=module.c_proj(z.reshape_as(x));head_replay.append(float((rebuilt-output[0]).norm()/output[0].norm()))
  headwrite=F.linear(z[:,:,head],module.c_proj.weight[:,head*128:(head+1)*128])
  if captured['mode']=='calibration':
   if index==0:calibration.append(z[:,:,head].mean((0,1)).detach())
   return output
  if captured['mode']=='native':
   scalar=head_scalar(x,captured['tokens'],producer,index);contribution=scalar[...,None]*producer['output_coefficients'][index]*producer['shared_output']
   reference=(scales[index]*(headwrite.double()@consumer['current_readers'].T))@merge[index].T
   component_replay.append(float((contribution-reference).norm()/reference.norm()));captured['components'][index]=contribution
  if captured['mode'] in ('wholehead','meanhead') and index==0:
   removed=output[0]-headwrite;zeroed=z.clone();zeroed[:,:,head]=0
   if captured['mode']=='meanhead':
    removed=removed+F.linear(mean_head,module.c_proj.weight[:,head*128:(head+1)*128])[None,None,:];zeroed[:,:,head]=mean_head
   direct=module.c_proj(zeroed.reshape_as(x));subtract_replay.append(float((removed-direct).norm()/direct.norm()));return removed,output[1]
  return output
 def before_last(module,args):captured['residual17']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def after_last_attention(module,args,out):captured.update(current17=args[0],pre17=captured['residual17']+out[0])
 handles=[model.transformer.h[8].attn.register_forward_hook(lambda module,args,out:hook(0,2,module,args,out)),model.transformer.h[9].attn.register_forward_hook(lambda module,args,out:hook(1,8,module,args,out)),last.register_forward_pre_hook(before_last),last.attn.register_forward_hook(after_last_attention)]
 def logits_from_state(h):return 30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
 def tail(z):return logits_from_state(z+last.mlp(F.rms_norm(z,(1152,))))
 def measure(logits):return -logits.log_softmax(-1)[0,198],logits[0,198]-logits[0,11]
 def forward(tokens):
  nonlocal body_count
  x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  body_count+=1;return logits_from_state(x[:,-1])
 try:
  captured.update(mode='calibration')
  for batch in range(0,24,4):forward(torch.tensor(mean_rows[batch:batch+4],device='cuda'))
  assert len(calibration)==6
  mean_head=torch.stack(calibration).mean(0)
  for i,row in enumerate(rows):
   tokens=torch.tensor([row['ids']],device='cuda');captured.update(mode='native',tokens=tokens,components={});baseline=forward(tokens);ce[i,0],margins[i,0]=[v.cpu() for v in measure(baseline)]
   current=captured['current17'];pre=captured['pre17'][:,-1];rho=(captured['residual17'].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();length=tokens.shape[1];q=current[:,-1];qr=rotary(length-1,128).cuda()
   changes=torch.stack([captured['components'][0],captured['components'][1],captured['components'][0]+captured['components'][1]])
   writes=torch.zeros(4,1,1152,device='cuda')
   for pos in range(length):
    x=current[:,pos];f=x.double()@consumer['current_readers'].T+consumer['token_reads'][tokens[:,pos]];rotation=(qr.T@rotary(pos,128).cuda()).float()
    base=execute_reading_head(q,x,f,rotation,consumer);ref=execute_mixed_token_head(q,x,tokens[:,pos],rotation,consumer);branch_replay.append(float((base-ref).norm()/ref.norm().clamp_min(1e-30)));writes[0]+=base
    for arm in range(3):writes[arm+1]+=execute_reading_head(q,x,f-changes[arm,:,pos]/rho[:,pos],rotation,consumer)
   noedit=tail(pre);logit_replay.append(float((noedit-baseline).norm()/baseline.norm()));noedit_ce=measure(noedit)[0];ce_replay.append(float((noedit_ce-ce[i,0].cuda()).abs()/ce[i,0].abs().clamp_min(1e-30).cuda()))
   for arm in range(3):ce[i,arm+1],margins[i,arm+1]=[v.cpu() for v in measure(tail(pre+writes[arm+1]-writes[0]))]
   captured.update(mode='wholehead',components={});whole=forward(tokens);ce[i,4],margins[i,4]=[v.cpu() for v in measure(whole)]
   captured.update(mode='meanhead',components={});whole=forward(tokens);ce[i,5],margins[i,5]=[v.cpu() for v in measure(whole)]
 finally:
  for handle in handles:handle.remove()
 assert body_count==150
 old=torch.load(P/'SCALAR_PRODUCERS_NEWLINE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 authored_replay=float((ce[:16,:5]-old['ce']).norm()/old['ce'].norm())
 old_effect=old['ce']-old['ce'][:,:1];new_effect=ce[:16,:5]-ce[:16,:1]
 authored_effect_replay=float((new_effect-old_effect).norm()/old_effect.norm())
 cells=[]
 for pool in ('authored','fineweb'):
  for family in range(2):
   ix=[i for i,r in enumerate(rows) if r['pool']==pool and r['family']==family];delta=ce[ix]-ce[ix,:1];count=len(ix)
   cap=float(margins[ix,0].mean())>=.2 and int((margins[ix,0]>0).sum())>=3*count//4 and float(ce[ix,0].mean())<=5
   positive=float(delta[:,5].mean())>=.02 and int((delta[:,5]>0).sum())>=count//2
   preserve=all(float(delta[:,arm].abs().mean())<=.02 and float(delta[:,arm].abs().max())<=.1 for arm in (1,2,3))
   cells.append(dict(pool=pool,family=family,count=count,native_capability=cap,meanhead_positive_control=positive,candidate_preservation=preserve,baseline_mean_ce=float(ce[ix,0].mean()),baseline_mean_margin=float(margins[ix,0].mean()),baseline_positive_margin_count=int((margins[ix,0]>0).sum()),mean_ce_change=delta.mean(0).tolist(),meanabs_ce_change=delta.abs().mean(0).tolist(),maxabs_ce_change=delta.abs().amax(0).tolist(),positive_ce_change_count=(delta>0).sum(0).tolist(),mean_margin_change=(margins[ix]-margins[ix,:1]).mean(0).tolist()))
 A=max(head_replay+subtract_replay+component_replay+branch_replay)<=1e-5 and max(logit_replay+ce_replay+[authored_replay,authored_effect_replay])<=1e-4
 natural=[c for c in cells if c['pool']=='fineweb'];B=A and all(c['native_capability'] and c['meanhead_positive_control'] for c in natural);C=B and all(c['candidate_preservation'] for c in natural)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'checks':dict(head_replay_max=max(head_replay),subtraction_replay_max=max(subtract_replay),component_replay_max=max(component_replay),branch_replay_max=max(branch_replay),logit_replay_max=max(logit_replay),ce_replay_max=max(ce_replay),authored_v1_ce_replay=authored_replay,authored_v1_effect_replay=authored_effect_replay),'cells':cells,'arms':['native','remove_regional_edge8_2','remove_regional_edge9_8','remove_joint_regional_edges','remove_wholehead8_2','mean_wholehead8_2'],'body_forwards':body_count,'seconds':time.perf_counter()-tic,'source_shas':binding,'scope':'Frozen factor validation, authored/FineWeb context crossed with actual zero/mean wholehead interventions. Natural rows selected by true labels without model scores. B/C inference requires natural capability and mean-head control. No corpus OOD or global removal claim.'}
 torch.save(dict(ce=ce,margins=margins,mean_head=mean_head.cpu()),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
