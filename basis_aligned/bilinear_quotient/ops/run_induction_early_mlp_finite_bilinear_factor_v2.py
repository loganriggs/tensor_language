#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_finite_bilinear_instrument pred_b_discovery_compact_factor pred_c_confirm_compact_factor pred_d_interaction_compression
"""Remove exact finite bilinear response factors from induction MLP8--12."""
from __future__ import annotations
import hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np

RUNNER=Path(__file__).resolve();OPS=RUNNER.parent;ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(OPS),str(ROOT)]
import run_induction_typed_consumer_group_response_v4 as prior
from circuit_fast_screen_managed_runner import atomic_create_json
from fp32_add_observation_v1 import observe
p=prior.p
ROWS=prior.ROWS;PREREG=POLY/'INDUCTION_EARLY_MLP_FINITE_BILINEAR_FACTOR_V2_PREREGISTRATION.md';BINDING=POLY/'INDUCTION_EARLY_MLP_FINITE_BILINEAR_FACTOR_V2_BINDING.json';OUT=POLY/'INDUCTION_EARLY_MLP_FINITE_BILINEAR_FACTOR_V2_RESULT.json'
SITES=(8,9,10,11,12);ARMS=('self','joint','quadratic','left','right','cross','full');CANDIDATES=('quadratic','left','right','cross');BATCH=32;PRICE={'forwards':24,'sequences':768,'backwards':0,'fits':0}
PREDICTION_REGISTRY={'pred_a_exact_finite_bilinear_instrument':None,'pred_b_discovery_compact_factor':None,'pred_c_confirm_compact_factor':None,'pred_d_interaction_compression':None}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_bound():
 b=json.loads(BINDING.read_text());assert all(digest(path)==expected for path,expected in b['files'].items());assert b['sites']==list(SITES) and b['arms']==list(ARMS) and b['price']==PRICE
 return prior.load_rows()
def plan():
 _r,f,rows,endpoints=load_bound();return {'schema':'induction_early_mlp_finite_bilinear_factor_v2_plan','model_loaded':False,'gpu_accessed':False,'queue_touched':False,'rows':len(rows),'endpoints':len(endpoints),'sites':list(SITES),'arms':list(ARMS),'candidate_order':list(CANDIDATES),'price':PRICE,'predicates':list(PREDICTION_REGISTRY)}
def capture_native(runtime,endpoints):
 torch=runtime.torch;device=next(runtime.model.parameters()).device;all_logits=[];all_terms=[];all_writes=[];all_states=[]
 for start in range(0,len(endpoints),BATCH):
  specs=endpoints[start:start+BATCH];tokens=torch.as_tensor(p.fixed_tokens(specs,'token_ids'),dtype=torch.long,device=device);terms=np.zeros((len(specs),4,p.RESIDUAL),np.float32);writes=np.zeros((len(specs),36,p.RESIDUAL),np.float32);states=np.zeros((len(specs),len(SITES),p.RESIDUAL),np.float32)
  def attention(event):
   if event.site in (5,7,8):
    write,first,captured,_=runtime.r585.factorize_attention_event(event,specs,torch=torch,functional=runtime.functional,induction=runtime.induction)
    for local,row in enumerate(captured):
     for index,site in enumerate(p.SITES):
      if int(site[1])==event.site:terms[local,index]=row[site]['term'].numpy()
   else:write,first=event.block.attn(event.state,event.first_value)
   for local,spec in enumerate(specs):writes[local,2*event.site]=write[local,int(spec['final_position'])].float().detach().cpu().numpy()
   return write,first
  def mlp(event):
   write=event.block.mlp(event.state)
   for local,spec in enumerate(specs):
    q=int(spec['final_position']);writes[local,2*event.site+1]=write[local,q].float().detach().cpu().numpy()
    if event.site in SITES:states[local,SITES.index(event.site)]=event.state[local,q].float().detach().cpu().numpy()
   return write
  with torch.inference_mode():logits=runtime.facade.forward_with_dispatch(runtime.model,tokens,attention,mlp,require_production=False)
  all_logits.append(np.stack([logits[i,int(s['final_position'])].float().cpu().numpy() for i,s in enumerate(specs)]));all_terms.append(terms);all_writes.append(writes);all_states.append(states)
 return tuple(np.concatenate(x) for x in (all_logits,all_terms,all_writes,all_states))
def execute(runtime,tokens,specs,target,native_states,arm):
 torch=runtime.torch;F=runtime.functional;device=next(runtime.model.parameters()).device;token_tensor=torch.as_tensor(tokens,dtype=torch.long,device=device);before=np.zeros((len(specs),3,p.RESIDUAL),np.float32);total=np.zeros_like(before);after=np.zeros_like(before);active=np.zeros(len(specs),bool);closures=[];norms=np.zeros(len(specs),np.float64)
 def attention(event):
  if event.site in (5,7,8):
   write,first,captured,_=runtime.r585.factorize_attention_event(event,specs,torch=torch,functional=F,induction=runtime.induction);changed=write.clone();indices=[i for i,s in enumerate(p.SITES) if int(s[1])==event.site];li=(5,7,8).index(event.site)
   for local,spec in enumerate(specs):
    q=int(spec['final_position']);pieces=[np.subtract(target[local,i],captured[local][p.SITES[i]]['term'].numpy(),dtype=np.float32) for i in indices];delta=torch.as_tensor(np.stack(pieces),device=device).sum(0);before[local,li]=changed[local,q].float().cpu().numpy();total[local,li]=delta.float().cpu().numpy();changed[local,q]+=delta.to(changed.dtype);after[local,li]=changed[local,q].float().cpu().numpy();active[local]|=bool(delta.float().norm().cpu()>1e-6)
   write=changed
  else:write,first=event.block.attn(event.state,event.first_value)
  return write,first
 def mlp(event):
  write=event.block.mlp(event.state)
  if event.site in SITES and arm not in ('self','joint'):
   changed=write.clone();mlp=event.block.mlp;si=SITES.index(event.site)
   for local,spec in enumerate(specs):
    q=int(spec['final_position']);x1=event.state[local,q].float();x0=torch.as_tensor(native_states[local,si],device=device);d=x1-x0;lw=mlp.Left.weight.float();rw=mlp.Right.weight.float();dw=mlp.Down.weight.float();l0=F.linear(x0,lw);r0=F.linear(x0,rw);ld=F.linear(d,lw);rd=F.linear(d,rw);left=F.linear(ld*r0,dw);right=F.linear(l0*rd,dw);quad=F.linear(ld*rd,dw);joint=left+right+quad;direct=F.linear(F.linear(x1,lw)*F.linear(x1,rw)-l0*r0,dw);x1d=x1.double();x0d=x0.double();dd=x1d-x0d;lwd=mlp.Left.weight.double();rwd=mlp.Right.weight.double();dwd=mlp.Down.weight.double();l0d=F.linear(x0d,lwd);r0d=F.linear(x0d,rwd);ldd=F.linear(dd,lwd);rdd=F.linear(dd,rwd);jointd=F.linear(ldd*r0d+l0d*rdd+ldd*rdd,dwd);directd=F.linear(F.linear(x1d,lwd)*F.linear(x1d,rwd)-l0d*r0d,dwd);closures.append(float((jointd-directd).square().sum()/directd.square().sum().clamp_min(1e-30)));component={'quadratic':quad,'left':left,'right':right,'cross':left+right,'full':joint}[arm];changed[local,q]=(changed[local,q].float()-component).to(changed.dtype);norms[local]+=float(component.double().norm().cpu())**2
   write=changed
  return write
 with torch.inference_mode():logits=runtime.facade.forward_with_dispatch(runtime.model,token_tensor,attention,mlp,require_production=False)
 selected=np.stack([logits[i,int(s['final_position'])].float().cpu().numpy() for i,s in enumerate(specs)]);obs=observe(before,total,after);assert obs['passed'];return selected,active,np.sqrt(norms),max(closures,default=0.0),obs
def main():
 _r,frozen,rows,endpoints=load_bound()
 if os.environ.get('BQLIB_DRYRUN')=='1' and os.environ.get('BQLIB_NO_MODEL')=='1' and os.environ.get('EXPECTED_SHA256'):print(json.dumps(plan(),sort_keys=True));return
 assert not OUT.exists();signal.alarm(600);runtime=prior.R594ModelExecutor(p,_r);runtime.torch.set_num_threads(2);counts=[0,0]
 def count(_m,args,_o):counts[0]+=1;counts[1]+=len(args[0])
 handle=runtime.model.transformer.h[0].attn.register_forward_hook(count);tic=time.perf_counter()
 try:
  endpoint_logits,endpoint_terms,_writes,endpoint_states=capture_native(runtime,endpoints);index={x['endpoint_id']:i for i,x in enumerate(endpoints)};ri=np.array([index[x['recipient_endpoint_id']] for x in rows]);di=np.array([index[x['donor_endpoint_id']] for x in rows]);specs=[endpoints[i] for i in ri];tokens=p.fixed_tokens(specs,'token_ids');native=endpoint_logits[ri];recipient=endpoint_terms[ri];donor=endpoint_terms[di];states=endpoint_states[ri];arms={};active={};norms={};closures=[];observations=[]
  for arm in ARMS:
   target=recipient if arm=='self' else donor;bb=[];aa=[];nn=[]
   for start in range(0,len(rows),BATCH):
    stop=start+BATCH;values=execute(runtime,tokens[start:stop],specs[start:stop],target[start:stop],states[start:stop],arm);logits,act,norm,closure,obs=values;bb.append(logits);aa.append(act);nn.append(norm);closures.append(closure);observations.append(obs)
   arms[arm]=np.concatenate(bb);active[arm]=np.concatenate(aa);norms[arm]=np.concatenate(nn)
 finally:handle.remove()
 replay_abs=float(np.max(np.abs(arms['self'].astype(np.float64)-native.astype(np.float64))));replay_rel=float(np.linalg.norm(arms['self'].astype(np.float64)-native.astype(np.float64))/max(np.linalg.norm(native.astype(np.float64)),1e-30));metrics={a:prior.metrics(native,v,rows) for a,v in arms.items() if a!='self'};disc=set(frozen['discovery_group_ids']);confirm=set(frozen['confirm_group_ids']);reports={s:{a:prior.cell_report(v,g) for a,v in metrics.items()} for s,g in [('discovery',disc),('confirm',confirm)]};decisions={}
 for split in reports:
  decisions[split]={}
  for arm in ARMS[2:]:
   passed,reductions=prior.passes(reports[split][arm],reports[split]['joint']);decisions[split][arm]={'passed':bool(passed),'reductions':reductions}
 selected=next((a for a in CANDIDATES if decisions['discovery'][a]['passed']),None);instrument=counts==[PRICE['forwards'],PRICE['sequences']] and replay_abs<=1e-3 and replay_rel<=1e-5 and max(closures)<=1e-10 and all(o['passed'] for o in observations) and all(float(np.mean(active[a]))>=.75 for a in ARMS[1:]);confirm_pass=bool(selected and decisions['confirm'][selected]['passed']);pred={'pred_a_exact_finite_bilinear_instrument':bool(instrument),'pred_b_discovery_compact_factor':bool(instrument and selected),'pred_c_confirm_compact_factor':bool(instrument and confirm_pass),'pred_d_interaction_compression':bool(instrument and confirm_pass and selected=='quadratic')};terminal='invalid' if not instrument else 'finite_bilinear_factor' if confirm_pass else 'finite_bilinear_factor_null'
 result={'schema':'induction_early_mlp_finite_bilinear_factor_v2_result','terminal':terminal,'predictions':pred,'selected':selected,'instrument':{'native_replay_max_abs':replay_abs,'native_replay_relative':replay_rel,'maximum_bilinear_closure_relative_squared_error':max(closures),'active_fraction':{a:float(np.mean(active[a])) for a in ARMS[1:]},'mean_removed_norm':{a:float(np.mean(norms[a])) for a in ARMS[2:]},'maximum_rounding_residual':max(o['maximum_rounding_residual'] for o in observations)},'reports':reports,'decisions':decisions,'price':{'forwards':counts[0],'sequences':counts[1],'backwards':0,'fits':0,'weight_updates':0},'rows_sha256':digest(ROWS),'binding_sha256':digest(BINDING),'runner_sha256':digest(RUNNER),'checkpoint_sha256':runtime.checkpoint_sha256,'wall_seconds':time.perf_counter()-tic};atomic_create_json(OUT,result);print(json.dumps({'terminal':terminal,'predictions':pred,'selected':selected,'decisions':decisions,'price':result['price']},indent=2));assert instrument
if __name__=='__main__':main()
