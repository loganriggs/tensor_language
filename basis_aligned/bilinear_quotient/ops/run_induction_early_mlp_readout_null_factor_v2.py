#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 18forwards576seq; induction early-MLP readout-null response factor;0updates.
"""Split the frozen early-MLP response into answer-parallel and readout-null factors."""
from __future__ import annotations
import hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).resolve(); OPS=RUNNER.parent; ROOT=RUNNER.parents[3]; POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(OPS),str(ROOT)]
import run_induction_typed_consumer_group_response_v4 as prior
from circuit_fast_screen_managed_runner import atomic_create_json
from fp32_add_observation_v1 import observe
p=prior.p
ROWS=prior.ROWS; PREREG=POLY/'INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_PREREGISTRATION.md'; BINDING=POLY/'INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_BINDING.json'; OUT=POLY/'INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_RESULT.json'
MODULES=prior.CANDIDATE_GROUPS['early_mlp']; MODES=('self','joint','full','parallel','null'); BATCH=32
PREDICTION_REGISTRY={'pred_a_exact_instrument':None,'pred_b_readout_null_answer_preserving':None,'pred_c_readout_null_removes_collateral':None,'pred_d_parallel_null_partition':None}
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def axes(runtime,specs):
 u=runtime.model.lm_head.weight.detach().float(); rows=[]
 for spec in specs:
  v=u[int(spec['recipient_answer_id'])]-u[int(spec['recipient_other_answer_id'])]; rows.append(v/v.norm().clamp_min(1e-30))
 return runtime.torch.stack(rows)
def execute(runtime,tokens,specs,target,native_writes,mode,readout_axes):
 torch=runtime.torch; device=next(runtime.model.parameters()).device; token_tensor=torch.as_tensor(tokens,dtype=torch.long,device=device); before=np.zeros((len(specs),3,p.RESIDUAL),np.float32); total=np.zeros_like(before); after=np.zeros_like(before); active=np.zeros(len(specs),bool); response=np.zeros(len(specs),np.float64); parallel_norm=np.zeros(len(specs),np.float64); null_norm=np.zeros(len(specs),np.float64); partition=[]
 restore={prior.module_index(x) for x in MODULES} if mode in {'full','parallel','null'} else set()
 def attention(event):
  if event.site in (5,7,8):
   write,first,captured,_=runtime.r585.factorize_attention_event(event,specs,torch=torch,functional=runtime.functional,induction=runtime.induction); changed=write.clone(); indices=[i for i,s in enumerate(p.SITES) if int(s[1])==event.site]; li=(5,7,8).index(event.site)
   for local,spec in enumerate(specs):
    query=int(spec['final_position']); pieces=[np.subtract(target[local,i],captured[local][p.SITES[i]]['term'].numpy(),dtype=np.float32) for i in indices]; delta=torch.as_tensor(np.stack(pieces),device=device).sum(0); before[local,li]=changed[local,query].float().detach().cpu().numpy(); total[local,li]=delta.float().detach().cpu().numpy(); changed[local,query]+=delta.to(changed.dtype); after[local,li]=changed[local,query].float().detach().cpu().numpy(); active[local]|=bool(delta.float().norm().detach().cpu()>1e-6)
   write=changed
  else: write,first=event.block.attn(event.state,event.first_value)
  return write,first
 def mlp(event):
  write=event.block.mlp(event.state); index=2*event.site+1
  if index in restore:
   write=write.clone()
   for local,spec in enumerate(specs):
    query=int(spec['final_position']); old=write[local,query].float(); delta=torch.as_tensor(native_writes[local,index],device=device)-old; axis=readout_axes[local]; par=torch.dot(delta,axis)*axis; nul=delta-par; component={'full':delta,'parallel':par,'null':nul}[mode]; write[local,query]=(old+component).to(write.dtype); response[local]+=float(component.double().norm().detach().cpu())**2; parallel_norm[local]+=float(par.double().norm().detach().cpu())**2; null_norm[local]+=float(nul.double().norm().detach().cpu())**2; partition.append(float((delta-par-nul).abs().max().detach().cpu()))
  return write
 with torch.inference_mode(): logits=runtime.facade.forward_with_dispatch(runtime.model,token_tensor,attention,mlp,require_production=False)
 selected=np.stack([logits[i,int(spec['final_position'])].float().cpu().numpy() for i,spec in enumerate(specs)]); observation=observe(before,total,after); assert observation['passed']
 return selected,active,np.sqrt(response),np.sqrt(parallel_norm),np.sqrt(null_norm),max(partition,default=0.0),observation
def report_pass(report,joint):
 reductions={cell:1-value['median_vocab_rms']/joint[cell]['median_vocab_rms'] for cell,value in report.items()}; answer=all(v['mean_ce_damage']<=.10 and v['correct_fraction']>=.75 for v in report.values()); collateral=all(v>=.25 for v in reductions.values()); return answer,collateral,reductions
def main():
 binding=json.loads(BINDING.read_text()); assert all(digest(path)==expected for path,expected in binding['files'].items()); r585,frozen,rows,endpoints=prior.load_rows(); assert binding['price']=={'forwards':18,'sequences':576}
 if os.environ.get('BQLIB_DRYRUN') == '1' and os.environ.get('BQLIB_NO_MODEL') == '1' and os.environ.get('EXPECTED_SHA256'):
  print(json.dumps({'dryrun':True,'model_loaded':False,'gpu_accessed':False,'rows':len(rows),'endpoints':len(endpoints),'modules':MODULES,'modes':MODES,'price':binding['price'],'predicates':list(PREDICTION_REGISTRY)})); return
 assert not OUT.exists(); signal.alarm(600); runtime=prior.R594ModelExecutor(p,r585); runtime.torch.set_num_threads(2); counts=[0,0]
 def count(_m,args,_o): counts[0]+=1; counts[1]+=len(args[0])
 handle=runtime.model.transformer.h[0].attn.register_forward_hook(count); tic=time.perf_counter()
 try:
  endpoint_logits,endpoint_terms,endpoint_writes=prior.capture_native(runtime,endpoints); endpoint_index={row['endpoint_id']:i for i,row in enumerate(endpoints)}; recipient=np.array([endpoint_index[row['recipient_endpoint_id']] for row in rows]); donor=np.array([endpoint_index[row['donor_endpoint_id']] for row in rows]); specs=[endpoints[i] for i in recipient]; tokens=p.fixed_tokens(specs,'token_ids'); native=endpoint_logits[recipient]; recipient_terms=endpoint_terms[recipient]; donor_terms=endpoint_terms[donor]; native_writes=endpoint_writes[recipient]; readout_axes=axes(runtime,rows); arms={}; active={}; restored={}; parnorm={}; nullnorm={}; partitions=[]; observations=[]
  for mode in MODES:
   target=recipient_terms if mode=='self' else donor_terms; batches=[]; aa=[]; rr=[]; pp=[]; nn=[]
   for start in range(0,len(rows),BATCH):
    stop=start+BATCH; values=execute(runtime,tokens[start:stop],specs[start:stop],target[start:stop],native_writes[start:stop],mode,readout_axes[start:stop]); logits,act,response,pn,nnorm,part,obs=values; batches.append(logits); aa.append(act); rr.append(response); pp.append(pn); nn.append(nnorm); partitions.append(part); observations.append(obs)
   arms[mode]=np.concatenate(batches); active[mode]=np.concatenate(aa); restored[mode]=np.concatenate(rr); parnorm[mode]=np.concatenate(pp); nullnorm[mode]=np.concatenate(nn)
 finally: handle.remove()
 max_abs=float(np.max(np.abs(arms['self'].astype(np.float64)-native.astype(np.float64)))); relative=float(np.linalg.norm(arms['self'].astype(np.float64)-native.astype(np.float64))/max(np.linalg.norm(native.astype(np.float64)),1e-30)); metrics={name:prior.metrics(native,arm,rows) for name,arm in arms.items() if name!='self'}; discovery=set(frozen['discovery_group_ids']); confirm=set(frozen['confirm_group_ids']); reports={split:{name:prior.cell_report(items,groups) for name,items in metrics.items()} for split,groups in [('discovery',discovery),('confirm',confirm)]}; decisions={}
 for split in reports:
  answer,collateral,reductions=report_pass(reports[split]['null'],reports[split]['joint']); decisions[split]={'answer_pass':answer,'collateral_pass':collateral,'reductions':reductions}
 instrument=counts==[18,576] and max_abs<=1e-3 and relative<=1e-5 and max(partitions)<=1e-5 and all(o['passed'] for o in observations) and all(float(np.mean(active[x]))>=.75 for x in ('joint','full','parallel','null')) and all(float(np.mean(restored[x]>1e-6))>=.75 for x in ('full','parallel','null'))
 pred_b=all(x['answer_pass'] for x in decisions.values()); pred_c=all(x['collateral_pass'] for x in decisions.values()); pred_d=max(partitions)<=1e-5
 predictions={'pred_a_exact_instrument':bool(instrument),'pred_b_readout_null_answer_preserving':bool(instrument and pred_b),'pred_c_readout_null_removes_collateral':bool(instrument and pred_c),'pred_d_parallel_null_partition':bool(instrument and pred_d)}; terminal='readout_null_screen' if all(predictions.values()) else ('invalid' if not instrument else 'readout_null_factor_null')
 result={'schema':'induction_early_mlp_readout_null_factor_v2_result','terminal':terminal,'predictions':predictions,'instrument':{'self_native_max_abs':max_abs,'self_native_relative':relative,'maximum_partition_error':max(partitions),'active_fraction':{x:float(np.mean(active[x])) for x in ('joint','full','parallel','null')},'response_active_fraction':{x:float(np.mean(restored[x]>1e-6)) for x in ('full','parallel','null')},'maximum_rounding_residual':float(max(o['maximum_rounding_residual'] for o in observations))},'reports':reports,'decisions':decisions,'component_norms':{x:{'mean_parallel_norm':float(np.mean(parnorm[x])),'mean_null_norm':float(np.mean(nullnorm[x])),'mean_installed_norm':float(np.mean(restored[x]))} for x in ('full','parallel','null')},'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'backwards':0,'weight_updates':0},'claim_boundary':'Opened-row explanatory factorization of fixed early MLP8-12 native response using one answer-pair unembedding axis per row; no extracted autonomous consumer or fresh confirmation.','rows_sha256':digest(ROWS),'binding_sha256':digest(BINDING),'runner_sha256':digest(RUNNER),'checkpoint_sha256':runtime.checkpoint_sha256,'wall_seconds':time.perf_counter()-tic}
 atomic_create_json(OUT,result); print(json.dumps({'terminal':terminal,'predictions':predictions,'decisions':decisions,'component_norms':result['component_norms'],'price':result['price']},indent=2)); assert instrument
if __name__=='__main__': main()
