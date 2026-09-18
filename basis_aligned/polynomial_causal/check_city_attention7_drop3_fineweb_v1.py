#!/usr/bin/env python3
# BQGATE:800 sequence-equivalent forwards;360 batched block calls;600seconds;CPU only;no fitting.
"""pred_a replay abs<=1e-4/rel<=1e-5 and instrument checks;
pred_b effecterror<=.05/livecap>=90; pred_c controls<=.5target;
pred_d positive>=.90/mean>=.02; pred_e target>=2nullmedian/16beaten.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_ATTENTION7_DROP3_FINEWEB_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 audit=json.loads((P/(STEM+'_ROW_AUDIT.json')).read_text());assert audit['passes'] and audit['row_sha256']==hashlib.sha256((P/(STEM+'_ROWS.json')).read_bytes()).hexdigest()
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('800 sequence-equivalents;360 block calls;CPU fresh one-input generator and16nulls');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 cert=json.loads((P/'CITY_ATTENTION7_DROP3_PACK_V1_RESULT.json').read_text())
 assert all(cert[k] for k in ['pred_a','pred_b','pred_c'])
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(600)
 torch.set_num_threads(2)
 from fastload import load_model_fast
 model=load_model_fast().eval();assert all(p.device.type=='cpu' for p in model.parameters())
 from city_attention7_drop3_v1 import execute
 from city_residual6_single_input_v1 import execute as full_execute
 from types import MethodType
 program={'attention7':torch.load(P/(STEM+'_PACKED_ATTENTION7.pt'),weights_only=True),
          'readers':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True),
          'head8':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)}
 full_program={**program,'attention7':torch.load(P/(STEM+'_ATTENTION7.pt'),weights_only=True)}
 ids=torch.zeros(40,max(len(r['ids']) for r in groups),dtype=torch.long)
 for i,row in enumerate(groups):ids[i,:len(row['ids'])]=torch.tensor(row['ids'])
 masks=torch.zeros_like(ids,dtype=torch.bool)
 for i,row in enumerate(groups):masks[i,row['destination_positions']]=True
 assert len({r['city_position'] for r in groups})==1
 city=groups[0]['city_position'];cache={};attn=model.transformer.h[8].attn
 original_attention=attn.squared_attention
 def capture_attention(module,q,k,v,q2,k2):
  result=original_attention(q,k,v,q2,k2)
  cache['queries']=torch.stack([q[:,:,2],q2[:,:,2]],2).clone()
  route=((q[:,:,2]*k[:,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[:,city,None,2]).sum(-1)/128)
  cache['native_delta']=-F.linear(route[...,None]*v[:,city,None,2],module.c_proj.weight[:,256:384])*masks[...,None]
  return result
 def prefix_capture(module,args):cache['residual6']=args[0].clone()
 attn.squared_attention=MethodType(capture_attention,attn)
 h=model.transformer.h[7].register_forward_pre_hook(prefix_capture)
 calls=0
 try:
  x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
  for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
 finally:
  h.remove();attn.squared_attention=original_attention
 last=x[torch.arange(40),torch.tensor([len(r['ids'])-1 for r in groups])]
 capture_scores=30*torch.tanh(model.lm_head(F.rms_norm(last,(1152,)))/30)
 capture_values=torch.zeros(40,10,dtype=torch.float64)
 for i,row in enumerate(groups):
  for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):capture_values[i,j]=capture_scores[i,left]-capture_scores[i,right]
 candidates=[];query_errors=[];candidate_query_errors=[];write_errors=[];fixtures=[]
 for i,row in enumerate(groups):
  t=len(row['ids']);inputs={'residual6':cache['residual6'][i:i+1,:t],'token_ids':ids[i:i+1,:t],'city':city,'destination':masks[i,:t]}
  delta,queries=execute(program,**inputs,return_queries=True);reference=cache['queries'][i:i+1,:t].double()
  candidate_query_errors.append(float((queries-reference).norm()/reference.norm()))
  _,native_queries=full_execute(full_program,**inputs,return_queries=True)
  query_errors.append(float((native_queries-reference).norm()/reference.norm()))
  native=cache['native_delta'][i:i+1,:t]
  write_errors.append(float((delta-native.double()).norm()/native.double().norm()))
  candidates.append(delta);fixtures.append({'inputs':inputs,'native_delta':native,'candidate_delta':delta})
 preflight={'query_gate':max(query_errors)<=1e-4,'finite':all(bool(torch.isfinite(w).all()) for w in candidates),'max_query_error':max(query_errors),'max_write_error':max(write_errors),'query_errors':query_errors,'write_errors':write_errors,'candidate_query_errors':candidate_query_errors}
 (P/(STEM+'_PREFLIGHT.json')).write_text(json.dumps(preflight,indent=2)+'\n')
 assert preflight['query_gate'] and preflight['finite'],preflight
 writes=[torch.stack([cache['native_delta'][i:i+1,:w.shape[1]].double(),w.double()]) for i,w in enumerate(candidates)]
 outside=max(float(w[:,:, [i for i in range(len(r['ids'])) if i not in r['destination_positions']]].abs().max()) for w,r in zip(writes,groups))
 print('Fresh capture and query preflight pass; starting19 installed arms',flush=True)
 edit=torch.zeros(40,ids.shape[1],1152);state={};norm_errors=[]
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 handle=model.transformer.h[8].attn.register_forward_hook(post8)
 values=torch.zeros(19,40,10,dtype=torch.float64);count=40;arm_seconds=[]
 try:
  for arm in range(19):
   arm_start=time.perf_counter();state['arm']=arm;edit.zero_()
   for i,row in enumerate(groups):
    if arm==0:continue
    if arm<=2:d=writes[i][arm-1].float()
    else:
     gen=torch.Generator().manual_seed(18099000+1000*(arm-3)+row['context_id'])
     random=torch.randn(candidates[i].shape,dtype=torch.float32,generator=gen).double()
     norm=candidates[i].double().norm(dim=-1,keepdim=True)
     d=(random*norm/random.norm(dim=-1,keepdim=True).clamp_min(1e-30)*(1 if row['cue']=='British' else -1)).float()
     norm_errors.append(float(((d.double().norm(dim=-1,keepdim=True)-norm).abs()/norm.clamp_min(1e-8)).max()))
    edit[i,:d.shape[1]]=d[0]
   x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   last=x[torch.arange(40),torch.tensor([len(r['ids'])-1 for r in groups])]
   scores=30*torch.tanh(model.lm_head(F.rms_norm(last,(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
   count+=40;arm_seconds.append(time.perf_counter()-arm_start)
   print(f'arm {arm+1}/19: {arm_seconds[-1]:.3f}s; total {time.perf_counter()-start:.1f}s',flush=True)
 finally:handle.remove()
 v=expand(values,mapping,6);e=v-v[:1]
 diff=values[0]-capture_values;absolute=float(diff.abs().max());relative=float(diff.norm()/capture_values.norm())
 rms=e.square().mean(1).sqrt();base=v[0,:,0][::2]-v[0,:,0][1::2];cap=base>=.1;records={}
 for arm,name in [(1,'full'),(2,'one_input')]:
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  records[name]={'target_rms':float(rms[arm,0]),'target_over_full':float(rms[arm,0]/rms[1,0]),'control_over_target':(rms[arm,1:]/rms[arm,0]).tolist(),'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean())}
 actual=e[1];approx=e[2];error=float((approx[:,0]-actual[:,0]).norm()/actual[:,0].norm())
 nr=rms[3:,0];ordered=nr.sort().values;median=float((ordered[7]+ordered[8])/2);g=records['one_input']
 strata={}
 for natural in [True,False]:
  idx=[i for i,row in enumerate(rows) if row['is_untouched_natural_arm']==natural]
  strata['untouched' if natural else 'substituted']=float((approx[idx,0]-actual[idx,0]).norm()/actual[idx,0].norm())
 row_audit=json.loads((P/(STEM+'_ROW_AUDIT.json')).read_text())
 baseline=json.loads((P/(STEM+'_BASELINE.json')).read_text())
 baseline_errors={}
 for name,idx in [('all',list(range(len(rows)))),('untouched',[i for i,r in enumerate(rows) if r['is_untouched_natural_arm']]),('substituted',[i for i,r in enumerate(rows) if not r['is_untouched_natural_arm']])]:
  truth=actual[idx,0];baseline_errors[name]={'opened_global_mean_error':float((truth-baseline['opened_mean_effect']).norm()/truth.norm()),'zero_effect_error':float(truth.norm()/truth.norm())}
 documents=[]
 for context in sorted({r['context_id'] for r in rows}):
  idx=[i for i,r in enumerate(rows) if r['context_id']==context];b=v[0,idx,0][::2]-v[0,idx,0][1::2];capable=b>=.1
  native_pair=v[1,idx,0][::2]-v[1,idx,0][1::2];candidate_pair=v[2,idx,0][::2]-v[2,idx,0][1::2]
  documents.append({'context_id':context,'capable_endpoints':int(capable.sum()),'native_attenuation':((b[capable]-native_pair[capable])/b[capable]).tolist(),'candidate_attenuation':((b[capable]-candidate_pair[capable])/b[capable]).tolist(),'effect_error':float((approx[idx,0]-actual[idx,0]).norm()/actual[idx,0].norm())})
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and row_audit['passes'] and len(rows)==240 and preflight['query_gate'] and outside==0 and max(norm_errors)<=1e-5 and bool(torch.isfinite(v).all()) and count==800 and calls==360 and not torch.cuda.is_initialized(),
    'pred_b':error<=.05 and g['target_rms']>=1e-5 and int(cap.sum())>=90,
    'pred_c':max(g['control_over_target'])<=.5,
    'pred_d':g['positive_fraction']>=.90 and g['mean_attenuation']>=.02,
    'pred_e':g['target_rms']>=2*median and bool((nr<g['target_rms']).all()),
    'baseline_errors':baseline_errors,'document_diagnostics':documents,'preflight':preflight,'arms':records,'target_error':error,'prediction_error_strata':strata,'capable_pairs':int(cap.sum()),
    'target_over_null_median':g['target_rms']/median,'nulls_beaten':int((nr<g['target_rms']).sum()),'null_readout_rms':rms[3:].tolist(),
    'replay_max_abs':absolute,'replay_relative':relative,'max_outside':outside,'max_null_norm_error':max(norm_errors),'body_forwards':count,'batched_block_calls':calls,'arm_seconds':arm_seconds,'floating_scalars':sum(t.numel() for pack in program.values() for t in pack.values() if t.is_floating_point()),'native_input_arrays':1,'native_input_scalars_T32':36864,'seconds':time.perf_counter()-start,
    'scope':'Fresh20document cross-corpus FineWeb CPU physically packed attention7 drop-head3 single-residual6 generator confirmation; approximate mixed8 RMS; full native MLP8/suffix external. Fixed spelling probes, not actual next-token labels. No independent composition or total compression claim.','source_shas':binding}
 torch.save({'values':v,'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
