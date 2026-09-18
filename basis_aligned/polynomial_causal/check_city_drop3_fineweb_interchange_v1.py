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
STEM='CITY_DROP3_FINEWEB_INTERCHANGE_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/('CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json')).read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 audit=json.loads((P/('CITY_ATTENTION7_DROP3_FINEWEB_V1_ROW_AUDIT.json')).read_text());assert audit['passes'] and audit['row_sha256']==hashlib.sha256((P/('CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json')).read_bytes()).hexdigest()
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('800 sequence-equivalents;360 block calls;CPU opened packed coupled interchange and16nulls');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 cert=json.loads((P/'CITY_ATTENTION7_DROP3_PACK_V1_RESULT.json').read_text())
 assert all(cert[k] for k in ['pred_a','pred_b','pred_c'])
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(600)
 torch.set_num_threads(2)
 from fastload import load_model_fast
 model=load_model_fast().eval();assert all(p.device.type=='cpu' for p in model.parameters())
 from city_drop3_fields_v1 import execute,city_write
 from city_residual6_fields_v1 import execute as native_execute
 from city_attention7_drop3_v1 import execute as removal
 from types import MethodType
 program={'attention7':torch.load(P/('CITY_ATTENTION7_DROP3_FINEWEB_V1_PACKED_ATTENTION7.pt'),weights_only=True),
          'readers':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True),
          'head8':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)}
 full_program={**program,'attention7':torch.load(P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ATTENTION7.pt',weights_only=True)}
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
  cache['fields']={name:x[:,:,2].clone() for name,x in zip(['q1','k1','q2','k2','value'],[q,k,q2,k2,v])}
  route=((q[:,:,2]*k[:,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[:,city,None,2]).sum(-1)/128)
  cache['native_removal']=-F.linear(route[...,None]*v[:,city,None,2],module.c_proj.weight[:,256:384])*masks[...,None]
  donor=torch.arange(q.shape[0])^1
  donor_route=((q[:,:,2]*k[donor,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[donor,city,None,2]).sum(-1)/128)
  cache['native_delta']=F.linear(donor_route[...,None]*v[donor,city,None,2]-route[...,None]*v[:,city,None,2],module.c_proj.weight[:,256:384])*masks[...,None]
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
 generated=[execute(program,cache['residual6'][i:i+1,:len(row['ids'])],ids[i:i+1,:len(row['ids'])]) for i,row in enumerate(groups)]
 native_fields=[{name:x[i:i+1,:len(row['ids'])].double() for name,x in cache['fields'].items()} for i,row in enumerate(groups)]
 candidates=[];query_errors=[];write_errors=[];fixtures=[];self_errors=[];generator_errors=[];native_swap_errors=[]
 for i,row in enumerate(groups):
  t=len(row['ids']);donor=i^1;mask=masks[i,:t];head=program['head8']
  inputs={'recipient_residual6':cache['residual6'][i:i+1,:t],'recipient_tokens':ids[i:i+1,:t],
          'donor_residual6':cache['residual6'][donor:donor+1,:t],'donor_tokens':ids[donor:donor+1,:t],'city':city,'destination':mask}
  own=city_write(generated[i],generated[i],head,city,mask)
  delta=city_write(generated[i],generated[donor],head,city,mask)-own
  queries=torch.stack([generated[i]['q1'],generated[i]['q2']],2);reference=cache['queries'][i:i+1,:t].double()
  fields=native_execute(full_program,inputs['recipient_residual6'],inputs['recipient_tokens'])
  native_queries=torch.stack([fields['q1'],fields['q2']],2)
  query_errors.append(float((native_queries-reference).norm()/reference.norm()))
  native=cache['native_delta'][i:i+1,:t]
  own_native=city_write(native_fields[i],native_fields[i],head,city,mask)
  reference_removal=cache['native_removal'][i:i+1,:t].double()
  self_errors.append(float((-own_native-reference_removal).norm()/reference_removal.norm()))
  old_generated=removal(program,inputs['recipient_residual6'],inputs['recipient_tokens'],city,mask)
  generator_errors.append(float((-own-old_generated).norm()/old_generated.norm()))
  native_factored=city_write(native_fields[i],native_fields[donor],head,city,mask)-own_native
  native_swap_errors.append(float((native_factored-native.double()).norm()/native.double().norm()))
  write_errors.append(float((delta-native.double()).norm()/native.double().norm()))
  candidates.append(delta);fixtures.append({'inputs':inputs,'native_delta':native,'candidate_delta':delta})
 preflight={'query_gate':max(query_errors)<=1e-4,'self_removal_gate':max(self_errors)<=1e-4 and max(generator_errors)<=1e-10,
            'native_swap_gate':max(native_swap_errors)<=1e-4,'finite':all(bool(torch.isfinite(w).all()) for w in candidates),
            'max_query_error':max(query_errors),'max_write_error':max(write_errors),'max_native_self_removal_error':max(self_errors),
            'max_generator_self_removal_error':max(generator_errors),'max_native_swap_error':max(native_swap_errors),
            'query_errors':query_errors,'write_errors':write_errors}
 (P/(STEM+'_PREFLIGHT.json')).write_text(json.dumps(preflight,indent=2)+'\n')
 assert all(preflight[k] for k in ['query_gate','self_removal_gate','native_swap_gate','finite']),preflight
 writes=[torch.stack([cache['native_delta'][i:i+1,:w.shape[1]].double(),w.double()]) for i,w in enumerate(candidates)]
 outside=max(float(w[:,:, [i for i in range(len(r['ids'])) if i not in r['destination_positions']]].abs().max()) for w,r in zip(writes,groups))
 print('Opened capture and query preflight pass; starting19 installed arms',flush=True)
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
     gen=torch.Generator().manual_seed(18100000+1000*(arm-3)+row['context_id'])
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
 for arm,name in [(1,'native_swap'),(2,'generated_swap')]:
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  records[name]={'target_rms':float(rms[arm,0]),'target_over_full':float(rms[arm,0]/rms[1,0]),'control_over_target':(rms[arm,1:]/rms[arm,0]).tolist(),'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean())}
 actual=e[1];approx=e[2];error=float((approx[:,0]-actual[:,0]).norm()/actual[:,0].norm())
 nr=rms[3:,0];ordered=nr.sort().values;median=float((ordered[7]+ordered[8])/2);g=records['generated_swap']
 strata={}
 for natural in [True,False]:
  idx=[i for i,row in enumerate(rows) if row['is_untouched_natural_arm']==natural]
  strata['untouched' if natural else 'substituted']=float((approx[idx,0]-actual[idx,0]).norm()/actual[idx,0].norm())
 row_audit=json.loads((P/('CITY_ATTENTION7_DROP3_FINEWEB_V1_ROW_AUDIT.json')).read_text())
 directions={}
 for cue in ['British','American']:
  idx=[i for i,r in enumerate(rows) if r['cue']==cue];sign=-1 if cue=='British' else 1
  directions[cue]={'native_mean_margin_change':float(actual[idx,0].mean()),'generated_mean_margin_change':float(approx[idx,0].mean()),'fraction_toward_donor':float((sign*approx[idx,0]>0).double().mean())}
 documents=[]
 for context in sorted({r['context_id'] for r in rows}):
  idx=[i for i,r in enumerate(rows) if r['context_id']==context];b=v[0,idx,0][::2]-v[0,idx,0][1::2];capable=b>=.1
  native_pair=v[1,idx,0][::2]-v[1,idx,0][1::2];candidate_pair=v[2,idx,0][::2]-v[2,idx,0][1::2]
  documents.append({'context_id':context,'capable_endpoints':int(capable.sum()),'native_attenuation':((b[capable]-native_pair[capable])/b[capable]).tolist(),'candidate_attenuation':((b[capable]-candidate_pair[capable])/b[capable]).tolist(),'effect_error':float((approx[idx,0]-actual[idx,0]).norm()/actual[idx,0].norm())})
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and row_audit['passes'] and len(rows)==240 and all(preflight[k] for k in ['query_gate','self_removal_gate','native_swap_gate']) and outside==0 and max(norm_errors)<=1e-5 and bool(torch.isfinite(v).all()) and count==800 and calls==360 and not torch.cuda.is_initialized(),
    'pred_b':error<=.05 and g['target_rms']>=1e-5 and int(cap.sum())>=90,
    'pred_c':max(g['control_over_target'])<=.5,
    'pred_d':g['positive_fraction']>=.90 and g['mean_attenuation']>=.02,
    'pred_e':g['target_rms']>=2*median and bool((nr<g['target_rms']).all()),
    'recipient_directions':directions,'document_diagnostics':documents,'preflight':preflight,'arms':records,'target_error':error,'prediction_error_strata':strata,'capable_pairs':int(cap.sum()),
    'target_over_null_median':g['target_rms']/median,'nulls_beaten':int((nr<g['target_rms']).sum()),'null_readout_rms':rms[3:].tolist(),
    'replay_max_abs':absolute,'replay_relative':relative,'max_outside':outside,'max_null_norm_error':max(norm_errors),'body_forwards':count,'batched_block_calls':calls,'arm_seconds':arm_seconds,'floating_scalars':sum(t.numel() for pack in program.values() for t in pack.values() if t.is_floating_point()),'logical_native_contexts':2,'native_input_scalars_T32_pair':73728,'seconds':time.perf_counter()-start,
    'scope':'Opened20document FineWeb packed coupled city K1/K2/fullV interchange with recipient queries. Two native residual6 contexts; approximate mixed8 RMS; native MLP8/suffix external. Fixed spelling probes. No independent composition or simplicity claim.','source_shas':binding}
 torch.save({'values':v,'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
