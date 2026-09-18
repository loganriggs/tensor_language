#!/usr/bin/env python3
# BQGATE:760 sequence-equivalent forwards;342 batched block calls;600seconds;CPU only;no fitting.
"""pred_a replay abs<=1e-4/rel<=1e-5 and instrument checks;
pred_b effecterror<=.35/livecap>=90; pred_c controls<=.5target;
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
STEM='CITY_RESIDUAL6_SINGLE_INPUT_CPU_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('760 sequence-equivalents;342 block calls;CPU one-input generator and16nulls');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 cert=json.loads((P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_CPU_RESULT.json').read_text())
 assert all(cert[k] for k in ['query_gate','write_gate','support_finite_unknown_gate'])
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(600)
 torch.set_num_threads(2)
 from fastload import load_model_fast
 model=load_model_fast().eval();assert all(p.device.type=='cpu' for p in model.parameters())
 candidates=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_WRITES.pt',weights_only=True)['writes']
 refs=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 writes=[torch.stack([r['inputs']['delta'].double(),w.double()]) for w,r in zip(candidates,refs,strict=True)]
 outside=max(float(w[:,:, [i for i in range(len(r['ids'])) if i not in r['destination_positions']]].abs().max()) for w,r in zip(writes,groups))
 ids=torch.zeros(40,max(len(r['ids']) for r in groups),dtype=torch.long)
 for i,row in enumerate(groups):ids[i,:len(row['ids'])]=torch.tensor(row['ids'])
 edit=torch.zeros(40,ids.shape[1],1152);state={};norm_errors=[]
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 handle=model.transformer.h[8].attn.register_forward_hook(post8)
 values=torch.zeros(19,40,10,dtype=torch.float64);count=0;calls=0;arm_seconds=[]
 try:
  for arm in range(19):
   arm_start=time.perf_counter();state['arm']=arm;edit.zero_()
   for i,row in enumerate(groups):
    if arm==0:continue
    if arm<=2:d=writes[i][arm-1].float()
    else:
     gen=torch.Generator().manual_seed(18093000+1000*(arm-3)+row['context_id'])
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
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['values']
 diff=v[:2]-old[:2];absolute=float(diff.abs().max());relative=float(diff.norm()/old[:2].norm())
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
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and outside==0 and max(norm_errors)<=1e-5 and bool(torch.isfinite(v).all()) and count==760 and calls==342 and not torch.cuda.is_initialized(),
    'pred_b':error<=.35 and g['target_rms']>=1e-5 and int(cap.sum())>=90,
    'pred_c':max(g['control_over_target'])<=.5,
    'pred_d':g['positive_fraction']>=.90 and g['mean_attenuation']>=.02,
    'pred_e':g['target_rms']>=2*median and bool((nr<g['target_rms']).all()),
    'arms':records,'target_error':error,'prediction_error_strata':strata,'capable_pairs':int(cap.sum()),
    'target_over_null_median':g['target_rms']/median,'nulls_beaten':int((nr<g['target_rms']).sum()),'null_readout_rms':rms[3:].tolist(),
    'replay_max_abs':absolute,'replay_relative':relative,'max_outside':outside,'max_null_norm_error':max(norm_errors),'body_forwards':count,'batched_block_calls':calls,'arm_seconds':arm_seconds,'floating_scalars':cert['floating_scalars'],'native_input_arrays':1,'native_input_scalars_T32':36864,'seconds':time.perf_counter()-start,
    'scope':'Opened CPU single-residual6 generator screen; approximate mixed8 RMS. Full native MLP8/suffix external. No supplied query or RMS. No fresh transfer, standalone certification, independent composition or total compression claim.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
