#!/usr/bin/env python3
# BQGATE:520 sequence-equivalent forwards;234 batched block calls;300seconds;CPU only;no fitting.
"""pred_a native/full/generated replay abs<=1e-4,rel<=1e-5 and instruments;
pred_b cheaper subset effecterror<=.05,positive>=.90,mean>=.02,controls<=.5;
pred_c registered minimum-price selection with positive saving.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_ATTENTION7_OMISSION_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('520 sequence-equivalents;234 block calls;CPU13 omission arms');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 cert=json.loads((P/'CITY_ATTENTION7_OMISSION_V1_CPU_RESULT.json').read_text())
 assert all(cert[k] for k in ['pred_a','pred_b'])
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300)
 torch.set_num_threads(2)
 from fastload import load_model_fast
 model=load_model_fast().eval();assert all(p.device.type=='cpu' for p in model.parameters())
 payload=torch.load(P/'CITY_ATTENTION7_OMISSION_V1_FP32_WRITES.pt',weights_only=True);candidates=payload['writes'];head_sets=payload['head_sets']
 refs=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 writes=[torch.cat([r['inputs']['delta'].double()[None],w.double()]) for w,r in zip(candidates,refs,strict=True)]
 outside=max(float(w[:,:, [i for i in range(len(r['ids'])) if i not in r['destination_positions']]].abs().max()) for w,r in zip(writes,groups))
 ids=torch.zeros(40,max(len(r['ids']) for r in groups),dtype=torch.long)
 for i,row in enumerate(groups):ids[i,:len(row['ids'])]=torch.tensor(row['ids'])
 edit=torch.zeros(40,ids.shape[1],1152);state={}
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 handle=model.transformer.h[8].attn.register_forward_hook(post8)
 values=torch.zeros(13,40,10,dtype=torch.float64);count=0;calls=0;arm_seconds=[]
 try:
  for arm in range(13):
   arm_start=time.perf_counter();state['arm']=arm;edit.zero_()
   for i,row in enumerate(groups):
    if arm==0:continue
    d=writes[i][arm-1].float()
    edit[i,:d.shape[1]]=d[0]
   x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   last=x[torch.arange(40),torch.tensor([len(r['ids'])-1 for r in groups])]
   scores=30*torch.tanh(model.lm_head(F.rms_norm(last,(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
   count+=40;arm_seconds.append(time.perf_counter()-arm_start)
   print(f'arm {arm+1}/13: {arm_seconds[-1]:.3f}s; total {time.perf_counter()-start:.1f}s',flush=True)
 finally:handle.remove()
 v=expand(values,mapping,6);e=v-v[:1];old=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['values']
 diff=v[:2]-old[:2];absolute=float(diff.abs().max());relative=float(diff.norm()/old[:2].norm())
 rms=e.square().mean(1).sqrt();base=v[0,:,0][::2]-v[0,:,0][1::2];cap=base>=.1
 old_cpu=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_CPU_V1_ARTIFACT.pt',weights_only=True)['values']
 full_diff=v[2]-old_cpu[2];full_abs=float(full_diff.abs().max());full_rel=float(full_diff.norm()/old_cpu[2].norm())
 records=[]
 for arm,heads in enumerate(head_sets,start=2):
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  error=float((e[arm,:,0]-e[1,:,0]).norm()/e[1,:,0].norm());control=(rms[arm,1:]/rms[arm,0]).tolist()
  positive=float((atten>0).double().mean());mean=float(atten.mean())
  passes=error<=.05 and positive>=.90 and mean>=.02 and int(cap.sum())>=90 and max(control)<=.5
  records.append({'arm':arm,'heads':heads,'target_error':error,'target_rms':float(rms[arm,0]),'control_over_target':control,'positive_fraction':positive,'mean_attenuation':mean,'passes_screen':passes,'potential_compact_floats':cert['potential_compact_float_counts'][arm-2]})
 eligible=[r for r in records if r['passes_screen'] and len(r['heads'])<9]
 selected=min(eligible,key=lambda r:(len(r['heads']),r['target_error'],r['heads'])) if eligible else None
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and full_abs<=1e-4 and full_rel<=1e-5 and outside==0 and bool(torch.isfinite(v).all()) and count==520 and calls==234 and not torch.cuda.is_initialized(),
    'pred_b':bool(eligible),'pred_c':selected is not None and selected['potential_compact_floats']<cert['potential_compact_float_counts'][0],
    'records':records,'selected':selected,'capable_pairs':int(cap.sum()),'replay_max_abs':absolute,'replay_relative':relative,'full_generator_replay_abs':full_abs,'full_generator_replay_relative':full_rel,'max_outside':outside,
    'body_forwards':count,'batched_block_calls':calls,'arm_seconds':arm_seconds,'native_input_arrays':1,'native_input_scalars_T32':36864,'seconds':time.perf_counter()-start,
    'scope':'Opened CPU omission selection screen inside city-write generator, full native MLP8/suffix. Prices require compact packing; no selected-subset fresh/selectivity-null/composition claim.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
