#!/usr/bin/env python3
# BQGATE:200 sequence-equivalent forwards;90 batched block calls;180seconds;CPU only;no fitting.
"""pred_a replay and instrument gates; pred_b mixed >.5present and all negative
on four registered reversal documents; pred_c self positive>=.90 overall.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_SOURCE7_SELF_CROSS_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_SOURCE7_FRESH_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('200 sequence-equivalents;90 block calls;CPU5 source-product arms');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 cert=json.loads((P/'CITY_SOURCE7_SELF_CROSS_V1_CPU_RESULT.json').read_text())
 assert all(cert[k] for k in ['pred_a','pred_b'])
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(180)
 torch.set_num_threads(2)
 from fastload import load_model_fast
 model=load_model_fast().eval();assert all(p.device.type=='cpu' for p in model.parameters())
 candidates=torch.load(P/'CITY_SOURCE7_SELF_CROSS_V1_WRITES.pt',weights_only=True)['writes']
 writes=[torch.cat([w,(w[1]+w[2])[None]]) for w in candidates]
 outside=max(float(w[:,:, [i for i in range(len(r['ids'])) if i not in r['destination_positions']]].abs().max()) for w,r in zip(writes,groups))
 ids=torch.zeros(40,max(len(r['ids']) for r in groups),dtype=torch.long)
 for i,row in enumerate(groups):ids[i,:len(row['ids'])]=torch.tensor(row['ids'])
 edit=torch.zeros(40,ids.shape[1],1152);state={}
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 handle=model.transformer.h[8].attn.register_forward_hook(post8)
 values=torch.zeros(5,40,10,dtype=torch.float64);count=0;calls=0;arm_seconds=[]
 try:
  for arm in range(5):
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
   print(f'arm {arm+1}/5: {arm_seconds[-1]:.3f}s; total {time.perf_counter()-start:.1f}s',flush=True)
 finally:handle.remove()
 v=expand(values,mapping,6);e=v-v[:1]
 old=torch.load(P/'CITY_SOURCE7_FRESH_V1_ARTIFACT.pt',weights_only=True)['values'][[0,2]]
 diff=v[:2]-old;absolute=float(diff.abs().max());relative=float(diff.norm()/old.norm())
 joint=v[4]-v[1];joint_abs=float(joint.abs().max());joint_rel=float(joint.norm()/v[1].norm())
 base=v[0,:,0][::2]-v[0,:,0][1::2];cap=base>=.1;rms=e.square().mean(1).sqrt();records={}
 for arm,name in [(1,'present'),(2,'self'),(3,'mixed'),(4,'joint')]:
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  records[name]={'target_rms':float(rms[arm,0]),'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean()),'control_over_target':(rms[arm,1:]/rms[arm,0]).tolist()}
 docs=[]
 for context in sorted({r['context_id'] for r in rows}):
  idx=[i for i,r in enumerate(rows) if r['context_id']==context];b=v[0,idx,0][::2]-v[0,idx,0][1::2];capable=b>=.1
  attenuation={}
  for arm,name in [(1,'present'),(2,'self'),(3,'mixed')]:
   pair=v[arm,idx,0][::2]-v[arm,idx,0][1::2];attenuation[name]=((b[capable]-pair[capable])/b[capable]).tolist()
  docs.append({'context_id':context,'mixed_over_present_effect':float(e[3,idx,0].norm()/e[1,idx,0].norm()),'attenuations':attenuation})
 reversed_docs=[d for d in docs if d['context_id'] in [4,8,11,19]]
 interactions={}
 for label,ids_set in [('all',set(range(20))),('reversed',{4,8,11,19}),('other',set(range(20))-{4,8,11,19})]:
  idx=[i for i,r in enumerate(rows) if r['context_id'] in ids_set]
  interactions[label]=float((e[1,idx,0]-e[2,idx,0]-e[3,idx,0]).norm()/min(e[2,idx,0].norm(),e[3,idx,0].norm()))
 r={'pred_a':absolute<=1e-4 and relative<=1e-5 and joint_abs<=1e-4 and joint_rel<=1e-5 and outside==0 and bool(torch.isfinite(v).all()) and count==200 and calls==90 and not torch.cuda.is_initialized(),
    'pred_b':len(reversed_docs)==4 and all(d['mixed_over_present_effect']>.5 and len(d['attenuations']['mixed'])==6 and all(a<0 for a in d['attenuations']['mixed']) for d in reversed_docs),
    'pred_c':int(cap.sum())==120 and records['self']['positive_fraction']>=.90,
    'arms':records,'documents':docs,'interaction_over_smaller':interactions,'capable_pairs':int(cap.sum()),'replay_max_abs':absolute,'replay_relative':relative,'joint_max_abs':joint_abs,'joint_relative':joint_rel,'max_outside':outside,
    'body_forwards':count,'batched_block_calls':calls,'arm_seconds':arm_seconds,'seconds':time.perf_counter()-start,
    'scope':'Opened diagnostic of city-side MLP7 self/mixed terms, full native queries/denominators and suffix. No fresh/null/selectivity or independent composition promotion.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
